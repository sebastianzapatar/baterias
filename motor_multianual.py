"""Motor multianual del proyecto PV + BESS.

Sustituye la aproximación de escalar el beneficio del año 1 por la degradación
del PV. Aquí cada año se despacha con su propio recurso solar, y el beneficio y
el ciclado salen de ese despacho.

Qué hace y qué no
-----------------
* Recorre el horizonte año a año con ``PV_t = PV_0 · (1 − d)^(t−1)``, de modo que
  **el año 1 opera a placa**: el perfil de entrada representa el primer año de
  operación.
* Lleva dos contadores del bloque instalado: el throughput acumulado en ciclos
  equivalentes (EFC) y la edad, ambos continuos.
* Cuando cualquiera de los dos alcanza su límite dispara una
  ``renovacion_bloque_almacenamiento``: un flujo de CAPEX discreto y el reinicio
  de ambos contadores. El número de renovaciones no es un parámetro: sale de los
  contadores.
* **No** aplica desvanecimiento de capacidad del BESS ni valor residual. Los dos
  son cero en el baseline y se dejan como sensibilidades futuras.

Localización intraanual del evento (C3A.1)
------------------------------------------
El cruce del límite de EFC se busca **hora a hora** sobre el despacho del año,
no al cierre anual. Antes se comparaba el EFC anual agregado y se reiniciaba el
contador a cero al terminar el año, lo que borraba todo el throughput ocurrido
entre el cruce y el 31 de diciembre. En el caso de estrés eso perdía 287,4 EFC.

Ahora el throughput del año se reparte: la parte anterior al cruce agota el
bloque saliente y **la parte posterior se asigna al bloque nuevo**, que empieza
el año siguiente con ese saldo. La edad también se reparte: si el evento ocurre
en la fracción ``f`` del año, el bloque saliente envejece ``f`` y el entrante
llega al cierre con ``1 − f``. La lógica admite varios eventos dentro de un
mismo año.

**Aproximación declarada**: el instante físico de la renovación se determina
dentro del año para conservar correctamente EFC y edad, mientras que el
desembolso se registra en el flujo financiero del año correspondiente. No hay
descuento fraccionario intraanual.

Por qué el evento no se llama «fin de vida»
-------------------------------------------
Los 7.200 ciclos son una envolvente de utilización y de vida contractual. No se
conoce el estado de salud asociado, así que el modelo no le atribuye ninguno.

Una propiedad que conviene tener presente
-----------------------------------------
Con desvanecimiento cero y un precio sombra constante, el despacho de un año
depende **solo del factor de escala del PV**, no del historial de renovaciones.
Por eso el beneficio y el EFC de cada año son función únicamente de ``t``, y la
interpolación entre años ancla está bien fundada: lo único delicado es localizar
con exactitud el año en que se cruza el umbral, y para eso el motor no
interpola nunca.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import costos

EVENTO = "renovacion_bloque_almacenamiento"

COLUMNAS = [
    "Anio", "Escala_PV", "Explicito", "Beneficio_COP", "EFC_anio",
    "EFC_acum_antes", "EFC_acum_despues", "Edad_BESS",
    "Renovaciones_en_anio", "Motivo_renovacion", "Costo_renovacion_COP",
]


def _interpolar(t, anclas, valores):
    """Interpolación lineal entre los dos años ancla que rodean a `t`."""
    return float(np.interp(t, anclas, valores))


def _payback(flujos, wacc):
    """Primer año en que el acumulado descontado cruza a positivo."""
    acum = np.cumsum([f / (1 + wacc) ** k for k, f in enumerate(flujos)])
    cruces = np.flatnonzero(acum >= 0)
    return int(cruces[0]) if len(cruces) else np.nan


def simular_horizonte(
    despachar,
    pv_pu,
    load,
    c_imp,
    c_exp,
    estrategia,
    pv_MWp,
    power_MW,
    cap_MWh,
    *,
    DoD,
    eff_rt,
    c_deg_COP_MWh,
    capex_inicial_COP,
    opex_anual_COP,
    horizonte=None,
    wacc=None,
    degradacion_pv=None,
    ciclos_vida=None,
    vida_calendario=None,
    anios_ancla=None,
    escenario_trayectoria="medio",
    trm=None,
    dt=1.0,
    evaluar_anio=None,
):
    """Simula el horizonte completo y devuelve la tabla anual y el flujo de caja.

    `despachar` se recibe por argumento: el motor no reimplementa el despacho.

    `evaluar_anio(t, escala_pv)` es una interfaz opcional de contabilidad completa.
    Devuelve `beneficio_COP`, `efc_h` (perfil horario) y `agregados` (opcional).
    Permite incluir térmica y contrafactuales compatibles sin simular compras
    ficticias de red. Si se omite, conserva la ruta histórica sin térmica.
    El llamador debe validar el despacho y expresar el beneficio en COP reales.

    `anios_ancla` es la lista de años que se simulan siempre. Los demás se
    interpolan, salvo cuando el EFC proyectado del segmento alcanza el límite o
    cuando el contador de edad va a cruzar la vida calendario: esos años se
    simulan explícitamente, porque el evento hay que localizarlo hora a hora.
    """
    horizonte = costos.HORIZONTE_ANIOS if horizonte is None else horizonte
    wacc = costos.WACC if wacc is None else wacc
    degradacion_pv = (costos.DEGRADACION_PV_ANUAL if degradacion_pv is None
                      else degradacion_pv)
    ciclos_vida = costos.CICLOS_VIDA_BESS if ciclos_vida is None else ciclos_vida
    vida_calendario = (costos.VIDA_CALENDARIO_BESS if vida_calendario is None
                       else vida_calendario)
    trm = costos.TRM_COP_USD if trm is None else trm

    tiene_bess = cap_MWh > 0 and power_MW > 0
    util = cap_MWh * DoD if tiene_bess else 1.0

    anclas = sorted(set(anios_ancla or range(1, horizonte + 1)) | {1, horizonte})
    anclas = [a for a in anclas if 1 <= a <= horizonte]

    cache = {}
    n_despachos = 0

    def anio_explicito(t):
        """Despacho real del año `t`. Memoizado: es función solo de `t`.

        Devuelve también el perfil horario de EFC, que es lo que permite
        localizar el cruce dentro del año.
        """
        nonlocal n_despachos
        if t in cache:
            return cache[t]
        escala = (1.0 - degradacion_pv) ** (t - 1)
        if evaluar_anio is not None:
            resultado = evaluar_anio(int(t), float(escala))
            beneficio = float(resultado['beneficio_COP'])
            efc_h = np.asarray(resultado['efc_h'], dtype=float)
            if (efc_h.ndim != 1 or len(efc_h) != len(load)
                    or not np.isfinite(efc_h).all() or (efc_h < 0).any()
                    or not np.isfinite(beneficio)):
                raise ValueError('Contabilidad anual o throughput no válidos')
            if not tiene_bess and np.any(efc_h):
                raise ValueError('No puede existir throughput sin BESS')
            costo_sin, costo_con = beneficio, 0.0
            agr = dict(resultado.get('agregados', {}))
            n_despachos += 1
        else:
            d = despachar(pv_pu * pv_MWp * escala, load, c_imp, c_exp, estrategia,
                          power_MW=power_MW, cap_MWh=cap_MWh, c_deg=c_deg_COP_MWh)
            costo_sin = float((d.Load_MW.to_numpy() * c_imp).sum())
            costo_con = float((d.Import_MW.to_numpy() * c_imp).sum()
                              - (d.Export_MW.to_numpy() * c_exp).sum())
            if tiene_bess:
                efc_h = d.Descarga_MW.to_numpy() * dt / util
            else:
                efc_h = np.zeros(len(d))
            n_despachos += 1
            agr = {
                "Importacion_MWh": float(d.Import_MW.sum()),
                "Exportacion_MWh": float(d.Export_MW.sum()),
                "Carga_BESS_MWh": float(d.Carga_MW.sum()) if "Carga_MW" in d else 0.0,
                "Descarga_BESS_MWh": float(d.Descarga_MW.sum()),
                "Vertimiento_MWh": float(d.Vertimiento_MW.sum()) if "Vertimiento_MW" in d else 0.0,
                "Autoconsumo_pct": (
                    (1.0 - float(d.Red_a_Load_MW.sum()) / float(d.Load_MW.sum())) * 100.0
                    if "Red_a_Load_MW" in d and float(d.Load_MW.sum()) > 0 else np.nan),
            }
        cache[t] = (costo_sin - costo_con, float(efc_h.sum()), escala, efc_h, agr)
        return cache[t]

    for a in anclas:
        anio_explicito(a)
    ben_ancla = [cache[a][0] for a in anclas]
    efc_ancla = [cache[a][1] for a in anclas]
    agr_anio1 = cache[1][4] if 1 in cache else {}

    filas = []
    renovaciones = []
    efc_acum, edad = 0.0, 0.0
    efc_retirado = 0.0          # throughput consumido por bloques ya retirados
    forzar_explicito = set()

    def _renovar(t, motivo, fraccion, efc_antes, edad_antes):
        """Registra el evento y devuelve su costo. No reinicia contadores."""
        costo = costos.costo_renovacion_COP(
            power_MW, cap_MWh, int(t), escenario=escenario_trayectoria, trm=trm)
        renovaciones.append({
            "anio": int(t), "motivo": motivo, "fraccion_anio": float(fraccion),
            "hora_del_anio": float(fraccion) * len(cache[t][3]) if t in cache else np.nan,
            "EFC_bloque_antes": float(efc_antes),
            "edad_bloque_antes": float(edad_antes),
            "costo_COP": float(costo), "anio_indice": int(t),
        })
        return costo

    def procesar(t, explicito):
        """Añade el año `t` a la tabla, localizando los eventos dentro del año."""
        nonlocal efc_acum, edad, efc_retirado
        if explicito:
            ben, efc_total, escala, efc_h, _agr = anio_explicito(t)
        else:
            ben = _interpolar(t, anclas, ben_ancla)
            efc_total = _interpolar(t, anclas, efc_ancla)
            escala = (1.0 - degradacion_pv) ** (t - 1)
            efc_h = None

        antes = efc_acum
        eventos_anio, motivos, costo_anio = 0, [], 0.0
        ultimo_anio = (t == horizonte)

        if not tiene_bess:
            efc_acum, edad = 0.0, edad + 1.0
        elif efc_h is None:
            # Año interpolado. Por construcción del recorrido, ningún evento
            # puede caer aquí: los años donde el EFC o la edad pueden cruzar se
            # fuerzan a explícitos antes de llegar a este punto.
            efc_acum += efc_total
            edad += 1.0
        else:
            n_h = len(efc_h)
            acum_h = np.cumsum(efc_h)

            def _efc_hasta(fraccion):
                """EFC del año acumulado hasta `fraccion`, con resolución
                SUB-HORARIA: la hora en curso se prorratea."""
                x = max(0.0, min(1.0, fraccion)) * n_h
                entero = int(np.floor(x + 1e-12))
                completo = float(acum_h[entero - 1]) if entero > 0 else 0.0
                if entero >= n_h:
                    return float(acum_h[-1])
                return completo + (x - entero) * float(efc_h[entero])

            pos = 0.0                    # fracción del año ya recorrida
            base = 0.0                   # EFC del año ya asignado
            while True:
                # ¿dónde cruza el límite de EFC, contando desde `pos`?
                falta = ciclos_vida - efc_acum
                idx = int(np.searchsorted(acum_h - base, falta, side="left"))
                if idx < n_h and efc_h[idx] > 0:
                    # Reparto exacto DENTRO de la hora del cruce: el bloque
                    # saliente consume justo `falta`, ni un ciclo más. Sin esto
                    # rebasaría el límite por hasta una hora de throughput y la
                    # identidad de conservación no cerraría.
                    previo = float(acum_h[idx - 1] - base) if idx > 0 else 0.0
                    f_efc = (idx + (falta - previo) / float(efc_h[idx])) / n_h
                else:
                    f_efc = np.inf
                # ¿dónde cruza la vida calendario?
                f_cal = pos + (vida_calendario - edad)
                if f_cal > 1.0 + 1e-12:
                    f_cal = np.inf

                f = min(f_efc, f_cal)
                if not np.isfinite(f) or f > 1.0 + 1e-12:
                    break                                  # sin eventos este año
                f = min(f, 1.0)
                motivo = "EFC" if f_efc <= f_cal else "calendario"

                # Al cerrar el horizonte no se renueva: no hay operación después.
                if ultimo_anio and f >= 1.0 - 1e-9:
                    break

                consumido = falta if motivo == "EFC" else (_efc_hasta(f) - base)
                efc_bloque = efc_acum + consumido
                edad_bloque = edad + (f - pos)

                costo_anio += _renovar(t, motivo, f, efc_bloque, edad_bloque)
                efc_retirado += efc_bloque
                eventos_anio += 1
                motivos.append(motivo)

                # El bloque nuevo hereda el throughput y el tiempo restantes.
                base += consumido
                efc_acum, edad = 0.0, 0.0
                pos = f
                if pos >= 1.0 - 1e-12:
                    break

            efc_acum += float(efc_h.sum()) - base
            edad += (1.0 - pos)

        filas.append({
            "Anio": t, "Escala_PV": escala, "Explicito": bool(explicito),
            "Beneficio_COP": ben, "EFC_anio": efc_total,
            # Sin ambigüedad: `antes` es el acumulado del bloque activo AL ENTRAR
            # al año y `despues` el que queda AL CERRARLO. El EFC justo antes de
            # cada evento vive en la lista `renovaciones`.
            "EFC_acum_antes": antes,
            "EFC_acum_despues": efc_acum, "Edad_BESS": edad,
            "Renovaciones_en_anio": eventos_anio,
            "Motivo_renovacion": "+".join(motivos),
            "Costo_renovacion_COP": costo_anio,
        })
        return eventos_anio

    # --- recorrido por segmentos --------------------------------------
    procesar(1, True)
    for k in range(len(anclas) - 1):
        a, b = anclas[k], anclas[k + 1]
        if b <= a:
            continue
        proyectado, critico = efc_acum, False
        for t in range(a + 1, b + 1):
            proyectado += _interpolar(t, anclas, efc_ancla)
            if tiene_bess and proyectado >= ciclos_vida:
                critico = True
                break
        # La edad es determinística: si va a cruzar dentro del segmento, esos
        # años también se simulan, para poder repartir el año del evento.
        # El cruce que cae EXACTAMENTE al cerrar el horizonte no genera evento
        # -no hay operación posterior-, así que no hace falta simularlo: es el
        # caso normal de un bloque que llega íntegro al año 20.
        if tiene_bess:
            cruza = edad + (b - a)
            if cruza > vida_calendario + 1e-12 or (
                    cruza >= vida_calendario - 1e-12 and b < horizonte):
                critico = True
        for t in range(a + 1, b + 1):
            procesar(t, critico or t in anclas or t in forzar_explicito)

    tabla = pd.DataFrame(filas, columns=COLUMNAS)

    flujos = [-float(capex_inicial_COP)]
    for _, f in tabla.iterrows():
        flujos.append(float(f.Beneficio_COP) - float(opex_anual_COP)
                      - float(f.Costo_renovacion_COP))
    # VALOR_RESIDUAL_BASE = 0: no se acredita nada al terminar el horizonte.

    vpn = float(sum(f / (1 + wacc) ** k for k, f in enumerate(flujos)))

    return {
        "tabla": tabla,
        "flujos": flujos,
        "VPN_COP": vpn,
        "TIR": costos.tir(flujos),
        "renovaciones": renovaciones,
        "n_renovaciones": len(renovaciones),
        "EFC_total_descargado": float(tabla.EFC_anio.sum()),
        "EFC_en_bloques_retirados": float(efc_retirado),
        "EFC_bloque_activo_final": float(efc_acum),
        "Edad_BESS_final": float(edad),
        "anio1": agr_anio1,
        "Beneficio_anio1_COP": float(cache[1][0]) if 1 in cache else np.nan,
        "EFC_anio1": float(cache[1][1]) if 1 in cache else np.nan,
        "Costo_renovaciones_nominal_COP": float(
            sum(r["costo_COP"] for r in renovaciones)),
        "Costo_renovaciones_VP_COP": float(sum(
            r["costo_COP"] / (1 + wacc) ** r["anio"] for r in renovaciones)),
        "Anios_renovacion": [r["anio"] for r in renovaciones],
        "Payback_anios": _payback(flujos, wacc),
        "n_despachos": n_despachos,
        "anios_ancla": anclas,
        "anios_explicitos": sorted(cache),
    }
