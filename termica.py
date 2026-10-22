# -*- coding: utf-8 -*-
"""Modelo de integración térmica del usuario industrial: red + cuatro unidades a gas.

Motor del cuaderno 11. Aquí están los parámetros de las unidades, los despachos
y sus validadores; `termica_barridos.py` los ejecuta y cachea a CSV en
`resultados_nb11/`, y el cuaderno solo lee esos CSV.

Casos que resuelve, del más simple al más completo:

    0A       orden de mérito horario, sin mínimos técnicos
    0B       mínimos técnicos, enumeración exacta de los 16 compromisos por hora
    0C       memoria temporal: un arranque y una parada por unidad y día,
             permanencias de 1 o 2 h, programación dinámica exacta
    eventos  0C diario con 10 o 20 minutos sin generar por arranque o parada
    cargo    eventos con un cargo por arranque, uniforme o por MW

Convenciones: potencias en MW, precios en USD/MWh, paso de 1 h, costos en USD.
La demanda y las señales T0/T1 vienen del cuaderno 02, que es la fuente de
verdad, a través de `benchmark_optimo.cargar_entorno_nb02()`.

Estado inicial por defecto **libre**: el año continúa una operación anterior y
la primera hora no cuenta ni cobra arranques. El cierre tampoco fuerza paradas.
"""

from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd

DT = 1.0                     # h
TOL_P = 1e-9                 # MW
# Margen exclusivo de redondeo: 64 épsilon de float64 por escala de potencia.
# Es mucho menor que TOL_P; no se redondean cargas ni se relajan validadores.
TOL_REDONDEO_REL = 64 * np.finfo(float).eps
TOL_COSTO = 1e-9             # USD por hora de 1 h

UNIDADES = ["G1", "G2", "G3", "G4"]
FUENTES = UNIDADES + ["Red"]
SENALES = ["T0", "T1"]
ESCENARIOS_GAS = {"historico": 8.0, "actual": 16.0, "futuro_2033": 20.0}   # USD/MMBtu
GASES = tuple(ESCENARIOS_GAS.values())

# Parámetros declarados por el usuario. La potencia disponible limita el
# despacho; la nominal se conserva como referencia.
PARAMETROS_UNIDADES = [
    # unidad, nominal MW, disponible MW, heat rate MMBtu/MWh, CV adicional USD/MWh, mínimo técnico MW
    ("G1", 10.0, 8.0, 8.0000, 9.28, 2.40),
    ("G2", 10.0, 8.0, 8.0000, 9.28, 2.40),
    ("G3", 7.0, 6.0, 10.0000, 9.28, 1.80),
    ("G4", 4.0, 3.2, 8.0576, 9.28, 0.96),
]

# Referencia documental de fabricante (§16 del cuaderno), consultada el 12/09/2026.
HR_FABRICANTE_BTU_KWH = np.array([9638., 9638., 11089., 12240.])
FECHA_CONSULTA_FABRICANTE = "2026-09-12"
BASE_CALORIFICA_FABRICANTE = ["No explicita en poster"] * 3 + ["PCI/LHV"]

# Sensibilidad del cargo por arranque (§26). Parámetros hipotéticos, no costos reales.
MINUTOS_EVENTO = (0, 10, 20)
TARIFAS_ARRANQUE = [0., 1., 5., 10., 25., 100., 250., 1000.]   # USD por arranque
CARGOS_POR_MW = [0., 1., 5., 10., 25., 100.]                    # USD por MW disponible por arranque


def unidades_historicas():
    """Tabla de las cuatro unidades con los HR históricos."""
    return pd.DataFrame(PARAMETROS_UNIDADES,
                        columns=["Unidad", "P_nom_MW", "P_disp_MW", "HeatRate_MMBtu_MWh",
                                 "CV_adicional_USD_MWh", "P_min_MW"]).set_index("Unidad")


def costo_marginal(gen, precio_gas_usd):
    """CV_g = HeatRate_g * PrecioGas + CV_adicional_g, en USD/MWh."""
    return gen.HeatRate_MMBtu_MWh * precio_gas_usd + gen.CV_adicional_USD_MWh


def unidades_fabricante(gen):
    """Tabla anonimizada con los HR de fabricante; demás parámetros sin cambio."""
    fab = gen.drop(columns=["P_nom_MW"]).copy(deep=True)
    fab["Origen_HR"] = "Referencia documental de fabricante; identidad omitida por confidencialidad"
    fab["HR_publicado_Btu_kWh"] = HR_FABRICANTE_BTU_KWH
    fab["HeatRate_MMBtu_MWh"] = fab.HR_publicado_Btu_kWh / 1000
    fab["CV_USD_MWh"] = costo_marginal(fab, 16.)
    fab["Costo_combustible_USD_MWh"] = fab.HeatRate_MMBtu_MWh * 16.
    fab["Fecha_consulta"] = FECHA_CONSULTA_FABRICANTE
    fab["Base_calorifica_documento"] = BASE_CALORIFICA_FABRICANTE
    fab["Convencion_precio_gas"] = "Misma base energetica que HR; facturacion real no verificada"
    fab["Calibrado_sitio"] = False
    fab["HR_constante"] = True
    return fab


TOL_SENAL_COP_MWH = 1e-6  # misma aritmética en ambas construcciones; diferencia observada ~3,5e-10


def cargar_entradas(version="corregida"):
    """Demanda y señales T0/T1 en USD/MWh para la línea térmica.

    Fuente canónica de `C_IMP_USD`: `tarifas_termica_v2.construir` (T0 con punta
    N3 [06:00, 21:00) y T1 = bolsa + margen a ingreso neutro), a partir de la
    demanda y la bolsa del cuaderno 02. Es la representación con la que se
    calcularon y firmaron las trayectorias guardadas de nb11–nb13: la huella de
    `huella_entradas` compara bytes, y la señal del cuaderno 02 difiere en
    ~3,5e−10 COP/MWh por el orden de las operaciones. Usarla invalidaba cachés
    válidas (incidencia del 13/09/2026, `CORRECCION_TARIFAS.md`).

    El cuaderno 02 sigue siendo la fuente de verdad de la tarifa y se comprueba
    de forma explícita: si alguna señal o el margen difieren de la construcción
    en más de `TOL_SENAL_COP_MWH`, se lanza ValueError y no se continúa.
    """
    if version != "corregida":
        raise ValueError("Versión tarifaria desconocida")
    import benchmark_optimo as BO
    import costos
    from tarifas_termica_v2 import construir, VERSION
    ns = BO.cargar_entorno_nb02()
    tarifas = ns["TARIFAS"]
    nombre = {k[:2]: k for k in tarifas}
    trm = costos.TRM_COP_USD
    fechas = pd.DatetimeIndex(ns["datos"]["Fecha"])
    load = np.asarray(ns["LOAD"], dtype=float)
    bolsa = ns["datos"]["Bolsa_COP_kWh"].to_numpy(dtype=float)*1000
    control = construir(fechas, load, bolsa)
    for s in SENALES:
        fuente = np.asarray(tarifas[nombre[s]], dtype=float)
        dif = float(np.max(np.abs(fuente - control[s])))
        if not dif <= TOL_SENAL_COP_MWH:
            raise ValueError(
                f"La señal {s} del cuaderno 02 difiere de la reconstruida por tarifas_termica_v2.construir "
                f"hasta en {dif:.6g} COP/MWh (tolerancia {TOL_SENAL_COP_MWH}). Revisar la tarifa del cuaderno 02 "
                f"o tarifas_termica_v2 ({VERSION}); no se sustituye la fuente.")
    if abs(float(ns["margen_COP_MWh"]) - control["Margen_COP_MWh"]) > TOL_SENAL_COP_MWH:
        raise ValueError("El margen de T1 del cuaderno 02 difiere del reconstruido por tarifas_termica_v2.construir")
    return dict(LOAD=load, FECHA=fechas, TARIFAS=tarifas, NOMBRE=nombre, TRM=trm,
                C_IMP_USD={s: np.asarray(control[s], dtype=float) / trm for s in SENALES},
                BOLSA_COP_MWH=bolsa, VERSION_TARIFA=VERSION,
                MARGEN_COP_MWH=float(ns["margen_COP_MWh"]), PUNTA=control["Punta"])


# ============================================================== 0A · mérito

def despachar_merito(load, c_imp_usd, gen):
    """Orden de mérito sin binarias.

    Devuelve (despacho, costos horarios, orden, capacidades ordenadas, despacho
    ordenado). La capacidad de la red se fija en la demanda de cada hora: es el
    máximo que puede necesitarse y evita operar con infinitos en el acumulado.
    """
    fuentes = list(gen.index) + ["Red"]
    n = len(load)
    costos_h = np.column_stack(
        [np.full(n, float(gen.loc[u, "CV_USD_MWh"])) for u in gen.index]
        + [np.asarray(c_imp_usd, dtype=float)])
    caps_h = np.column_stack(
        [np.full(n, float(gen.loc[u, "P_disp_MW"])) for u in gen.index]
        + [np.asarray(load, dtype=float)])
    orden = np.argsort(costos_h, axis=1, kind="stable")
    caps_ord = np.take_along_axis(caps_h, orden, axis=1)
    acum_previo = np.cumsum(caps_ord, axis=1) - caps_ord
    restante = np.maximum(0.0, np.asarray(load)[:, None] - acum_previo)
    desp_ord = np.minimum(caps_ord, restante)
    desp = np.empty_like(desp_ord)
    np.put_along_axis(desp, orden, desp_ord, axis=1)
    return (pd.DataFrame(desp, columns=fuentes), pd.DataFrame(costos_h, columns=fuentes),
            orden, caps_ord, desp_ord)


def validar_merito(despacho, costo_h, orden, caps_ord, desp_ord, load, gen, tol=1e-9):
    """Las ocho validaciones del 0A. Devuelve la lista de fallos; vacía = correcto."""
    d = despacho
    fallos = []
    total = d.sum(axis=1).to_numpy()
    p_disp_total = float(gen.P_disp_MW.sum())
    res_bal = np.abs(total - load).max()
    if res_bal > tol:
        fallos.append("balance horario, residuo {:.2e} MW".format(res_bal))
    for u in gen.index:
        exceso = d[u].max() - gen.loc[u, "P_disp_MW"]
        if exceso > tol:
            fallos.append("{} supera su potencia disponible en {:.2e} MW".format(u, exceso))
    if (d.to_numpy() < -tol).any():
        fallos.append("hay potencias negativas")
    if d["Red"].min() < -tol:
        fallos.append("importacion de red negativa")
    term = d[list(gen.index)].sum(axis=1).to_numpy()
    if term.max() - p_disp_total > tol:
        fallos.append("termico total {:.4f} MW supera {:.1f} MW".format(term.max(), p_disp_total))
    if (total - load).max() > tol:
        fallos.append("hay exportacion: el despacho excede la demanda")
    # Si una fuente despacha, todas las estrictamente más baratas de esa hora
    # tienen que estar saturadas.
    co = costo_h.to_numpy()
    co_ord = np.take_along_axis(co, orden, axis=1)
    for k in range(d.shape[1] - 1):
        mas_barata_estricta = co_ord[:, k] < co_ord[:, k + 1] - 1e-12
        usa_la_cara = desp_ord[:, k + 1] > tol
        holgura = caps_ord[:, k] - desp_ord[:, k]
        malo = mas_barata_estricta & usa_la_cara & (holgura > 1e-6)
        if malo.any():
            fallos.append("orden de merito violado en {} horas (posicion {})".format(int(malo.sum()), k))
    e_fuentes = float(d.sum().sum()) * DT
    e_demanda = float(np.sum(load)) * DT
    if abs(e_fuentes - e_demanda) > 1e-6:
        fallos.append("energia anual {:.6f} vs demanda {:.6f} MWh".format(e_fuentes, e_demanda))
    return fallos


def kpis_merito(despacho, gen, gas_usd, c_imp_usd, tol=1e-9):
    """Indicadores anuales del 0A. Energías en MWh, costos en USD."""
    d = despacho
    unidades = list(gen.index)
    n = len(d)
    e = d.sum() * DT
    e_term = float(e[unidades].sum())
    e_red = float(e["Red"])
    e_total = e_term + e_red
    gas = e[unidades] * gen.HeatRate_MMBtu_MWh
    combustible = gas * gas_usd
    adicional = e[unidades] * gen.CV_adicional_USD_MWh
    costo_red = float((d["Red"].to_numpy() * c_imp_usd).sum() * DT)
    por_unidad = pd.DataFrame({
        "Energia_MWh": e[unidades],
        "Participacion_pct": e[unidades] / e_total * 100,
        "Horas_operacion": (d[unidades] > tol).sum(),
        "Factor_planta_pct": e[unidades] / (gen.P_disp_MW * n * DT) * 100,
        "Gas_MMBtu": gas,
        "Combustible_USD": combustible,
        "CV_adicional_USD": adicional,
        "Costo_total_USD": combustible + adicional,
    })
    resumen = {
        "Energia_termica_MWh": e_term,
        "Participacion_termica_pct": e_term / e_total * 100,
        "Energia_red_MWh": e_red,
        "Participacion_red_pct": e_red / e_total * 100,
        "Gas_total_MMBtu": float(gas.sum()),
        "Costo_combustible_USD": float(combustible.sum()),
        "Costo_variable_adicional_USD": float(adicional.sum()),
        "Costo_total_termico_USD": float(combustible.sum() + adicional.sum()),
        "Costo_compras_red_USD": costo_red,
        "Costo_total_abastecimiento_USD": float(combustible.sum() + adicional.sum()) + costo_red,
    }
    resumen["Costo_medio_USD_MWh"] = resumen["Costo_total_abastecimiento_USD"] / e_total
    return por_unidad, pd.Series(resumen)


# ============================================================== 0B · mínimos

def enumerar_termico(demanda, precio_red, cv, pmin, pmax, *, permitir_red=True):
    """Óptimo de horas independientes; arrays en MW y USD/MWh, paso 1 h.

    Retorna potencias (cuatro unidades y red), compromiso, costo seleccionado,
    costos de los 16 estados y máscara de factibilidad. No modifica los
    argumentos. permitir_red=False se usa solo para el contrafactual térmico:
    una hora no factible conserva costo infinito.

    Desempate: costos iguales dentro de TOL_COSTO se resuelven maximizando
    lexicográficamente el despacho G1, G2, G3, G4; si coincide, menos unidades
    comprometidas.
    """
    carga = np.atleast_1d(np.asarray(demanda, dtype=float))
    red = np.broadcast_to(np.asarray(precio_red, dtype=float), carga.shape)
    cv, pmin, pmax = [np.asarray(v, dtype=float) for v in (cv, pmin, pmax)]
    if carga.ndim != 1 or any(v.shape != (4,) for v in (cv, pmin, pmax)):
        raise ValueError('Se requieren series unidimensionales y cuatro unidades')
    if not all(np.isfinite(v).all() for v in (carga, red, cv, pmin, pmax)):
        raise ValueError('Entradas no finitas')
    if (carga < 0).any() or (pmin < 0).any() or (pmax < pmin).any():
        raise ValueError('Demanda o límites físicos inválidos')
    estados = np.array(sorted(product((0, 1), repeat=4),
                              key=lambda u: (sum(u), tuple(-x for x in u))), dtype=int)
    n = len(carga)
    costos_h = np.column_stack([np.broadcast_to(cv, (n, 4)), red])
    orden = np.argsort(costos_h, axis=1, kind='stable')
    candidatos = np.full((n, 16), np.inf)
    factibles = np.zeros((n, 16), dtype=bool)
    mejor = np.full(n, np.inf)
    potencia = np.zeros((n, 5))
    elegido = np.zeros(n, dtype=int)
    for k, u in enumerate(estados):
        base = pmin * u
        residual = carga - base.sum()
        tol = TOL_REDONDEO_REL * np.maximum(1., np.maximum(np.abs(carga), np.abs(base).sum()))
        factible = residual >= -tol
        if not permitir_red:
            factible &= carga <= np.sum(pmax*u)
        factibles[:, k] = factible
        r = np.maximum(residual, 0.0)
        capacidades = np.column_stack([np.broadcast_to((pmax-pmin)*u, (n, 4)),
                                       r if permitir_red else np.zeros(n)])
        cap_ord = np.take_along_axis(capacidades, orden, axis=1)
        previo = np.cumsum(cap_ord, axis=1) - cap_ord
        incremental_ord = np.minimum(cap_ord, np.maximum(r[:, None]-previo, 0.0))
        despacho = np.zeros((n, 5))
        np.put_along_axis(despacho, orden, incremental_ord, axis=1)
        despacho[:, :4] += base
        costo = np.sum(despacho * costos_h, axis=1)
        candidatos[factible, k] = costo[factible]
        # Comparación lexicográfica de potencias sin costos artificiales.
        lex_mayor = np.zeros(n, dtype=bool)
        iguales = np.ones(n, dtype=bool)
        for j in range(4):
            lex_mayor |= iguales & (despacho[:, j] > potencia[:, j] + TOL_P)
            iguales &= np.abs(despacho[:, j]-potencia[:, j]) <= TOL_P
        empate = np.abs(costo-mejor) <= TOL_COSTO
        tomar = factible & ((costo < mejor-TOL_COSTO) | (empate & lex_mayor))
        potencia[tomar] = despacho[tomar]
        mejor[tomar] = costo[tomar]
        elegido[tomar] = k
    return dict(potencia=potencia, compromiso=estados[elegido], costo=mejor,
                costos_estados=candidatos, factibles=factibles,
                estado=elegido, estados=estados)


def despachar_hora(demanda, precio_red, cv, pmin, pmax):
    """Interfaz escalar del algoritmo exacto para una hora de 1 h."""
    r = enumerar_termico([demanda], precio_red, cv, pmin, pmax)
    return {k: (v[0].copy() if hasattr(v[0], 'copy') else v[0])
            for k, v in r.items() if k != 'estados'}


def validar_enumeracion(r, carga, red, cv, pmin, pmax):
    p, u = r['potencia'], r['compromiso']
    assert np.isfinite(p).all() and np.isfinite(r['costo']).all()
    assert np.isin(u, [0, 1]).all()
    residuo = float(np.max(np.abs(p.sum(axis=1)-carga)))
    assert residuo <= TOL_P, ('Balance', residuo)
    assert (p >= -TOL_P).all()  # incluye generación y red
    assert (p[:, :4] <= pmax + TOL_P).all()
    assert (np.abs(p[:, :4][u == 0]) <= TOL_P).all()
    assert (p[:, :4] >= pmin*u - TOL_P).all()
    assert (p[:, :4].sum(axis=1) <= pmax.sum()+TOL_P).all()
    assert (p.sum(axis=1) <= carga+TOL_P).all()  # no exportación ni vertimiento
    assert abs(p.sum()-np.sum(carga)) <= 1e-6
    assert r['factibles'].any(axis=1).all()
    assert r['factibles'][np.arange(len(carga)), r['estado']].all()
    brecha = r['costo'][:, None]-r['costos_estados']
    assert (brecha[r['factibles']] <= TOL_COSTO).all()
    recalculado = (p[:, :4]*cv).sum(axis=1)+p[:, 4]*red
    np.testing.assert_allclose(r['costo'], recalculado, rtol=0, atol=TOL_COSTO)
    return dict(Residuo_balance_max_MW=residuo,
                Brecha_optimo_max_USD_h=float(np.max(brecha[r['factibles']])),
                Diferencia_energia_MWh=float(p.sum()-np.sum(carga)))


def indicadores(p, gas, precio_red, carga, heat_rate, cvom):
    """Indicadores anuales de una trayectoria (cuatro unidades y red)."""
    hr = np.asarray(heat_rate, dtype=float)
    cv = hr*gas+cvom
    energia = p.sum(axis=0)*DT
    gas_total = float(energia[:4]@hr)
    costo_termico = float(energia[:4]@cv)
    costo_red = float(p[:, 4]@precio_red*DT)
    out = dict(Costo_total_USD=costo_termico+costo_red,
               Costo_medio_USD_MWh=(costo_termico+costo_red)/(carga.sum()*DT),
               Energia_termica_MWh=energia[:4].sum(), Energia_red_MWh=energia[4],
               Participacion_termica_pct=100*energia[:4].sum()/carga.sum()/DT,
               Participacion_red_pct=100*energia[4]/carga.sum()/DT,
               Gas_total_MMBtu=gas_total, Energia_G1_G2_MWh=energia[0]+energia[1],
               Energia_G4_MWh=energia[3], Energia_G3_MWh=energia[2],
               Costo_termico_USD=costo_termico, Costo_red_USD=costo_red)
    out.update({f'Horas_operacion_{u}': int((p[:, j] > TOL_P).sum())
                for j, u in enumerate(UNIDADES)})
    return out


def bloques(p):
    """Columnas G1+G2, G4, G3 y red: G1 y G2 son económicamente idénticas."""
    return np.column_stack([p[:, 0]+p[:, 1], p[:, 3], p[:, 2], p[:, 4]])


# ============================================================== 0C · compromiso temporal

def resolver_compromiso_temporal(costos_estados, estados, fechas, minimo_h=1,
                                 limite_diario=True, inicial='libre'):
    """DP exacta del modelo horario, con costos fijos por compromiso.

    Estado local: (encendido, duración saturada, arranques hoy, paradas hoy).
    Las transiciones se factorizan por unidad; el costo horario acopla las cuatro.
    Inicial 'libre': la primera hora toma cualquier compromiso sin evento y con
    permanencias cumplidas. Con un vector de cuatro binarios, ese es el estado
    previo y los cambios de la primera hora cuentan. Terminal: permanencia
    cumplida, sin exigir apagado ni inventar una parada fuera del horizonte.
    Desempate: menos arranques, sin penalización monetaria artificial.

    Solo admite permanencias de 1 o 2 h: con 3 h el espacio llega a 18**4
    estados, más de lo que indexa uint16.
    """
    c = np.asarray(costos_estados, float)
    estados = np.asarray(estados, int)
    fechas = pd.DatetimeIndex(fechas)
    m = int(minimo_h)
    if m != minimo_h or m < 1 or m > 2:
        raise ValueError('Esta implementación auditada admite permanencias de 1 o 2 horas')
    if c.shape != (len(fechas), 16) or estados.shape != (16, 4):
        raise ValueError('Se requieren dieciséis compromisos de cuatro unidades')
    if np.isnan(c).any() or np.isneginf(c).any() or not np.isfinite(c).any(axis=1).all():
        raise ValueError('Costos inválidos o una hora sin compromiso factible')
    if len(fechas) < 1 or not fechas.is_unique or not fechas.is_monotonic_increasing:
        raise ValueError('Calendario inválido')
    if not ((fechas[1:] - fechas[:-1]) == pd.Timedelta(hours=1)).all():
        raise ValueError('Se requiere paso horario continuo')
    libre = isinstance(inicial, str)
    if libre and inicial != 'libre':
        raise ValueError("inicial debe ser 'libre' o un vector de cuatro binarios")
    if not libre:
        inicial = np.asarray(inicial, int)
        if inicial.shape != (4,) or not np.isin(inicial, [0, 1]).all():
            raise ValueError('Estado inicial inválido')
    # Combinaciones compatibles con la alternancia de arranques/paradas.
    locales = [(u, edad, a, b) for u in (0, 1) for edad in range(1, m+1)
               for a, b in ([(0, 0), (1, 0), (1, 1)] if u else [(0, 0), (0, 1), (1, 1)])]
    if not limite_diario:
        locales = [(u, edad, 0, 0) for u in (0, 1) for edad in range(1, m+1)]
    lookup = {v: i for i, v in enumerate(locales)}
    n = len(locales)
    shape = (n,)*4
    total = n**4
    coords = np.array(np.unravel_index(np.arange(total), shape)).T
    u_global = np.asarray(locales)[coords, 0]
    codigo_estado = {tuple(u): k for k, u in enumerate(estados)}
    indices_estado = np.array([codigo_estado[tuple(u)] for u in u_global])
    terminal = (np.asarray(locales)[coords, 1] == m).all(axis=1)
    ident = np.arange(total, dtype=np.uint16).reshape(shape)
    # Mapas de predecesores por eje: no se construye una matriz global cuadrática.
    mapas = {}
    for nuevo_dia in (False, True):
        entradas = [[] for _ in locales]
        for i, (u, edad, a, b) in enumerate(locales):
            if nuevo_dia:
                a = b = 0
            for v in (u, 1-u):
                if v != u and edad < m:
                    continue
                arr, par = int(v > u), int(v < u)
                aa, bb = (a+arr, b+par) if limite_diario else (0, 0)
                if aa > 1 or bb > 1:
                    continue
                destino = (v, min(edad+1, m) if v == u else 1, aa, bb)
                entradas[lookup[destino]].append((i, arr))
        ancho = max(map(len, entradas))
        prev = np.zeros((ancho, n), int)
        starts = np.zeros((ancho, n), int)
        valid = np.zeros((ancho, n), bool)
        for j, opciones in enumerate(entradas):
            for k, (i, arr) in enumerate(opciones):
                prev[k, j], starts[k, j], valid[k, j] = i, arr, True
        mapas[nuevo_dia] = prev, starts, valid
    valor = np.full(shape, np.inf)
    n_arr = np.full(shape, 10**8, np.int32)
    # Hasta 363 MB con permanencia de dos horas; se libera en cada corrida.
    historia = np.empty((len(c), total), np.uint16)
    dias = fechas.normalize()
    if libre:
        # La primera hora toma cualquier compromiso, sin evento ni contador diario
        # y con permanencias cumplidas: el año continúa una operación anterior.
        for k, u in enumerate(estados):
            idx = tuple(lookup[(int(x), m, 0, 0)] for x in u)
            valor[idx], n_arr[idx] = c[0, k], 0
        historia[0] = ident.ravel()
        primera = 1
    else:
        inicio = tuple(lookup[(int(u), m, 0, 0)] for u in inicial)
        valor[inicio], n_arr[inicio] = 0., 0
        primera = 0
    for t in range(primera, len(c)):
        nuevo_dia = t == 0 or dias[t] != dias[t-1]
        prev, starts, valid = mapas[nuevo_dia]
        origen = ident
        for eje in range(4):
            mejor = np.full(shape, np.inf)
            mejor_arr = np.full(shape, 10**8, np.int32)
            mejor_orig = np.zeros(shape, np.uint16)
            dims = [1]*4
            dims[eje] = n
            for k in range(len(prev)):
                candidato = np.take(valor, prev[k], axis=eje)
                arr = np.take(n_arr, prev[k], axis=eje) + starts[k].reshape(dims)
                tomar = valid[k].reshape(dims) & ((candidato < mejor) |
                    ((candidato == mejor) & (arr < mejor_arr)))
                np.copyto(mejor, candidato, where=tomar)
                np.copyto(mejor_arr, arr, where=tomar)
                np.copyto(mejor_orig, np.take(origen, prev[k], axis=eje), where=tomar)
            valor, n_arr, origen = mejor, mejor_arr, mejor_orig
        valor = valor + c[t, indices_estado].reshape(shape)
        historia[t] = origen.ravel()
    finales = np.where(terminal, valor.ravel(), np.inf)
    optimo = finales.min()
    if not np.isfinite(optimo):
        raise ValueError('No existe trayectoria temporal factible')
    candidatos = np.flatnonzero(finales == optimo)
    actual = candidatos[np.argmin(n_arr.ravel()[candidatos])]
    seleccion = np.empty(len(c), int)
    for t in range(len(c)-1, -1, -1):
        seleccion[t] = indices_estado[actual]
        actual = int(historia[t, actual])
    if libre:
        local0 = np.asarray(locales)[coords[actual]]
        assert (local0[:, 1] == m).all() and not local0[:, 2:].any()
        inicial = estados[seleccion[0]]
    else:
        assert actual == np.ravel_multi_index(inicio, shape)
    costos = c[np.arange(len(c)), seleccion]
    np.testing.assert_allclose(costos.sum(), optimo, rtol=0, atol=1e-5)
    return dict(estado=seleccion, compromiso=estados[seleccion], costo=costos,
                costo_dp=float(optimo), estados_dp=total, inicial=np.array(inicial, int),
                inicial_libre=libre, minimo_h=m, limite_diario=limite_diario)


def potencia_compromiso(carga, red, cv, pmin, pmax, compromiso):
    """Despacho económico condicionado a la trayectoria seleccionada."""
    carga, red = np.asarray(carga), np.asarray(red)
    base = compromiso*np.asarray(pmin)
    residual = carga-base.sum(axis=1)
    assert (residual >= -1e-9).all()
    costos = np.column_stack([np.broadcast_to(cv, base.shape), red])
    orden = np.argsort(costos, axis=1, kind='stable')
    cap = np.column_stack([compromiso*(np.asarray(pmax)-pmin), np.maximum(residual, 0)])
    cap_ord = np.take_along_axis(cap, orden, axis=1)
    antes = np.cumsum(cap_ord, axis=1)-cap_ord
    inc = np.minimum(cap_ord, np.maximum(residual[:, None]-antes, 0))
    p = np.zeros_like(cap)
    np.put_along_axis(p, orden, inc, axis=1)
    p[:, :4] += base
    return p


def rearranques_tras_demanda_baja(compromiso, carga, pmin, inicial=None):
    """Diagnóstico descriptivo: rearranques tras horas de demanda menor que Pmin, por unidad.

    Sin exportación, una demanda menor que el Pmin de una unidad la obliga a
    **estar apagada** en esa hora. Eso solo obliga a una **parada** si estaba
    encendida en la hora anterior, y **no obliga a volver a encenderla**: la red
    puede cubrir la demanda después. Se cuentan los arranques cuyo periodo
    apagado previo sigue a una operación y contiene al menos una hora con
    demanda menor que el Pmin de la unidad.

    El conteo depende de la trayectoria elegida y un cargo por arranque
    suficientemente alto puede eliminarlo. No es una cota de arranques
    inevitables ni se resta de los arranques para medir reducciones.
    """
    u = np.asarray(compromiso, int)
    inicial = u[0] if inicial is None else np.asarray(inicial, int)
    imposible = np.asarray(carga, float)[:, None] < np.asarray(pmin, float)[None, :]
    previo = np.vstack([inicial, u[:-1]])
    forzados = np.zeros(u.shape[1], int)
    for j in range(u.shape[1]):
        ya_encendida, vio = bool(inicial[j]), False
        for t in np.flatnonzero((u[:, j] != previo[:, j]) | imposible[:, j]):
            if u[t, j] and not previo[t, j]:
                forzados[j] += int(ya_encendida and vio)
                ya_encendida, vio = True, False
                continue
            if previo[t, j] and not u[t, j]:
                ya_encendida, vio = True, False
            if imposible[t, j] and ya_encendida:
                vio = True
    return forzados


def metricas_ciclado(compromiso, fechas, inicial=None, carga=None, pmin=None):
    """Eventos al inicio de la hora; rachas finales se reportan censuradas.

    inicial=None equivale a estado inicial libre: la primera hora no es evento.
    Con carga y pmin se agrega el diagnóstico de rearranques tras demanda baja.
    """
    u = np.asarray(compromiso, int)
    inicial = u[0].copy() if inicial is None else np.asarray(inicial, int)
    delta = np.diff(np.vstack([inicial, u]), axis=0)
    arr, par = (delta == 1).astype(int), (delta == -1).astype(int)
    dias = pd.DatetimeIndex(fechas).normalize()
    diario = pd.DataFrame({**{f'Arranques_G{j+1}': arr[:, j] for j in range(4)},
                           **{f'Paradas_G{j+1}': par[:, j] for j in range(4)}}, index=dias).groupby(level=0).sum()
    filas, rachas = [], []
    for j in range(4):
        cambios = np.r_[0, np.flatnonzero(np.diff(u[:, j]))+1, len(u)]
        for a, b in zip(cambios[:-1], cambios[1:]):
            rachas.append(dict(Unidad=f'G{j+1}', Encendida=int(u[a, j]), Inicio=fechas[a],
                Fin_exclusivo=fechas[b-1]+pd.Timedelta(hours=1), Duracion_h=int(b-a),
                Censura_izquierda=bool(a == 0 and u[0, j] == inicial[j]),
                Censura_derecha=bool(b == len(u))))
        rr = [r for r in rachas if r['Unidad'] == f'G{j+1}' and r['Encendida'] == 1
              and not r['Censura_izquierda'] and not r['Censura_derecha']]
        filas.append(dict(Unidad=f'G{j+1}', Horas_operacion=int(u[:, j].sum()),
            Arranques=int(arr[:, j].sum()), Arranques_internos=int(arr[1:, j].sum()),
            Arranque_inicial=int(arr[0, j]), Paradas=int(par[:, j].sum()),
            Max_arranques_dia=int(diario[f'Arranques_G{j+1}'].max()),
            Max_paradas_dia=int(diario[f'Paradas_G{j+1}'].max()),
            Dias_mas_un_arranque=int((diario[f'Arranques_G{j+1}'] > 1).sum()),
            Dias_mas_una_parada=int((diario[f'Paradas_G{j+1}'] > 1).sum()),
            Max_arranques_24h=int(pd.Series(arr[:, j]).rolling(24, min_periods=1).sum().max()),
            Encendidos_completos_1h=sum(r['Duracion_h'] == 1 for r in rr),
            Min_encendido_completo_h=min((r['Duracion_h'] for r in rr), default=np.nan)))
    tabla = pd.DataFrame(filas)
    if carga is not None:
        tabla['Rearranques_tras_demanda_baja'] = rearranques_tras_demanda_baja(u, carga, pmin, inicial)
    return tabla, diario.rename_axis('Dia').reset_index(), pd.DataFrame(rachas)


def validar_temporal(r, carga, red, cv, pmin, pmax, fechas):
    u, p = r['compromiso'], r['potencia']
    assert np.isfinite(p).all() and (p >= -1e-9).all()
    assert (p[:, :4] >= u*pmin-1e-9).all() and (p[:, :4] <= u*pmax+1e-9).all()
    residuo = np.max(np.abs(p.sum(axis=1)-carga))
    assert residuo < 1e-9
    np.testing.assert_allclose((p[:, :4]*cv).sum(axis=1)+p[:, 4]*red, r['costo'], rtol=0, atol=1e-8)
    filas, diario, rachas = metricas_ciclado(u, fechas, r['inicial'])
    if r['limite_diario']:
        assert (diario.filter(regex='Arranques|Paradas').to_numpy() <= 1).all()
    # Incluye la racha final: no se aceptan permanencias pendientes al cerrar el año.
    assert (rachas.loc[~rachas.Censura_izquierda, 'Duracion_h'] >= r['minimo_h']).all()
    return dict(Estado='OK', Residuo_balance_max_MW=float(residuo),
                Error_costo_dp_USD=float(abs(r['costo'].sum()-r['costo_dp'])))


def contraste_milp_compromiso(carga, red, cv, pmin, pmax, fechas, minimo_h=1, minutos=0,
                              costo_arranque=0., limite_diario=True, inicial_libre=True):
    """Contraste independiente: formulación de potencia/u/arranque/parada con HiGHS.

    Variables por hora: cuatro potencias, red, cuatro u, cuatro arranques y y
    cuatro paradas z. Con inicial_libre la primera hora no tiene evento ni enlace
    con un estado previo; si no, el estado previo es apagado. El despacho continuo
    se recalcula con las binarias fijadas, para no confundir tolerancias de
    factibilidad del MILP con diferencias de costo. Solo para horizontes cortos.
    """
    from scipy.optimize import milp, linprog, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix, vstack
    tau = minutos/60.
    n = len(carga)
    nv = 17*n
    obj = np.tile(np.r_[cv, 0., np.zeros(12)], n)
    obj[np.arange(n)*17+4] = red
    cargos = np.broadcast_to(np.asarray(costo_arranque, float), (4,))
    lo = np.zeros(nv); hi = np.full(nv, np.inf)
    integer = np.zeros(nv, int)
    for t in range(n):
        obj[17*t+9:17*t+13] = cargos
        hi[17*t+5:17*t+17] = 1
        integer[17*t+5:17*t+17] = 1
    filas, lbs, ubs = [], [], []

    def add(items, lower=-np.inf, upper=np.inf):
        filas.append(items); lbs.append(lower); ubs.append(upper)

    for t in range(n):
        add({17*t+j: 1 for j in range(5)}, carga[t], carga[t])
        for j in range(4):
            p, u, y, z = 17*t+j, 17*t+5+j, 17*t+9+j, 17*t+13+j
            add({p: 1, u: -pmax[j]}, upper=0)
            if tau:
                disponibilidad = {p: 1, u: -pmax[j], y: tau*pmax[j]}
                if t+1 < n:
                    disponibilidad[17*(t+1)+13+j] = tau*pmax[j]
                add(disponibilidad, upper=0)
            add({p: -1, u: pmin[j]}, upper=0)
            if t == 0 and inicial_libre:
                hi[y] = hi[z] = 0
            else:
                eq = {u: 1, y: -1, z: 1}
                if t:
                    eq[17*(t-1)+5+j] = -1
                add(eq, 0, 0)
            add({y: 1, z: 1}, upper=1)
            up = {17*k+9+j: 1 for k in range(max(0, t-minimo_h+1), t+1)}
            up[u] = -1
            add(up, upper=0)
            down = {17*k+13+j: 1 for k in range(max(0, t-minimo_h+1), t+1)}
            down[u] = 1
            add(down, upper=1)
            if t > n-minimo_h:
                hi[y] = hi[z] = 0
    if limite_diario:
        dias = pd.DatetimeIndex(fechas).normalize()
        for dia in dias.unique():
            indices = np.flatnonzero(dias == dia)
            for j in range(4):
                for offset in (9, 13):
                    add({17*t+offset+j: 1 for t in indices}, upper=1)
    a = lil_matrix((len(filas), nv))
    for i, row in enumerate(filas):
        for j, v in row.items():
            a[i, j] = v
    res = milp(obj, integrality=integer, bounds=Bounds(lo, hi),
               constraints=LinearConstraint(a.tocsr(), lbs, ubs),
               options={'mip_rel_gap': 0., 'time_limit': 60.})
    assert res.success, res.message
    lo[integer == 1] = np.rint(res.x[integer == 1])
    hi[integer == 1] = lo[integer == 1]
    matriz = a.tocsr(); inferior = np.asarray(lbs); superior = np.asarray(ubs)
    igualdad = inferior == superior
    arriba = np.isfinite(superior) & ~igualdad
    abajo = np.isfinite(inferior) & ~igualdad
    lp = linprog(obj, A_ub=vstack([matriz[arriba], -matriz[abajo]]),
                 b_ub=np.r_[superior[arriba], -inferior[abajo]],
                 A_eq=matriz[igualdad], b_eq=superior[igualdad], bounds=list(zip(lo, hi)),
                 method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                          'dual_feasibility_tolerance': 1e-9})
    assert lp.success, lp.message
    return float(lp.fun)


# ============================================================== eventos y cargo de arranque

@lru_cache(maxsize=1)
def estructura_eventos():
    """Estados locales (u, arranque en esta hora, arranques hoy, paradas hoy)."""
    locales = [(0, 0, a, b) for a, b in [(0, 0), (0, 1), (1, 1)]]
    locales += [(1, 0, a, b) for a, b in [(0, 0), (1, 0), (1, 1)]]
    locales += [(1, 1, a, b) for a, b in [(1, 0), (1, 1)]]
    local = np.array(locales, int)
    lookup = {tuple(x): i for i, x in enumerate(locales)}
    coord = np.array(list(product(range(8), repeat=4)), int)
    atributos = local[coord]
    u, inicio = atributos[:, :, 0], atributos[:, :, 1]
    pesos = np.array([512, 64, 8, 1])
    pesos_perfil = np.array([64, 16, 4, 1])
    compromisos = np.array(list(product((0, 1), repeat=4)), int)
    mapas = {}
    for nuevo_dia in (False, True):
        origenes, destinos, perfiles, arranques = [], [], [], []
        for siguiente in compromisos:
            arr = (siguiente > u).astype(int)
            par = (siguiente < u).astype(int)
            a = arr if nuevo_dia else atributos[:, :, 2]+arr
            b = par if nuevo_dia else atributos[:, :, 3]+par
            valid = ((a <= 1) & (b <= 1)).all(axis=1)
            origen = np.flatnonzero(valid)
            destino_local = np.array([[lookup[(int(siguiente[j]), int(arr[i, j]), int(a[i, j]), int(b[i, j]))]
                                       for j in range(4)] for i in origen])
            # Perfil de la hora que termina: 0 apagada, 1 sin evento,
            # 2 un evento y 3 ambos eventos (arranque y posterior parada).
            perfil = np.where(u[origen], 1+inicio[origen]+par[origen], 0)@pesos_perfil
            origenes.extend(origen); destinos.extend(destino_local@pesos)
            perfiles.extend(perfil); arranques.extend(arr[origen].sum(axis=1))
        mapas[nuevo_dia] = tuple(np.asarray(v, int) for v in (origenes, destinos, perfiles, arranques))
    # Libre: cualquier compromiso sin evento en la primera hora.
    # Apagado: las unidades encendidas en la primera hora cuentan su arranque.
    iniciales = {'libre': np.asarray([sum(lookup[(int(x), 0, 0, 0)]*pesos[j] for j, x in enumerate(v))
                                      for v in compromisos]),
                 'apagado': np.asarray([sum(lookup[(int(x), int(x), int(x), 0)]*pesos[j] for j, x in enumerate(v))
                                        for v in compromisos])}
    # No se impone parada al finalizar el año: se supone continuidad después.
    perfil_final = np.where(u, 1+inicio, 0)@pesos_perfil
    return u, iniciales, perfil_final, mapas


def despacho_capacidad_eventos(carga, red, cv, pmin, capacidad, u):
    base = np.asarray(pmin)*u
    residual = np.asarray(carga)-base.sum(axis=1)
    tol = TOL_REDONDEO_REL * np.maximum(1., np.maximum(np.abs(carga), np.abs(base).sum(axis=1)))
    factible = (residual >= -tol) & (capacidad >= base-1e-12).all(axis=1)
    costos = np.column_stack([np.broadcast_to(cv, base.shape), red])
    orden = np.argsort(costos, axis=1, kind='stable')
    cap = np.column_stack([np.maximum(capacidad-base, 0), np.maximum(residual, 0)])
    cap_ord = np.take_along_axis(cap, orden, axis=1)
    previo = np.cumsum(cap_ord, axis=1)-cap_ord
    inc = np.minimum(cap_ord, np.maximum(residual[:, None]-previo, 0))
    p = np.zeros_like(cap)
    np.put_along_axis(p, orden, inc, axis=1)
    p[:, :4] += base
    costo = np.where(factible, (p*costos).sum(axis=1), np.inf)
    return p, costo


def resolver_eventos(carga, red, cv, pmin, pmax, fechas, minutos, costo_arranque=0., inicial='libre'):
    """DP con costo condicionado a arranque actual y parada siguiente.

    Paso 1 h. Un arranque y una parada por unidad/día calendario.
    Inicial 'libre' (sin evento en la primera hora) o 'apagado'; final sin
    parada forzada. Sin permanencias adicionales.
    Descuentos aditivos de tiempo sin generar; no modela trayectoria real.
    costo_arranque: escalar o cuatro cargos USD/arranque.
    costo conserva el componente energético; costo_total agrega los arranques.
    costo_dp corresponde al objetivo completo. Cero recupera el modelo previo.
    """
    carga, red, cv, pmin, pmax = [np.asarray(x, float) for x in (carga, red, cv, pmin, pmax)]
    fechas = pd.DatetimeIndex(fechas)
    if minutos not in (0, 10, 20):
        raise ValueError('Escenarios auditados: 0, 10 o 20 minutos por evento')
    if len(carga) != len(red) or len(carga) != len(fechas) or not len(carga):
        raise ValueError('Series incompatibles')
    if any(x.shape != (4,) for x in (cv, pmin, pmax)):
        raise ValueError('Se requieren cuatro unidades')
    if not all(np.isfinite(x).all() for x in (carga, red, cv, pmin, pmax)):
        raise ValueError('Entradas no finitas')
    if (carga < 0).any() or (pmin < 0).any() or (pmax < pmin).any():
        raise ValueError('Entradas físicas inválidas')
    if not fechas.is_unique or not ((fechas[1:]-fechas[:-1]) == pd.Timedelta(hours=1)).all():
        raise ValueError('Se requiere calendario horario continuo')
    tarifa_arr = np.broadcast_to(np.asarray(costo_arranque, float), (4,)).copy()
    if not np.isfinite(tarifa_arr).all() or (tarifa_arr < 0).any():
        raise ValueError('Costo de arranque debe ser finito y no negativo, en USD/evento')
    if inicial not in ('libre', 'apagado'):
        raise ValueError("inicial debe ser 'libre' o 'apagado'")
    tau = minutos/60.
    perfiles = np.array(list(product(range(4), repeat=4)), int)
    costos = np.empty((len(carga), 256))
    for k, perfil in enumerate(perfiles):
        u = np.broadcast_to(perfil > 0, (len(carga), 4))
        factor = np.where(perfil > 0, 1-tau*(perfil-1), 0)
        capacidad = np.broadcast_to(pmax*factor, u.shape)
        _, costos[:, k] = despacho_capacidad_eventos(carga, red, cv, pmin, capacidad, u)
    u, iniciales, perfil_final, mapas = estructura_eventos()
    n_estados = len(u)
    valor = np.full(n_estados, np.inf)
    arr_acum = np.full(n_estados, 10**8, int)
    iniciales = iniciales[inicial]
    if inicial == 'apagado':
        valor[iniciales] = u[iniciales]@tarifa_arr
        arr_acum[iniciales] = u[iniciales].sum(axis=1)
    else:
        valor[iniciales] = 0.
        arr_acum[iniciales] = 0
    historia = np.zeros((len(carga), n_estados), np.uint16)
    dias = fechas.normalize()
    cargos_transicion = {dia: ((u[datos[1]] > u[datos[0]]).astype(float)@tarifa_arr)
                         for dia, datos in mapas.items()}
    for t in range(len(carga)-1):
        src, dst, perfil, arr = mapas[dias[t+1] != dias[t]]
        candidatos = valor[src]+costos[t, perfil]+cargos_transicion[dias[t+1] != dias[t]]
        mejor = np.full(n_estados, np.inf)
        np.minimum.at(mejor, dst, candidatos)
        empate = np.isfinite(candidatos) & (candidatos == mejor[dst])
        conteo = arr_acum[src]+arr
        mejor_arr = np.full(n_estados, 10**8, int)
        np.minimum.at(mejor_arr, dst[empate], conteo[empate])
        empate &= conteo == mejor_arr[dst]
        pred = np.full(n_estados, n_estados, int)
        np.minimum.at(pred, dst[empate], src[empate])
        historia[t+1] = np.where(pred < n_estados, pred, 0)
        valor, arr_acum = mejor, mejor_arr
    final = valor+costos[-1, perfil_final]
    optimo = final.min()
    if not np.isfinite(optimo):
        raise ValueError('No existe trayectoria factible')
    candidatos = np.flatnonzero(final == optimo)
    actual = candidatos[np.argmin(arr_acum[candidatos])]
    elegido = np.empty(len(carga), int)
    for t in range(len(carga)-1, -1, -1):
        elegido[t] = actual
        if t:
            actual = int(historia[t, actual])
    assert actual in iniciales
    compromiso = u[elegido]
    estado_previo = compromiso[0].copy() if inicial == 'libre' else np.zeros(4, int)
    delta = np.diff(np.vstack([estado_previo, compromiso]), axis=0)
    arr, par = (delta == 1).astype(int), (delta == -1).astype(int)
    par_siguiente = np.vstack([par[1:], np.zeros(4, int)])
    capacidad = pmax*(compromiso-tau*arr-tau*par_siguiente)
    p, costo = despacho_capacidad_eventos(carga, red, cv, pmin, capacidad, compromiso)
    cargo_arranques = arr@tarifa_arr
    total_horario = costo+cargo_arranques
    np.testing.assert_allclose(total_horario.sum(), optimo, rtol=0, atol=1e-5)
    return dict(potencia=p, compromiso=compromiso, costo=costo, costo_dp=float(optimo),
                arranques=arr, paradas=par, parada_siguiente=par_siguiente,
                capacidad_horaria=capacidad, minutos=minutos,
                costo_arranques=cargo_arranques, costo_total=total_horario,
                costo_por_arranque=tarifa_arr, inicial=estado_previo,
                inicial_libre=inicial == 'libre', minimo_h=1, limite_diario=True)


def validar_eventos(r, carga, red, cv, pmin, pmax, fechas):
    out = validar_temporal(r, carga, red, cv, pmin, pmax, fechas)
    assert (r['potencia'][:, :4] <= r['capacidad_horaria']+1e-9).all()
    assert (r['capacidad_horaria'] >= -1e-9).all()
    assert not r['parada_siguiente'][-1].any()
    out['Exceso_capacidad_evento_MW'] = float(np.maximum(r['potencia'][:, :4]-r['capacidad_horaria'], 0).max())
    return out


def validar_cargo_arranque(r, carga, red, cv, pmin, pmax, fechas):
    # Las validaciones previas siguen verificando exclusivamente energía.
    energetico = dict(r, costo_dp=float(r['costo'].sum()))
    v = validar_eventos(energetico, carga, red, cv, pmin, pmax, fechas)
    esperado = r['arranques']@r['costo_por_arranque']
    np.testing.assert_allclose(esperado, r['costo_arranques'], rtol=0, atol=1e-8)
    np.testing.assert_allclose(r['costo']+esperado, r['costo_total'], rtol=0, atol=1e-8)
    error = abs(r['costo_total'].sum()-r['costo_dp'])
    assert error < 1e-5
    v['Error_objetivo_total_USD'] = float(error)
    return v


def umbrales_arranques(filas, valores, ctx):
    """Intervalos del barrido donde los arranques totales caen por primera vez
    y donde bajan al menos 10 % y 50 % frente al cargo cero."""
    base = filas[0]['Arranques']
    salida = []
    for objetivo in (0, 10, 50):
        nombre = ('Primera reduccion de arranques' if objetivo == 0
                  else f'Reduccion de al menos {objetivo} pct de arranques')
        if base == 0:
            inferior, superior, estado = np.nan, np.nan, 'Sin arranques con cargo cero'
        else:
            hits = [i for i, f in enumerate(filas)
                    if (f['Arranques'] < base if objetivo == 0 else f['Reduccion_arranques_pct'] >= objetivo)]
            if hits:
                i = hits[0]
                inferior, superior = (valores[i-1] if i else 0.), valores[i]
                estado = 'Umbral acotado por barrido; extremos sujetos a empates'
            else:
                inferior, superior = valores[-1], np.nan
                estado = 'No alcanzado hasta el máximo explorado; no demuestra imposibilidad'
        salida.append(dict(**ctx, Objetivo=nombre, Arranques_cargo_cero=base,
                           Limite_inferior=inferior, Limite_superior=superior, Estado=estado))
    return salida
