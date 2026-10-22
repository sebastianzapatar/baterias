# -*- coding: utf-8 -*-
"""Políticas integradas A/B/D con PV + BESS + térmica + red (`REGLAS_TERMICA.md`).

Implementado:
- `evaluador_dia`: evaluador térmico diario común (DP exacta del problema condicionado).
- `despachar_termica_encadenada`: aplica `evaluador_dia` por día calendario heredando el estado.
- `simular_a_int`: `A_INT` (regla original A del cuaderno 02 para el BESS) y `SIN_BESS_DP`
  (BESS nulo), con térmica y red por el evaluador diario.
- `simular_b_int` y `ciclo_soc_b_int`: `B_INT`, mérito diario sobre el suministro térmico y de
  red (segmentos S divisibles y bloques U indivisibles), con SOC inicial explícito o cíclico.
- `simular_d_int` y `ciclo_soc_d_int`: `D_INT`, B_INT más exportación del BESS cuando venderla vale más
  que conservarla para atender demanda posterior. Construye un plan diario antes de su ejecución;
  el recorrido horario fija el prefijo del BESS, mientras la térmica de todo el día sigue siendo provisional.
- `validar_evaluador`, `validar_politica`, `validar_b_int` y `validar_d_int`: validaciones independientes.

Problema que resuelve `evaluador_dia` para una ventana horaria continua:

    min  Σ_t [ Σ_j CV_j·p_jt + λ_t·g_t + Σ_j K_j·α_jt ]
    s. a.  Σ_j p_jt + g_t = r_t,   g_t ≥ 0                  (red sin límite de importación)
           Pmin_j·u_jt ≤ p_jt ≤ Pmax_j·u_jt                  (sin sumidero: no hay exportación térmica)
           α_jt − β_jt = u_jt − u_j,t−1,   α_jt + β_jt ≤ 1
           por unidad y día calendario: Σ α ≤ 1 y Σ β ≤ 1, contando los eventos ya usados
           antes de la ventana en ese mismo día

con información perfecta del residual y de los precios de la ventana (declarada), estado
térmico heredado, compromiso final libre y valor terminal cero; 0 minutos por evento.
La solución es exacta para ese problema condicionado; encadenar ventanas no equivale al
óptimo conjunto de varios días ni al óptimo anual.

Estado previo:
- 'libre': solo si la ventana empieza en la primera hora del año. Esa hora toma cualquier
  compromiso sin evento ni cargo, igual que `Inicial-sin-arranque` en PyPSA.
- dict(u, arranques_usados, paradas_usadas, fecha): `fecha` es la hora anterior al inicio.
  Si la ventana empieza en otro día calendario los contadores vuelven a cero; si empieza
  dentro del mismo día se respetan. El compromiso se hereda siempre. Los binarios deben
  ser exactamente 0 o 1 y localmente coherentes, también antes de reiniciarlos.
"""
import heapq
import math
import time
from collections import Counter
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd

from termica import despacho_capacidad_eventos
import disponibilidad_termica as DT

# Estado local por unidad: (encendida, arranque ya usado hoy, parada ya usada hoy).
# Solo seis combinaciones son alcanzables: apagada (0,0), (0,1), (1,1); encendida (0,0), (1,0), (1,1).
LOCALES = ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1))
COMPROMISOS = np.array(list(product((0, 1), repeat=4)), int)   # 16 compromisos G1..G4
# Tolerancia ABSOLUTA de costo (USD), sin componente relativa. Errores observados: 1,46e−10 USD
# entre la DP y el MILP; las reconstrucciones usan math.fsum, cuyo error es del orden de
# ε·|suma| (≈ 2e−9 USD para una suma de 11 millones de USD). 1e−6 USD deja más de dos órdenes de
# margen sobre esos errores y detecta cualquier alteración de 0,01 USD.
TOL_OBJETIVO_USD = 1e-6
TOL_MW = 1e-7              # balance de la DP
TOL_FLUJO_MW = 1e-6        # flujos PV/BESS; igual que termica_reglas.TOL_MW
TOL_SOC_MWh = 1e-5         # dinámica y límites del SOC; igual que termica_reglas.TOL_MWh
TOL_CIERRE_MWh = 1e-3      # cierre cíclico de `validar` del cuaderno 02; igual que termica_reglas
CASOS_A = {'A_INT': 'A', 'SIN_BESS_DP': 'SIN_BESS'}   # caso integrado → regla original del cuaderno 02


def _binario(x, nombre, forma):
    """Arreglo con valores finitos exactamente 0 o 1; nunca redondea ni trunca."""
    a = np.asarray(x, dtype=float)
    if a.shape != forma:
        raise ValueError(f'{nombre}: forma {a.shape}; se esperaba {forma}')
    if not np.isfinite(a).all() or not np.isin(a, (0., 1.)).all():
        raise ValueError(f'{nombre}: se requieren valores finitos exactamente iguales a 0 o 1 (sin redondeo)')
    return a.astype(int)


@lru_cache(maxsize=1)
def _tablas():
    """Transiciones de los 6⁴ = 1.296 estados térmicos para cada compromiso siguiente."""
    lookup = {x: i for i, x in enumerate(LOCALES)}
    coord = np.array(list(product(range(len(LOCALES)), repeat=4)), int)
    local = np.array(LOCALES, int)[coord]
    u, a, b = local[..., 0], local[..., 1], local[..., 2]
    pesos = len(LOCALES) ** np.arange(3, -1, -1)
    trans = {}
    for nuevo_dia in (False, True):
        destinos = np.full((16, len(coord)), -1, int)
        arranques = np.zeros((16, len(coord), 4), int)
        for k, v in enumerate(COMPROMISOS):
            arr = (v > u).astype(int)
            par = (v < u).astype(int)
            aa = arr if nuevo_dia else a + arr
            bb = par if nuevo_dia else b + par
            validos = np.flatnonzero(((aa <= 1) & (bb <= 1)).all(axis=1))
            for s in validos:
                destinos[k, s] = sum(lookup[(int(v[j]), int(aa[s, j]), int(bb[s, j]))]*pesos[j] for j in range(4))
            arranques[k] = arr
        trans[nuevo_dia] = (destinos, arranques)
    return u, a, b, pesos, lookup, trans


def _indice(u, a, b):
    _, _, _, pesos, lookup, _ = _tablas()
    try:
        return int(sum(lookup[(int(u[j]), int(a[j]), int(b[j]))]*pesos[j] for j in range(4)))
    except KeyError as e:
        raise ValueError(f'Estado local incoherente (u, arranque usado, parada usada) = {e}') from None


def _validar_calendario(fechas):
    """Calendario horario continuo, sin duplicados; lo exigen tanto el motor como los validadores."""
    fechas = pd.DatetimeIndex(fechas)
    if not len(fechas) or not fechas.is_unique or not ((fechas[1:] - fechas[:-1]) == pd.Timedelta(hours=1)).all():
        raise ValueError('Se requiere calendario horario continuo')
    return fechas


def _estado_previo(estado_previo, fechas):
    """Valida el estado previo de una ventana y devuelve (libre, u0, a0, b0, nuevo_dia).

    'libre' solo con ese valor exacto y solo si la ventana empieza el 1 de enero a las 00:00.
    Un diccionario debe estar fechado exactamente una hora antes del inicio; compromiso y contadores
    se validan (finitos, binarios exactos y coherentes) ANTES de reiniciar los contadores a medianoche."""
    if isinstance(estado_previo, str):
        if estado_previo != 'libre':
            raise ValueError("estado_previo debe ser 'libre' o un diccionario de estado")
        f0 = fechas[0]
        if not (f0.month == 1 and f0.day == 1 and f0.hour == 0 and f0.minute == 0 and f0.second == 0):
            raise ValueError(f'El estado libre sin evento solo se admite en la primera hora del año; la ventana empieza en {f0}')
        return True, None, np.zeros(4, int), np.zeros(4, int), True
    if not isinstance(estado_previo, dict):
        raise ValueError("estado_previo debe ser 'libre' o un diccionario de estado")
    faltan = [k for k in ('u', 'arranques_usados', 'paradas_usadas', 'fecha') if k not in estado_previo]
    if faltan:
        raise ValueError(f'Faltan claves del estado previo: {faltan}')
    u0, a0, b0 = [_binario(estado_previo[k], f'estado_previo[{k}]', (4,))
                  for k in ('u', 'arranques_usados', 'paradas_usadas')]
    _indice(u0, a0, b0)   # coherencia local antes de cualquier reinicio
    fecha_previa = pd.Timestamp(estado_previo['fecha'])
    if fecha_previa != fechas[0] - pd.Timedelta(hours=1):
        raise ValueError(f'El estado previo es de {fecha_previa}; la ventana empieza en {fechas[0]}')
    nuevo = bool(fecha_previa.normalize() != fechas[0].normalize())
    if nuevo:   # medianoche: contadores a cero, compromiso heredado
        a0, b0 = np.zeros(4, int), np.zeros(4, int)
    return False, u0, a0, b0, nuevo


def _disponibilidad_ventana(disponible, n):
    """None (flota siempre disponible) o matriz booleana (n, 4): True si la unidad puede comprometerse.

    Acepta un vector (n,) para la flota completa o una matriz (n, 4) por unidad. Exige tipo booleano:
    una disponibilidad fraccionaria no tiene significado en este modelo."""
    if disponible is None:
        return None
    d = np.asarray(disponible)
    if d.dtype != bool:
        raise ValueError('La disponibilidad debe ser booleana')
    if d.shape == (n,):
        d = np.repeat(d[:, None], 4, axis=1)
    if d.shape != (n, 4):
        raise ValueError('La disponibilidad debe tener forma (n,) o (n, 4)')
    return d


def _paso(valor, n_arr, costo_hora, transicion, cargos):
    """Un paso de la DP. Desempate: menor costo → menos arranques → menor índice de origen."""
    destinos, arranques = transicion
    vivos = np.isfinite(valor)
    origen, destino, candidato, conteo = [], [], [], []
    for k in range(16):
        if not np.isfinite(costo_hora[k]):
            continue
        s = np.flatnonzero(vivos & (destinos[k] >= 0))
        arr = arranques[k][s]
        origen.append(s)
        destino.append(destinos[k][s])
        candidato.append(valor[s] + arr@cargos + costo_hora[k])
        conteo.append(n_arr[s] + arr.sum(axis=1))
    nuevo = np.full(len(valor), np.inf)
    nuevo_arr = np.full(len(valor), 10**9, int)
    pred = np.full(len(valor), -1, int)
    if not origen:
        return nuevo, nuevo_arr, pred
    origen, destino, candidato, conteo = map(np.concatenate, (origen, destino, candidato, conteo))
    if not len(origen):   # ningún estado vivo tiene una transición válida: la ventana es infactible
        return nuevo, nuevo_arr, pred
    orden = np.lexsort((origen, conteo, candidato, destino))
    d = destino[orden]
    primero = orden[np.r_[True, d[1:] != d[:-1]]]
    nuevo[destino[primero]] = candidato[primero]
    nuevo_arr[destino[primero]] = conteo[primero]
    pred[destino[primero]] = origen[primero]
    return nuevo, nuevo_arr, pred


def evaluador_dia(residual, precio_red, cv, pmin, pmax, fechas, costo_arranque, estado_previo, disponible=None):
    """Compromiso y despacho térmico + red exactos para una ventana horaria con estado heredado.

    Pensado para un día calendario; admite cualquier ventana continua (dentro de un día, con
    los eventos ya usados, o de varios días para contrastes). Devuelve potencia (G1..G4, Red),
    compromiso, arranques, paradas, costos horarios, objetivo y `estado_final`, que es el
    `estado_previo` de la ventana siguiente. Si no hay trayectoria factible lanza ValueError.

    `disponible` (versión 2): None o booleano (n,) o (n, 4). En una hora no disponible solo se admiten
    compromisos con esa unidad apagada; la parada forzada es una parada ordinaria (cuenta en el límite
    diario, sin cargo) y el rearranque posterior paga el cargo por arranque."""
    r = np.asarray(residual, float)
    red = np.broadcast_to(np.asarray(precio_red, float), r.shape).astype(float)
    cv, pmin, pmax = [np.asarray(x, float) for x in (cv, pmin, pmax)]
    cargos = np.broadcast_to(np.asarray(costo_arranque, float), (4,)).copy()
    if r.ndim != 1 or not len(r) or len(pd.DatetimeIndex(fechas)) != len(r):
        raise ValueError('Residual, precios y fechas deben ser series horarias de igual longitud')
    if any(x.shape != (4,) for x in (cv, pmin, pmax)):
        raise ValueError('Se requieren cuatro unidades')
    if not all(np.isfinite(x).all() for x in (r, red, cv, pmin, pmax, cargos)):
        raise ValueError('Entradas no finitas')
    if (r < 0).any() or (pmin < 0).any() or (pmax <= 0).any() or (pmin > pmax).any() or (cargos < 0).any():
        raise ValueError('Residual negativo o parámetros físicos inválidos')
    fechas = _validar_calendario(fechas)
    U, A, B, _, _, trans = _tablas()
    dias = fechas.normalize().to_numpy()  # mismo calendario; evita crear Timestamp en el bucle
    libre, u0, a0, b0, nuevo0 = _estado_previo(estado_previo, fechas)
    n = len(r)
    costo_h = np.empty((n, 16))
    for k, v in enumerate(COMPROMISOS):
        uu = np.broadcast_to(v, (n, 4))
        _, costo_h[:, k] = despacho_capacidad_eventos(r, red, cv, pmin, pmax*uu, uu)
    disp = _disponibilidad_ventana(disponible, n)
    if disp is not None:
        costo_h[(COMPROMISOS.astype(bool)[None, :, :] & ~disp[:, None, :]).any(axis=2)] = np.inf
    N = len(U)
    historia = np.full((n, N), -1, int)
    if libre:
        valor, n_arr = np.full(N, np.inf), np.full(N, 10**9, int)
        for k, v in enumerate(COMPROMISOS):
            if np.isfinite(costo_h[0, k]):
                d = _indice(v, np.zeros(4), np.zeros(4))
                valor[d], n_arr[d] = costo_h[0, k], 0
        inicio = None
    else:
        inicio = _indice(u0, a0, b0)
        valor, n_arr = np.full(N, np.inf), np.full(N, 10**9, int)
        valor[inicio], n_arr[inicio] = 0., 0
        valor, n_arr, historia[0] = _paso(valor, n_arr, costo_h[0], trans[False], cargos)
    for t in range(1, n):
        valor, n_arr, historia[t] = _paso(valor, n_arr, costo_h[t], trans[bool(dias[t] != dias[t-1])], cargos)
    vivos = np.flatnonzero(np.isfinite(valor))
    if not len(vivos):
        raise ValueError('No existe trayectoria térmica factible con ese estado previo, eventos y residual')
    final = vivos[np.lexsort((vivos, n_arr[vivos], valor[vivos]))[0]]
    estados = np.empty(n, int)
    estados[-1] = final
    for t in range(n-1, 0, -1):
        estados[t-1] = historia[t, estados[t]]
    if not libre:
        assert historia[0, estados[0]] == inicio
    compromiso = U[estados]
    previo = compromiso[0] if libre else u0
    delta = np.diff(np.vstack([previo, compromiso]), axis=0)
    arranques, paradas = (delta == 1).astype(int), (delta == -1).astype(int)
    p, costo = despacho_capacidad_eventos(r, red, cv, pmin, pmax*compromiso, compromiso)
    assert np.isfinite(costo).all()
    costo_arr = arranques@cargos
    total = costo + costo_arr
    objetivo = float(valor[final])
    assert abs(math.fsum(total) - objetivo) <= TOL_OBJETIVO_USD
    np.testing.assert_array_equal(A[estados[-1]], _contadores(arranques, dias, a0, nuevo0)[-1])
    np.testing.assert_array_equal(B[estados[-1]], _contadores(paradas, dias, b0, nuevo0)[-1])
    return dict(potencia=p, compromiso=compromiso, arranques=arranques, paradas=paradas,
                costo_energia=costo, costo_arranques=costo_arr, costo_total=total, objetivo=objetivo,
                estado_final=dict(u=compromiso[-1].copy(), arranques_usados=A[estados[-1]].copy(),
                                  paradas_usadas=B[estados[-1]].copy(), fecha=fechas[-1]),
                estado_inicial=dict(u=None if libre else u0.copy(), arranques_usados=a0, paradas_usadas=b0,
                                    inicial_libre=libre, contadores_reiniciados=bool(nuevo0)),
                estados_dp=N, disponible=disp)


def _contadores(eventos, dias, usados, nuevo0):
    """Eventos acumulados por día calendario al final de cada hora, con los usados antes de la ventana."""
    acumulado = np.zeros_like(eventos)
    actual = np.zeros(4, int) if (usados is None or nuevo0) else np.asarray(usados, int).copy()
    for t in range(len(eventos)):
        if t and dias[t] != dias[t-1]:
            actual = np.zeros(4, int)
        actual = actual + eventos[t]
        acumulado[t] = actual
    return acumulado


def validar_evaluador(res, residual, precio_red, cv, pmin, pmax, fechas, costo_arranque, estado_previo, tol=1e-6,
                      disponible=None):
    """Validación independiente de la DP: calendario, estado previo, binarios, balance, límites,
    eventos con estado heredado, estado final y costo (tolerancia absoluta `TOL_OBJETIVO_USD`).
    No supone que el resultado proviene de una llamada válida al motor."""
    r = np.asarray(residual, float)
    n = len(r)
    red = np.broadcast_to(np.asarray(precio_red, float), r.shape)
    cv, pmin, pmax = [np.asarray(x, float) for x in (cv, pmin, pmax)]
    cargos = np.broadcast_to(np.asarray(costo_arranque, float), (4,))
    fechas = _validar_calendario(fechas)
    assert len(fechas) == n, 'fechas y residual de distinta longitud'
    libre, u_prev, a0, b0, nuevo0 = _estado_previo(estado_previo, fechas)
    p = np.asarray(res['potencia'], float)
    assert p.shape == (n, 5) and np.isfinite(p).all(), 'potencia con forma o valores inválidos'
    u = _binario(res['compromiso'], 'compromiso', (n, 4))
    arr = _binario(res['arranques'], 'arranques', (n, 4))
    par = _binario(res['paradas'], 'paradas', (n, 4))
    disp = _disponibilidad_ventana(disponible, n)
    if disp is not None:
        assert not (u.astype(bool) & ~disp).any(), 'unidad comprometida en una hora sin disponibilidad'
    residuo_balance = float(np.abs(p.sum(axis=1) - r).max())
    assert residuo_balance <= tol, ('balance', residuo_balance)
    assert (p >= -tol).all(), 'potencia negativa'
    assert (p[:, :4] >= u*pmin - tol).all() and (p[:, :4] <= u*pmax + tol).all(), 'mínimos técnicos o máximos'
    dias = fechas.normalize()
    previo = u[0] if libre else u_prev
    delta = np.diff(np.vstack([previo, u]), axis=0)
    np.testing.assert_array_equal(arr, (delta == 1).astype(int), err_msg='arranques incoherentes con el compromiso')
    np.testing.assert_array_equal(par, (delta == -1).astype(int), err_msg='paradas incoherentes con el compromiso')
    eventos_max = 0
    for dia in dias.unique():
        m = dias == dia
        extra_a = a0 if dia == dias[0] else 0
        extra_b = b0 if dia == dias[0] else 0
        na = arr[m].sum(axis=0) + extra_a
        nb = par[m].sum(axis=0) + extra_b
        assert (na <= 1).all() and (nb <= 1).all(), ('límite diario', str(dia.date()), na, nb)
        eventos_max = max(eventos_max, int(max(na.max(), nb.max())))
    # Estado final: compromiso de la última hora, eventos acumulados del último día y fecha exacta.
    ef = res['estado_final']
    uf = _binario(ef['u'], 'estado_final[u]', (4,))
    af = _binario(ef['arranques_usados'], 'estado_final[arranques_usados]', (4,))
    bf = _binario(ef['paradas_usadas'], 'estado_final[paradas_usadas]', (4,))
    ultimo = dias == dias[-1]
    hereda = dias[-1] == dias[0]
    assert np.array_equal(uf, u[-1]), 'estado_final: compromiso distinto del de la última hora'
    assert np.array_equal(af, arr[ultimo].sum(axis=0) + (a0 if hereda else 0)), 'estado_final: arranques usados incoherentes'
    assert np.array_equal(bf, par[ultimo].sum(axis=0) + (b0 if hereda else 0)), 'estado_final: paradas usadas incoherentes'
    assert pd.Timestamp(ef['fecha']) == fechas[-1], 'estado_final: fecha distinta de la última hora'
    costo = math.fsum(np.r_[p[:, :4]@cv, p[:, 4]*red, arr@cargos])
    objetivo = float(res['objetivo'])
    assert np.isfinite(objetivo), 'objetivo no finito'
    error = abs(costo - objetivo)
    assert error <= TOL_OBJETIVO_USD, ('costo reconstruido', error)
    return dict(Residuo_balance_MW=residuo_balance, Eventos_dia_max=eventos_max, Error_costo_USD=error)


def segmentos_dias(fechas):
    """Tramos (inicio, fin) de cada día calendario presente en la ventana."""
    dias = pd.DatetimeIndex(fechas).normalize()
    cortes = np.flatnonzero(np.r_[True, dias[1:] != dias[:-1]])
    return [(int(a), int(z)) for a, z in zip(cortes, np.r_[cortes[1:], len(dias)])]


def despachar_termica_encadenada(residual, precio_red, cv, pmin, pmax, fechas, costo_arranque, estado_previo,
                                 disponible=None):
    """`evaluador_dia` por día calendario, heredando compromiso y reiniciando contadores a medianoche.

    No es el óptimo conjunto de la ventana: cada día optimiza con valor terminal cero."""
    fechas = pd.DatetimeIndex(fechas)
    r = np.asarray(residual, float)
    red = np.broadcast_to(np.asarray(precio_red, float), r.shape)
    disp = _disponibilidad_ventana(disponible, len(r))
    segmentos, estado = [], estado_previo
    inicio = time.perf_counter()
    for a, z in segmentos_dias(fechas):
        t = time.perf_counter()
        d = None if disp is None else disp[a:z]
        ev = evaluador_dia(r[a:z], red[a:z], cv, pmin, pmax, fechas[a:z], costo_arranque, estado, disponible=d)
        segmentos.append(dict(inicio=a, fin=z, estado_previo=estado, resultado=ev, segundos=time.perf_counter() - t))
        estado = ev['estado_final']

    def junta(k):
        return np.concatenate([s['resultado'][k] for s in segmentos])
    return dict(potencia=junta('potencia'), compromiso=junta('compromiso'), arranques=junta('arranques'),
                paradas=junta('paradas'), costo_energia=junta('costo_energia'), costo_arranques=junta('costo_arranques'),
                objetivo=math.fsum(s['resultado']['objetivo'] for s in segmentos), segmentos=segmentos,
                estado_inicial=estado_previo, estado_final=estado, segundos=time.perf_counter() - inicio)


def simular_a_int(caso, env, fechas, pv, load, c_imp_cop, c_exp_cop, trm, b, p_exp_max, P, costo_arranque,
                  estado_termico, disponible=None):
    """A_INT o SIN_BESS_DP en una ventana: BESS y PV por la regla original A del cuaderno 02,
    térmica y red por `evaluador_dia` encadenado sobre el residual de cada día.

    - A_INT: `despachar(..., 'A Autoconsumo')` con el BESS vigente (PV directo primero, carga solo
      con excedente solar, descarga a demanda, sin exportación desde el BESS, exportación del PV
      sobrante con el precio y límite comunes). SOC por el punto fijo cíclico original de la ventana.
    - SIN_BESS_DP: la misma regla con potencia y energía nulas.
    El costo devuelto es el de esta trayectoria determinista: no es una cota del óptimo conjunto."""
    import termica_reglas as TR   # reglas originales del cuaderno 02 y sus comprobaciones; no se modifican
    if caso not in CASOS_A:
        raise ValueError(f'caso debe ser uno de {tuple(CASOS_A)}')
    fechas = pd.DatetimeIndex(fechas)
    t = time.perf_counter()
    d = TR.despachar_regla(env, pv, load, c_imp_cop, c_exp_cop, CASOS_A[caso], b, p_exp_max)
    segundos_bess = time.perf_counter() - t
    bb = TR.bess_de(CASOS_A[caso], b)
    carga = d.Load_MW.to_numpy(float)
    pv_directo = d.PV_a_Load_MW.to_numpy(float)
    dd = d.Descarga_Demanda_MW.to_numpy(float)
    de = d.Descarga_Red_MW.to_numpy(float)
    export_total = d.Export_MW.to_numpy(float)
    residual = carga - pv_directo - dd
    if residual.min() < -TOL_FLUJO_MW:
        raise ValueError('La regla deja un residual negativo')
    residual = np.maximum(residual, 0.)
    ci_usd = np.asarray(c_imp_cop, float)/trm
    ce_usd = np.asarray(c_exp_cop, float)/trm
    cv = np.asarray(P['cv'], float)
    disp = DT.alinear(disponible, fechas)
    ter = despachar_termica_encadenada(residual, ci_usd, cv, P['pmin'], P['pmax'], fechas, costo_arranque, estado_termico,
                                       disponible=disp)
    p = ter['potencia']
    ingreso = math.fsum((export_total*ce_usd).tolist())
    soc_ini = float(d.attrs['SOC_inicial_MWh'])
    return dict(Caso=caso, Tipo='politica', Fechas=fechas, PV=d.PV_MW.to_numpy(float), Carga=carga,
                PV_directo=pv_directo, Carga_BESS=d.Carga_PV_MW.to_numpy(float), Descarga_demanda=dd,
                Descarga_export=de, Export_PV=export_total - de, Vertimiento=d.Vertimiento_MW.to_numpy(float),
                SOC=d.SOC_MWh.to_numpy(float), SOC_inicial=soc_ini,
                Frontera_SOC=dict(tipo='ciclica_regla_A', valor=soc_ini,
                                  cierre_regla_MWh=float(d.attrs['cierre_ciclico_MWh'])),
                BESS=bb, Residual=residual, potencia=p, compromiso=ter['compromiso'], arranques=ter['arranques'],
                paradas=ter['paradas'], C_imp_USD=ci_usd, C_exp_USD=ce_usd, Cargo=costo_arranque, P_EXP_MAX=p_exp_max,
                Costo_combustible_CVOM_USD=math.fsum((p[:, :4]@cv).tolist()), Costo_red_USD=math.fsum((p[:, 4]*ci_usd).tolist()),
                Costo_arranques_USD=math.fsum(ter['costo_arranques'].tolist()), Costo_termica_red_arranques_USD=ter['objetivo'],
                Ingreso_exportacion_USD=ingreso, Objetivo=ter['objetivo'] - ingreso, segmentos=ter['segmentos'],
                estado_termico_inicial=estado_termico, estado_termico_final=ter['estado_final'],
                Segundos_BESS=segundos_bess, Segundos_termica=ter['segundos'], despacho_regla=d, Disponible=disp)


def _mismo_estado(x, y):
    if isinstance(x, str) or isinstance(y, str):
        return x == y
    return all(np.array_equal(_binario(x[k], k, (4,)), _binario(y[k], k, (4,)))
               for k in ('u', 'arranques_usados', 'paradas_usadas')) and pd.Timestamp(x['fecha']) == pd.Timestamp(y['fecha'])


def validar_politica(res, P, exportar_bess=False, tol_mw=TOL_FLUJO_MW, tol_mwh=TOL_SOC_MWh):
    """Validación integrada de una trayectoria de política (A_INT, SIN_BESS_DP, B_INT).

    Calendario continuo y estado térmico inicial admisible (libre solo al inicio exacto del año).
    PV y BESS: prioridad solar, balance del excedente, carga solo solar, potencias, sin simultaneidad,
    sin exportación del BESS salvo `exportar_bess`, dinámica, límites y frontera declarada del SOC.
    Red y térmica: balance de demanda, sin exportación térmica, importación y exportación no simultáneas,
    y `validar_evaluador` por día con continuidad exacta del estado heredado.
    Costo: combustible + CVOM, red y arranques reconstruidos por separado, total, ingreso y objetivo,
    con tolerancia absoluta."""
    fechas = _validar_calendario(res['Fechas'])
    n = len(fechas)
    _estado_previo(res['estado_termico_inicial'], fechas)
    b = res['BESS']
    cv, pmin, pmax = (np.asarray(P[k], float) for k in ('cv', 'pmin', 'pmax'))
    cargos = np.broadcast_to(np.asarray(res['Cargo'], float), (4,))
    serie = {}
    for k in ('PV', 'Carga', 'PV_directo', 'Carga_BESS', 'Descarga_demanda', 'Descarga_export', 'Export_PV',
              'Vertimiento', 'SOC', 'C_imp_USD', 'C_exp_USD'):
        serie[k] = np.asarray(res[k], float)
        assert serie[k].shape == (n,) and np.isfinite(serie[k]).all(), f'{k}: forma o valores inválidos'
    pv, load, xd = serie['PV'], serie['Carga'], serie['PV_directo']
    c, dd, de, xpv, vert, soc = (serie[k] for k in ('Carga_BESS', 'Descarga_demanda', 'Descarga_export',
                                                     'Export_PV', 'Vertimiento', 'SOC'))
    ci, ce = serie['C_imp_USD'], serie['C_exp_USD']
    p = np.asarray(res['potencia'], float)
    assert p.shape == (n, 5) and np.isfinite(p).all(), 'potencia con forma o valores inválidos'
    u = _binario(res['compromiso'], 'compromiso', (n, 4))
    arr = _binario(res['arranques'], 'arranques', (n, 4))
    par = _binario(res['paradas'], 'paradas', (n, 4))
    # PV y BESS
    assert float(np.abs(xd - np.minimum(pv, load)).max()) <= tol_mw, 'prioridad solar: PV directo distinto de min(PV, demanda)'
    exc = pv - xd
    for nombre, x in (('carga', c), ('descarga a demanda', dd), ('descarga a exportación', de), ('exportación PV', xpv),
                      ('vertimiento', vert), ('red y térmica', p)):
        assert (x >= -tol_mw).all(), f'{nombre} negativo'
    assert (c <= exc + tol_mw).all(), 'el BESS carga más que el excedente solar'
    residuo_pv = float(np.abs(exc - c - xpv - vert).max())
    assert residuo_pv <= tol_mw, ('balance del excedente PV', residuo_pv)
    assert (c <= b['P_MW'] + tol_mw).all() and (dd + de <= b['P_MW'] + tol_mw).all(), 'potencia del BESS'
    if not exportar_bess:
        assert (de <= tol_mw).all(), 'exportación del BESS no permitida en esta política'
    simultaneas = int(((c > tol_mw) & ((dd + de) > tol_mw)).sum())
    assert simultaneas == 0, ('carga y descarga simultáneas', simultaneas)
    E = b['E_MWh']
    residuo_soc = 0.
    if n:
        previo = np.r_[res['SOC_inicial'], soc[:-1]]
        residuo_soc = float(np.abs(soc - (previo + b['eta_c']*c - (dd + de)/b['eta_d'])).max())
    assert residuo_soc <= tol_mwh, ('dinámica del SOC', residuo_soc)
    assert (soc >= E*b['soc_min_pu'] - tol_mwh).all() and (soc <= E*b['soc_max_pu'] + tol_mwh).all(), 'límites del SOC'
    frontera = res['Frontera_SOC']
    assert abs(res['SOC_inicial'] - frontera['valor']) <= tol_mwh, 'SOC inicial distinto de la frontera declarada'
    if frontera['tipo'].startswith('ciclica'):
        cierre = abs(float(soc[-1]) - res['SOC_inicial'])
        assert cierre <= TOL_CIERRE_MWh, ('cierre cíclico del SOC', cierre)
    elif frontera['tipo'] == 'fija':
        cierre = abs(float(soc[-1]) - frontera['valor'])
        assert cierre <= tol_mwh, ('SOC final distinto de la frontera fija', cierre)
    elif frontera['tipo'] == 'abierta_prueba':
        # Prueba de funcionamiento: SOC inicial declarado y final calculado. No es una frontera de comparación.
        cierre = float('nan')
    else:
        raise AssertionError(f"tipo de frontera no reconocido: {frontera['tipo']}")
    # Demanda, red y térmica
    residual = load - xd - dd
    assert residual.min() >= -tol_mw, 'residual negativo'
    residuo_dem = float(np.abs(xd + dd + p.sum(axis=1) - load).max())
    assert residuo_dem <= tol_mw, ('balance de demanda', residuo_dem)
    termica, red = p[:, :4].sum(axis=1), p[:, 4]
    assert (termica <= np.maximum(residual, 0.) + tol_mw).all(), 'térmica por encima del residual (exportación térmica)'
    exporta = (xpv + de) > tol_mw
    assert not (exporta & (red > tol_mw)).any(), 'importación y exportación simultáneas'
    assert not (exporta & (termica > tol_mw)).any(), 'térmica generando en hora de exportación'
    assert not (exporta & (ce <= 0)).any(), 'exportación con precio no positivo'
    assert ((xpv + de) <= res['P_EXP_MAX'] + tol_mw).all(), 'límite de exportación'
    # Térmica por día con estado heredado
    segmentos = res['segmentos']
    assert [s['inicio'] for s in segmentos] == [a for a, _ in segmentos_dias(fechas)], 'segmentos no son días calendario'
    assert [s['fin'] for s in segmentos] == [z for _, z in segmentos_dias(fechas)], 'segmentos no cubren la ventana'
    estado = res['estado_termico_inicial']
    disp = _disponibilidad_ventana(res.get('Disponible'), n)
    for s in segmentos:
        a, z = s['inicio'], s['fin']
        assert _mismo_estado(s['estado_previo'], estado), f'estado heredado incoherente al inicio de {fechas[a]}'
        ev = dict(s['resultado'], potencia=p[a:z], compromiso=u[a:z], arranques=arr[a:z], paradas=par[a:z])
        validar_evaluador(ev, np.maximum(residual[a:z], 0.), ci[a:z], cv, pmin, pmax, fechas[a:z], cargos, s['estado_previo'],
                          disponible=None if disp is None else disp[a:z])
        estado = s['resultado']['estado_final']
    assert _mismo_estado(res['estado_termico_final'], estado), 'estado térmico final incoherente'
    previo_u = u[0] if isinstance(res['estado_termico_inicial'], str) else _binario(res['estado_termico_inicial']['u'], 'u', (4,))
    delta = np.diff(np.vstack([previo_u, u]), axis=0)
    assert np.array_equal(arr, (delta == 1).astype(int)), 'arranques no derivados del compromiso (arranque gratuito o ficticio)'
    assert np.array_equal(par, (delta == -1).astype(int)), 'paradas no derivadas del compromiso'
    # Costo completo, por componentes
    for k in ('Costo_combustible_CVOM_USD', 'Costo_red_USD', 'Costo_arranques_USD', 'Costo_termica_red_arranques_USD',
              'Ingreso_exportacion_USD', 'Objetivo'):
        assert np.isfinite(float(res[k])), f'{k} no finito'
    combustible = math.fsum((p[:, :4]@cv).tolist())
    costo_red = math.fsum((red*ci).tolist())
    costo_arr = math.fsum((arr@cargos).tolist())
    costo_termica = math.fsum(np.r_[p[:, :4]@cv, red*ci, arr@cargos].tolist())
    ingreso = math.fsum(((xpv + de)*ce).tolist())
    errores = dict(
        Error_combustible_CVOM_USD=abs(combustible - float(res['Costo_combustible_CVOM_USD'])),
        Error_red_USD=abs(costo_red - float(res['Costo_red_USD'])),
        Error_arranques_USD=abs(costo_arr - float(res['Costo_arranques_USD'])),
        Error_costo_termica_USD=abs(costo_termica - float(res['Costo_termica_red_arranques_USD'])),
        Error_segmentos_USD=abs(math.fsum(s['resultado']['objetivo'] for s in segmentos) - costo_termica),
        Error_ingreso_USD=abs(ingreso - float(res['Ingreso_exportacion_USD'])),
        Error_objetivo_USD=abs(costo_termica - ingreso - float(res['Objetivo'])))
    for k, e in errores.items():
        assert e <= TOL_OBJETIVO_USD, (k, e)
    return dict(Caso=res['Caso'], Residuo_PV_MW=residuo_pv, Residuo_demanda_MW=residuo_dem, Residuo_SOC_MWh=residuo_soc,
                Horas_simultaneas=simultaneas, Dias=len(segmentos), Cierre_SOC_MWh=cierre, **errores, Estado='OK')


# ============================================================== B_INT: mérito del suministro térmico y de red

TOL_DEFICIT_MW = 1e-9     # umbral de déficit de `despachar` en el cuaderno 02
TOL_PLAN_MWh = 1e-9       # SOC mínimo admitido al planificar (más estricto que TOL_SOC_MWh de la validación)
# Descarga mínima para aceptar un segmento S. Sin ella, la bisección acepta fracciones ~1e−10 que solo caben
# dentro de TOL_PLAN_MWh: no son decisiones sino residuos numéricos. Se rechazan como 'S_sin_energia'.
MIN_DESCARGA_MW = 1e-6
FUENTES = ('G1', 'G2', 'G3', 'G4', 'Red')


def trayectoria_soc(q, eps, soc0, b, export=None):
    """SOC al final de cada hora con carga voraz del excedente solar, descarga `q` (MW entregados a demanda) y,
    opcionalmente, exportación del BESS `export` (MW entregados al punto de exportación, solo D_INT).
    Cronológica: la energía de una recarga posterior solo está disponible después de cargarla."""
    q, eps = np.asarray(q, float), np.asarray(eps, float)
    x = np.zeros(len(q)) if export is None else np.asarray(export, float)
    e_max = b['E_MWh']*b['soc_max_pu']
    e, c, soc = float(soc0), np.zeros(len(q)), np.zeros(len(q))
    for t in range(len(q)):
        if eps[t] > 0:
            c[t] = min(eps[t], b['P_MW'], max(e_max - e, 0.)/b['eta_c'])
        e = e + b['eta_c']*c[t] - (q[t] + x[t])/b['eta_d']
        soc[t] = e
    return c, soc


def _soc_factible(q, eps, soc0, b, export=None):
    return bool((trayectoria_soc(q, eps, soc0, b, export)[1] >= b['E_MWh']*b['soc_min_pu'] - TOL_PLAN_MWh).all())


def _segmentos_hora(t, p, u, precio, cv, pmin, eta_d, version):
    """Segmentos S de una hora: red y generación por encima de los mínimos, de la fuente más cara a la más barata."""
    fuentes = [(float(precio), 4, float(p[4]))] if p[4] > TOL_MW else []
    fuentes += [(float(cv[j]), j, float(p[j] - pmin[j])) for j in range(4) if u[j] and p[j] - pmin[j] > TOL_MW]
    fuentes.sort(key=lambda x: (-x[0], x[1]))
    return [dict(tipo='S', hora=int(t), horas=[int(t)], fuente=FUENTES[k], unidad=k if k < 4 else None, rango=r,
                 version=int(version), mw=m, costo=v, valor_USD=m*v, energia_MWh=m/eta_d, densidad=v*eta_d)
            for r, (v, k, m) in enumerate(fuentes)]


def candidatos_sistema(delta, precio, cv, pmin, pmax, cargos, referencia, b):
    """Candidatos de B_INT sobre el despacho térmico de referencia del día (sin BESS).

    S (divisibles): por hora con déficit, la red a λ_t y cada unidad comprometida por encima de su mínimo a CV_j.
      Descargar dentro de esos segmentos no obliga a apagar unidades.
    U (indivisibles): bloque completo de la unidad j que ARRANCA en el día; retirarlo evita su arranque.
      Descarga q_τ = min(P, p⁰_jτ); el resto de la hora se redespacha sin la unidad.
    `valor_USD` es el valor atribuido para ordenar, con el resto del compromiso de referencia fijo:
    no es un ahorro exacto ni aditivo entre candidatos."""
    n = len(delta)
    p0, u0, c0 = referencia['potencia'], referencia['compromiso'], referencia['costo_energia']
    cand = []
    for t in range(n):
        if delta[t] > TOL_DEFICIT_MW:
            cand += _segmentos_hora(t, p0[t], u0[t], precio[t], cv, pmin, b['eta_d'], 0)
    for j in range(4):
        for h1 in np.flatnonzero(referencia['arranques'][:, j] == 1):
            h2 = int(h1)
            while h2 + 1 < n and u0[h2 + 1, j] == 1:
                h2 += 1
            horas = np.arange(h1, h2 + 1)
            q = np.minimum(b['P_MW'], p0[horas, j])
            u1 = u0[horas].copy()
            u1[:, j] = 0
            p1, c1 = despacho_capacidad_eventos(delta[horas] - q, precio[horas], cv, pmin, pmax*u1, u1)
            if not np.isfinite(c1).all() or q.sum() <= 0:
                continue
            valor = math.fsum((c0[horas] - c1).tolist()) + float(cargos[j])
            energia = float(q.sum())/b['eta_d']
            if valor > 0:
                cand.append(dict(tipo='U', hora=int(h1), horas=[int(h) for h in horas], fuente=FUENTES[j], unidad=j,
                                 rango=j, version=0, q=q, valor_USD=valor, energia_MWh=energia, densidad=valor/energia,
                                 p_tras=p1, u_tras=u1))
    return cand


def seleccionar_plan(candidatos, delta, eps, soc0, b, precio, cv, pmin, q_fijo=None, export=None, desde=-1):
    """Una sola pasada por densidad de valor (USD por MWh de SOC), con verificación cronológica del SOC.

    Orden: densidad desc → U antes que S → hora más temprana → rango/unidad → orden de creación.
    - U: se acepta completo si no se solapa con candidatos ya aceptados y cabe en SOC y potencia; nunca se fracciona.
      Al aceptarlo, los segmentos S de sus horas se regeneran con el despacho sin la unidad (sin doble conteo).
    - S: dentro de cada hora, en orden de mérito (el siguiente solo si el anterior entró completo); se acepta la
      mayor fracción que mantiene el SOC ≥ mínimo en todas las horas y la potencia ≤ P.
    Devuelve la descarga por hora, los aceptados (con su valor atribuido) y el conteo de rechazos por motivo.

    `q_fijo` (descargas ya fijadas en el plan), `export` (exportaciones del BESS ya decididas, MW entregados) y
    `desde` (solo se consideran candidatos posteriores a esa hora) conservan el prefijo del BESS al construir
    el plan diario de D_INT. No fijan la trayectoria térmica. Por defecto el comportamiento es el de B_INT."""
    n = len(delta)
    P = b['P_MW']
    q = np.zeros(n) if q_fijo is None else np.asarray(q_fijo, float).copy()
    exportado = np.zeros(n) if export is None else np.asarray(export, float)
    version = np.zeros(n, int)
    completos = np.zeros(n, int)
    parcial = np.zeros(n, bool)
    ocupada = np.zeros(n, bool)
    cands = list(candidatos)
    heap = []

    def meter(k):
        c = cands[k]
        heapq.heappush(heap, (-c['densidad'], 0 if c['tipo'] == 'U' else 1, c['hora'], c['rango'], k))
    for k in range(len(cands)):
        meter(k)
    aceptados, rechazos = [], Counter()
    while heap:
        k = heapq.heappop(heap)[-1]
        c = cands[k]
        if c['tipo'] == 'U':
            h = np.asarray(c['horas'])
            if h.min() <= desde:
                rechazos['U_ya_ejecutado'] += 1
                continue
            if ocupada[h].any():
                rechazos['U_solapado'] += 1
                continue
            prueba = q.copy()
            prueba[h] += c['q']
            if (prueba[h] > P + TOL_MW).any():
                rechazos['U_potencia'] += 1
                continue
            if not _soc_factible(prueba, eps, soc0, b, exportado):
                rechazos['U_no_cabe_SOC'] += 1
                continue
            q = prueba
            ocupada[h] = True
            aceptados.append(dict(tipo='U', fuente=c['fuente'], horas=list(c['horas']), q=[float(x) for x in c['q']],
                                  fraccion=1.0, valor_atribuido_USD=float(c['valor_USD']), densidad=float(c['densidad'])))
            for i, t in enumerate(h):
                version[t] += 1
                completos[t] = 0
                for s in _segmentos_hora(t, c['p_tras'][i], c['u_tras'][i], precio[t], cv, pmin, b['eta_d'], version[t]):
                    cands.append(s)
                    meter(len(cands) - 1)
            continue
        t = c['hora']
        if t <= desde:
            rechazos['S_ya_ejecutado'] += 1
            continue
        if c['version'] != version[t]:
            rechazos['S_reemplazado'] += 1
            continue
        if parcial[t] or c['rango'] != completos[t]:
            rechazos['S_orden_horario'] += 1
            continue
        f_max = min(1., (P - q[t] - exportado[t])/c['mw'])
        if f_max <= 0:
            rechazos['S_potencia'] += 1
            parcial[t] = True
            continue

        def factible(f):
            prueba = q.copy()
            prueba[t] += f*c['mw']
            return _soc_factible(prueba, eps, soc0, b, exportado)
        if factible(f_max):
            f = f_max
        else:
            lo, hi = 0., f_max
            for _ in range(60):
                mid = 0.5*(lo + hi)
                if factible(mid):
                    lo = mid
                else:
                    hi = mid
            f = lo
        if f*c['mw'] < MIN_DESCARGA_MW:
            rechazos['S_sin_energia'] += 1
            parcial[t] = True
            continue
        q[t] += f*c['mw']
        ocupada[t] = True
        if f < 1.:
            parcial[t] = True
        else:
            completos[t] += 1
        aceptados.append(dict(tipo='S', fuente=c['fuente'], horas=[t], q=[f*c['mw']], fraccion=f, mw=c['mw'],
                              costo=c['costo'], valor_atribuido_USD=f*c['valor_USD'], densidad=float(c['densidad'])))
    return q, aceptados, rechazos


def planificar_dia_b(delta, eps, precio, fechas, cv, pmin, pmax, cargos, estado_termico, soc0, b, disponible=None):
    """Plan de B_INT para un día (o su tramo inicial): referencia → candidatos → selección → despacho realizado.

    Si el despacho térmico del plan no es factible con el estado heredado se lanza ValueError: no se reduce el
    plan en silencio, no se violan mínimos ni se regalan eventos."""
    referencia = evaluador_dia(delta, precio, cv, pmin, pmax, fechas, cargos, estado_termico, disponible=disponible)
    candidatos = candidatos_sistema(delta, precio, cv, pmin, pmax, cargos, referencia, b)
    q, aceptados, rechazos = seleccionar_plan(candidatos, delta, eps, soc0, b, precio, cv, pmin)
    residual = np.maximum(delta - q, 0.)
    try:
        final = evaluador_dia(residual, precio, cv, pmin, pmax, fechas, cargos, estado_termico, disponible=disponible)
    except ValueError as e:
        raise ValueError(f'Plan B_INT de {fechas[0]} térmicamente infactible con el estado heredado '
                         f'(no se reduce ni se fuerza): {e}') from None
    return dict(q=q, referencia=referencia, final=final, aceptados=aceptados, rechazos=dict(rechazos),
                fecha=fechas[0], n_candidatos_S=sum(c['tipo'] == 'S' for c in candidatos),
                n_candidatos_U=sum(c['tipo'] == 'U' for c in candidatos),
                objetivo_referencia_USD=referencia['objetivo'], objetivo_final_USD=final['objetivo'],
                valor_atribuido_USD=math.fsum(a['valor_atribuido_USD'] for a in aceptados),
                ahorro_redespacho_USD=referencia['objetivo'] - final['objetivo'])


def simular_b_int(fechas, pv, load, c_imp_usd, c_exp_usd, b, p_exp_max, P, costo_arranque, estado_termico, soc_inicial,
                  frontera='abierta_prueba', disponible=None):
    """B_INT en una ventana con SOC inicial EXPLÍCITO. Cada día: PV directo primero, carga voraz del excedente,
    plan de descarga a demanda por mérito del sistema y térmica/red por `evaluador_dia` con estado heredado.

    Información perfecta del día; no usa días posteriores. `frontera` solo etiqueta el SOC inicial
    ('abierta_prueba' para pruebas de funcionamiento; `ciclo_soc_b_int` usa 'ciclica_b_int').
    El costo es el de esta trayectoria determinista: no es una cota del óptimo conjunto."""
    fechas = _validar_calendario(fechas)
    pv, load = np.asarray(pv, float), np.asarray(load, float)
    ci, ce = np.asarray(c_imp_usd, float), np.asarray(c_exp_usd, float)
    n = len(fechas)
    if any(x.shape != (n,) or not np.isfinite(x).all() for x in (pv, load, ci, ce)) or (pv < 0).any() or (load < 0).any():
        raise ValueError('Series de B_INT inválidas')
    disp = DT.alinear(disponible, fechas)
    e_min, e_max = b['E_MWh']*b['soc_min_pu'], b['E_MWh']*b['soc_max_pu']
    if not (e_min <= float(soc_inicial) <= e_max):
        raise ValueError(f'SOC inicial {soc_inicial} fuera de [{e_min}, {e_max}]')
    cv, pmin, pmax = (np.asarray(P[k], float) for k in ('cv', 'pmin', 'pmax'))
    cargos = np.broadcast_to(np.asarray(costo_arranque, float), (4,))
    x = np.minimum(pv, load)
    delta, eps = load - x, pv - x
    q, c, soc = np.zeros(n), np.zeros(n), np.zeros(n)
    segmentos, dias = [], []
    estado, e = estado_termico, float(soc_inicial)
    inicio = time.perf_counter()
    for a, z in segmentos_dias(fechas):
        t = time.perf_counter()
        plan = planificar_dia_b(delta[a:z], eps[a:z], ci[a:z], fechas[a:z], cv, pmin, pmax, cargos, estado, e, b,
                                disponible=None if disp is None else disp[a:z])
        q[a:z] = plan['q']
        c[a:z], soc[a:z] = trayectoria_soc(plan['q'], eps[a:z], e, b)
        e = float(soc[z - 1])
        segmentos.append(dict(inicio=a, fin=z, estado_previo=estado, resultado=plan['final'], segundos=time.perf_counter() - t))
        dias.append({k: v for k, v in plan.items() if k not in ('referencia', 'final')})
        estado = plan['final']['estado_final']
    sobrante = eps - c
    xpv = np.where(ce > 0, np.minimum(sobrante, p_exp_max), 0.)
    vert = sobrante - xpv
    p = np.concatenate([s['resultado']['potencia'] for s in segmentos])
    arr = np.concatenate([s['resultado']['arranques'] for s in segmentos])
    costo_termica = math.fsum(s['resultado']['objetivo'] for s in segmentos)
    ingreso = math.fsum((xpv*ce).tolist())
    return dict(Caso='B_INT', Tipo='politica', Fechas=fechas, PV=pv, Carga=load, PV_directo=x, Carga_BESS=c,
                Descarga_demanda=q, Descarga_export=np.zeros(n), Export_PV=xpv, Vertimiento=vert, SOC=soc,
                SOC_inicial=float(soc_inicial), Frontera_SOC=dict(tipo=frontera, valor=float(soc_inicial)), BESS=b,
                Residual=np.maximum(delta - q, 0.), potencia=p,
                compromiso=np.concatenate([s['resultado']['compromiso'] for s in segmentos]), arranques=arr,
                paradas=np.concatenate([s['resultado']['paradas'] for s in segmentos]), C_imp_USD=ci, C_exp_USD=ce,
                Cargo=costo_arranque, P_EXP_MAX=p_exp_max,
                Costo_combustible_CVOM_USD=math.fsum((p[:, :4]@cv).tolist()), Costo_red_USD=math.fsum((p[:, 4]*ci).tolist()),
                Costo_arranques_USD=math.fsum((arr@cargos).tolist()), Costo_termica_red_arranques_USD=costo_termica,
                Ingreso_exportacion_USD=ingreso, Objetivo=costo_termica - ingreso, segmentos=segmentos, dias_b=dias,
                Valor_atribuido_USD=math.fsum(d['valor_atribuido_USD'] for d in dias),
                Ahorro_redespacho_USD=math.fsum(d['ahorro_redespacho_USD'] for d in dias),
                estado_termico_inicial=estado_termico, estado_termico_final=estado,
                Segundos_simulacion=time.perf_counter() - inicio, Disponible=disp)


def ciclo_soc_b_int(fechas, pv, load, c_imp_usd, c_exp_usd, b, p_exp_max, P, costo_arranque, estado_termico,
                    max_iter=25, tol=1e-4, disponible=None):
    """Punto fijo cíclico del SOC de B_INT, separado de la simulación con SOC explícito.

    Empieza en el SOC mínimo, como `despachar` del cuaderno 02, con tolerancia 1e−4 MWh y 25 iteraciones por
    defecto; el estado térmico inicial es el mismo en todas las iteraciones. Devuelve CONVERGE con la
    trayectoria cuyo SOC inicial cierra, CICLO si el SOC final repite un SOC inicial anterior, o NO_CONVERGE al
    agotar el límite. NO_CONVERGE no demuestra que no exista punto fijo: con un límite mayor puede aparecer un
    CICLO. En CICLO y NO_CONVERGE no hay trayectoria: no se promedia ni se ajusta la última hora."""
    def simular(e):
        return simular_b_int(fechas, pv, load, c_imp_usd, c_exp_usd, b, p_exp_max, P, costo_arranque, estado_termico, e,
                             frontera='ciclica_b_int', disponible=disponible)
    return ciclo_soc_politica(simular, b, max_iter=max_iter, tol=tol)


def validar_b_int(res, P, exportar_bess=False):
    """`validar_politica` más los controles propios de B_INT (y de D_INT con `exportar_bess=True`).

    - la descarga realizada es exactamente la suma de los candidatos aceptados;
    - los bloques U están completos (fracción 1) y no comparten horas con otros aceptados;
    - un segmento S de la unidad j no coexiste con un bloque U de j en la misma hora;
    - las fracciones S están en (0, 1] y el valor atribuido de un S nunca incluye cargos de arranque;
    - valor atribuido y ahorro de redespacho coinciden con sus definiciones y se reportan por separado."""
    v = validar_politica(res, P, exportar_bess=exportar_bess)
    fechas = pd.DatetimeIndex(res['Fechas'])
    assert len(res['dias_b']) == len(res['segmentos'])
    valores, ahorros = [], []
    for seg, dia in zip(res['segmentos'], res['dias_b']):
        a, z = seg['inicio'], seg['fin']
        assert pd.Timestamp(dia['fecha']) == fechas[a], 'plan diario desalineado'
        q = np.zeros(z - a)
        horas_u, fuentes_s = set(), set()
        for acc in dia['aceptados']:
            horas = list(acc['horas'])
            if acc['tipo'] == 'U':
                assert acc['fraccion'] == 1.0, 'bloque U fraccionado'
                assert not (set(horas) & horas_u), 'bloques U solapados'
                horas_u |= set(horas)
            else:
                assert 0. < acc['fraccion'] <= 1., 'fracción S fuera de (0, 1]'
                assert len(horas) == 1 and abs(acc['q'][0] - acc['fraccion']*acc['mw']) <= TOL_MW, 'segmento S incoherente'
                assert abs(acc['valor_atribuido_USD'] - acc['q'][0]*acc['costo']) <= TOL_OBJETIVO_USD, 'valor de S con cargos'
                fuentes_s.add((horas[0], acc['fuente']))
            for i, h in enumerate(horas):
                q[h] += acc['q'][i]
        for acc in dia['aceptados']:
            if acc['tipo'] == 'U':
                assert not any((h, acc['fuente']) in fuentes_s for h in acc['horas']), 'S y U de la misma unidad en la misma hora'
        assert float(np.abs(q - res['Descarga_demanda'][a:z]).max()) <= TOL_MW, 'descarga distinta de los candidatos aceptados'
        assert abs(dia['valor_atribuido_USD'] - math.fsum(x['valor_atribuido_USD'] for x in dia['aceptados'])) <= TOL_OBJETIVO_USD
        assert abs(dia['objetivo_final_USD'] - seg['resultado']['objetivo']) <= TOL_OBJETIVO_USD, 'objetivo realizado distinto'
        assert abs(dia['ahorro_redespacho_USD'] - (dia['objetivo_referencia_USD'] - dia['objetivo_final_USD'])) <= TOL_OBJETIVO_USD
        valores.append(dia['valor_atribuido_USD'])
        ahorros.append(dia['ahorro_redespacho_USD'])
    assert abs(res['Valor_atribuido_USD'] - math.fsum(valores)) <= TOL_OBJETIVO_USD
    assert abs(res['Ahorro_redespacho_USD'] - math.fsum(ahorros)) <= TOL_OBJETIVO_USD
    return dict(v, Valor_atribuido_USD=res['Valor_atribuido_USD'], Ahorro_redespacho_USD=res['Ahorro_redespacho_USD'])


# ============================================================== D_INT: mérito del sistema y exportación del BESS

# Cantidades candidatas de exportación evaluadas en cada hora elegible, como fracción del máximo admisible.
# Es un conjunto FINITO: no se afirma optimalidad continua. Siempre incluye exportar cero.
CANDIDATOS_EXPORTACION = (0., 0.25, 0.5, 0.75, 1.)
TOL_BENEFICIO_USD = 1e-6


def exportacion_pv(eps, c, ce, p_exp_max):
    """PV sobrante exportado: min(excedente − carga, límite) si el precio es positivo. El PV tiene prioridad
    sobre la exportación del BESS, que solo usa la capacidad restante."""
    eps, c, ce = np.asarray(eps, float), np.asarray(c, float), np.asarray(ce, float)
    return np.where(ce > 0, np.minimum(np.maximum(eps - c, 0.), p_exp_max), 0.)


def _ventana_madrugada(delta, eps, precio, ce, fechas, z, extension_periodica, disponible=None):
    """Madrugada siguiente hasta la primera recarga solar (excluida), para valorar la energía conservada.

    - Si quedan horas en la ventana, son las horas reales desde `z`.
    - Si `z` es el final de la ventana y `extension_periodica`, se usa la madrugada del PRIMER día con fechas
      que continúan la ventana: sirve solo para estimar valor. Sus flujos, ingresos y costos no se suman a la
      ventana y no recibe un estado térmico libre: hereda el estado con el que termina el día evaluado."""
    n = len(delta)
    if z < n:
        fin = z
        while fin < n and eps[fin] <= 0:
            fin += 1
        idx = np.arange(z, fin)
        origen = 'ventana'
    elif extension_periodica:
        fin = 0
        while fin < n and eps[fin] <= 0:
            fin += 1
        idx = np.arange(0, fin)
        origen = 'extension_periodica'
    else:
        return None
    if not len(idx):
        return None
    f = fechas[idx] if origen == 'ventana' else pd.date_range(fechas[-1] + pd.Timedelta(hours=1), periods=len(idx), freq='h')
    return dict(delta=delta[idx], eps=eps[idx], precio=precio[idx], ce=ce[idx], fechas=f, origen=origen, horas=len(idx),
                disponible=None if disponible is None else disponible[idx])


def _futuro_d(ctx, t, y):
    """Costo futuro estimado si en la hora `t` se exportan `y` MW entregados desde el BESS.

    Replanifica el BESS solo después de `t`, pero evalúa de nuevo la térmica de TODO el día, incluidas horas
    anteriores a `t`, respetando el estado heredado al inicio del día. Es construcción previa del plan diario:
    no presupone que esas decisiones térmicas ya se ejecutaron. Añade la madrugada siguiente hasta la primera
    recarga solar. Incluye red, térmica, arranques, carga solar cronológica e ingreso por PV posterior a `t`.
    La continuación se valora con el plan de B, sin exportaciones futuras del BESS; al final de la ventana se
    usa la extensión periódica declarada solo para valorar, con estado térmico heredado. Es una ESTIMACIÓN,
    no el costo realizado ni el valor exacto del año."""
    export = ctx['export'].copy()
    export[t] = y
    q, aceptados, _ = seleccionar_plan(ctx['candidatos'], ctx['delta'], ctx['eps'], ctx['soc0'], ctx['b'],
                                       ctx['precio'], ctx['cv'], ctx['pmin'], q_fijo=ctx['q_exe'], export=export, desde=t)
    c, soc = trayectoria_soc(q, ctx['eps'], ctx['soc0'], ctx['b'], export)
    ev = evaluador_dia(np.maximum(ctx['delta'] - q, 0.), ctx['precio'], ctx['cv'], ctx['pmin'], ctx['pmax'],
                       ctx['fechas'], ctx['cargos'], ctx['estado'], disponible=ctx.get('disponible'))
    xpv = exportacion_pv(ctx['eps'], c, ctx['ce'], ctx['p_exp'])
    post = slice(t + 1, len(q))
    costo = ev['objetivo'] - math.fsum((xpv[post]*ctx['ce'][post]).tolist())
    mad = ctx['madrugada']
    if mad is not None:
        plan = planificar_dia_b(mad['delta'], mad['eps'], mad['precio'], mad['fechas'], ctx['cv'], ctx['pmin'],
                                ctx['pmax'], ctx['cargos'], ev['estado_final'], float(soc[-1]), ctx['b'],
                                disponible=mad.get('disponible'))
        costo += plan['objetivo_final_USD']
    return dict(costo=costo, q=q, aceptados=aceptados, export=export, soc_fin=float(soc[-1]))


def decidir_exportacion_d(ctx, t, y_max):
    """Elige cuánto exportar en la hora `t` entre un conjunto finito de cantidades, incluida la de exportar cero.

        beneficio(y) = y·c_exp_t − [costo_futuro(y) − costo_futuro(0)]

    con `y` en MW entregados durante la hora (x = y/η_d MWh de SOC). Se exige beneficio estrictamente positivo:
    ante empate o beneficio nulo se conserva la energía; entre cantidades de igual beneficio se toma la menor."""
    base = _futuro_d(ctx, t, 0.)
    mejor = dict(y=0., x_soc=0., ingreso=0., delta_futuro=0., beneficio=0., plan=base)
    evaluadas = []
    for f in CANDIDATOS_EXPORTACION:
        y = f*y_max
        if f > 0 and y < MIN_DESCARGA_MW:
            continue
        plan = base if f == 0 else _futuro_d(ctx, t, y)
        ingreso = y*ctx['ce'][t]
        delta_futuro = plan['costo'] - base['costo']
        beneficio = ingreso - delta_futuro
        evaluadas.append(dict(fraccion=f, y_MW=y, ingreso_USD=ingreso, delta_futuro_USD=delta_futuro,
                              beneficio_USD=beneficio))
        if beneficio > mejor['beneficio'] + TOL_BENEFICIO_USD:
            mejor = dict(y=y, x_soc=y/ctx['b']['eta_d'], ingreso=ingreso, delta_futuro=delta_futuro,
                         beneficio=beneficio, plan=plan)
    return mejor, evaluadas


def simular_d_int(fechas, pv, load, c_imp_usd, c_exp_usd, b, p_exp_max, P, costo_arranque, estado_termico, soc_inicial,
                  frontera='abierta_prueba', exportar_bess=True, extension_periodica=True, disponible=None):
    """D_INT en una ventana con SOC inicial EXPLÍCITO: B_INT más exportación del BESS cuando venderla vale más
    que conservarla. Con `exportar_bess=False` reproduce B_INT.

    Construye el plan antes de ejecutar cada día: plan provisional de B → recorrido horario interno que fija
    carga, descarga y exportación del BESS y replantea sus horas posteriores → despacho térmico definitivo
    al cerrar el PLAN diario. Solo el prefijo del BESS queda fijo durante el recorrido; cada valoración puede
    revisar la térmica de todo el día, conservando el estado heredado de la víspera. No es un controlador en
    tiempo real. Usa información del día y de la madrugada siguiente (extensión periódica al final, solo para
    valorar); los flujos de esa continuación no se contabilizan en la ventana.
    Elegibilidad de la exportación del BESS: sin déficit tras el PV directo, sin carga en la misma hora, precio
    de exportación positivo, y dentro de SOC, potencia de descarga y capacidad de exportación restante tras el
    PV sobrante."""
    fechas = _validar_calendario(fechas)
    pv, load = np.asarray(pv, float), np.asarray(load, float)
    ci, ce = np.asarray(c_imp_usd, float), np.asarray(c_exp_usd, float)
    n = len(fechas)
    if any(x.shape != (n,) or not np.isfinite(x).all() for x in (pv, load, ci, ce)) or (pv < 0).any() or (load < 0).any():
        raise ValueError('Series de D_INT inválidas')
    disp = DT.alinear(disponible, fechas)
    e_min, e_max = b['E_MWh']*b['soc_min_pu'], b['E_MWh']*b['soc_max_pu']
    if not (e_min <= float(soc_inicial) <= e_max):
        raise ValueError(f'SOC inicial {soc_inicial} fuera de [{e_min}, {e_max}]')
    cv, pmin, pmax = (np.asarray(P[k], float) for k in ('cv', 'pmin', 'pmax'))
    cargos = np.broadcast_to(np.asarray(costo_arranque, float), (4,))
    x_dir = np.minimum(pv, load)
    delta, eps = load - x_dir, pv - x_dir
    q_tot, c_tot, soc_tot, exp_tot = (np.zeros(n) for _ in range(4))
    segmentos, dias = [], []
    estado, e = estado_termico, float(soc_inicial)
    inicio = time.perf_counter()
    for a, z in segmentos_dias(fechas):
        t0 = time.perf_counter()
        dd, ee, pp, cc, ff = delta[a:z], eps[a:z], ci[a:z], ce[a:z], fechas[a:z]
        dsub = None if disp is None else disp[a:z]
        referencia = evaluador_dia(dd, pp, cv, pmin, pmax, ff, cargos, estado, disponible=dsub)
        candidatos = candidatos_sistema(dd, pp, cv, pmin, pmax, cargos, referencia, b)
        q, aceptados, rechazos = seleccionar_plan(candidatos, dd, ee, e, b, pp, cv, pmin)
        export = np.zeros(z - a)
        q_exe = np.zeros(z - a)
        ctx = dict(candidatos=candidatos, delta=dd, eps=ee, precio=pp, ce=cc, fechas=ff, cv=cv, pmin=pmin, pmax=pmax,
                   cargos=cargos, estado=estado, soc0=e, b=b, p_exp=p_exp_max, q_exe=q_exe, export=export,
                   disponible=dsub,
                   madrugada=_ventana_madrugada(delta, eps, ci, ce, fechas, z, extension_periodica, disponible=disp))
        c_dia, soc_dia, exportaciones = np.zeros(z - a), np.zeros(z - a), []
        e_hora = e
        for i in range(z - a):
            c_i = min(ee[i], b['P_MW'], max(e_max - e_hora, 0.)/b['eta_c']) if ee[i] > 0 else 0.
            if exportar_bess and dd[i] <= TOL_DEFICIT_MW and c_i <= TOL_FLUJO_MW and cc[i] > 0:
                xpv_i = min(max(ee[i] - c_i, 0.), p_exp_max)
                y_max = min(b['P_MW'] - q[i], max(e_hora - e_min, 0.)*b['eta_d'], max(p_exp_max - xpv_i, 0.))
                if y_max > MIN_DESCARGA_MW:
                    mejor, evaluadas = decidir_exportacion_d(ctx, i, y_max)
                    exportaciones.append(dict(hora=int(a + i), fecha=ff[i], y_MW=mejor['y'], x_SOC_MWh=mejor['x_soc'],
                                              ingreso_USD=mejor['ingreso'], delta_futuro_USD=mejor['delta_futuro'],
                                              beneficio_USD=mejor['beneficio'], y_max_MW=y_max,
                                              horas_madrugada=(0 if ctx['madrugada'] is None else ctx['madrugada']['horas']),
                                              origen_madrugada=(None if ctx['madrugada'] is None else ctx['madrugada']['origen']),
                                              evaluadas=evaluadas,
                                              decision='exporta' if mejor['y'] > 0 else 'conserva'))
                    if mejor['y'] > 0:   # replanificación del BESS: solo cambian sus horas posteriores
                        export = mejor['plan']['export'].copy()
                        q = mejor['plan']['q'].copy()
                        aceptados = [x for x in aceptados if max(x['horas']) <= i] + mejor['plan']['aceptados']
                        ctx['export'] = export
            q_exe[i] = q[i]
            ctx['q_exe'] = q_exe
            e_hora = e_hora + b['eta_c']*c_i - (q[i] + export[i])/b['eta_d']
            c_dia[i], soc_dia[i] = c_i, e_hora
        c_ref, soc_ref = trayectoria_soc(q, ee, e, b, export)
        assert np.abs(c_ref - c_dia).max() <= TOL_FLUJO_MW and np.abs(soc_ref - soc_dia).max() <= TOL_SOC_MWh
        final = evaluador_dia(np.maximum(dd - q, 0.), pp, cv, pmin, pmax, ff, cargos, estado, disponible=dsub)
        q_tot[a:z], c_tot[a:z], soc_tot[a:z], exp_tot[a:z] = q, c_dia, soc_dia, export
        segmentos.append(dict(inicio=a, fin=z, estado_previo=estado, resultado=final, segundos=time.perf_counter() - t0))
        dias.append(dict(fecha=ff[0], q=q, aceptados=aceptados, rechazos=dict(rechazos),
                         n_candidatos_S=sum(x['tipo'] == 'S' for x in candidatos),
                         n_candidatos_U=sum(x['tipo'] == 'U' for x in candidatos),
                         objetivo_referencia_USD=referencia['objetivo'], objetivo_final_USD=final['objetivo'],
                         valor_atribuido_USD=math.fsum(x['valor_atribuido_USD'] for x in aceptados),
                         ahorro_redespacho_USD=referencia['objetivo'] - final['objetivo'],
                         exportaciones=exportaciones))
        e = float(soc_dia[-1])
        estado = final['estado_final']
    xpv = exportacion_pv(eps, c_tot, ce, p_exp_max)
    vert = np.maximum(eps - c_tot, 0.) - xpv
    p = np.concatenate([s['resultado']['potencia'] for s in segmentos])
    arr = np.concatenate([s['resultado']['arranques'] for s in segmentos])
    costo_termica = math.fsum(s['resultado']['objetivo'] for s in segmentos)
    ingreso = math.fsum(((xpv + exp_tot)*ce).tolist())
    return dict(Caso='D_INT' if exportar_bess else 'D_INT_sin_exportacion', Tipo='politica', Fechas=fechas, PV=pv,
                Carga=load, PV_directo=x_dir, Carga_BESS=c_tot, Descarga_demanda=q_tot, Descarga_export=exp_tot,
                Export_PV=xpv, Vertimiento=vert, SOC=soc_tot, SOC_inicial=float(soc_inicial),
                Frontera_SOC=dict(tipo=frontera, valor=float(soc_inicial)), BESS=b,
                Residual=np.maximum(delta - q_tot, 0.), potencia=p,
                compromiso=np.concatenate([s['resultado']['compromiso'] for s in segmentos]), arranques=arr,
                paradas=np.concatenate([s['resultado']['paradas'] for s in segmentos]), C_imp_USD=ci, C_exp_USD=ce,
                Cargo=costo_arranque, P_EXP_MAX=p_exp_max,
                Costo_combustible_CVOM_USD=math.fsum((p[:, :4]@cv).tolist()), Costo_red_USD=math.fsum((p[:, 4]*ci).tolist()),
                Costo_arranques_USD=math.fsum((arr@cargos).tolist()), Costo_termica_red_arranques_USD=costo_termica,
                Ingreso_exportacion_USD=ingreso, Objetivo=costo_termica - ingreso, segmentos=segmentos, dias_b=dias,
                Valor_atribuido_USD=math.fsum(d['valor_atribuido_USD'] for d in dias),
                Ahorro_redespacho_USD=math.fsum(d['ahorro_redespacho_USD'] for d in dias),
                Exportaciones_BESS_MWh=float(exp_tot.sum()), Ingreso_export_BESS_USD=math.fsum((exp_tot*ce).tolist()),
                Extension_periodica=extension_periodica, estado_termico_inicial=estado_termico,
                estado_termico_final=estado, Segundos_simulacion=time.perf_counter() - inicio, Disponible=disp)


def ciclo_soc_politica(simular, b, max_iter=25, tol=1e-4):
    """Punto fijo cíclico del SOC de una política, separado de la simulación con SOC explícito.

    `simular(e)` devuelve la trayectoria que empieza en `e`. Empieza en el SOC mínimo, como `despachar` del
    cuaderno 02. Devuelve CONVERGE con la trayectoria cuyo SOC inicial cierra, CICLO si el SOC final repite un
    SOC inicial anterior, o NO_CONVERGE al agotar `max_iter`. NO_CONVERGE significa que no se alcanzó el cierre
    con ese límite, no que no exista punto fijo: con más iteraciones puede detectarse un CICLO.
    En CICLO y NO_CONVERGE no hay trayectoria: no se promedia ni se ajusta la última hora."""
    e = b['E_MWh']*b['soc_min_pu']
    visitados = {round(e, 6): 0}
    historial = []
    for k in range(1, max_iter + 1):
        res = simular(e)
        e_fin = float(res['SOC'][-1])
        historial.append(dict(iteracion=k, soc_inicial=e, soc_final=e_fin, residuo=e_fin - e))
        if abs(e_fin - e) <= tol:
            return dict(estado='CONVERGE', iteraciones=k, soc_inicial=e, soc_final=e_fin, residuo=e_fin - e,
                        historial=historial, max_iter=max_iter, resultado=res)
        clave = round(e_fin, 6)
        if clave in visitados:
            return dict(estado='CICLO', iteraciones=k, periodo=k - visitados[clave], soc_inicial=e, soc_final=e_fin,
                        residuo=e_fin - e, historial=historial, max_iter=max_iter, resultado=None)
        visitados[clave] = k
        e = e_fin
    return dict(estado='NO_CONVERGE', iteraciones=max_iter, soc_inicial=historial[-1]['soc_inicial'],
                soc_final=historial[-1]['soc_final'], residuo=historial[-1]['residuo'], historial=historial,
                max_iter=max_iter, resultado=None)


def ciclo_soc_d_int(fechas, pv, load, c_imp_usd, c_exp_usd, b, p_exp_max, P, costo_arranque, estado_termico,
                    max_iter=25, tol=1e-4, exportar_bess=True, extension_periodica=True, disponible=None):
    """Cierre cíclico del SOC de D_INT, con el mismo procedimiento y diagnósticos que B_INT."""
    def simular(e):
        return simular_d_int(fechas, pv, load, c_imp_usd, c_exp_usd, b, p_exp_max, P, costo_arranque, estado_termico, e,
                             frontera='ciclica_d_int', exportar_bess=exportar_bess,
                             extension_periodica=extension_periodica, disponible=disponible)
    return ciclo_soc_politica(simular, b, max_iter=max_iter, tol=tol)


def validar_d_int(res, P):
    """`validar_b_int` con exportación del BESS permitida, más la elegibilidad temporal de cada exportación:
    sin déficit tras el PV directo, sin carga simultánea, precio positivo, y dentro de potencia y de la
    capacidad de exportación restante tras el PV sobrante."""
    v = validar_b_int(res, P, exportar_bess=True)
    de = np.asarray(res['Descarga_export'], float)
    dd = np.asarray(res['Descarga_demanda'], float)
    c = np.asarray(res['Carga_BESS'], float)
    ce = np.asarray(res['C_exp_USD'], float)
    xpv = np.asarray(res['Export_PV'], float)
    delta = np.asarray(res['Carga'], float) - np.asarray(res['PV_directo'], float)
    horas = de > TOL_FLUJO_MW
    assert (delta[horas] <= TOL_DEFICIT_MW).all(), 'exportación del BESS en hora con déficit tras el PV directo'
    assert (c[horas] <= TOL_FLUJO_MW).all(), 'exportación del BESS con carga simultánea'
    assert (ce[horas] > 0).all(), 'exportación del BESS con precio no positivo'
    assert (de + dd <= res['BESS']['P_MW'] + TOL_FLUJO_MW).all(), 'potencia de descarga del BESS'
    assert (de + xpv <= res['P_EXP_MAX'] + TOL_FLUJO_MW).all(), 'capacidad de exportación'
    registradas = math.fsum(x['y_MW'] for d in res['dias_b'] for x in d.get('exportaciones', []) if x['decision'] == 'exporta')
    assert abs(registradas - float(de.sum())) <= TOL_FLUJO_MW, 'exportaciones registradas distintas de las ejecutadas'
    ingreso_bess = math.fsum((de*ce).tolist())
    assert abs(ingreso_bess - float(res['Ingreso_export_BESS_USD'])) <= TOL_OBJETIVO_USD
    return dict(v, Exportaciones_BESS_MWh=float(de.sum()), Horas_exportacion_BESS=int(horas.sum()),
                Ingreso_export_BESS_USD=ingreso_bess)
