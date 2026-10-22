# -*- coding: utf-8 -*-
"""Pruebas cortas del evaluador térmico diario común (`termica_politicas.evaluador_dia`).

    .venv/Scripts/python.exe -B termica_politicas_pruebas.py --salida resultados_politicas/evaluador

Cada ventana se contrasta contra un MILP independiente (SciPy/HiGHS) con los mismos datos,
estado previo, eventos disponibles y condición terminal libre. No hay corridas anuales.
Escribe `evaluador_pruebas.csv` en `--salida`; no sobrescribe.
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import termica as T
import termica_politicas as TPOL
import termica_pv as TP

TOL_MILP_USD = 1e-5
K = TP.CARGO_ARRANQUE_USD
# Datos fijos de los ejemplos sintéticos originales; independientes del escenario vigente.
PAR = TP.parametros_termicos(gas=16, cvom=9.28)
CV, PMIN, PMAX = (np.asarray(PAR[k], float) for k in ('cv', 'pmin', 'pmax'))
# T0 del cuaderno 02 en USD/MWh (N3: punta 06:00–21:00 = 659,9496 y fuera = 653,2247 COP/kWh; TRM 3.126,08).
PUNTA, FUERA = 659.9496e3/3126.08, 653.2247e3/3126.08
FILAS = []


def ok(prueba, **detalle):
    FILAS.append(dict(Prueba=prueba, Estado='OK', **detalle))
    print('OK', prueba, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in detalle.items()}, flush=True)


def precio_t0(fechas):
    h = pd.DatetimeIndex(fechas).hour
    return np.where((h >= 6) & (h <= 20), PUNTA, FUERA)


def estado(u, a, b, fecha):
    """Estado previo de prueba. No convierte a enteros: un 0,9 llega intacto al componente probado."""
    return dict(u=np.asarray(u), arranques_usados=np.asarray(a), paradas_usadas=np.asarray(b), fecha=pd.Timestamp(fecha))


def milp_ventana(residual, red, fechas, cargos, estado_previo, cv=CV, pmin=PMIN, pmax=PMAX, disponible=None):
    """MILP independiente: potencia, red, u, arranque y, parada z por hora; estado previo y eventos usados.

    `disponible` (versión 2): None o booleano (n,) o (n, 4); fija u = 0 en las horas no disponibles."""
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import lil_matrix, vstack
    r = np.asarray(residual, float)
    red = np.broadcast_to(np.asarray(red, float), r.shape)
    cargos = np.broadcast_to(np.asarray(cargos, float), (4,))
    fechas = pd.DatetimeIndex(fechas)
    n = len(r)
    nv = 17*n          # p1..p4, red, u1..u4, y1..y4, z1..z4
    c = np.zeros(nv)
    lo, hi = np.zeros(nv), np.full(nv, np.inf)
    entero = np.zeros(nv, int)
    for t in range(n):
        c[17*t:17*t+4] = cv
        c[17*t+4] = red[t]
        c[17*t+9:17*t+13] = cargos
        hi[17*t+5:17*t+17] = 1
        entero[17*t+5:17*t+17] = 1
    if disponible is not None:
        d = np.asarray(disponible, bool)
        d = np.repeat(d[:, None], 4, axis=1) if d.ndim == 1 else d
        for t, j in zip(*np.nonzero(~d)):
            hi[17*t+5+j] = 0.
    filas, li, ls = [], [], []

    def add(d, a=-np.inf, b=np.inf):
        filas.append(d); li.append(a); ls.append(b)

    libre = isinstance(estado_previo, str)
    for t in range(n):
        add({17*t+j: 1. for j in range(5)}, r[t], r[t])
        for j in range(4):
            p, u, y, z = 17*t+j, 17*t+5+j, 17*t+9+j, 17*t+13+j
            add({p: 1., u: -pmax[j]}, b=0.)
            add({p: -1., u: pmin[j]}, b=0.)
            add({y: 1., z: 1.}, b=1.)
            if t == 0 and libre:
                hi[y] = hi[z] = 0.
            elif t == 0:
                add({u: 1., y: -1., z: 1.}, float(estado_previo['u'][j]), float(estado_previo['u'][j]))
            else:
                add({u: 1., u - 17: -1., y: -1., z: 1.}, 0., 0.)
    dias = fechas.normalize()
    dia_previo = None if libre else pd.Timestamp(estado_previo['fecha']).normalize()
    for dia in dias.unique():
        idx = np.flatnonzero(dias == dia)
        for j in range(4):
            usado_a = int(estado_previo['arranques_usados'][j]) if dia == dia_previo else 0
            usado_b = int(estado_previo['paradas_usadas'][j]) if dia == dia_previo else 0
            add({17*t+9+j: 1. for t in idx}, b=1. - usado_a)
            add({17*t+13+j: 1. for t in idx}, b=1. - usado_b)
    a = lil_matrix((len(filas), nv))
    for i, d in enumerate(filas):
        for k, v in d.items():
            a[i, k] = v
    a = a.tocsr()
    res = milp(c, integrality=entero, bounds=Bounds(lo, hi), constraints=LinearConstraint(a, li, ls),
               options={'mip_rel_gap': 0., 'time_limit': 120.})
    if res.status == 2:
        return None
    assert res.success, res.message
    # Pulido: binarias fijas y LP continuo con tolerancias estrictas.
    lo2, hi2 = lo.copy(), hi.copy()
    lo2[entero == 1] = hi2[entero == 1] = np.rint(res.x[entero == 1])
    li, ls = np.asarray(li), np.asarray(ls)
    ig = li == ls
    arriba, abajo = np.isfinite(ls) & ~ig, np.isfinite(li) & ~ig
    lp = linprog(c, A_ub=vstack([a[arriba], -a[abajo]]), b_ub=np.r_[ls[arriba], -li[abajo]], A_eq=a[ig], b_eq=ls[ig],
                 bounds=list(zip(lo2, hi2)), method='highs',
                 options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    assert lp.success, lp.message
    return dict(objetivo=float(lp.fun), compromiso=np.rint(res.x[entero == 1]).reshape(n, 12)[:, :4].astype(int))


def contrastar(nombre, residual, red, fechas, estado_previo, cargos=K, **extra):
    ev = TPOL.evaluador_dia(residual, red, CV, PMIN, PMAX, fechas, cargos, estado_previo)
    v = TPOL.validar_evaluador(ev, residual, red, CV, PMIN, PMAX, fechas, cargos, estado_previo)
    m = milp_ventana(residual, red, fechas, cargos, estado_previo)
    assert m is not None, (nombre, 'MILP infactible con DP factible')
    dif = abs(ev['objetivo'] - m['objetivo'])
    assert dif <= TOL_MILP_USD, (nombre, ev['objetivo'], m['objetivo'])
    ok(nombre, Horas=len(residual), Objetivo_DP_USD=ev['objetivo'], Objetivo_MILP_USD=m['objetivo'],
       Diferencia_MILP_USD=dif, Arranques=int(ev['arranques'].sum()), Paradas=int(ev['paradas'].sum()),
       Red_MWh=float(ev['potencia'][:, 4].sum()), Termica_MWh=float(ev['potencia'][:, :4].sum()),
       Eventos_dia_max=v['Eventos_dia_max'], **extra)
    return ev


def infactible(nombre, residual, red, fechas, estado_previo, **extra):
    try:
        TPOL.evaluador_dia(residual, red, CV, PMIN, PMAX, fechas, K, estado_previo)
    except ValueError as e:
        assert 'factible' in str(e), e
    else:
        raise AssertionError(f'{nombre}: la DP aceptó un caso infactible')
    assert milp_ventana(residual, red, fechas, K, estado_previo) is None, f'{nombre}: el MILP sí encuentra solución'
    ok(nombre, Horas=len(residual), Objetivo_DP_USD=np.nan, Objetivo_MILP_USD=np.nan, Diferencia_MILP_USD=0.,
       Resultado='infactible en DP y MILP', **extra)


def dia(fecha):
    return pd.date_range(fecha, periods=24, freq='h')


def apagado_vispera(fecha):
    return estado([0]*4, [0]*4, [0]*4, pd.Timestamp(fecha) - pd.Timedelta(hours=1))


def casos_sinteticos():
    f = dia('2025-03-10')
    red = precio_t0(f)
    # 1. Arranque evitado por compras de red: 1,5 MW a las 21:00. Por energía, G4 (205,12) es más
    #    barata que la red (208,96), pero con su arranque cuesta 407,68 frente a 313,44.
    r = np.zeros(24); r[21] = 1.5
    hora = T.despachar_hora(1.5, red[21], CV, PMIN, PMAX)
    assert tuple(hora['compromiso']) == (0, 0, 0, 1)
    ev = contrastar('arranque_evitado_por_red', r, red, f, apagado_vispera(f[0]),
                    Compromiso_solo_energia='G4', Costo_esperado_USD=1.5*FUERA)
    assert ev['arranques'].sum() == 0 and abs(ev['objetivo'] - 1.5*FUERA) < 1e-9
    # 2. Mínimos técnicos: residuales entre combinaciones de Pmin.
    r = np.array([10.5, 8.5, 3.0, 2.2, 1.0, 0.5, 9.6, 16.4, 17.2, 19.1, 4.4, 2.4] + [7.]*12)
    ev = contrastar('minimos_tecnicos', r, red, f, estado([1, 1, 0, 0], [0]*4, [0]*4, f[0] - pd.Timedelta(hours=1)))
    assert (ev['potencia'][:, :4] >= ev['compromiso']*PMIN - 1e-9).all()
    # 3. Caída del residual a cero (caso F): G1 viene encendida; a las 05:00 el residual cae bajo su Pmin,
    #    06–08 vuelve a 6 MW y 09–15 hay excedente PV. Reencender G1 dejaría una unidad sin parada disponible.
    r = np.r_[[5.]*5, 2.0, [6.]*3, [0.]*7, [0.]*8]
    prev = estado([1, 0, 0, 0], [0]*4, [0]*4, f[0] - pd.Timedelta(hours=1))
    ev = contrastar('caida_residual_cero_caso_F_voraz', r, red, f, prev)
    g1 = ev['compromiso'][:, 0]
    assert not (ev['arranques'][:, 0].sum() and ev['paradas'][:, 0].sum() > 0 and g1[9:].any())
    ev_piso = contrastar('caida_residual_cero_caso_F_con_piso', np.r_[[5.]*5, 2.4, 5.6, 6., 6., [0.]*15], red, f, prev)
    ok('caso_F_resumen', Costo_voraz_USD=ev['objetivo'], Costo_con_piso_USD=ev_piso['objetivo'],
       Ahorro_piso_USD=ev['objetivo'] - ev_piso['objetivo'])
    # 4. Límites diarios agotados en una ventana que empieza a las 12:00, con eventos usados.
    fm = pd.date_range('2025-03-10 12:00', periods=12, freq='h')
    prev_m = estado([1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], '2025-03-10 11:00')   # G1 reencendida; G2 sin eventos
    rm = np.r_[[12.]*5, [4.]*7]
    ev = contrastar('limites_diarios_agotados_ventana_12h', rm, precio_t0(fm), fm, prev_m)
    assert ev['arranques'][:, 1].sum() == 0, 'G2 ya usó su arranque'
    sin_eventos = TPOL.evaluador_dia(rm, precio_t0(fm), CV, PMIN, PMAX, fm, K,
                                     estado([1, 0, 1, 0], [0]*4, [0]*4, '2025-03-10 11:00'))
    ok('eventos_usados_cambian_la_solucion', Costo_con_eventos_usados_USD=ev['objetivo'],
       Costo_si_se_reiniciaran_USD=sin_eventos['objetivo'])
    infactible('limites_agotados_unidad_bloqueada_residual_cero', np.r_[[6.]*3, [0.]*9], precio_t0(fm), fm, prev_m)
    try:
        TPOL.evaluador_dia(rm, precio_t0(fm), CV, PMIN, PMAX, fm, K, estado([1, 0, 1, 0], [0]*4, [0]*4, '2025-03-09 23:00'))
    except ValueError as e:
        assert 'estado previo' in str(e)
    else:
        raise AssertionError('aceptó un estado previo que no es la hora anterior')
    ok('rechaza_estado_previo_discontinuo')
    # 5. Continuidad a medianoche: G1 bloqueada la víspera puede parar a las 00:00; G3 sigue sin arranque.
    f2 = dia('2025-03-11')
    prev = estado([1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], '2025-03-10 23:00')
    r2 = np.r_[[3.]*6, [0.]*18]
    ev = contrastar('continuidad_medianoche', r2, precio_t0(f2), f2, prev)
    assert ev['estado_inicial']['contadores_reiniciados'] and ev['arranques'][0].sum() == 0, 'sin arranque gratuito a las 00:00'
    assert ev['compromiso'][6:].sum() == 0, 'con contadores reiniciados todas las unidades pueden parar ante residual cero'
    # El mismo residual dentro del mismo día (G1 sin parada disponible) es infactible.
    f_mismo = pd.date_range('2025-03-10 18:00', periods=6, freq='h')
    infactible('sin_reinicio_dentro_del_dia_es_infactible', np.r_[[3.]*2, [0.]*4], precio_t0(f_mismo), f_mismo,
               estado([1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], '2025-03-10 17:00'))
    fv = pd.date_range('2025-03-10 20:00', periods=10, freq='h')
    rv = np.r_[[9.]*4, [3.]*6]
    contrastar('ventana_que_cruza_medianoche', rv, precio_t0(fv), fv, estado([1, 1, 0, 0], [1, 1, 0, 0], [0]*4, '2025-03-10 19:00'))
    # 6. Primera hora del año: estado libre sin evento, igual que resolver_eventos auditado.
    fa = dia('2025-01-01')
    ra = np.full(24, 5.)
    ev = contrastar('primera_hora_del_anio_libre', ra, precio_t0(fa), fa, 'libre')
    ref = T.resolver_eventos(ra, precio_t0(fa), CV, PMIN, PMAX, fa, 0, costo_arranque=K, inicial='libre')
    assert abs(ref['costo_dp'] - ev['objetivo']) < 1e-6 and ev['arranques'].sum() == 0
    ok('libre_igual_a_resolver_eventos_auditado', Objetivo_resolver_eventos_USD=ref['costo_dp'])
    try:
        TPOL.evaluador_dia(ra, precio_t0(dia('2025-01-02')), CV, PMIN, PMAX, dia('2025-01-02'), K, 'libre')
    except ValueError:
        ok('libre_rechazado_fuera_de_la_primera_hora')
    # 7. Sin generación térmica conveniente: red a 120 USD/MWh y residual cero con unidades encendidas.
    ev = contrastar('sin_termica_red_barata', np.full(24, 5.), np.full(24, 120.), f, apagado_vispera(f[0]))
    assert ev['potencia'][:, :4].sum() == 0 and ev['arranques'].sum() == 0
    ev = contrastar('sin_termica_residual_cero', np.zeros(24), red, f, estado([1, 1, 1, 1], [0]*4, [0]*4, f[0] - pd.Timedelta(hours=1)))
    assert ev['objetivo'] == 0 and ev['paradas'][0].sum() == 4


def encadenado_frente_a_conjunto():
    """Dos días: la DP de cada día con valor terminal cero no ve que conviene arrancar a las 23:00."""
    f = pd.date_range('2025-03-10', periods=48, freq='h')
    r = np.zeros(48); r[23] = 2.2; r[24:30] = 2.2
    red = precio_t0(f)
    prev = apagado_vispera(f[0])
    d1 = contrastar('encadenado_dia_1', r[:24], red[:24], f[:24], prev)
    d2 = contrastar('encadenado_dia_2', r[24:], red[24:], f[24:], d1['estado_final'])
    conjunto = contrastar('conjunto_dos_dias', r, red, f, prev)
    diferencia = d1['objetivo'] + d2['objetivo'] - conjunto['objetivo']
    assert diferencia > 1., 'el caso debe mostrar la pérdida del valor terminal cero'
    ok('encadenado_distinto_del_conjunto_sintetico', Costo_encadenado_USD=d1['objetivo'] + d2['objetivo'],
       Costo_conjunto_USD=conjunto['objetivo'], Diferencia_USD=diferencia)


def ejemplos_propuesta():
    """E1, E2 y F de la propuesta con datos completos y reoptimización de todas las unidades."""
    f = dia('2025-03-10')
    red = precio_t0(f)
    prev = apagado_vispera(f[0])
    # E1: 20:00 (punta) 6,5 MW y 21:00 (fuera) 1,5 MW; 1,5 MWh descargables.
    base = np.zeros(24); base[20], base[21] = 6.5, 1.5
    tarifa = base.copy(); tarifa[20] -= 1.5          # B original: descarga en la hora de mayor c_imp
    sistema = base.copy(); sistema[21] -= 1.5        # B integrada: guarda para la hora sin térmica factible
    e_t = contrastar('E1_descarga_por_tarifa', tarifa, red, f, prev)
    e_s = contrastar('E1_descarga_por_sistema', sistema, red, f, prev)
    ok('E1_resumen', Costo_tarifa_USD=e_t['objetivo'], Costo_sistema_USD=e_s['objetivo'], Ahorro_USD=e_t['objetivo'] - e_s['objetivo'])
    # E2: día completo. 05–06: 1,5 MW con red a 180 USD/MWh (menor que G3, mayor que G1); 13:00: nube, 3 MW;
    #     17–22: 8 MW; resto 0 (excedente PV o PV = demanda). Unidades apagadas la víspera. 3 MWh descargables.
    r = np.zeros(24); r[5:7] = 1.5; r[13] = 3.; r[17:23] = 8.
    red2 = red.copy(); red2[5:7] = 180.
    sin = contrastar('E2_sin_BESS', r, red2, f, prev)
    madrugada = r.copy(); madrugada[5:7] = 0.
    nube = r.copy(); nube[13] = 0.
    e_m = contrastar('E2_descarga_05_06', madrugada, red2, f, prev)
    e_n = contrastar('E2_descarga_13', nube, red2, f, prev)
    marginal_05 = red2[5]
    marginal_13 = CV[0] if sin['compromiso'][13].any() else red2[13]
    ok('E2_resumen', Costo_sin_BESS_USD=sin['objetivo'], Ahorro_descarga_05_06_USD=sin['objetivo'] - e_m['objetivo'],
       Ahorro_descarga_13_USD=sin['objetivo'] - e_n['objetivo'], Valor_marginal_05_06_USD_MWh=marginal_05,
       Valor_marginal_13_USD_MWh=marginal_13,
       Eleccion_marginal='05-06' if marginal_05 > marginal_13 else '13',
       Eleccion_reoptimizada='05-06' if e_m['objetivo'] < e_n['objetivo'] else '13',
       Diferencia_USD=e_m['objetivo'] - e_n['objetivo'])


def dias_reales():
    """Días reales SIN_BESS (demanda − PV directo), T0 y T1, encadenados y en ventana conjunta."""
    E = TP.cargar_entradas_pv()
    residual_anual = np.asarray(E['LOAD'], float) - np.minimum(np.asarray(E['PV_MW'], float), np.asarray(E['LOAD'], float))
    for inicio, dias_n in (('2025-01-01', 3), ('2025-04-10', 3), ('2025-12-29', 3)):
        a = E['FECHA'].get_loc(pd.Timestamp(inicio))
        for s in ('T0', 'T1'):
            estado_d = 'libre' if inicio == '2025-01-01' else apagado_vispera(inicio)
            inicial = estado_d
            total = 0.
            for d in range(dias_n):
                sl = slice(a + 24*d, a + 24*(d+1))
                ev = contrastar(f'real_{inicio}_{s}_dia{d+1}', residual_anual[sl], E['C_IMP_USD'][s][sl], E['FECHA'][sl], estado_d)
                total += ev['objetivo']
                estado_d = ev['estado_final']
            sl = slice(a, a + 24*dias_n)
            conj = contrastar(f'real_{inicio}_{s}_conjunto_{dias_n}d', residual_anual[sl], E['C_IMP_USD'][s][sl],
                              E['FECHA'][sl], inicial)
            assert conj['objetivo'] <= total + TOL_MILP_USD
            ok(f'real_{inicio}_{s}_encadenado_frente_a_conjunto', Costo_encadenado_USD=total,
               Costo_conjunto_USD=conj['objetivo'], Diferencia_USD=total - conj['objetivo'],
               Estado_inicial='libre' if inicio == '2025-01-01' else 'apagadas la víspera (supuesto de prueba)')


def aleatorias(n=40, semilla=20260915):
    rng = np.random.default_rng(semilla)
    iguales = infactibles = 0
    for i in range(n):
        mitad = i % 3 == 2
        horas = 12 if mitad else 24
        inicio = pd.Timestamp('2025-06-02 12:00' if mitad else '2025-06-02 00:00') + pd.Timedelta(days=i)
        f = pd.date_range(inicio, periods=horas, freq='h')
        r = np.round(rng.choice([0., 1., 2.], size=horas, p=[.25, .15, .6]) * rng.uniform(0, 12, size=horas), 2)
        red = rng.choice([120., 170., FUERA, PUNTA, 240.], size=horas)
        loc = [TPOL.LOCALES[k] for k in rng.integers(0, 6, size=4)]
        u, a_, b_ = (np.array([x[q] for x in loc]) for q in range(3))
        prev = estado(u, a_, b_, inicio - pd.Timedelta(hours=1))
        m = milp_ventana(r, red, f, K, prev)
        try:
            ev = TPOL.evaluador_dia(r, red, CV, PMIN, PMAX, f, K, prev)
        except ValueError:
            assert m is None, f'aleatoria {i}: DP infactible y MILP factible'
            infactibles += 1
            continue
        TPOL.validar_evaluador(ev, r, red, CV, PMIN, PMAX, f, K, prev)
        assert m is not None and abs(ev['objetivo'] - m['objetivo']) <= TOL_MILP_USD, (i, ev['objetivo'], m)
        iguales += 1
    ok('aleatorias_contra_MILP', Casos=n, Factibles_iguales=iguales, Infactibles_en_ambos=infactibles, Semilla=semilla)


def _rechaza(funcion):
    try:
        funcion()
    except (ValueError, AssertionError):
        return True
    return False


def rechazos_evaluador():
    """Binarios no exactos, estado final alterado y costo alterado en 0,01 USD deben rechazarse."""
    f = dia('2025-03-10')
    red = precio_t0(f)
    r = np.r_[[5.]*5, 2.0, [6.]*3, [0.]*15]
    prev = estado([1, 0, 0, 0], [0]*4, [0]*4, f[0] - pd.Timedelta(hours=1))
    rechazados = []
    for nombre, cambio in (('u_previo_0_9', dict(u=[0.9, 0, 0, 0])), ('u_previo_2', dict(u=[2, 0, 0, 0])),
                           ('arranque_usado_0_5', dict(arranques_usados=[0.5, 0, 0, 0])),
                           ('parada_usada_nan', dict(paradas_usadas=[np.nan, 0, 0, 0]))):
        malo = dict(prev, **cambio)
        assert _rechaza(lambda: TPOL.evaluador_dia(r, red, CV, PMIN, PMAX, f, K, malo)), nombre
        rechazados.append(nombre)
    ev = TPOL.evaluador_dia(r, red, CV, PMIN, PMAX, f, K, prev)
    TPOL.validar_evaluador(ev, r, red, CV, PMIN, PMAX, f, K, prev)
    # Ventana dentro del día con eventos heredados: el estado final debe sumar los eventos previos.
    fm = pd.date_range('2025-03-10 12:00', periods=12, freq='h')
    prev_m = estado([1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], '2025-03-10 11:00')
    rm = np.r_[[12.]*5, [4.]*7]
    ev_m = TPOL.evaluador_dia(rm, precio_t0(fm), CV, PMIN, PMAX, fm, K, prev_m)
    TPOL.validar_evaluador(ev_m, rm, precio_t0(fm), CV, PMIN, PMAX, fm, K, prev_m)
    assert ev_m['estado_final']['arranques_usados'][0] == 1 and ev_m['estado_final']['paradas_usadas'][1] == 1
    ceros = np.argwhere(ev['compromiso'] == 0)[0]

    def fraccion(q):
        q['compromiso'] = q['compromiso'].astype(float)
        q['compromiso'][tuple(ceros)] = 0.9

    def voltea(clave):
        def f_(q):
            q['estado_final'][clave] = 1 - np.asarray(q['estado_final'][clave])
        return f_

    alteraciones = {
        'compromiso_0_9': (ev, r, red, f, prev, fraccion),
        'estado_final_u': (ev, r, red, f, prev, voltea('u')),
        'estado_final_arranques': (ev, r, red, f, prev, voltea('arranques_usados')),
        'estado_final_paradas': (ev, r, red, f, prev, voltea('paradas_usadas')),
        'estado_final_sin_eventos_previos': (ev_m, rm, precio_t0(fm), fm, prev_m,
                                             lambda q: q['estado_final'].__setitem__('arranques_usados', np.zeros(4, int))),
        'estado_final_fecha': (ev, r, red, f, prev, lambda q: q['estado_final'].__setitem__('fecha', f[-1] + pd.Timedelta(hours=1))),
        'objetivo_mas_0_01_USD': (ev, r, red, f, prev, lambda q: q.__setitem__('objetivo', q['objetivo'] + 0.01)),
    }
    for nombre, (base, rr, pr, ff, ee, cambio) in alteraciones.items():
        q = copy.deepcopy(base)
        cambio(q)
        assert _rechaza(lambda: TPOL.validar_evaluador(q, rr, pr, CV, PMIN, PMAX, ff, K, ee)), f'aceptó {nombre}'
        rechazados.append(nombre)
    ok('rechazos_binarios_estado_final_y_costo', Rechazados=';'.join(rechazados), Casos=len(rechazados),
       Tolerancia_objetivo_absoluta_USD=TPOL.TOL_OBJETIVO_USD)


def comparar_referencia(tabla, referencia, excluir_prefijos=('Segundos', 'Referencia')):
    """Las filas deterministas de una corrida anterior deben conservarse sin cambios (mismas soluciones).
    Se excluyen tiempos, rutas de referencia y las filas que a su vez comparaban contra otra referencia."""
    import io
    ref = pd.read_csv(referencia, float_precision='round_trip')
    ref = ref[~ref.Prueba.str.startswith('referencia')]
    buf = io.StringIO()
    tabla.to_csv(buf, index=False)
    buf.seek(0)
    nueva = pd.read_csv(buf, float_precision='round_trip').set_index('Prueba')
    faltan = sorted(set(ref.Prueba) - set(nueva.index))
    assert not faltan, ('faltan pruebas de la referencia', faltan)
    ref = ref.set_index('Prueba')
    dif_max, distintas = 0., []
    for prueba in ref.index:
        for col in ref.columns:
            if col.startswith(excluir_prefijos):
                continue
            a, b = ref.at[prueba, col], nueva.at[prueba, col] if col in nueva.columns else np.nan
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (float, int, np.floating, np.integer)) and not isinstance(a, bool):
                d = abs(float(a) - float(b))
                dif_max = max(dif_max, d)
                if d > 1e-9:
                    distintas.append((prueba, col, a, b))
            elif str(a) != str(b):
                distintas.append((prueba, col, a, b))
    assert not distintas, ('soluciones distintas de la referencia', distintas[:10])
    return dict(Referencia=str(referencia), Pruebas_comparadas=len(ref), Diferencia_maxima=dif_max)


def controles_calendario_estado():
    """validar_evaluador comprueba por sí mismo calendario, estado libre, fecha previa y contadores previos."""
    fa = dia('2025-01-01')
    ra = np.full(24, 5.)
    ev = TPOL.evaluador_dia(ra, precio_t0(fa), CV, PMIN, PMAX, fa, K, 'libre')
    TPOL.validar_evaluador(ev, ra, precio_t0(fa), CV, PMIN, PMAX, fa, K, 'libre')
    rechazados = []
    abril = dia('2025-04-10')
    casos = {
        'enero_trasladado_a_abril_con_estado_libre': (ev, ra, abril, 'libre'),
        'estado_libre_con_otro_texto': (ev, ra, fa, 'Libre'),
    }
    f = dia('2025-03-10')
    r = np.r_[[5.]*5, 2.0, [6.]*3, [0.]*15]
    prev = estado([1, 0, 0, 0], [0]*4, [0]*4, '2025-03-09 23:00')
    ev2 = TPOL.evaluador_dia(r, precio_t0(f), CV, PMIN, PMAX, f, K, prev)
    TPOL.validar_evaluador(ev2, r, precio_t0(f), CV, PMIN, PMAX, f, K, prev)
    casos['fecha_previa_no_contigua'] = (ev2, r, f, estado([1, 0, 0, 0], [0]*4, [0]*4, '2025-03-09 22:00'))
    casos['contadores_fraccionarios_de_la_vispera'] = (ev2, r, f, estado([1, 0, 0, 0], [0.5, 0, 0, 0], [0]*4, '2025-03-09 23:00'))
    casos['contadores_incoherentes_de_la_vispera'] = (ev2, r, f, estado([0, 0, 0, 0], [1, 0, 0, 0], [0]*4, '2025-03-09 23:00'))
    casos['calendario_con_hueco'] = (ev2, r, f.delete(12).append(pd.DatetimeIndex([f[-1] + pd.Timedelta(hours=1)])), prev)
    for nombre, (base, rr, ff, ee) in casos.items():
        assert _rechaza(lambda: TPOL.validar_evaluador(copy.deepcopy(base), rr, precio_t0(ff), CV, PMIN, PMAX, ff, K, ee)), nombre
        rechazados.append(nombre)
    # El motor también rechaza los contadores fraccionarios de la víspera aunque luego se reinicien.
    assert _rechaza(lambda: TPOL.evaluador_dia(r, precio_t0(f), CV, PMIN, PMAX, f, K, casos['contadores_fraccionarios_de_la_vispera'][3]))
    rechazados.append('motor_contadores_fraccionarios_de_la_vispera')
    ok('rechazos_calendario_y_estado_previo', Rechazados=';'.join(rechazados), Casos=len(rechazados))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--salida', type=Path, default=Path('resultados_politicas/evaluador'))
    parser.add_argument('--referencia', type=Path, help='CSV de una corrida anterior cuyas filas deben coincidir')
    args = parser.parse_args()
    destino = args.salida/'evaluador_pruebas.csv'
    if destino.exists():
        raise SystemExit(f'{destino} ya existe; no se sobrescribe. Usar otra --salida.')
    casos_sinteticos()
    encadenado_frente_a_conjunto()
    ejemplos_propuesta()
    aleatorias()
    dias_reales()
    rechazos_evaluador()
    controles_calendario_estado()
    if args.referencia:
        ok('referencia_sin_cambios', **comparar_referencia(pd.DataFrame(FILAS), args.referencia))
    args.salida.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(FILAS)
    tabla.to_csv(destino, index=False)
    contrastes = tabla['Diferencia_MILP_USD'].dropna()
    print(f"{len(tabla)} pruebas OK; diferencia máxima frente al MILP {contrastes.max():.2e} USD; escrito {destino}", flush=True)


if __name__ == '__main__':
    main()
