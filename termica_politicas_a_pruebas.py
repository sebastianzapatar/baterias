# -*- coding: utf-8 -*-
"""Pruebas cortas de A_INT y SIN_BESS_DP (`termica_politicas.simular_a_int`). No ejecuta el año.

    .venv/Scripts/python.exe -B termica_politicas_a_pruebas.py --salida resultados_politicas/<carpeta_nueva>

1. A_INT reproduce exactamente los flujos del BESS de A original (`despachar` del cuaderno 02).
2. SIN_BESS_DP: flujos y estados del BESS en cero.
3. Ventanas reales de 72 h (enero, abril, diciembre) × T0/T1: validación integrada, continuidad del
   estado térmico entre días, DP = MILP en cada día, encadenado frente a óptimo conjunto de 72 h,
   térmica apagada con residual cero, costos e ingresos reconstruidos y tiempos.
   Enero parte del estado libre (primera hora del año). Abril y diciembre usan un estado previo
   explícito SINTÉTICO (unidades apagadas la víspera): supuesto de prueba, no el estado de la
   trayectoria anual.
4. Ventana parcial que empieza a las 12:00 con eventos ya usados (estado sintético).
5. Rechazo de estados y resultados alterados.
6. Enero (única ventana compatible: estado térmico libre como PyPSA, misma frontera de SOC):
   OPT_AB y SIN_BESS secuencial en PyPSA frente a A_INT y SIN_BESS_DP.

Escribe `a_int_pruebas.csv` en `--salida`; no sobrescribe.
"""
import argparse
import copy
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import termica_politicas as TPOL
import termica_politicas_pruebas as TPP
import termica_reglas as TR


def milp_par(ctx):
    """Costos y límites del MILP de contraste: los mismos de la trayectoria (escenario vigente)."""
    return {k: ctx['P'][k] for k in ('cv', 'pmin', 'pmax')}

FILAS = []
TOL_MILP_USD = TPP.TOL_MILP_USD
TOL_PYPSA_USD = 1e-3      # tolerancia de optimalidad y factibilidad de PyPSA/HiGHS usada en termica_reglas
VENTANAS = (('2025-01-01 00:00', 72), ('2025-04-10 00:00', 72), ('2025-12-29 00:00', 72))


def ok(prueba, **detalle):
    FILAS.append(dict(Prueba=prueba, Estado='OK', **detalle))
    print('OK', prueba, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in detalle.items()}, flush=True)


def estado_sintetico(inicio):
    """Supuesto de prueba: unidades apagadas y sin eventos la víspera. No es el estado de la trayectoria anual."""
    return TPP.estado([0]*4, [0]*4, [0]*4, pd.Timestamp(inicio) - pd.Timedelta(hours=1))


def ventana(ctx, inicio, horas, senal):
    E = ctx['E']
    a = E['FECHA'].get_loc(pd.Timestamp(inicio))
    sl = slice(a, a + horas)
    fechas = E['FECHA'][sl]
    return dict(fechas=fechas, pv=np.asarray(E['PV_MW'])[sl], load=np.asarray(E['LOAD'])[sl],
                ci=ctx['C_IMP_COP'][senal][sl], ce=ctx['C_EXP_COP'][sl], env=TR.entorno(ctx['ns'], fechas))


def simular(ctx, caso, w, estado):
    return TPOL.simular_a_int(caso, w['env'], w['fechas'], w['pv'], w['load'], w['ci'], w['ce'], ctx['TRM'], ctx['b'],
                              ctx['P_EXP_MAX'], ctx['P'], TR.CARGO, estado)


def igualdad_flujos(ctx):
    """A_INT = A original: misma llamada al `despachar` del cuaderno 02, sin pasar por termica_politicas."""
    b = ctx['b']
    for inicio, horas in VENTANAS[:2]:
        for s in ('T0', 'T1'):
            w = ventana(ctx, inicio, horas, s)
            estado = 'libre' if inicio.startswith('2025-01-01') else estado_sintetico(inicio)
            res = simular(ctx, 'A_INT', w, estado)
            d = w['env']['despachar'](np.asarray(w['pv'], float), np.asarray(w['load'], float), np.asarray(w['ci'], float),
                                      np.asarray(w['ce'], float), TR.REGLAS['A'], power_MW=b['P_MW'], cap_MWh=b['E_MWh'],
                                      c_deg=TR.C_DEG, p_exp_max=ctx['P_EXP_MAX'])
            pares = (('Carga_BESS', d.Carga_PV_MW), ('Descarga_demanda', d.Descarga_Demanda_MW),
                     ('Descarga_export', d.Descarga_Red_MW), ('Export_PV', d.Export_MW - d.Descarga_Red_MW),
                     ('Vertimiento', d.Vertimiento_MW), ('SOC', d.SOC_MWh), ('PV_directo', d.PV_a_Load_MW))
            for clave, col in pares:
                np.testing.assert_array_equal(res[clave], col.to_numpy(float), err_msg=clave)
            assert res['SOC_inicial'] == d.attrs['SOC_inicial_MWh']
            ok(f'igualdad_flujos_A_INT_A_{inicio[:10]}_{s}', Horas=horas, Diferencia_maxima_MW=0.,
               SOC_inicial_MWh=res['SOC_inicial'], Carga_MWh=float(res['Carga_BESS'].sum()),
               Descarga_MWh=float(res['Descarga_demanda'].sum()))


def ventanas_reales(ctx):
    P = ctx['P']
    for inicio, horas in VENTANAS:
        enero = inicio.startswith('2025-01-01')
        for s in ('T0', 'T1'):
            w = ventana(ctx, inicio, horas, s)
            estado = 'libre' if enero else estado_sintetico(inicio)
            for caso in ('SIN_BESS_DP', 'A_INT'):
                t = time.perf_counter()
                res = simular(ctx, caso, w, estado)
                t_sim = time.perf_counter() - t
                t = time.perf_counter()
                v = TPOL.validar_politica(res, P)
                t_val = time.perf_counter() - t
                if caso == 'SIN_BESS_DP':
                    for k in ('Carga_BESS', 'Descarga_demanda', 'Descarga_export', 'SOC'):
                        assert (res[k] == 0).all(), k
                    assert res['SOC_inicial'] == 0 and res['BESS']['P_MW'] == 0 and res['BESS']['E_MWh'] == 0
                    np.testing.assert_array_equal(res['Residual'], w['load'] - np.minimum(w['pv'], w['load']))
                # Continuidad: cada día hereda exactamente el estado final del anterior; sin arranques gratuitos.
                segs = res['segmentos']
                for k in range(1, len(segs)):
                    assert TPOL._mismo_estado(segs[k]['estado_previo'], segs[k-1]['resultado']['estado_final'])
                    assert not segs[k]['resultado']['estado_inicial']['inicial_libre']
                    assert segs[k]['resultado']['estado_inicial']['contadores_reiniciados']
                medianoche = [sg['inicio'] for sg in segs[1:]]
                arranques_medianoche = int(res['arranques'][medianoche].sum())
                cambios_medianoche = int((res['compromiso'][medianoche] > res['compromiso'][[m - 1 for m in medianoche]]).sum())
                assert arranques_medianoche == cambios_medianoche
                # Residual cero: sin sumidero, la térmica debe estar apagada.
                cero = res['Residual'] <= 1e-9
                assert not res['compromiso'][cero].any()
                # DP = MILP en cada día, con el mismo estado heredado.
                t = time.perf_counter()
                dif_milp = 0.
                for sg in segs:
                    a, z = sg['inicio'], sg['fin']
                    m = TPP.milp_ventana(res['Residual'][a:z], res['C_imp_USD'][a:z], w['fechas'][a:z], TR.CARGO, sg['estado_previo'],
                                         **milp_par(ctx))
                    dif_milp = max(dif_milp, abs(m['objetivo'] - sg['resultado']['objetivo']))
                t_milp = time.perf_counter() - t
                assert dif_milp <= TOL_MILP_USD, dif_milp
                # Encadenado frente a óptimo conjunto de la ventana (misma DP y MILP de 72 h).
                conj = TPOL.evaluador_dia(res['Residual'], res['C_imp_USD'], P['cv'], P['pmin'], P['pmax'], w['fechas'], TR.CARGO, estado)
                m72 = TPP.milp_ventana(res['Residual'], res['C_imp_USD'], w['fechas'], TR.CARGO, estado, **milp_par(ctx))
                assert abs(conj['objetivo'] - m72['objetivo']) <= TOL_MILP_USD
                dif_conj = res['Costo_termica_red_arranques_USD'] - conj['objetivo']
                assert dif_conj >= -TOL_MILP_USD
                ingreso = math.fsum(((res['Export_PV'] + res['Descarga_export'])*res['C_exp_USD']).tolist())
                ok(f'{caso}_{inicio[:10]}_{horas}h_{s}', Caso=caso, Senal=s, Horas=horas, Dias=len(segs),
                   Estado_termico_inicial='libre (primera hora del año)' if enero else 'sintético: apagadas la víspera',
                   SOC_inicial_MWh=res['SOC_inicial'], SOC_final_MWh=float(res['SOC'][-1]),
                   Costo_trayectoria_USD=res['Objetivo'], Costo_termica_red_arranques_USD=res['Costo_termica_red_arranques_USD'],
                   Ingreso_exportacion_USD=res['Ingreso_exportacion_USD'], Ingreso_recalculado_USD=ingreso,
                   Error_objetivo_USD=v['Error_objetivo_USD'], Error_ingreso_USD=v['Error_ingreso_USD'],
                   Diferencia_MILP_dias_USD=dif_milp, Costo_conjunto_72h_USD=conj['objetivo'],
                   Diferencia_encadenado_conjunto_USD=dif_conj, Arranques=int(res['arranques'].sum()),
                   Arranques_medianoche=arranques_medianoche, Horas_residual_cero=int(cero.sum()),
                   Red_MWh=float(res['potencia'][:, 4].sum()), Termica_MWh=float(res['potencia'][:, :4].sum()),
                   Descarga_MWh=float(res['Descarga_demanda'].sum()), Export_PV_MWh=float(res['Export_PV'].sum()),
                   Segundos_simulacion=t_sim, Segundos_BESS=res['Segundos_BESS'], Segundos_termica=res['Segundos_termica'],
                   Segundos_por_dia=res['Segundos_termica']/len(segs), Segundos_validacion=t_val, Segundos_MILP_dias=t_milp)


def ventana_parcial(ctx):
    """Ventana que empieza a las 12:00 con eventos ya usados hoy (estados SINTÉTICOS de prueba)."""
    inicio = '2025-04-10 12:00'
    w = ventana(ctx, inicio, 36, 'T0')
    # Factible: G1 arrancó hoy y sigue encendida (puede parar una vez); G2 arrancó y paró hoy (no puede rearrancar).
    prev = TPP.estado([1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], '2025-04-10 11:00')
    res = simular(ctx, 'A_INT', w, prev)
    v = TPOL.validar_politica(res, ctx['P'])
    primer = res['segmentos'][0]['resultado']
    ei = primer['estado_inicial']
    assert not ei['contadores_reiniciados'] and ei['arranques_usados'][0] == 1 and ei['paradas_usadas'][1] == 1
    assert primer['arranques'][:, :2].sum() == 0, 'G1 y G2 ya usaron su arranque de hoy'
    assert res['segmentos'][1]['resultado']['estado_inicial']['contadores_reiniciados']
    a, z = res['segmentos'][0]['inicio'], res['segmentos'][0]['fin']
    m = TPP.milp_ventana(res['Residual'][a:z], res['C_imp_USD'][a:z], w['fechas'][a:z], TR.CARGO, prev, **milp_par(ctx))
    dif = abs(m['objetivo'] - primer['objetivo'])
    assert dif <= TOL_MILP_USD
    # Misma trayectoria del BESS (no depende de la térmica) con G2 bloqueada: encendida y con su parada ya usada.
    bloqueado = TPP.estado([1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], '2025-04-10 11:00')
    residual_d1, fechas_d1 = res['Residual'][a:z], w['fechas'][a:z]
    try:
        TPOL.evaluador_dia(residual_d1, res['C_imp_USD'][a:z], ctx['P']['cv'], ctx['P']['pmin'], ctx['P']['pmax'],
                           fechas_d1, TR.CARGO, bloqueado)
        factible_dp = True
    except ValueError:
        factible_dp = False
    factible_milp = TPP.milp_ventana(residual_d1, res['C_imp_USD'][a:z], fechas_d1, TR.CARGO, bloqueado,
                                     **milp_par(ctx)) is not None
    assert factible_dp == factible_milp
    ok('ventana_parcial_eventos_usados_A_INT', Estado_previo='sintético 11:00: G1 arrancó hoy; G2 arrancó y paró hoy',
       Costo_trayectoria_USD=res['Objetivo'], Arranques=int(res['arranques'].sum()), Diferencia_MILP_USD=dif,
       Error_objetivo_USD=v['Error_objetivo_USD'], Horas_residual_cero_dia_1=int((residual_d1 <= 1e-9).sum()),
       Variante_G2_bloqueada='factible' if factible_dp else 'infactible', MILP_coincide_factibilidad=True)
    try:
        simular(ctx, 'A_INT', w, TPP.estado([1, 0, 0, 0], [1, 0, 0, 0], [0]*4, '2025-04-10 10:00'))
    except ValueError:
        ok('ventana_parcial_rechaza_estado_de_otra_hora')
    try:
        simular(ctx, 'A_INT', ventana(ctx, '2025-04-10 00:00', 24, 'T0'), 'libre')
    except ValueError:
        ok('estado_libre_rechazado_en_abril')


def rechazos(ctx):
    w = ventana(ctx, '2025-01-01 00:00', 72, 'T0')
    res = simular(ctx, 'A_INT', w, 'libre')
    TPOL.validar_politica(res, ctx['P'])
    t_carga = int(np.flatnonzero(res['Carga_BESS'] > 1)[0])
    t_red = int(np.flatnonzero(res['potencia'][:, 4] > 1e-6)[0]) if (res['potencia'][:, 4] > 1e-6).any() else 0
    t_exc = int(np.flatnonzero(res['PV'] - res['PV_directo'] > 1)[0])
    cero = tuple(np.argwhere(res['compromiso'] == 0)[0])

    def poner(clave, idx, delta):
        def f(q):
            q[clave] = np.array(q[clave], float)
            q[clave][idx] += delta
        return f

    def estado_seg(q):
        e = q['segmentos'][1]['estado_previo']
        q['segmentos'][1]['estado_previo'] = dict(e, u=1 - np.asarray(e['u']))

    def frac(q):
        q['compromiso'] = q['compromiso'].astype(float)
        q['compromiso'][cero] = 0.9

    alteraciones = {
        'carga_mayor_que_excedente': poner('Carga_BESS', t_carga, 1e3),
        'soc_alterado': poner('SOC', 10, 1.),
        'exportacion_desde_BESS': poner('Descarga_export', t_exc, 0.5),
        'red_alterada': poner('potencia', (t_red, 4), 0.1),
        'exportacion_PV_en_hora_con_red': poner('Export_PV', t_red, 0.5),
        'frontera_SOC_declarada': lambda q: q['Frontera_SOC'].__setitem__('valor', q['Frontera_SOC']['valor'] + 1.),
        'estado_heredado_dia_2': estado_seg,
        'estado_termico_final': lambda q: q['estado_termico_final'].__setitem__('fecha', q['Fechas'][-1] - pd.Timedelta(hours=1)),
        'compromiso_0_9': frac,
        'objetivo_mas_0_01_USD': lambda q: q.__setitem__('Objetivo', q['Objetivo'] + 0.01),
        'ingreso_mas_0_01_USD': lambda q: q.__setitem__('Ingreso_exportacion_USD', q['Ingreso_exportacion_USD'] + 0.01),
    }
    rechazados = []
    for nombre, cambio in alteraciones.items():
        q = copy.deepcopy(res)
        cambio(q)
        assert TPP._rechaza(lambda: TPOL.validar_politica(q, ctx['P'])), f'aceptó {nombre}'
        rechazados.append(nombre)
    # Diccionario sin pasar por TPP.estado, que convierte a enteros y truncaría 0,9 antes de llegar al evaluador.
    crudo = dict(u=[0.9, 0, 0, 0], arranques_usados=[0]*4, paradas_usadas=[0]*4, fecha=pd.Timestamp('2025-04-09 23:00'))
    for nombre, e in (('estado_u_0_9', crudo),
                      ('estado_incoherente', TPP.estado([0, 0, 0, 0], [1, 0, 0, 0], [0]*4, '2025-04-10 05:00'))):
        wv = ventana(ctx, '2025-04-10 00:00' if nombre == 'estado_u_0_9' else '2025-04-10 06:00', 24, 'T0')
        assert TPP._rechaza(lambda: simular(ctx, 'A_INT', wv, e)), nombre
        rechazados.append(nombre)
    ok('rechazos_A_INT_estados_y_resultados_alterados', Rechazados=';'.join(rechazados), Casos=len(rechazados))


def comparacion_pypsa_enero(ctx):
    """Enero es compatible: estado térmico libre sin evento en la primera hora (igual que PyPSA), final libre,
    mismos límites diarios y la frontera de SOC de A_INT impuesta a OPT_AB."""
    P, trm, b = ctx['P'], ctx['TRM'], ctx['b']
    for s in ('T0', 'T1'):
        w = ventana(ctx, '2025-01-01 00:00', 72, s)
        ci, ce = w['ci']/trm, w['ce']/trm
        a_int = simular(ctx, 'A_INT', w, 'libre')
        sin_dp = simular(ctx, 'SIN_BESS_DP', w, 'libre')
        soc0 = a_int['SOC_inicial']
        cierre = abs(float(a_int['SOC'][-1]) - soc0)
        # La A original cierra su SOC con tolerancia; OPT_AB lo fija exacto. Esa energía vale a lo sumo
        # cierre/η_d por el mayor precio de reposición de la ventana.
        tol_frontera = cierre/b['eta_d']*max(float(ci.max()), float(np.max(P['cv'])), float(ce.max())) + TOL_PYPSA_USD
        t = time.perf_counter()
        oab = TR.optimo('OPT_AB', w['fechas'], w['pv'], w['load'], ci, ce, P, b, ctx['P_EXP_MAX'], soc0_MWh=soc0,
                        segundos=300, brecha=0.)
        t_oab = time.perf_counter() - t
        TR.validar_integrado(oab, P, False, soc0)
        assert oab['Cota'] <= a_int['Objetivo'] + tol_frontera
        d0 = TR.despachar_regla(w['env'], w['pv'], w['load'], w['ci'], w['ce'], 'SIN_BESS', b, ctx['P_EXP_MAX'])
        seq = TR.acoplar('SIN_BESS', d0, w['fechas'], ci, ce, P, b, segundos=300, brecha=0.)
        TR.validar_integrado(seq, P, False)
        conj = TPOL.evaluador_dia(sin_dp['Residual'], ci, P['cv'], P['pmin'], P['pmax'], w['fechas'], TR.CARGO, 'libre')
        assert seq['Objetivo'] <= sin_dp['Objetivo'] + TOL_PYPSA_USD
        assert abs((conj['objetivo'] - sin_dp['Ingreso_exportacion_USD']) - seq['Objetivo']) <= TOL_PYPSA_USD
        ok(f'comparacion_PyPSA_enero_{s}', Senal=s, SOC_frontera_MWh=soc0, Cierre_SOC_A_INT_MWh=cierre,
           Tolerancia_frontera_USD=tol_frontera, Costo_A_INT_USD=a_int['Objetivo'], OPT_AB_USD=oab['Objetivo'],
           Cota_OPT_AB_USD=oab['Cota'], Distancia_A_INT_OPT_AB_USD=a_int['Objetivo'] - oab['Objetivo'],
           Costo_SIN_BESS_DP_USD=sin_dp['Objetivo'], SIN_BESS_PyPSA_72h_USD=seq['Objetivo'],
           Perdida_gestor_diario_USD=sin_dp['Objetivo'] - seq['Objetivo'],
           DP_conjunta_72h_menos_PyPSA_USD=(conj['objetivo'] - sin_dp['Ingreso_exportacion_USD']) - seq['Objetivo'],
           Segundos_OPT_AB=t_oab)
    ok('comparacion_PyPSA_abril_diciembre_no_directa',
       Motivo='estado térmico inicial sintético explícito frente al estado libre sin evento de PyPSA; '
              'las fronteras térmicas no coinciden y no se compara el costo')


def controles_estado_y_componentes(ctx):
    """validar_politica hereda los controles de estado y reconstruye cada componente económico por separado."""
    w = ventana(ctx, '2025-01-01 00:00', 72, 'T0')
    res = simular(ctx, 'A_INT', w, 'libre')
    TPOL.validar_politica(res, ctx['P'])
    rechazados = []
    trasladada = copy.deepcopy(res)
    desplazamiento = pd.Timestamp('2025-04-10') - pd.Timestamp('2025-01-01')
    trasladada['Fechas'] = res['Fechas'] + desplazamiento
    try:
        TPOL.validar_politica(trasladada, ctx['P'])
    except ValueError as e:
        assert 'primera hora del año' in str(e), e
    else:
        raise AssertionError('aceptó una trayectoria de enero trasladada a abril con estado libre')
    rechazados.append('enero_trasladado_a_abril_con_estado_libre')
    for nombre, cambios in (('combustible_mas_10_red_menos_10', {'Costo_combustible_CVOM_USD': 10., 'Costo_red_USD': -10.}),
                            ('arranques_mas_10_red_menos_10', {'Costo_arranques_USD': 10., 'Costo_red_USD': -10.}),
                            ('combustible_mas_10_arranques_menos_10', {'Costo_combustible_CVOM_USD': 10., 'Costo_arranques_USD': -10.}),
                            ('combustible_no_finito', {'Costo_combustible_CVOM_USD': np.inf})):
        q = copy.deepcopy(res)
        for k, dlt in cambios.items():
            q[k] = q[k] + dlt
        assert TPP._rechaza(lambda: TPOL.validar_politica(q, ctx['P'])), nombre
        rechazados.append(nombre)
    ok('rechazos_estado_trasladado_y_componentes_economicos', Rechazados=';'.join(rechazados), Casos=len(rechazados))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--salida', type=Path, required=True)
    parser.add_argument('--referencia', type=Path, help='CSV de una corrida anterior cuyas métricas deterministas deben coincidir')
    args = parser.parse_args()
    destino = args.salida/'a_int_pruebas.csv'
    if destino.exists():
        raise SystemExit(f'{destino} ya existe; no se sobrescribe. Usar otra --salida.')
    for nombre in ('pypsa', 'linopy'):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    inicio = time.perf_counter()
    ctx = TR.entradas()
    igualdad_flujos(ctx)
    ventanas_reales(ctx)
    ventana_parcial(ctx)
    rechazos(ctx)
    comparacion_pypsa_enero(ctx)
    controles_estado_y_componentes(ctx)
    if args.referencia:
        ok('referencia_sin_cambios', **TPP.comparar_referencia(pd.DataFrame(FILAS), args.referencia))
    args.salida.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(FILAS)
    tabla.to_csv(destino, index=False)
    print(f'{len(tabla)} pruebas OK en {time.perf_counter() - inicio:.0f} s; escrito {destino}', flush=True)


if __name__ == '__main__':
    main()
