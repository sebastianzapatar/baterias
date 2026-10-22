# -*- coding: utf-8 -*-
"""Experimento: comparación con SOC cíclico libre de A_INT, B_INT y OPT_AB (ventanas cortas).

    .venv/Scripts/python.exe -B termica_politicas_ciclica.py --salida resultados_politicas/<carpeta_nueva>

Convención del experimento (no corrige las corridas anteriores con SOC fijo):
    SOC_final = SOC_inicial   y   SOC_min ≤ SOC_t ≤ SOC_max
para cada política y para el benchmark. A_INT conserva el ciclo natural de la regla A original
(`despachar` del cuaderno 02), B_INT su propio ciclo (`ciclo_soc_b_int`) y OPT_AB elige su SOC inicial
(`frontera_soc='ciclica'`). No se exige el mismo valor numérico de SOC: cada trayectoria devuelve al final
la energía con la que empieza (operación periódica), no es una puesta en marcha desde un inventario común.
Solo el BESS es cíclico: el estado térmico sigue libre sin evento en la primera hora del año, con límites
diarios y compromiso terminal libre.

Pruebas:
1. Sintética de 48 h (desde el 1 de enero, estado térmico libre): A_INT y B_INT cierran con SOC iniciales
   distintos; identidad energética con eficiencias y Δt = 1 h; sin energía neta del inventario; ambas
   pertenecen al benchmark cíclico común (decisiones fijadas en el mismo modelo PyPSA) y OPT_AB acota.
2. Enero real, 72 h, T0 y T1: SIN_BESS_DP, A_INT, B_INT y OPT_AB cíclico; validaciones; pertenencia de
   A_INT y B_INT con todas sus decisiones fijadas; descomposición con solo el BESS fijado (horizonte térmico).
Escribe `ciclica_pruebas.csv`, `comparacion_enero.csv` y `ejecucion.json` en `--salida`; no sobrescribe.
"""
import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import termica_politicas as TPOL
import termica_politicas_a_pruebas as TPA
import termica_politicas_pruebas as TPP
import termica_reglas as TR

FILAS, COMPARACION = [], []
TOL_FIJAR_MIN = 1e-7          # banda mínima de las igualdades fijadas (MW, MWh)
TOL_ENERGIA_MWh = 1e-6        # identidad energética del ciclo
DT_H = 1.0                    # duración de cada intervalo (h)


def ok(prueba, **detalle):
    FILAS.append(dict(Prueba=prueba, Estado='OK', **detalle))
    print('OK', prueba, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in detalle.items()}, flush=True)


def identidad_energia(res, b):
    """η_c·Σc·Δt − Σ(d/η_d)·Δt = SOC_final − SOC_inicial y cierre cíclico: sin energía neta del inventario."""
    carga_soc = b['eta_c']*math.fsum((np.asarray(res['Carga_BESS'], float)*DT_H).tolist())
    descarga_soc = math.fsum(((np.asarray(res['Descarga_demanda'], float) + np.asarray(res['Descarga_export'], float))
                              / b['eta_d']*DT_H).tolist())
    delta = float(res['SOC'][-1]) - float(res['SOC_inicial'])
    residuo = carga_soc - descarga_soc - delta
    assert abs(residuo) <= TOL_ENERGIA_MWh, ('identidad energética', residuo)
    assert abs(delta) <= TPOL.TOL_CIERRE_MWh, ('cierre cíclico', delta)
    return dict(Energia_cargada_SOC_MWh=carga_soc, Energia_descargada_SOC_MWh=descarga_soc,
                Energia_neta_inventario_MWh=-delta, Residuo_identidad_MWh=residuo)


def fijar_desde(res, termica=True):
    """Decisiones de la política para fijarlas en OPT_AB. La banda absorbe el cierre cíclico permitido."""
    cierre = abs(float(res['SOC'][-1]) - float(res['SOC_inicial']))
    d = dict(carga=res['Carga_BESS'], descarga_demanda=res['Descarga_demanda'], export_pv=res['Export_PV'],
             soc=res['SOC'], tol=max(TOL_FIJAR_MIN, 2*cierre))
    if termica:
        d.update(potencia=res['potencia'], compromiso=res['compromiso'])
    return d


def resolver_opt(w, P, b, p_exp, ci, ce, fijar=None, segundos=300):
    t = time.perf_counter()
    r = TR.optimo('OPT_AB', w['fechas'], w['pv'], w['load'], ci, ce, P, b, p_exp, frontera_soc='ciclica', cargo=TR.CARGO,
                  segundos=segundos, brecha=0., aceptar_limite_tiempo=True, fijar=fijar)
    r['Segundos_totales'] = time.perf_counter() - t
    TR.validar_integrado(r, P, False)
    assert r['Frontera_SOC_modo'] == 'ciclica' and abs(r['Residuo_cierre_SOC_MWh']) <= 1e-6
    assert abs(r['Residuo_primera_hora_SOC_MWh']) <= 1e-6, 'la primera hora no parte del SOC de cierre'
    return r


def pertenencia(res, w, P, b, p_exp, ci, ce):
    """La trayectoria completa (BESS, SOC, exportación PV, potencias y compromiso) es factible en OPT_AB cíclico
    y su objetivo coincide con el costo realizado, dentro del efecto de la banda."""
    fx = fijar_desde(res, termica=True)
    r = resolver_opt(w, P, b, p_exp, ci, ce, fijar=fx, segundos=120)
    assert r['Condicion'].endswith('/optimal'), r['Condicion']
    tol_obj = fx['tol']*math.fsum((np.sum(P['cv']) + ci + np.maximum(ce, 0.) + 1.).tolist()) + 1e-6
    dif = abs(r['Objetivo'] - res['Objetivo'])
    assert dif <= tol_obj, ('objetivo con decisiones fijadas', dif, tol_obj)
    assert np.array_equal(r['compromiso'], np.asarray(res['compromiso'], int))
    assert np.abs(r['SOC'] - res['SOC']).max() <= fx['tol'] + 1e-9
    return r, dif, tol_obj


def indicadores(nombre, res, P, extra=None):
    """Energías entregadas por ruta (MW × DT_H); la descarga total incluye demanda y exportación.

    Las descargas ya están del lado de entrega: no se aplica de nuevo eta_d. La energía retirada
    del SOC se comprueba por separado en identidad_energia. El modelo usa intervalos de una hora.
    """
    p = np.asarray(res['potencia'], float)
    arr = np.asarray(res['arranques']).sum(axis=0)
    cv = np.asarray(P['cv'], float)
    ci, ce = np.asarray(res['C_imp_USD'], float), np.asarray(res['C_exp_USD'], float)
    combustible = math.fsum((p[:, :4]@cv).tolist())
    red = math.fsum((p[:, 4]*ci).tolist())
    arranques = float(TR.CARGO*arr.sum())
    ingreso = math.fsum(((np.asarray(res['Export_PV']) + np.asarray(res['Descarga_export']))*ce).tolist())
    costo = combustible + red + arranques - ingreso
    assert abs(costo - float(res['Objetivo'])) <= 1e-6, (nombre, costo, res['Objetivo'])
    soc_ini = float(res['SOC_inicial'])
    descarga_demanda = float(np.sum(res['Descarga_demanda']))*DT_H
    descarga_export = float(np.sum(res['Descarga_export']))*DT_H
    fila = dict(Caso=nombre, Costo_realizado_USD=costo, Combustible_CVOM_USD=combustible, Red_USD=red,
                Arranques_USD=arranques, Ingreso_exportacion_USD=ingreso, Arranques=int(arr.sum()),
                Arranques_G1_G2=int(arr[:2].sum()), Arranques_G3=int(arr[2]), Arranques_G4=int(arr[3]),
                Termica_MWh=float(p[:, :4].sum()), Red_MWh=float(p[:, 4].sum()), Carga_BESS_MWh=float(np.sum(res['Carga_BESS'])),
                Descarga_demanda_MWh=descarga_demanda, Descarga_export_MWh=descarga_export,
                Descarga_BESS_MWh=descarga_demanda + descarga_export,
                Export_PV_MWh=float(np.sum(res['Export_PV'])),
                SOC_inicial_MWh=soc_ini, SOC_final_MWh=float(res['SOC'][-1]), Cierre_SOC_MWh=float(res['SOC'][-1]) - soc_ini)
    fila.update(extra or {})
    return fila


def ventana_sintetica(ctx, noche=2.4, excedente=40., tarde=12.):
    """48 h desde el 1 de enero (estado térmico libre), datos sintéticos repetidos cada día.
    00–07: `noche` MW; 09–15: excedente `excedente` MW; 17–23: `tarde` MW. Con 2,4 / 40 / 12 MW, A_INT y B_INT
    convergen a SOC cíclicos distintos (elegido en un barrido corto de 18 combinaciones)."""
    fechas = pd.date_range('2025-01-01', periods=48, freq='h')
    h = fechas.hour.to_numpy()
    delta = np.where(h <= 7, noche, np.where(h >= 17, tarde, 0.))
    eps = np.where((h >= 9) & (h <= 15), excedente, 0.)
    base = 2.
    load, pv = delta + base, eps + base
    ci = TPP.precio_t0(fechas)
    ce = np.full(48, 50.)
    return dict(fechas=fechas, pv=pv, load=load, ci=ci*ctx['TRM'], ce=ce*ctx['TRM'], env=TR.entorno(ctx['ns'], fechas))


def prueba_sintetica(ctx):
    P, b, trm = ctx['P'], ctx['b'], ctx['TRM']
    w = ventana_sintetica(ctx)
    ci, ce = w['ci']/trm, w['ce']/trm
    p_exp = ctx['P_EXP_MAX']
    a = TPA.simular(ctx, 'A_INT', w, 'libre')
    TPOL.validar_politica(a, P)
    ciclo = TPOL.ciclo_soc_b_int(w['fechas'], w['pv'], w['load'], ci, ce, b, p_exp, P, TR.CARGO, 'libre')
    assert ciclo['estado'] == 'CONVERGE', ciclo['estado']
    bb = ciclo['resultado']
    TPOL.validar_b_int(bb, P)
    ea, eb = identidad_energia(a, b), identidad_energia(bb, b)
    assert abs(a['SOC_inicial'] - bb['SOC_inicial']) > 1., 'la prueba debe tener SOC iniciales distintos'
    ra, da, ta = pertenencia(a, w, P, b, p_exp, ci, ce)
    rb, db, tb = pertenencia(bb, w, P, b, p_exp, ci, ce)
    opt = resolver_opt(w, P, b, p_exp, ci, ce)
    assert opt['Cota'] <= a['Objetivo'] + 1e-3 and opt['Cota'] <= bb['Objetivo'] + 1e-3
    ok('sintetica_dos_ciclos_distintos_en_benchmark_comun',
       SOC_inicial_A_INT_MWh=a['SOC_inicial'], SOC_inicial_B_INT_MWh=bb['SOC_inicial'], Iteraciones_B_INT=ciclo['iteraciones'],
       Identidad_A_MWh=ea['Residuo_identidad_MWh'], Identidad_B_MWh=eb['Residuo_identidad_MWh'],
       Energia_neta_inventario_A_MWh=ea['Energia_neta_inventario_MWh'], Energia_neta_inventario_B_MWh=eb['Energia_neta_inventario_MWh'],
       Costo_A_INT_USD=a['Objetivo'], Costo_B_INT_USD=bb['Objetivo'],
       Objetivo_fijado_A_USD=ra['Objetivo'], Diferencia_fijado_A_USD=da, Tolerancia_fijado_A_USD=ta, Condicion_fijado_A=ra['Condicion'],
       Objetivo_fijado_B_USD=rb['Objetivo'], Diferencia_fijado_B_USD=db, Tolerancia_fijado_B_USD=tb, Condicion_fijado_B=rb['Condicion'],
       OPT_AB_ciclico_USD=opt['Objetivo'], Cota_OPT_AB_USD=opt['Cota'], Condicion_OPT_AB=opt['Condicion'],
       SOC_inicial_OPT_AB_MWh=opt['SOC_inicial'], SOC_fin_primera_hora_OPT_AB_MWh=opt['SOC_fin_primera_hora'])


def prueba_no_convergencia(ctx):
    """Con 2,4 / 10 / 8 MW el punto fijo de B_INT no se alcanza con el límite predeterminado de 25 iteraciones:
    se devuelve NO_CONVERGE, sin trayectoria y sin promediar. NO_CONVERGE no demuestra que no exista punto fijo:
    con un límite suficiente se detecta CICLO (ver la prueba del ciclo en `termica_politicas_d_pruebas.py`).
    B_INT no se modifica para evitarlo."""
    w = ventana_sintetica(ctx, noche=2.4, excedente=10., tarde=8.)
    trm = ctx['TRM']
    ciclo = TPOL.ciclo_soc_b_int(w['fechas'], w['pv'], w['load'], w['ci']/trm, w['ce']/trm, ctx['b'], ctx['P_EXP_MAX'],
                                 ctx['P'], TR.CARGO, 'libre')
    assert ciclo['estado'] in ('NO_CONVERGE', 'CICLO') and ciclo['resultado'] is None, ciclo['estado']
    residuos = [round(h['residuo'], 4) for h in ciclo['historial']]
    ok('sintetica_B_INT_ciclo_no_convergente_detectado', Estado_ciclo=ciclo['estado'], Iteraciones=ciclo['iteraciones'],
       Residuos_distintos=';'.join(str(r) for r in sorted(set(residuos))), Ultimo_SOC_inicial_MWh=ciclo['soc_inicial'],
       Ultimo_SOC_final_MWh=ciclo['soc_final'], Trayectoria_devuelta=False)


def enero(ctx):
    P, b, trm = ctx['P'], ctx['b'], ctx['TRM']
    p_exp = ctx['P_EXP_MAX']
    for s in ('T0', 'T1'):
        w = TPA.ventana(ctx, '2025-01-01 00:00', 72, s)
        ci, ce = np.asarray(w['ci'], float)/trm, np.asarray(w['ce'], float)/trm
        sin = TPA.simular(ctx, 'SIN_BESS_DP', w, 'libre')
        TPOL.validar_politica(sin, P)
        a = TPA.simular(ctx, 'A_INT', w, 'libre')
        TPOL.validar_politica(a, P)
        t = time.perf_counter()
        ciclo = TPOL.ciclo_soc_b_int(w['fechas'], w['pv'], w['load'], ci, ce, b, p_exp, P, TR.CARGO, 'libre')
        t_b = time.perf_counter() - t
        assert ciclo['estado'] == 'CONVERGE', ciclo['estado']
        bb = ciclo['resultado']
        TPOL.validar_b_int(bb, P)
        opt = resolver_opt(w, P, b, p_exp, ci, ce)
        optimo_probado = opt['Condicion'].endswith('/optimal')
        filas = {}
        for nombre, res in (('A_INT', a), ('B_INT', bb)):
            e = identidad_energia(res, b)
            r_fija, dif, tol = pertenencia(res, w, P, b, p_exp, ci, ce)
            solo_bess = resolver_opt(w, P, b, p_exp, ci, ce, fijar=fijar_desde(res, termica=False))
            horizonte = res['Objetivo'] - solo_bess['Objetivo']
            filas[nombre] = indicadores(nombre, res, P, dict(
                Senal=s, Pertenece_OPT_AB_ciclico=True, Condicion_fijado=r_fija['Condicion'], Objetivo_fijado_USD=r_fija['Objetivo'],
                Diferencia_fijado_USD=dif, Tolerancia_fijado_USD=tol, Residuo_identidad_MWh=e['Residuo_identidad_MWh'],
                Distancia_OPT_AB_min_USD=res['Objetivo'] - opt['Objetivo'], Distancia_OPT_AB_max_USD=res['Objetivo'] - opt['Cota'],
                Objetivo_solo_BESS_fijado_USD=solo_bess['Objetivo'], Cota_solo_BESS_fijado_USD=solo_bess['Cota'],
                Condicion_solo_BESS_fijado=solo_bess['Condicion'],
                Efecto_horizonte_termico_min_USD=horizonte, Efecto_horizonte_termico_max_USD=res['Objetivo'] - solo_bess['Cota'],
                Efecto_decisiones_BESS_min_USD=solo_bess['Cota'] - opt['Objetivo'],
                Efecto_decisiones_BESS_max_USD=solo_bess['Objetivo'] - opt['Cota'],
                Ahorro_vs_SIN_BESS_DP_USD=sin['Objetivo'] - res['Objetivo'],
                Iteraciones_ciclo=ciclo['iteraciones'] if nombre == 'B_INT' else np.nan,
                Segundos_simulacion=t_b if nombre == 'B_INT' else res['Segundos_BESS'] + res['Segundos_termica']))
        filas['SIN_BESS_DP'] = indicadores('SIN_BESS_DP', sin, P, dict(Senal=s))
        filas['OPT_AB_ciclico'] = indicadores('OPT_AB_ciclico', opt, P, dict(
            Senal=s, Condicion_solver=opt['Condicion'], Optimo_probado=optimo_probado, Objetivo_USD=opt['Objetivo'],
            Cota_USD=opt['Cota'], Brecha_solver=opt['Brecha_solver'], Segundos_solver=opt['Segundos'],
            SOC_fin_primera_hora_MWh=opt['SOC_fin_primera_hora'], Residuo_primera_hora_MWh=opt['Residuo_primera_hora_SOC_MWh'],
            Ahorro_vs_SIN_BESS_DP_USD=sin['Objetivo'] - opt['Objetivo']))
        for nombre in ('SIN_BESS_DP', 'A_INT', 'B_INT', 'OPT_AB_ciclico'):
            COMPARACION.append(filas[nombre])
        ok(f'enero_ciclico_{s}', Optimo_probado_OPT_AB=optimo_probado, Condicion_OPT_AB=opt['Condicion'],
           SOC_A_INT_MWh=a['SOC_inicial'], SOC_B_INT_MWh=bb['SOC_inicial'], SOC_OPT_AB_MWh=opt['SOC_inicial'],
           Costo_A_INT_USD=a['Objetivo'], Costo_B_INT_USD=bb['Objetivo'], OPT_AB_USD=opt['Objetivo'], Cota_OPT_AB_USD=opt['Cota'],
           A_INT_pertenece=True, B_INT_pertenece=True)


def huella():
    archivos = ['termica_politicas.py', 'termica_politicas_ciclica.py', 'termica_reglas.py', 'termica_pypsa.py', 'termica.py']
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        head = f'no disponible: {e}'
    return dict(git_head=head, sha256={a: hashlib.sha256(Path(a).read_bytes()).hexdigest() for a in archivos})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--salida', type=Path, required=True)
    args = parser.parse_args()
    destinos = [args.salida/n for n in ('ciclica_pruebas.csv', 'comparacion_enero.csv', 'ejecucion.json')]
    if any(p.exists() for p in destinos):
        raise SystemExit(f'{args.salida} ya tiene resultados; no se sobrescriben. Usar otra --salida.')
    for nombre in ('pypsa', 'linopy'):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    inicio = time.perf_counter()
    ctx = TR.entradas()
    prueba_sintetica(ctx)
    prueba_no_convergencia(ctx)
    enero(ctx)
    args.salida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(FILAS).to_csv(destinos[0], index=False)
    pd.DataFrame(COMPARACION).to_csv(destinos[1], index=False)
    destinos[2].write_text(json.dumps(dict(huella(), convencion='SOC cíclico libre por trayectoria (solo BESS)',
                                           segundos=time.perf_counter() - inicio), indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(FILAS)} pruebas OK en {time.perf_counter() - inicio:.0f} s; escrito {args.salida}', flush=True)


if __name__ == '__main__':
    main()
