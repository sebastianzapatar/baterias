"""Cierre anual de A_INT/B_INT/D_INT y benchmarks conjuntos OPT_AB/OPT_D.

Usa los motores vigentes sin modificar políticas, tarifas, tamaños ni restricciones.
SOC cíclico libre por trayectoria; estado térmico libre solo al iniciar el año.
Cada caso se conserva en NPZ (series) y JSON comprimido (diagnóstico completo sin pickle).
Las etapas se pueden ejecutar por separado sin sobrescribir casos terminados.
"""
import argparse
import gzip
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import termica_politicas as POL
import termica_politicas_ciclica as CIC
import termica_reglas as REG

POLITICAS = ('A_INT', 'B_INT', 'D_INT')
BENCHMARKS = ('OPT_AB', 'OPT_D')
SERIES = ('PV', 'Carga', 'PV_directo', 'Carga_BESS', 'Descarga_demanda', 'Descarga_export',
          'Export_PV', 'Vertimiento', 'SOC', 'potencia', 'compromiso', 'arranques', 'paradas',
          'C_imp_USD', 'C_exp_USD')
CODIGO = ('termica_politicas_anual.py', 'termica_politicas.py', 'termica_politicas_ciclica.py',
          'termica_reglas.py', 'termica_pypsa.py', 'termica_pv.py', 'termica_bess.py', 'termica.py',
          'benchmark_optimo.py', 'solar_tmy.py', 'costos.py', '02_Optimizacion_portafolio_PV_BESS.ipynb')


def codificar(obj):
    if isinstance(obj, np.ndarray):
        return {'__array__': obj.tolist(), 'dtype': str(obj.dtype)}
    if isinstance(obj, pd.DatetimeIndex):
        return {'__fechas__': obj.astype(str).tolist()}
    if isinstance(obj, pd.Timestamp):
        return {'__fecha__': obj.isoformat()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def decodificar(obj):
    if '__array__' in obj:
        return np.asarray(obj['__array__'], dtype=obj['dtype'])
    if '__fechas__' in obj:
        return pd.DatetimeIndex(obj['__fechas__'])
    if '__fecha__' in obj:
        return pd.Timestamp(obj['__fecha__'])
    return obj


def escribir_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=codificar), encoding='utf-8')


def leer_caso(salida, etiqueta):
    with gzip.open(salida / f'{etiqueta}.json.gz', 'rt', encoding='utf-8') as f:
        return json.load(f, object_hook=decodificar)


def guardar_caso(salida, etiqueta, res):
    # El DataFrame original de A es redundante; sus atributos conservan la información del cierre.
    copia = {k: v for k, v in res.items() if k != 'despacho_regla'}
    if 'despacho_regla' in res:
        copia['atributos_regla_A'] = dict(res['despacho_regla'].attrs)
    for extension in ('.npz', '.json.gz'):
        assert not (salida / f'{etiqueta}{extension}').exists(), f'Caso existente: {etiqueta}'
    np.savez_compressed(salida / f'{etiqueta}.npz', Fecha=res['Fechas'].to_numpy(),
                        **{k: np.asarray(res[k]) for k in SERIES}, SOC_inicial=res['SOC_inicial'],
                        Objetivo=res['Objetivo'])
    with gzip.open(salida / f'{etiqueta}.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(copia, f, ensure_ascii=False, default=codificar)
    leido = leer_caso(salida, etiqueta)
    with np.load(salida / f'{etiqueta}.npz', allow_pickle=False) as series:
        for k in SERIES:
            np.testing.assert_array_equal(series[k], leido[k], err_msg=k)
    return leido


def validar(res, parametros, p_exp):
    if res['Caso'] == 'D_INT':
        v = POL.validar_d_int(res, parametros)
    elif res['Caso'] == 'B_INT':
        v = POL.validar_b_int(res, parametros)
    elif res['Caso'] == 'A_INT':
        v = POL.validar_politica(res, parametros)
    else:
        v = REG.validar_integrado(res, parametros, res['Caso'] == 'OPT_D')
    assert (res['Export_PV'] + res['Descarga_export'] <= p_exp + POL.TOL_FLUJO_MW).all()
    energia = CIC.identidad_energia(res, res['BESS'])
    assert res['Fechas'][0] == pd.Timestamp('2025-01-01') and len(res['Fechas']) == 8760
    return dict(v, **energia)


def indicadores(res, senal, parametros, ciclo=None):
    f = CIC.indicadores(res['Caso'], res, parametros)
    f.update(Senal=senal, Horas=len(res['Fechas']),
             Horas_unidad_encendida=int(res['compromiso'].sum()),
             Horas_exportacion_BESS=int((res['Descarga_export'] > POL.TOL_FLUJO_MW).sum()),
             PV_directo_MWh=float(res['PV_directo'].sum()), Vertimiento_MWh=float(res['Vertimiento'].sum()),
             Ingreso_BESS_USD=float(res['Descarga_export'] @ res['C_exp_USD']))
    for j in range(4):
        f[f'Arranques_G{j+1}'] = int(res['arranques'][:, j].sum())
        f[f'Paradas_G{j+1}'] = int(res['paradas'][:, j].sum())
    if res['Caso'] in BENCHMARKS:
        f.update(Cota_USD=res['Cota'], Brecha_solver=res['Brecha_solver'], Condicion=res['Condicion'],
                 Segundos=res['Segundos'], Exclusion=res['Exclusion'], Horas_reparadas=res['Horas_reparadas'],
                 Residuo_primera_hora_SOC_MWh=res['Residuo_primera_hora_SOC_MWh'])
    else:
        f.update(Condicion=ciclo['estado'], Iteraciones_ciclo=ciclo['iteraciones'],
                 Segundos=ciclo['segundos'], Residuo_ciclo_MWh=ciclo['residuo'])
    return f


def progreso(salida, etapa, **datos):
    registro = dict(fecha=pd.Timestamp.now().isoformat(), etapa=etapa, **datos)
    escribir_json(salida / 'progreso.json', registro)
    print(json.dumps(registro, ensure_ascii=False, default=codificar), flush=True)


def huella(ctx):
    E = ctx['E']
    archivos = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in CODIGO}
    entradas = {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in dict(
        pv=E['PV_MW'], demanda=E['LOAD'], fechas=E['FECHA'].to_numpy(),
        T0=ctx['C_IMP_COP']['T0'], T1=ctx['C_IMP_COP']['T1'], exportacion=ctx['C_EXP_COP']).items()}
    return dict(codigo=archivos, entradas=entradas)


def comparar(salida, filas):
    registros, valor_exportar = [], []
    por = {(f['Senal'], f['Caso']): f for f in filas}
    for senal in ('T0', 'T1'):
        for politica, benchmark in (('A_INT', 'OPT_AB'), ('B_INT', 'OPT_AB'), ('D_INT', 'OPT_D')):
            if (senal, politica) not in por or (senal, benchmark) not in por:
                continue
            p, o = por[senal, politica], por[senal, benchmark]
            cp, ub, lb = p['Costo_realizado_USD'], o['Costo_realizado_USD'], o['Cota_USD']
            assert lb <= cp + REG.TOL_USD, (senal, politica, 'cota mayor que política factible')
            registros.append(dict(Senal=senal, Politica=politica, Benchmark=benchmark,
                                   Costo_politica_USD=cp, Benchmark_factible_USD=ub, Benchmark_cota_USD=lb,
                                   Distancia_min_USD=max(0., cp-ub), Distancia_max_USD=max(0., cp-lb),
                                   Diferencia_vs_factible_USD=cp-ub,
                                   Distancia_min_pct=100*max(0., cp-ub)/ub if ub > 0 else np.nan,
                                   Distancia_max_pct=100*max(0., cp-lb)/lb if lb > 0 else np.nan,
                                   Condicion_benchmark=o['Condicion']))
        if (senal, 'OPT_AB') in por and (senal, 'OPT_D') in por:
            ab, d = por[senal, 'OPT_AB'], por[senal, 'OPT_D']
            assert d['Cota_USD'] <= ab['Costo_realizado_USD'] + REG.TOL_USD
            valor_exportar.append(dict(Senal=senal,
                Valor_exportacion_min_USD=max(0., ab['Cota_USD'] - d['Costo_realizado_USD']),
                Valor_exportacion_max_USD=max(0., ab['Costo_realizado_USD'] - d['Cota_USD']),
                Diferencia_costos_factibles_USD=ab['Costo_realizado_USD'] - d['Costo_realizado_USD']))
    if registros:
        pd.DataFrame(registros).to_csv(salida / 'comparacion.csv', index=False)
    if valor_exportar:
        pd.DataFrame(valor_exportar).to_csv(salida / 'valor_exportacion.csv', index=False)


def consolidar(salida, ctx):
    filas, validaciones = [], []
    for senal in ('T0', 'T1'):
        for caso in POLITICAS + BENCHMARKS:
            etiqueta = f'{senal}_{caso}'
            if not (salida / f'{etiqueta}.json.gz').exists():
                continue
            res = leer_caso(salida, etiqueta)
            v = validar(res, ctx['P'], ctx['P_EXP_MAX'])
            ciclo = json.loads((salida / f'{etiqueta}_ciclo.json').read_text(encoding='utf-8')) if caso in POLITICAS else None
            filas.append(indicadores(res, senal, ctx['P'], ciclo))
            validaciones.append(dict(Senal=senal, **v))
    if filas:
        pd.DataFrame(filas).to_csv(salida / 'resumen.csv', index=False)
        pd.DataFrame(validaciones).to_csv(salida / 'validaciones.csv', index=False)
        comparar(salida, filas)
    return filas


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--salida', required=True, type=Path)
    parser.add_argument('--etapa', required=True, choices=('politicas', 'benchmarks', 'validar'))
    parser.add_argument('--senales', nargs='+', choices=('T0', 'T1'), default=['T0', 'T1'])
    parser.add_argument('--casos', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--max-iter', type=int, default=25)
    parser.add_argument('--tol-ciclo', type=float, default=1e-4)
    parser.add_argument('--segundos', type=int, default=600)
    parser.add_argument('--brecha', type=float, default=1e-6)
    args = parser.parse_args()
    for logger in ('pypsa', 'linopy'):
        logging.getLogger(logger).setLevel(logging.ERROR)
    if args.etapa == 'validar':
        # La auditoría de trayectorias usa las condiciones registradas de esa corrida.
        # No carga entradas actuales ni afirma reproducir su optimización. Las etapas
        # que sí ejecutan modelos conservan la comprobación estricta de huella.
        meta = json.loads((args.salida / 'ejecucion.json').read_text(encoding='utf-8'),
                          object_hook=decodificar)
        ctx_guardado = dict(P=meta['parametros_termicos'], P_EXP_MAX=meta['P_EXP_MAX'])
        filas = consolidar(args.salida, ctx_guardado)
        assert filas, 'No hay trayectorias guardadas para validar'
        progreso(args.salida, 'etapa_terminada', nombre='validar', casos_validados=len(filas),
                 alcance='auditoría física y económica de series con parámetros guardados')
        return
    ctx = REG.entradas()
    E, P, b, trm = ctx['E'], ctx['P'], ctx['b'], ctx['TRM']
    fechas = pd.DatetimeIndex(E['FECHA'])
    assert fechas.equals(pd.date_range('2025-01-01', periods=8760, freq='h')), 'Se requiere el año horario completo'
    assert args.max_iter > 0 and args.segundos > 0 and 0 < args.tol_ciclo <= POL.TOL_CIERRE_MWh
    pv, load = np.asarray(E['PV_MW'], float), np.asarray(E['LOAD'], float)
    ce = ctx['C_EXP_COP']/trm
    env = REG.entorno(ctx['ns'], fechas)
    salida = args.salida
    salida.mkdir(parents=True, exist_ok=True)
    meta_path = salida / 'ejecucion.json'
    sello = huella(ctx)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'), object_hook=decodificar)
        assert meta['huella'] == sello, 'Código o entradas diferentes: usar otra carpeta'
        assert meta['max_iter'] == args.max_iter and meta['tol_ciclo_MWh'] == args.tol_ciclo
    else:
        assert args.etapa == 'politicas', 'Primero se deben ejecutar las políticas'
        head = subprocess.check_output(['git', '-c', f'safe.directory={Path.cwd().as_posix()}',
                                        'rev-parse', 'HEAD'], text=True).strip()
        meta = dict(inicio=pd.Timestamp.now().isoformat(), commit=head, huella=sello,
                    convencion='SOC cíclico libre por trayectoria; solo el BESS es cíclico',
                    termica_inicial='libre sin evento en la primera hora del año',
                    parametros_termicos=P, bess=b, TRM=trm, P_EXP_MAX=ctx['P_EXP_MAX'],
                    cargo_arranque_USD=REG.CARGO, minutos_eventos=0, limite_anual_arranques=None,
                    limite_diario='un arranque y una parada por unidad',
                    max_iter=args.max_iter, tol_ciclo_MWh=args.tol_ciclo,
                    tolerancia_validacion_cierre_MWh=POL.TOL_CIERRE_MWh,
                    exclusiones_benchmarks='binaria', horas=8760, python=sys.version)
        escribir_json(meta_path, meta)
    if args.etapa == 'politicas':
        for senal in args.senales:
            ci = ctx['C_IMP_COP'][senal]/trm
            for caso in POLITICAS:
                etiqueta = f'{senal}_{caso}'
                if (salida / f'{etiqueta}_ciclo.json').exists():
                    progreso(salida, 'politica_conservada', caso=etiqueta)
                    continue
                t0 = time.perf_counter()
                progreso(salida, 'inicio_politica', caso=etiqueta)
                if caso == 'A_INT':
                    res = POL.simular_a_int(caso, env, fechas, pv, load, ctx['C_IMP_COP'][senal],
                                            ctx['C_EXP_COP'], trm, b, ctx['P_EXP_MAX'], P, REG.CARGO, 'libre')
                    residuo = float(res['SOC'][-1] - res['SOC_inicial'])
                    ciclo = dict(estado='CONVERGE' if abs(residuo) <= POL.TOL_CIERRE_MWh else 'NO_CONVERGE',
                                 iteraciones=None, soc_inicial=res['SOC_inicial'], soc_final=float(res['SOC'][-1]),
                                 residuo=residuo, resultado=res, procedimiento='ciclo original de A del cuaderno 02',
                                 atributos_regla_A=dict(res['despacho_regla'].attrs))
                else:
                    contador = [0]
                    motor = POL.simular_b_int if caso == 'B_INT' else POL.simular_d_int
                    def simular(e):
                        contador[0] += 1
                        progreso(salida, 'iteracion_SOC', caso=etiqueta, iteracion=contador[0], soc_inicial_MWh=e)
                        r = motor(fechas, pv, load, ci, ce, b, ctx['P_EXP_MAX'], P, REG.CARGO, 'libre', e,
                                  frontera='ciclica_anual')
                        progreso(salida, 'fin_iteracion_SOC', caso=etiqueta, iteracion=contador[0],
                                 soc_inicial_MWh=e, soc_final_MWh=float(r['SOC'][-1]), residuo_MWh=float(r['SOC'][-1]-e))
                        return r
                    ciclo = POL.ciclo_soc_politica(simular, b, max_iter=args.max_iter, tol=args.tol_ciclo)
                res = ciclo.pop('resultado')
                ciclo['segundos'] = time.perf_counter() - t0
                if ciclo['estado'] == 'CONVERGE':
                    validar(res, P, ctx['P_EXP_MAX'])
                    leido = guardar_caso(salida, etiqueta, res)
                    v = validar(leido, P, ctx['P_EXP_MAX'])
                    escribir_json(salida / f'{etiqueta}_validacion.json', v)
                escribir_json(salida / f'{etiqueta}_ciclo.json', ciclo)
                progreso(salida, 'fin_politica', caso=etiqueta, estado=ciclo['estado'], residuo_MWh=ciclo['residuo'],
                         segundos=ciclo['segundos'], costo_USD=None if res is None else res['Objetivo'])
                consolidar(salida, ctx)
    elif args.etapa == 'benchmarks':
        for senal in args.senales:
            for politica in POLITICAS:
                c = json.loads((salida / f'{senal}_{politica}_ciclo.json').read_text(encoding='utf-8'))
                assert c['estado'] == 'CONVERGE', (senal, politica, c['estado'])
            consolidar(salida, ctx)
            ci = ctx['C_IMP_COP'][senal]/trm
            for caso in args.casos:
                etiqueta = f'{senal}_{caso}'
                if (salida / f'{etiqueta}.json.gz').exists():
                    progreso(salida, 'benchmark_conservado', caso=etiqueta)
                    continue
                progreso(salida, 'inicio_benchmark', caso=etiqueta, limite_segundos=args.segundos,
                         brecha_solicitada=args.brecha, exclusion='binaria')
                res = REG.optimo(caso, fechas, pv, load, ci, ce, P, b, ctx['P_EXP_MAX'],
                                 frontera_soc='ciclica', cargo=REG.CARGO, segundos=args.segundos,
                                 brecha=args.brecha, exclusion='binaria', aceptar_limite_tiempo=True)
                leido = guardar_caso(salida, etiqueta, res)
                v = validar(leido, P, ctx['P_EXP_MAX'])
                escribir_json(salida / f'{etiqueta}_validacion.json', v)
                escribir_json(salida / f'{etiqueta}_solver.json', dict(limite_segundos=args.segundos,
                    brecha_solicitada=args.brecha, condicion=res['Condicion'], objetivo=res['Objetivo'],
                    cota=res['Cota'], brecha=res['Brecha_solver'], segundos=res['Segundos']))
                progreso(salida, 'fin_benchmark', caso=etiqueta, condicion=res['Condicion'],
                         costo_USD=res['Objetivo'], cota_USD=res['Cota'], brecha=res['Brecha_solver'],
                         segundos=res['Segundos'])
                consolidar(salida, ctx)
    filas = consolidar(salida, ctx)
    progreso(salida, 'etapa_terminada', nombre=args.etapa, casos_validados=len(filas))


if __name__ == '__main__':
    main()
