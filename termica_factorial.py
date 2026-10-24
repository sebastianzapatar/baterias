"""Factorial anual con térmica: K1-K4 × T0/T1 × reglas y benchmarks.

Reutiliza los motores y K3 congelados. Cada proceso resuelve un caso completo;
solo el coordinador publica el progreso y consolida. No calcula VPN ni sizing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil

import termica_vpn as V
from termica_politicas_anual import decodificar

SALIDA = Path('resultados_factorial_termica/gas20_cvom_unidades_20260919')
K = {
    'K1': dict(f_PV=1., P_MW=50., E_MWh=400.),
    'K2': dict(f_PV=.918, P_MW=45., E_MWh=180.),
    'K3': dict(f_PV=1., P_MW=45., E_MWh=280.),
    'K4': dict(f_PV=.830, P_MW=50., E_MWh=100.),
}
CASOS = V.POLITICAS + V.BENCHMARKS
CODIGO = tuple(dict.fromkeys(V.CODIGO + ('termica_factorial.py',)))


def leer(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_hook=decodificar)


def identidad(k, s, c):
    return f'{k}_{s}_{c}'


def tareas():
    # K1 comparte exactamente el PV de K3: su referencia sin BESS se reutiliza.
    t = [(k, s, c) for k in K for s in ('T0', 'T1') for c in CASOS]
    t += [(k, s, 'PV_SIN_BESS') for k in ('K2', 'K3', 'K4') for s in ('T0', 'T1')]
    t += [('K3', s, 'RED_TERMICA') for s in ('T0', 'T1')]
    assert len(t) == len(set(t)) == 72
    return t


def contexto(k, senal):
    m = deepcopy(V.meta_fuente())
    d = list(V.cargar_entrada(senal))
    d[1] *= K[k]['f_PV']
    m['bess'].update(P_MW=K[k]['P_MW'], E_MWh=K[k]['E_MWh'])
    # Convención vigente: techo común a reglas y benchmark de cada K.
    m['P_EXP_MAX'] = float(d[1].max() + K[k]['P_MW'])
    assert abs(d[1].sum()/d[2].sum()-K[k]['f_PV']) < 1e-10
    assert len(d[0]) == 8760 and d[0][0] == pd.Timestamp('2025-01-01')
    assert ((d[0][1:]-d[0][:-1]) == pd.Timedelta(hours=1)).all()
    if k == 'K3':
        assert m['bess'] == V.meta_fuente()['bess']
        assert m['P_EXP_MAX'] == V.meta_fuente()['P_EXP_MAX']
    return d, m


def fuente_k3(s, c):
    base = V.SALIDA / 'benchmarks_600s' if c in V.BENCHMARKS else V.SALIDA
    return V.rutas(base, s, c, 1)


def preparar(salida, segundos=600., brecha=.001):
    salida = Path(salida)
    for carpeta in ('casos', 'logs', 'errores'):
        (salida/carpeta).mkdir(parents=True, exist_ok=True)
    vpn = leer(V.SALIDA/'configuracion.json')
    assert vpn['codigo'] == {f: V.sha(f) for f in V.CODIGO}, 'Cambió el motor del VPN congelado'
    assert leer(V.SALIDA/'trazabilidad_financiera.json')['estado'] == 'COMPLETO'
    fuentes = [V.FUENTE_AB/'ejecucion.json', V.SALIDA/'configuracion.json',
               Path('.build/factorial_R.py')]
    for s in ('T0', 'T1'):
        for c in CASOS + V.BASES:
            fuentes.extend(fuente_k3(s, c))
    cfg = dict(diseno=K, senales=['T0', 'T1'], casos=list(CASOS), region='Oriente',
               anio=2025, horas=8760, pv_k3_MWp=vpn['pv_MWp'], TRM=vpn['TRM'],
               parametros_termicos=vpn['parametros_termicos'],
               cargo_arranque_USD=V.meta_fuente()['cargo_arranque_USD'],
               segundos_solver=segundos, brecha_solver=brecha, hilos_solver=1,
               procesos_pypsa=2, procesos_reglas=1, total_archivos=72,
               p_exp='max(PV_K)+P_BESS_K; mismo límite por K para regla y benchmark',
               frontera='SOC cíclico libre por trayectoria; térmica libre solo al comenzar el año',
               alcance='operativo anual; K categórico, sin VPN ni redimensionamiento',
               codigo={f: V.sha(f) for f in CODIGO}, fuentes={str(f): V.sha(f) for f in fuentes})
    ruta = salida/'configuracion.json'
    if ruta.exists():
        # La serialización compara también los arrays sin ambigüedad booleana.
        assert json.dumps(leer(ruta), default=V.codificar, sort_keys=True) == json.dumps(cfg, default=V.codificar, sort_keys=True), 'Use otra salida: configuración distinta'
    else:
        V.json_guardar(ruta, cfg)
    return cfg


def comprobar_codigo(salida):
    cfg = leer(Path(salida)/'configuracion.json')
    assert cfg['codigo'] == {f: V.sha(f) for f in CODIGO}, 'Código modificado durante la corrida'
    return cfg


@contextmanager
def solver_un_hilo():
    """Opción de ejecución aislada al proceso; no cambia motores ni restricciones.

    PyPSA pasa opciones explícitas a linopy.Model.solve. Este adaptador agrega
    threads=1 y comprueba la opción efectiva en HiGHS. Se restaura al salir.
    """
    import linopy
    original = linopy.Model.solve
    registro = []

    def resolver(modelo, *args, **kwargs):
        kwargs['threads'] = 1
        resultado = original(modelo, *args, **kwargs)
        _, hilos = modelo.solver_model.getOptionValue('threads')
        assert hilos == 1, 'HiGHS no respetó el límite de hilos'
        registro.append(int(hilos))
        return resultado

    with patch.object(linopy.Model, 'solve', resolver):
        yield registro


def reconstruir(npz, fila, m):
    with np.load(npz, allow_pickle=False) as z:
        r = {k: z[k].copy() for k in V.SERIES}
        r.update(Fechas=pd.DatetimeIndex(z['Fecha']), SOC_inicial=float(z['SOC_inicial']),
                 Carga_Red=z['Carga_Red'].copy())
    b = dict(m['bess'])
    if fila['Caso'] in V.BASES:
        b.update(P_MW=0., E_MWh=0.)
    r['Carga_PV'] = r['Carga_BESS']-r['Carga_Red']
    r.update(Caso=fila['Caso'], Tipo='archivo_auditado', BESS=b,
             Cargo=m['cargo_arranque_USD'], P_EXP_MAX=m['P_EXP_MAX'],
             estado_termico_inicial='libre', Objetivo=fila['Costo_operativo_USD'],
             Cota=fila['Cota_USD'] if np.isfinite(fila['Cota_USD']) else fila['Costo_operativo_USD'],
             Condicion=fila['Condicion'], Brecha_solver=fila['Brecha_solver'])
    return r


def auditar_archivo(r, m, datos):
    p = m['parametros_termicos']
    assert r['Fechas'].equals(datos[0])
    assert len(r['SOC']) == 8760
    for clave, esperado in zip(('PV', 'Carga', 'C_imp_USD', 'C_exp_USD'), datos[1:]):
        if clave == 'PV' and r['Caso'] == 'RED_TERMICA':
            esperado = np.zeros_like(esperado)
        np.testing.assert_array_equal(r[clave], esperado, err_msg=clave)
    for clave in ('compromiso', 'arranques', 'paradas'):
        V.POL._binario(r[clave], clave, (8760, 4))
    if r['Caso'] in ('C_INT', 'F_INT', 'OPT_CF'):
        v = V.CF.validar_cf(r, p)
    else:
        assert abs(r['Carga_Red']).max() <= V.POL.TOL_FLUJO_MW
        v = V.REG.validar_integrado(r, p, r['Caso'] in ('D_INT', 'OPT_D'))
    assert (r['Export_PV']+r['Descarga_export'] <= m['P_EXP_MAX']+V.POL.TOL_FLUJO_MW).all()
    assert abs(r['SOC'][-1]-r['SOC_inicial']) <= V.REG.TOL_CIERRE_MWh
    V.contabilidad(r, p)
    return v


def guardar(salida, k, s, c, r, m, segundos, origen, hilos=None):
    etiqueta = identidad(k, s, c)
    npz = Path(salida)/'casos'/f'{etiqueta}.npz'
    js = npz.with_suffix('.json')
    assert not js.exists(), f'No se sobrescribe {etiqueta}'
    series = {clave: np.asarray(r[clave]) for clave in V.SERIES}
    series.update(Fecha=r['Fechas'].to_numpy(), SOC_inicial=np.asarray(r['SOC_inicial']),
                  Carga_Red=np.asarray(r.get('Carga_Red', np.zeros(8760))))
    tmp = npz.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **series)
    tmp.replace(npz)
    util = m['bess']['E_MWh']*m['bess']['DoD']
    cont = V.contabilidad(r, m['parametros_termicos'])
    fila = dict(K=k, Senal=s, Caso=c, Estado='OK', Origen=origen,
                 Segundos=segundos, Hilos_solver=hilos,
                 Condicion=r.get('Condicion', 'politica/CONVERGE'),
                 Cota_USD=float(r.get('Cota', np.nan)),
                 Brecha_solver=float(r.get('Brecha_solver', np.nan)),
                 SHA256_npz=V.sha(npz), P_EXP_MAX=m['P_EXP_MAX'],
                 P_MW=r['BESS']['P_MW'], E_MWh=r['BESS']['E_MWh'], f_PV=0. if c=='RED_TERMICA' else K[k]['f_PV'],
                 EFC=cont['Descarga_MWh']/util if c not in V.BASES else 0.,
                 Termica_MWh=float(r['potencia'][:, :4].sum()),
                 Red_MWh=float((r['potencia'][:, 4]+series['Carga_Red']).sum()),
                 Carga_red_MWh=float(series['Carga_Red'].sum()),
                 Export_BESS_MWh=float(r['Descarga_export'].sum()),
                 Export_PV_MWh=float(r['Export_PV'].sum()),
                 Vertimiento_MWh=float(r['Vertimiento'].sum()),
                 SOC_inicial_MWh=float(r['SOC_inicial']),
                 RAM_pico_MB=psutil.Process().memory_info().peak_wset/1e6 if os.name=='nt' else None,
                 **cont)
    V.json_guardar(js, fila)
    verificar(salida, k, s, c)
    return fila


def verificar(salida, k, s, c):
    js = Path(salida)/'casos'/f'{identidad(k,s,c)}.json'
    fila = leer(js)
    assert (fila['K'], fila['Senal'], fila['Caso'], fila['Estado']) == (k, s, c, 'OK')
    assert V.sha(js.with_suffix('.npz')) == fila['SHA256_npz']
    datos, m = contexto(k, s)
    r = reconstruir(js.with_suffix('.npz'), fila, m)
    auditar_archivo(r, m, datos)
    cont = V.contabilidad(r, m['parametros_termicos'])
    for nombre, valor in cont.items():
        assert abs(valor-fila[nombre]) <= 1e-3, (nombre, valor, fila[nombre])
    return fila


def reutilizar(salida):
    for k, s, c in tareas():
        if k != 'K3':
            continue
        if (Path(salida)/'casos'/f'{identidad(k,s,c)}.json').exists():
            verificar(salida, k, s, c)
            continue
        tic = time.perf_counter()
        npz, js = fuente_k3(s, c)
        fila = leer(js)
        assert V.sha(npz) == fila['SHA256_npz']
        datos, m = contexto(k, s)
        r = reconstruir(npz, fila, m)
        auditar_archivo(r, m, datos)
        # Evita convertir un costo de política en una supuesta cota inferior.
        r['Cota'] = fila['Cota_USD']
        guardar(salida, k, s, c, r, m, time.perf_counter()-tic,
                dict(tipo='K3_reutilizado', archivo=str(js), SHA256=V.sha(js)), None)


def ejecutar_caso(salida, k, s, c):
    cfg = comprobar_codigo(salida)
    if (Path(salida)/'casos'/f'{identidad(k,s,c)}.json').exists():
        return verificar(salida, k, s, c)
    datos, m = contexto(k, s)
    V.inicializar_worker(True)
    tic = time.perf_counter()
    if c in V.BENCHMARKS:
        with solver_un_hilo() as hilos:
            r = V.resolver(c, datos, m, cfg['segundos_solver'], cfg['brecha_solver'])
        assert hilos == [1], f'Número inesperado de soluciones: {hilos}'
    else:
        r = V.resolver(c, datos, m, cfg['segundos_solver'], cfg['brecha_solver'])
    return guardar(salida, k, s, c, r, m, time.perf_counter()-tic,
                    dict(tipo='despacho_nuevo'), 1 if c in V.BENCHMARKS else None)


def progreso(salida, activos=(), fallos=(), etapa='EJECUTANDO'):
    filas = [leer(p) for p in (Path(salida)/'casos').glob('*.json')]
    filas = [f for f in filas if f['Estado']=='OK']
    registro = dict(fecha=pd.Timestamp.now().isoformat(), etapa=etapa,
                     completos=len(filas), total=72,
                     politicas=sum(f['Caso'] in V.POLITICAS for f in filas), total_politicas=40,
                     benchmarks=sum(f['Caso'] in V.BENCHMARKS for f in filas), total_benchmarks=24,
                     reutilizados=sum(f['Origen']['tipo']=='K3_reutilizado' for f in filas),
                     RAM_libre_GiB=psutil.virtual_memory().available/2**30,
                     activos=list(activos), fallos=list(fallos))
    V.json_guardar(Path(salida)/'progreso.json', registro)
    texto = ['# Avance del factorial térmico', '', f'Actualizado: {registro["fecha"]}',
              f'Estado: **{etapa}**', '', f'Casos guardados: **{len(filas)}/72**; K3 reutilizados: **{registro["reutilizados"]}**.',
              f'Reglas: {registro["politicas"]}/40. Benchmarks: {registro["benchmarks"]}/24. Referencias: {len(filas)-registro["politicas"]-registro["benchmarks"]}/8.',
              f'RAM disponible: {registro["RAM_libre_GiB"]:.2f} GiB.', '',
              'El porcentaje de casos no representa el porcentaje de tiempo.', '', 'Procesos activos:']
    texto += [f'- {a["caso"]}, PID {a["pid"]}, {a["segundos"]:.0f} s.' for a in activos]
    texto += ['', 'Incidencias: '+(', '.join(fallos) if fallos else 'ninguna registrada')]
    (Path(salida)/'PROGRESO.md').write_text('\n'.join(texto)+'\n', encoding='utf-8')
    return registro


def coordinar(salida):
    cfg = preparar(salida)
    reutilizar(salida)
    pendientes = [t for t in tareas() if not (salida/'casos'/f'{identidad(*t)}.json').exists()]
    # Alterna tipos; los CF más lentos entran temprano en sus respectivos cupos.
    orden = {c:i for i,c in enumerate(('OPT_CF','C_INT','OPT_D','F_INT','OPT_AB','D_INT','B_INT','A_INT','PV_SIN_BESS'))}
    pendientes.sort(key=lambda t: (orden.get(t[2], 99), t[0], t[1]))
    activos, fallos = [], []
    ultimo = None
    while pendientes or activos:
        for a in activos[:]:
            code = a['proceso'].poll()
            if code is None:
                continue
            a['log'].close()
            activos.remove(a)
            if code:
                fallos.append(identidad(*a['tarea']))
            else:
                verificar(salida, *a['tarea'])
        for t in pendientes[:]:
            mip = t[2] in V.BENCHMARKS
            nm = sum(a['mip'] for a in activos)
            nr = len(activos)-nm
            if (mip and nm >= cfg['procesos_pypsa']) or (not mip and nr >= cfg['procesos_reglas']):
                continue
            libre = psutil.virtual_memory().available/2**30
            margen = (5.5 if nm else 4.5) if mip else 2.0
            if libre < margen:
                continue  # No mata procesos activos; espera antes de iniciar otro.
            etiqueta = identidad(*t)
            log = (salida/'logs'/f'{etiqueta}.log').open('w', encoding='utf-8')
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONUTF8='1')
            cmd = [sys.executable, '-B', 'termica_factorial.py', '--salida', str(salida),
                   '--caso', *t]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            activos.append(dict(proceso=proc, log=log, tarea=t, mip=mip, inicio=time.perf_counter()))
            pendientes.remove(t)
            print(f'INICIO {etiqueta}, PID {proc.pid}, RAM libre {libre:.2f} GiB', flush=True)
            # Evita lanzar otro modelo antes de que el primero reserve memoria.
            time.sleep(3)
        vista = [dict(caso=identidad(*a['tarea']), pid=a['proceso'].pid,
                       segundos=time.perf_counter()-a['inicio']) for a in activos]
        r = progreso(salida, vista, fallos)
        if (r['completos'], tuple(fallos)) != ultimo:
            print(f'PROGRESO {r["completos"]}/72; fallos={fallos}', flush=True)
            ultimo = (r['completos'], tuple(fallos))
        if pendientes or activos:
            time.sleep(5)
    if fallos:
        progreso(salida, fallos=fallos, etapa='INCOMPLETO')
        raise RuntimeError(f'Casos fallidos, revisar sus logs: {fallos}')
    progreso(salida, etapa='VALIDANDO')
    from termica_factorial_informe import consolidar
    consolidar(salida)
    progreso(salida, etapa='COMPLETO')
    print('FACTORIAL TERMINADO, VALIDADO Y DOCUMENTADO', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--salida', type=Path, default=SALIDA)
    ap.add_argument('--preparar', action='store_true')
    ap.add_argument('--caso', nargs=3, metavar=('K', 'SENAL', 'CASO'))
    args = ap.parse_args()
    if args.caso:
        assert tuple(args.caso) in tareas()
        try:
            print(json.dumps(ejecutar_caso(args.salida, *args.caso), default=V.codificar), flush=True)
        except Exception as exc:
            V.json_guardar(args.salida/'errores'/f'{identidad(*args.caso)}.json',
                           dict(caso=args.caso, error=str(exc), traceback=traceback.format_exc()))
            raise
    elif args.preparar:
        preparar(args.salida)
        reutilizar(args.salida)
        print(progreso(args.salida, etapa='PREPARADO'), flush=True)
    else:
        args.salida.mkdir(parents=True, exist_ok=True)
        lock = args.salida/'coordinador.lock'
        if lock.exists():
            previo = leer(lock)
            if psutil.pid_exists(previo['pid']):
                raise RuntimeError(f'Coordinador existente, PID {previo["pid"]}')
            lock.rename(lock.with_name(f'coordinador_anterior_{previo["pid"]}.json'))
        with lock.open('x', encoding='utf-8') as f:
            json.dump(dict(pid=os.getpid(), fecha=pd.Timestamp.now().isoformat()), f)
        try:
            coordinar(args.salida)
        finally:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
