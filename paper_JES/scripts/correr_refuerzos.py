"""Coordinador de los refuerzos R1–R3: lanza cada caso en un proceso independiente.

Como máximo `--mip` benchmarks anuales simultáneos (memoria) y `--ligeros` procesos de reglas,
referencias u horizonte rodante. No relanza casos guardados; los fallos quedan en `errores/`.
Publica el avance en `resultados_refuerzos/PROGRESO.md`.

Uso:  OperacionBESS/.venv/bin/python paper_JES/scripts/correr_refuerzos.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

import comun as C
import r23_sensibilidades as R23

SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable


def tareas():
    t = []
    # R1 primero entre los ligeros: son los más largos de su grupo.
    for H in (24, 48):
        for acc in ('B', 'X'):
            for s in ('T0', 'T1'):
                t.append(dict(id=f'R1_RH{H}_{acc}_{s}', mip=False,
                              json=C.SALIDA/'R1'/'casos'/f'R1_RH{H}_{acc}_{s}.json',
                              cmd=[PY, '-W', 'ignore', str(SCRIPTS/'r1_horizonte_rodante.py'), '--caso', str(H), acc, s]))
    for rf, esc, s, c in R23.tareas():
        e = R23.etiqueta(rf, esc, s, c)
        t.append(dict(id=e, mip=c.startswith('OPT_'), json=C.SALIDA/rf/'casos'/f'{e}.json',
                      cmd=[PY, '-W', 'ignore', str(SCRIPTS/'r23_sensibilidades.py'), '--caso', rf, esc, s, c]))
    # Benchmarks primero (más lentos); dentro de cada grupo, el orden de la lista.
    t.sort(key=lambda x: not x['mip'])
    return t


def progreso(todas, activos, fallos, etapa):
    hechas = [x for x in todas if x['json'].exists()]
    txt = ['# Avance de los refuerzos R1–R3 (artículo JES)', '',
           f'Actualizado: {time.strftime("%Y-%m-%d %H:%M:%S")}', f'Estado: **{etapa}**', '',
           f'Casos guardados: **{len(hechas)}/{len(todas)}**',
           f'- R1: {sum(x["id"].startswith("R1") for x in hechas)}/8',
           f'- R2: {sum(x["id"].startswith("R2") for x in hechas)}/{sum(x["id"].startswith("R2") for x in todas)}',
           f'- R3: {sum(x["id"].startswith("R3") for x in hechas)}/{sum(x["id"].startswith("R3") for x in todas)}',
           f'RAM disponible: {psutil.virtual_memory().available/2**30:.1f} GiB', '', 'Activos:']
    txt += [f'- {a["id"]} (PID {a["proc"].pid}, {time.time()-a["inicio"]:.0f} s)' for a in activos]
    txt += ['', 'Fallos: ' + (', '.join(fallos) if fallos else 'ninguno')]
    C.SALIDA.mkdir(parents=True, exist_ok=True)
    (C.SALIDA/'PROGRESO.md').write_text('\n'.join(txt) + '\n', encoding='utf-8')
    return len(hechas)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--mip', type=int, default=3)
    ap.add_argument('--ligeros', type=int, default=3)
    a = ap.parse_args()
    todas = tareas()
    pendientes = [x for x in todas if not x['json'].exists()]
    (C.SALIDA/'logs').mkdir(parents=True, exist_ok=True)
    C.json_guardar(C.SALIDA/'configuracion.json', dict(
        fecha=time.strftime('%Y-%m-%d %H:%M:%S'), python=sys.version, tareas=[x['id'] for x in todas],
        codigo=C.huella_codigo(extra=[SCRIPTS/f for f in ('comun.py', 'r1_horizonte_rodante.py',
                                                            'r23_sensibilidades.py', 'correr_refuerzos.py')]),
        nota='Evaluador DP normal (sin numba); su equivalencia con el acelerado se verificó en la tesis. '
             'Reproducción base K3 gas 20/cargo 100: diferencias < 1e-6 USD en A/B/D, PV sin BESS y red+gas.'))
    activos, fallos = [], []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
    ultimo = None
    while pendientes or activos:
        for x in activos[:]:
            code = x['proc'].poll()
            if code is None:
                continue
            x['log'].close()
            activos.remove(x)
            if code or not x['json'].exists():
                fallos.append(x['id'])
            print(('FALLO ' if code else 'FIN ') + x['id'], f'{time.time()-x["inicio"]:.0f} s', flush=True)
        for x in pendientes[:]:
            nm = sum(y['mip'] for y in activos)
            nl = len(activos) - nm
            if (x['mip'] and nm >= a.mip) or (not x['mip'] and nl >= a.ligeros):
                continue
            if psutil.virtual_memory().available/2**30 < (5. if x['mip'] else 2.5):
                continue
            log = (C.SALIDA/'logs'/f'{x["id"]}.log').open('w', encoding='utf-8')
            x['proc'] = subprocess.Popen(x['cmd'], stdout=log, stderr=subprocess.STDOUT, env=env,
                                         cwd=str(C.OPBESS))
            x['log'], x['inicio'] = log, time.time()
            activos.append(x)
            pendientes.remove(x)
            print('INICIO', x['id'], flush=True)
            time.sleep(3)
        n = progreso(todas, activos, fallos, 'EJECUTANDO')
        if n != ultimo:
            print(f'PROGRESO {n}/{len(todas)}; fallos={fallos}', flush=True)
            ultimo = n
        time.sleep(5)
    progreso(todas, [], fallos, 'COMPLETO' if not fallos else 'INCOMPLETO')
    print('TERMINADO; fallos:', fallos, flush=True)


if __name__ == '__main__':
    main()
