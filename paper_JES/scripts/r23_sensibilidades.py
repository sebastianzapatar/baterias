"""Refuerzos R2 (precio del gas) y R3 (cargo por arranque) sobre C280 (K3 en el repositorio).

Cada caso es una anualidad completa resuelta con `termica_vpn.resolver`, el mismo motor de la
corrida cerrada del 20/09/2026. Solo cambian el precio del gas (y con él `cv`) o el cargo por arranque.
Las reglas A/B/D se validan con sus validadores y se comprueba su pertenencia al conjunto factible
del benchmark compatible con los mismos parámetros. Los niveles base (gas 20, cargo 100) se leen
de la corrida cerrada y no se recalculan.

Uso (desde cualquier carpeta):
    OperacionBESS/.venv/bin/python paper_JES/scripts/r23_sensibilidades.py --caso R2 g12 T0 OPT_AB
"""
from __future__ import annotations

import argparse
import time
import traceback

import comun as C
from comun import V, F, np

SEGUNDOS, BRECHA = 600., 1e-3
ESCENARIOS = {
    'R2': {'g12': dict(gas=12.), 'g16': dict(gas=16.), 'g24': dict(gas=24.)},
    'R3': {'su0': dict(cargo=0.), 'su500': dict(cargo=500.), 'su1000': dict(cargo=1000.)},
}
CASOS = {
    'R2': ('RED_TERMICA', 'PV_SIN_BESS', 'A_INT', 'B_INT', 'D_INT', 'OPT_AB', 'OPT_D'),
    'R3': ('RED_TERMICA', 'PV_SIN_BESS', 'A_INT', 'B_INT', 'D_INT', 'OPT_AB'),
}


def tareas():
    return [(rf, esc, s, c) for rf in ESCENARIOS for esc in ESCENARIOS[rf] for s in ('T0', 'T1')
            for c in CASOS[rf]]


def etiqueta(rf, esc, s, c):
    return f'{rf}_{esc}_{s}_{c}'


def ejecutar(rf, esc, s, c):
    carpeta = C.SALIDA / rf
    js = carpeta / 'casos' / f'{etiqueta(rf, esc, s, c)}.json'
    if js.exists():
        return C.leer_json(js)
    datos, m = C.contexto_c280(s, **ESCENARIOS[rf][esc])
    p, cargo = m['parametros_termicos'], m['cargo_arranque_USD']
    V.inicializar_worker(False)  # sin numba en el .venv; evaluador normal, verificado equivalente en la tesis
    tic = time.perf_counter()
    if c in V.BENCHMARKS:
        with F.solver_un_hilo() as hilos:
            r = V.resolver(c, datos, m, SEGUNDOS, BRECHA)
        assert hilos == [1]
    else:
        r = V.resolver(c, datos, m, SEGUNDOS, BRECHA)
    seg = time.perf_counter() - tic
    ind = C.indicadores(r, p, cargo)
    assert abs(ind['Costo_operativo_USD'] - r['Objetivo']) <= 1e-3, (ind['Costo_operativo_USD'], r['Objetivo'])
    V.contabilidad(r, p)  # misma aserción del motor cerrado
    fila = dict(Refuerzo=rf, Escenario=esc, Senal=s, Caso=c, Estado='OK', Gas_USD_MMBtu=p['gas'],
                CV_USD_MWh=np.asarray(p['cv']).tolist(), Cargo_arranque_USD=cargo,
                Condicion=r.get('Condicion', 'politica/CONVERGE'), Objetivo_USD=float(r['Objetivo']),
                Cota_USD=float(r.get('Cota', np.nan)) if c in V.BENCHMARKS + V.BASES else None,
                Brecha_solver=float(r.get('Brecha_solver', np.nan)) if c in V.BENCHMARKS + V.BASES else None,
                Segundos=seg, Segundos_limite=SEGUNDOS, Brecha_solicitada=BRECHA, **ind)
    if c in C.BENCH_DE:
        fila.update(C.pertenencia(r, C.BENCH_DE[c], datos, m, p))
    return C.guardar_caso(carpeta, etiqueta(rf, esc, s, c), r, fila)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--caso', nargs=4, metavar=('REFUERZO', 'ESCENARIO', 'SENAL', 'CASO'), required=True)
    a = ap.parse_args()
    t = tuple(a.caso)
    assert t in tareas(), t
    try:
        fila = ejecutar(*t)
        print('OK', etiqueta(*t), round(fila['Costo_operativo_USD'], 2), fila['Condicion'], flush=True)
    except Exception as exc:
        (C.SALIDA / t[0] / 'errores').mkdir(parents=True, exist_ok=True)
        C.json_guardar(C.SALIDA / t[0] / 'errores' / f'{etiqueta(*t)}.json',
                       dict(caso=t, error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
