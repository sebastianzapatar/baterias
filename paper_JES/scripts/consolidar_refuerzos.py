"""Consolida R1–R3 en CSV y un resumen Markdown para el manuscrito.

Distancias con cotas del solucionador (ec. 11 del artículo):
    ΔC ∈ [max(0, C−U), max(0, C−L)],  ΔC_rel ∈ [max(0, C/U−1), max(0, C/L−1)].
Descomposición R1 (ec. 12): C_regla − C*_OPT = (C_regla − C_RH) + (C_RH − C*_OPT), con el segundo
término acotado por [C_RH − U, C_RH − L]. Los niveles base (gas 20, cargo 100) se leen de la corrida
cerrada `resultados_correccion_numerica/20260920` sin recalcular.
"""
from __future__ import annotations

import math

import comun as C
from comun import np, pd

POT_TERMICA_MW = 25.2
NOMBRE = {'A_INT': 'SC', 'B_INT': 'DM', 'D_INT': 'DMX', 'OPT_AB': 'OPT-B', 'OPT_D': 'OPT-X',
          'PV_SIN_BESS': 'PV-only', 'RED_TERMICA': 'Grid+Gas'}
SENAL = {'T0': 'FLAT', 'T1': 'WMI'}


def fl(x, d=2):
    return math.floor(x*10**d)/10**d


def ce(x, d=2):
    return math.ceil(x*10**d)/10**d


def cerrado(senal, caso):
    f = C.caso_cerrado(senal, caso)
    L = f['Cota_USD'] if caso.startswith('OPT_') else f['Costo_operativo_USD']
    return dict(Costo=f['Costo_operativo_USD'], L=L, U=f['Costo_operativo_USD'], Arranques=f['Arranques'],
                Termica_MWh=f['Termica_MWh'], EFC=f.get('EFC', 0.), Brecha=f.get('Brecha_solver'))


def nuevo(rf, esc, senal, caso):
    f = C.leer_json(C.SALIDA/rf/'casos'/f'{rf}_{esc}_{senal}_{caso}.json')
    L = f['Cota_USD'] if caso.startswith('OPT_') else f['Costo_operativo_USD']
    return dict(Costo=f['Costo_operativo_USD'], L=L, U=f['Costo_operativo_USD'], Arranques=f['Arranques'],
                Termica_MWh=f['Termica_MWh'], EFC=f.get('EFC', 0.), Brecha=f.get('Brecha_solver'),
                Condicion=f.get('Condicion'), Pertenencia=f.get('Pertenencia'))


def distancia(c, b):
    return (max(0., c-b['U']), max(0., c-b['L']), 100*max(0., c/b['U']-1), 100*max(0., c/b['L']-1))


def tabla_sensibilidad(rf, niveles, base, campo, casos):
    filas = []
    for nivel, esc in niveles:
        for s in ('T0', 'T1'):
            R = {c: (cerrado(s, c) if esc == base else nuevo(rf, esc, s, c)) for c in casos}
            reglas = [c for c in ('A_INT', 'B_INT', 'D_INT') if c in R]
            orden = sorted(reglas, key=lambda c: R[c]['Costo'])
            for c in casos:
                r = R[c]
                fila = {campo: nivel, 'Senal': SENAL[s], 'Caso': NOMBRE[c], 'Costo_MUSD': r['Costo']/1e6,
                        'Arranques': r['Arranques'], 'FP_termico_pct': 100*r['Termica_MWh']/(POT_TERMICA_MW*8760),
                        'EFC': r['EFC'], 'Brecha_solver_pct': None if r['Brecha'] is None else 100*r['Brecha'],
                        'Orden_reglas': '>'.join(NOMBRE[x] for x in orden) if c == orden[0] else ''}
                if c.startswith('OPT_'):
                    fila.update(L_MUSD=r['L']/1e6, U_MUSD=r['U']/1e6)
                bench = C.BENCH_DE.get(c)
                if bench and bench in R:
                    dmin, dmax, pmin, pmax = distancia(r['Costo'], R[bench])
                    fila.update(Benchmark=NOMBRE[bench], Dist_kUSD_min=fl(dmin/1e3, 1), Dist_kUSD_max=ce(dmax/1e3, 1),
                                Dist_pct_min=fl(pmin), Dist_pct_max=ce(pmax))
                filas.append(fila)
    return pd.DataFrame(filas)


def tabla_r1():
    filas = []
    for s in ('T0', 'T1'):
        for acc, bench, reglas in (('B', 'OPT_AB', ('A_INT', 'B_INT')), ('X', 'OPT_D', ('D_INT',))):
            b = cerrado(s, bench)
            rh = {H: C.leer_json(C.SALIDA/'R1'/'casos'/f'R1_RH{H}_{acc}_{s}.json') for H in ('24', '48', '24A6')
                  if (C.SALIDA/'R1'/'casos'/f'R1_RH{H}_{acc}_{s}.json').exists() or H != '24A6'}
            for H, f in rh.items():
                c = f['Costo_operativo_USD']
                filas.append(dict(Senal=SENAL[s], Accion=acc, Benchmark=NOMBRE[bench], Caso=f'RH-{H}',
                                  Costo_MUSD=c/1e6, Arranques=f['Arranques'], EFC=f['EFC'],
                                  FP_termico_pct=100*f['Termica_MWh']/(POT_TERMICA_MW*8760),
                                  Dist_pct_min=fl(100*max(0., c/b['U']-1)), Dist_pct_max=ce(100*max(0., c/b['L']-1)),
                                  Dist_kUSD_min=fl(max(0., c-b['U'])/1e3, 1), Dist_kUSD_max=ce(max(0., c-b['L'])/1e3, 1),
                                  Cota_valida=bool(c >= b['L'] - 1e-6), Pertenencia=f.get('Pertenencia'),
                                  Retrocesos=len(f.get('Retrocesos', []))))
            for regla in reglas:
                cr = cerrado(s, regla)['Costo']
                gap_min, gap_max = max(0., cr-b['U']), max(0., cr-b['L'])
                for H, f in rh.items():
                    c = f['Costo_operativo_USD']
                    logica = cr - c
                    filas.append(dict(Senal=SENAL[s], Accion=acc, Benchmark=NOMBRE[bench],
                                      Caso=f'{NOMBRE[regla]} vs RH-{H}', Costo_MUSD=cr/1e6,
                                      Brecha_total_kUSD_min=fl(gap_min/1e3, 1), Brecha_total_kUSD_max=ce(gap_max/1e3, 1),
                                      Logica_kUSD=logica/1e3,
                                      Prevision_kUSD_min=fl(max(0., c-b['U'])/1e3, 1),
                                      Prevision_kUSD_max=ce(max(0., c-b['L'])/1e3, 1),
                                      Cierra_pct_min=fl(100*logica/gap_max) if gap_max > 0 else None,
                                      Cierra_pct_max=ce(100*logica/gap_min) if gap_min > 0 else None))
    return pd.DataFrame(filas)


def main():
    salida = C.SALIDA/'consolidado'
    salida.mkdir(parents=True, exist_ok=True)
    partes = {}
    try:
        partes['R1_horizonte_rodante'] = tabla_r1()
    except FileNotFoundError as e:
        print('R1 incompleto:', e)
    try:
        partes['R2_precio_gas'] = tabla_sensibilidad(
            'R2', [(12, 'g12'), (16, 'g16'), (20, 'base'), (24, 'g24')], 'base', 'Gas_USD_MMBtu',
            ('RED_TERMICA', 'PV_SIN_BESS', 'A_INT', 'B_INT', 'D_INT', 'OPT_AB', 'OPT_D'))
    except FileNotFoundError as e:
        print('R2 incompleto:', e)
    try:
        partes['R3_cargo_arranque'] = tabla_sensibilidad(
            'R3', [(0, 'su0'), (100, 'base'), (500, 'su500'), (1000, 'su1000')], 'base', 'Cargo_USD',
            ('RED_TERMICA', 'PV_SIN_BESS', 'A_INT', 'B_INT', 'D_INT', 'OPT_AB'))
    except FileNotFoundError as e:
        print('R3 incompleto:', e)
    md = ['# Refuerzos R1–R3: resultados consolidados', '',
          'Generado por `paper_JES/scripts/consolidar_refuerzos.py`. Niveles base leídos de la corrida cerrada.', '']
    for nombre, df in partes.items():
        df.to_csv(salida/f'{nombre}.csv', index=False)
        md += [f'## {nombre}', '', '```', df.to_string(index=False), '```', '']
    (salida/'RESUMEN.md').write_text('\n'.join(md), encoding='utf-8')
    print('Consolidado:', list(partes))


if __name__ == '__main__':
    main()
