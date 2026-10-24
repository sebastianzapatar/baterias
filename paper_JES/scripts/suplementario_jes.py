"""Tablas y figura del material suplementario (S1–S5) del artículo JES.

Genera fragmentos LaTeX en manuscript/sup_tablas/ y la figura S1 en manuscript/figures/.
Fuentes: series auditadas de K3 (20260920), CSV cerrados, resultados de carbono (malla 20260926),
resultados R1–R3 y tablas de la tesis transcritas (etapa sin gas y extensión; valores en mil millones COP,
convertidos con TRM = 3.126,08 COP/USD).
Uso: OperacionBESS/.venv/bin/python paper_JES/scripts/suplementario_jes.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.dates  # noqa: F401
import figuras_jes as FJ
plt = FJ.plt

RAIZ = Path(__file__).resolve().parents[1]
OP = RAIZ.parent / 'OperacionBESS'
# En el repositorio público, paper_JES/ está dentro de la raíz de OperacionBESS.
if not OP.is_dir():
    OP = RAIZ.parent
CASOS = OP / 'resultados_correccion_numerica/20260920/factorial/casos'
OUT = RAIZ / 'manuscript/sup_tablas'
TRM = 3126.08
REF = RAIZ / 'resultados_refuerzos'
usd = lambda cop_mm: cop_mm*1e9/TRM/1e6   # mil millones COP -> M USD


def escribir(nombre, filas):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{nombre}.tex').write_text('\n'.join(filas) + '\n', encoding='utf-8')
    print('OK', nombre)


# ---------------------------------------------------------------- S1: demanda
def s1():
    r = FJ.caso('K3', 'T0', 'A_INT')
    d = pd.Series(r['Carga'], index=pd.DatetimeIndex(r['Fecha']))
    dia = d.resample('D')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FJ.DOBLE, 62*FJ.MM), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.25))
    a1.fill_between(dia.mean().index, dia.min(), dia.max(), color=FJ.AZUL, alpha=0.12, lw=0, label='Daily min–max')
    a1.plot(dia.mean().index, dia.mean(), color=FJ.AZUL, lw=1.3, label='Daily mean')
    a1.set_ylabel('Demand (MW)'); a1.set_ylim(0, 25); FJ.ejes_limpios(a1); a1.legend(loc='lower left', ncol=2)
    a1.xaxis.set_major_formatter(FJ.matplotlib.dates.DateFormatter('%b'))
    FJ.etiqueta_panel(a1, 'a', x=-0.06)
    a2.plot(np.arange(1, 8761), np.sort(d.values)[::-1], color=FJ.AZUL, lw=1.5)
    a2.axhline(8, color=FJ.TXT2, lw=0.6, ls=':'); a2.text(8700, 8.4, '8 MW threshold', ha='right', fontsize=6, color=FJ.TXT2)
    a2.set_xlim(1, 8760); a2.set_ylim(0, 25); a2.set_xlabel('Hours (sorted)'); a2.set_ylabel('Demand (MW)'); FJ.ejes_limpios(a2)
    FJ.etiqueta_panel(a2, 'b', x=-0.12)
    FJ.guardar(fig, 'figS1_demand')
    p = np.percentile(d, [5, 50, 95]); mm = d.groupby(d.index.month).mean()
    h = d.index.hour; diurna = (h >= 6) & (h < 18)
    dd = d.diff().abs().dropna()
    filas = [
        r'Hourly values & 8,760\\',
        r'Zero and partial-recovery readings replaced & 79 (0.9\% of the year)\\',
        fr'Retained hours with positive demand $<$ 8~MW & {int(((d > 0) & (d < 8)).sum())} (1.3\% of the year)\\',
        r'Days with a repeated hourly profile & 28, in four blocks\\',
        fr'Annual energy & {d.sum():,.1f}~MWh\\',
        fr'Mean; maximum demand & {d.mean():.2f}~MW; {d.max():.2f}~MW\\',
        fr'Load factor & {d.mean()/d.max():.2f}\\',
        fr'Standard deviation; coefficient of variation & {d.std():.2f}~MW; {d.std()/d.mean():.3f}\\',
        fr'Percentiles 5, 50 and 95 & {p[0]:.2f}; {p[1]:.2f}; {p[2]:.2f}~MW\\',
        fr'Lowest and highest monthly mean & {mm.min():.2f}~MW (May); {mm.max():.2f}~MW (March)\\',
        fr'Mean 06:00--18:00 / mean of other hours & {d[diurna].mean()/d[~diurna].mean():.2f}\\',
        fr'Change between consecutive hours, percentiles 95 and 99 & {np.percentile(dd, 95):.2f}; {np.percentile(dd, 99):.2f}~MW\\',
        fr'Largest absolute change between consecutive hours & {dd.max():.2f}~MW\\',
    ]
    escribir('S1_demanda', filas)


# ---------------------------------------------------------------- S2: etapa sin gas (tesis)
def s2():
    reg = [('Caribe', 1574.3, 17.97, 100.83, 'May (4.01)', 'February (4.92)'),
           ('Antioquia', 1353.6, 15.45, 117.27, 'April (3.41)', 'February (4.06)'),
           ('Southwest', 1503.6, 17.16, 105.57, 'June (3.70)', 'January (4.63)'),
           ('Oriente', 1387.4, 15.84, 114.42, 'June (3.02)', 'January (4.52)'),
           ('Northeast', 1440.1, 16.44, 110.23, 'July (3.33)', 'January (4.62)')]
    escribir('S2_regiones', [fr'{a} & {b:,.1f} & {c:.2f} & {d:.2f} & {e} & {f}\\' for a, b, c, d, e, f in reg])
    # Tabla 15 de la tesis (mil millones COP): total/incremental, plana/bolsa
    t15 = [('PV only', 192.75, None, 180.93, None), ('SC', 285.79, 93.03, 298.20, 117.27),
           ('DM', 285.83, 93.08, 300.24, 119.31), ('DMX', 283.13, 90.38, 298.19, 117.26),
           ('AP', 136.58, -56.18, 107.71, -73.21), ('APS', 161.36, -31.39, 191.94, 11.01)]
    f = lambda x: '--' if x is None else (f'{usd(x):.2f}' if x >= 0 else f'$-{abs(usd(x)):.2f}$')
    escribir('S2_vpn_sin_gas', [fr'{n} & {f(a)} & {f(b)} & {f(c)} & {f(d)}\\' for n, a, b, c, d in t15])
    # Tabla 16 de la tesis
    t16 = [('Oriente, FLAT, C280', 285.83, 287.77, 294.20, 0.67, 2.24, 2.84),
           ('Oriente, WMI, C280', 300.24, 303.87, 312.34, 1.20, 2.79, 3.88),
           ('Oriente, FLAT, C100', 239.89, 240.73, 245.67, 0.35, 2.05, 2.34),
           ('Caribe, WMI, C280', 346.32, 349.58, 357.95, 0.93, 2.40, 3.25),
           ('Southwest, FLAT, C400', 291.31, 294.63, 304.38, 1.13, 3.31, 4.29),
           ('Antioquia, FLAT, C400', 250.04, 253.67, 263.28, 1.43, 3.79, 4.97)]
    for c, dm, b, full, d1, d2, d3 in t16:   # control de consistencia de la transcripción
        # La brecha total usa la mejor regla del contexto (puede no ser DM): solo se exige que no supere la de DM.
        assert abs(100*(b-dm)/b - d1) < 0.01 and abs(100*(full-b)/b - d2) < 0.01 and d3 <= 100*(full-dm)/full + 0.01, c
    escribir('S2_referentes_sin_gas', [fr'{c} & {usd(a):.2f} & {usd(b):.2f} & {usd(cc):.2f} & {d1:.2f} & {d2:.2f} & {d3:.2f}\\'
                                       for c, a, b, cc, d1, d2, d3 in t16])


# ---------------------------------------------------------------- S3: estado de los solucionadores
def s3():
    filas = []
    nombres = {'K1': 'C400', 'K3': 'C280', 'K2': 'C180', 'K4': 'C100'}
    for k in ('K3', 'K1', 'K2', 'K4'):
        for s, sn in (('T0', 'FLAT'), ('T1', 'WMI')):
            for c, cn in (('OPT_AB', 'OPT-B'), ('OPT_D', 'OPT-X'), ('OPT_CF', 'OPT-F')):
                j = json.loads((CASOS / f'{k}_{s}_{c}.json').read_text())
                cond = 'time limit' if 'time_limit' in j['Condicion'] else 'optimal'
                gap_req = '$10^{-6}$' if k == 'K3' else '$10^{-3}$'
                filas.append(fr'Factorial & {nombres[k]} & {sn} & {cn} & {gap_req} & {cond} & {100*j["Brecha_solver"]:.4f}\\')
    for rf in ('R2', 'R3'):
        for js in sorted((REF / rf / 'casos').glob('*_OPT_*.json')):
            j = json.loads(js.read_text())
            cond = 'time limit' if 'time_limit' in j['Condicion'] else 'optimal'
            nivel = f'gas {j["Gas_USD_MMBtu"]:g}' if rf == 'R2' else f'start-up {j["Cargo_arranque_USD"]:,.0f}'
            filas.append(fr'{rf} ({nivel}) & C280 & {"FLAT" if j["Senal"] == "T0" else "WMI"} & '
                         fr'{"OPT-B" if j["Caso"] == "OPT_AB" else "OPT-X"} & $10^{{-3}}$ & {cond} & {100*j["Brecha_solver"]:.4f}\\')
    escribir('S3_solver', filas)
    # Pertenencia de las trayectorias nuevas
    filas = []
    for rf in ('R1', 'R2', 'R3'):
        js = [json.loads(p.read_text()) for p in sorted((REF / rf / 'casos').glob('R*.json')) if not p.name.endswith('_ventanas.json')]
        js = [j for j in js if j['Caso'] in ('A_INT', 'B_INT', 'D_INT') or j['Caso'].startswith('RH')]
        ok = sum(j.get('Pertenencia') == 'OK' for j in js)
        dif = max(j['Pertenencia_dif_objetivo_USD'] for j in js); err = max(j['Pertenencia_error_flujos'] for j in js)
        filas.append(fr'{rf} & {len(js)} & {ok} & {dif:.2e} & {err:.2e}\\')
    escribir('S3_pertenencia', filas)


# ---------------------------------------------------------------- S4: extensión (tesis + CSV de carbono)
def s4():
    t12 = [('M01', 114.42, '45 / 280', 6.2, 157.61, 134.08), ('M02', 114.42, '45 / 360', 8.0, 190.64, 167.10),
           ('M03', 114.42, '45 / 180', 4.0, 116.33, 92.80), ('M04', 114.42, '45 / 120', 2.7, 91.56, 68.03),
           ('M05', 114.42, '35 / 280', 8.0, 149.11, 130.46), ('M06', 114.42, '25 / 280', 11.2, 140.61, 126.84),
           ('M07', 114.42, '25 / 180', 7.2, 99.33, 85.56), ('M08', 114.42, '25 / 120', 4.8, 74.56, 60.79),
           ('M09', 85.81, '45 / 280', 6.2, 157.61, 134.08), ('M10', 85.81, '25 / 180', 7.2, 99.33, 85.56),
           ('M11', 57.21, '45 / 280', 6.2, 157.61, 134.08), ('M12', 57.21, '25 / 120', 4.8, 74.56, 60.79)]
    escribir('S4_configuraciones', [fr'{a} & {b:.2f} & {c} & {d:.1f} & {usd(e):.2f} & {usd(f):.2f}\\' for a, b, c, d, e, f in t12])
    t24 = [('FLAT', 0, 'M01 (SC)', 260.60, 260.60, 'M05', 4.05, 287.14, 'M05', 9.56),
           ('FLAT', 10, 'M01 (SC)', 277.25, 259.94, 'M05', 4.36, 303.79, 'M05', 9.86),
           ('FLAT', 20, 'M01 (DM)', 292.60, 286.06, 'M05', 4.49, 319.14, 'M05', 10.00),
           ('FLAT', 50, 'M01 (DM)', 302.40, 286.06, 'M05', 4.70, 328.94, 'M05', 10.21),
           ('WMI', 0, 'M01 (SC)', 213.42, 213.42, 'M07', 1.71, 239.96, 'M05', 7.71),
           ('WMI', 10, 'M01 (SC)', 222.76, 213.61, 'M05', 2.32, 249.30, 'M05', 7.83),
           ('WMI', 20, 'M01 (SC)', 231.67, 214.31, 'M05', 2.43, 258.21, 'M05', 7.94),
           ('WMI', 50, 'M01 (SC)', 255.34, 219.18, 'M05', 2.70, 281.88, 'M05', 8.21)]
    escribir('S4_mejor', [fr'{s} & {p} & {b} & {usd(a):.2f} & {usd(f):.2f} & {c2} ({usd(d2):.2f}) & {usd(a3):.2f} & {c3} ({usd(d3):.2f})\\'
                          for s, p, b, a, f, c2, d2, a3, c3, d3 in t24])
    car = pd.read_csv(OP / 'resultados_carbono_v2/malla_20260926/benchmarks_finalistas.csv')
    esc = {'V2_p0': 0, 'V2_p10': 10, 'V2_p20': 20, 'V2_p50': 50}
    car['p'] = car.Escenario_piloto.map(esc)
    filas = []
    for (cfg, s, p), g in car.groupby(['Config', 'Senal', 'p'], sort=True):
        d = {}
        for _, r in g.iterrows():
            lo = math.floor(r.Diferencia_vs_factible_pct*100)/100; hi = math.ceil(r.Diferencia_vs_cota_pct*100)/100
            fin = '' if r.Regla_es_finalista else '*'
            d[r.Regla] = f'{lo:.2f}--{hi:.2f}{fin}'
        term = '; '.join(sorted({f"{'OPT-B' if r.Benchmark == 'OPT_AB' else 'OPT-X'}: {'tolerance' if r.Certificado_brecha_1e6 else 'time limit'}"
                                 for _, r in g.iterrows()}))
        filas.append(fr'{cfg} & {"FLAT" if s == "T0" else "WMI"} & {int(p)} & {term} & {d.get("A_INT", "--")} & '
                     fr'{d.get("B_INT", "--")} & {d.get("D_INT", "--")}\\')
    escribir('S4_distancias', filas)


# ---------------------------------------------------------------- S5: refuerzos (detalle)
def s5():
    c = REF / 'consolidado'
    r1 = pd.read_csv(c / 'R1_horizonte_rodante.csv')
    nom = {'RH-24': 'RH-24', 'RH-24A6': 'RH-24S', 'RH-48': 'RH-48'}
    filas = []
    for _, f in r1[r1.Caso.isin(nom)].iterrows():
        filas.append(fr'{nom[f.Caso]} & {f.Accion} & {f.Senal} & {f.Costo_MUSD:.4f} & {f.Dist_pct_min:.2f}--{f.Dist_pct_max:.2f} & '
                     fr'{int(f.Arranques)} & {f.EFC:.1f} & {f.FP_termico_pct:.1f}\\')
    escribir('S5_r1', filas)
    for nm, campo, fn, fmt in (('S5_r2', 'Gas_USD_MMBtu', 'R2_precio_gas.csv', lambda x: f'{x:g}'),
                               ('S5_r3', 'Cargo_USD', 'R3_cargo_arranque.csv', lambda x: f'{x:,.0f}')):
        df = pd.read_csv(c / fn)
        filas = []
        for _, f in df.iterrows():
            dist = f'{f.Dist_pct_min:.2f}--{f.Dist_pct_max:.2f}' if f.Dist_pct_min == f.Dist_pct_min else '--'
            cost = f'{f.Costo_MUSD:.4f}' if not f.Caso.startswith('OPT') else f'[{math.floor(f.L_MUSD*1e4)/1e4:.4f}, {math.ceil(f.U_MUSD*1e4)/1e4:.4f}]'
            filas.append(fr'{fmt(f[campo])} & {f.Senal} & {f.Caso} & {cost} & {int(f.Arranques)} & {abs(f.FP_termico_pct):.1f} & {f.EFC:.1f} & {dist}\\')
        escribir(nm, filas)


if __name__ == '__main__':
    for fn in (s1, s2, s3, s4, s5):
        fn()
