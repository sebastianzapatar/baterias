"""Figuras del artículo JES (inglés, USD reales 2026), en PDF vectorial y PNG a 300 dpi.

Fuentes (solo lectura):
- Series horarias auditadas: OperacionBESS/resultados_correccion_numerica/20260920/factorial/casos/*.npz
- VPN y bandas: paper_JES/fuentes/vpn_resumen.csv y OperacionBESS/.../vpn/vpn_intervalos.csv
- Distancias del factorial: paper_JES/fuentes/factorial_comparacion.csv
- Carbono (año 1, M01): OperacionBESS/resultados_carbono_v2/malla_20260926/benchmarks_finalistas.csv
- Refuerzos R1–R4: paper_JES/resultados_refuerzos/consolidado/*.csv

Paleta categórica validada con el validador de la guía de visualización (orden y CVD):
azul #2a78d6, naranja #eb6834, aqua #1baf7a, amarillo #eda100 (modo claro, superficie #fcfcfb).
Uso:  OperacionBESS/.venv/bin/python paper_JES/scripts/figuras_jes.py [--solo 4 5 ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
OP = RAIZ.parent / 'OperacionBESS'
# En el repositorio público, paper_JES/ está dentro de la raíz de OperacionBESS.
if not OP.is_dir():
    OP = RAIZ.parent
CASOS = OP / 'resultados_correccion_numerica/20260920/factorial/casos'
VPN_INT = OP / 'resultados_correccion_numerica/20260920/vpn/vpn_intervalos.csv'
CARBONO = OP / 'resultados_carbono_v2/malla_20260926/benchmarks_finalistas.csv'
REF = RAIZ / 'resultados_refuerzos/consolidado'
SALIDA = RAIZ / 'manuscript/figures'
TRM = 3126.08

# Tokens de color (modo claro; las figuras de revista son impresas en claro).
SUP = '#fcfcfb'
TXT = '#0b0b0b'
TXT2 = '#52514e'
GRID = '#e4e3df'
AZUL, NARANJA, AQUA, AMARILLO, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
GRIS = '#8a8984'
REGLA = {'SC': AZUL, 'DM': NARANJA, 'DMX': AQUA, 'AP': AMARILLO, 'APS': MAGENTA}
FUENTE = {'Direct PV': AMARILLO, 'BESS discharge': AZUL, 'Gas units': NARANJA, 'Grid purchases': AQUA}
CODIGO = {'SC': 'A_INT', 'DM': 'B_INT', 'DMX': 'D_INT', 'AP': 'C_INT', 'APS': 'F_INT',
          'OPT-B': 'OPT_AB', 'OPT-X': 'OPT_D', 'OPT-F': 'OPT_CF', 'Grid+Gas': 'RED_TERMICA', 'PV-only': 'PV_SIN_BESS'}
SENAL = {'FLAT': 'T0', 'WMI': 'T1'}
CONFIG = {'C400': 'K1', 'C280': 'K3', 'C180': 'K2', 'C100': 'K4'}
CV20 = np.array([202.04, 202.04, 237.10, 260.12])

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.edgecolor': GRIS, 'axes.linewidth': 0.6, 'axes.labelcolor': TXT,
    'xtick.color': TXT2, 'ytick.color': TXT2, 'text.color': TXT, 'axes.facecolor': SUP, 'figure.facecolor': 'white',
    'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.5, 'lines.linewidth': 1.4,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})
MM = 1/25.4
DOBLE, SIMPLE = 190*MM, 90*MM


def caso(k, s, c):
    with np.load(CASOS / f'{k}_{s}_{c}.npz', allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def guardar(fig, nombre):
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / f'{nombre}.pdf', bbox_inches='tight')
    fig.savefig(SALIDA / f'{nombre}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('OK', nombre)


def ejes_limpios(ax, eje='y'):
    ax.grid(True, axis=eje, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def etiqueta_panel(ax, letra, x=-0.02, y=1.04):
    ax.text(x, y, f'({letra})', transform=ax.transAxes, fontsize=8.5, fontweight='bold', ha='right', va='bottom')


def componentes(r, cv=CV20, cargo=100.):
    pot = r['potencia']
    cr = r.get('Carga_Red', np.zeros(len(pot)))
    gas = float((pot[:, :4] @ cv).sum()) + cargo*float(r['arranques'].sum())
    compras = float((pot[:, 4] + cr) @ r['C_imp_USD'])
    ingreso = float((r['Export_PV'] + r['Descarga_export']) @ r['C_exp_USD'])
    return gas, compras, ingreso


# ---------------------------------------------------------------- Fig. 1
def fig1():
    fig, ax = plt.subplots(figsize=(DOBLE*0.78, 78*MM))
    ax.set_xlim(0, 122); ax.set_ylim(0, 76); ax.axis('off')
    W, H = 22, 11
    cajas = {'PV': (14, 50, 'PV plant\n114.4 MWp', AMARILLO), 'BESS': (14, 14, 'BESS\n45 MW/280 MWh', AZUL),
             'Load': (58, 32, 'Industrial load\n158.7 GWh/yr', GRIS), 'Gas': (58, 58, 'Gas units G1–G4\n25.2 MW', NARANJA),
             'Grid': (108, 32, 'National\ngrid', AQUA)}
    ax.add_patch(FancyBboxPatch((2, 3), 86, 64, boxstyle='round,pad=0,rounding_size=2', fill=False, ec=GRIS, lw=0.8,
                                ls=(0, (4, 3))))
    ax.text(4, 4.5, 'Behind-the-meter boundary (user)', fontsize=6.5, color=TXT2, va='bottom')
    for x, y, t, c in cajas.values():
        ax.add_patch(FancyBboxPatch((x-W/2, y-H/2), W, H, boxstyle='round,pad=0.3,rounding_size=1.5',
                                    fc='white', ec=c, lw=1.6, zorder=3))
        ax.text(x, y, t, ha='center', va='center', fontsize=7, zorder=4)

    def flecha(pts, disc=False, txt=None, tx=None, ty=None, ha='center'):
        estilo = dict(color=TXT if disc else TXT2, lw=1.1, ls=(0, (3, 2)) if disc else '-')
        for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x0, x1], [y0, y1], **estilo, solid_capstyle='butt', zorder=2)
        ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=2,
                    arrowprops=dict(arrowstyle='-|>', mutation_scale=8, shrinkA=0, shrinkB=0, **estilo))
        if txt:
            ax.text(tx, ty, txt, fontsize=6.2, color=TXT2, ha=ha, va='center', zorder=5,
                    bbox=dict(fc='white', ec='none', pad=0.6))
    flecha([(25.3, 47), (46.7, 35.5)], txt='direct PV', tx=37, ty=44.5)
    flecha([(14, 44.2), (14, 19.8)], txt='surplus\ncharging', tx=14, ty=32)
    flecha([(25.3, 17), (46.7, 28.5)], txt='discharge', tx=37, ty=19.5)
    flecha([(58, 52.2), (58, 37.8)], txt='gas', tx=60.5, ty=45, ha='left')
    flecha([(96.7, 32), (69.3, 32)], txt='purchases', tx=81, ty=34.2)
    flecha([(14, 55.8), (14, 71), (108, 71), (108, 37.8)], txt='PV surplus export', tx=60, ty=71)
    flecha([(25.3, 11.5), (112, 11.5), (112, 26.2)], disc=True, txt='BESS export (OPT-X, OPT-F, DMX, AP, APS)', tx=62, ty=11.5)
    flecha([(104, 26.2), (104, 16.5), (25.3, 16.5)], disc=True, txt='grid charging (OPT-F, AP, APS)', tx=72, ty=16.5)
    guardar(fig, 'fig1_system')


# ---------------------------------------------------------------- Fig. 2
def fig2():
    a = caso('K3', 'T0', 'A_INT'); b = caso('K3', 'T1', 'A_INT')
    flat, wmi, exp = a['C_imp_USD'], b['C_imp_USD'], a['C_exp_USD']
    h = pd.DatetimeIndex(a['Fecha']).hour
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOBLE, 62*MM), gridspec_kw=dict(wspace=0.28))
    df = pd.DataFrame(dict(h=h, FLAT=flat, WMI=wmi, EXP=exp))
    g = df.groupby('h')
    ax1.fill_between(range(24), g.WMI.quantile(.1), g.WMI.quantile(.9), color=AQUA, alpha=0.12, lw=0,
                     label='WMI 10th–90th percentile')
    ax1.plot(range(24), g.FLAT.mean(), color=AZUL, lw=1.6, label='FLAT')
    ax1.plot(range(24), g.WMI.mean(), color=AQUA, lw=1.6, label='WMI (mean)')
    ax1.plot(range(24), g.EXP.mean(), color=GRIS, lw=1.4, ls='-', label='Export price (mean)')
    for cv, t, dy in ((CV20[0], 'G1/G2', -13), (CV20[2], 'G3', 4), (CV20[3], 'G4', 4)):
        ax1.axhline(cv, color=NARANJA, lw=0.9, ls=(0, (4, 2)))
        ax1.text(0.3, cv+dy, t, fontsize=6, color=TXT2, ha='left')
    ax1.set_xlim(0, 23); ax1.set_xticks(range(0, 24, 3)); ax1.set_xlabel('Hour of day')
    ax1.set_ylabel('USD/MWh'); ax1.set_ylim(0, 420); ejes_limpios(ax1)
    ax1.legend(loc='upper left', ncol=2, handlelength=1.6)
    etiqueta_panel(ax1, 'a')
    x = np.arange(1, 8761)
    for serie, c, t in ((flat, AZUL, 'FLAT'), (wmi, AQUA, 'WMI'), (exp, GRIS, 'Export price')):
        ax2.plot(x, np.sort(serie)[::-1], color=c, lw=1.6, label=t)
    caja = dict(fc=SUP, ec='none', pad=0.4)
    for cv, t, x, va in ((CV20[0], 'G1/G2 @ 20', 8700, 'top'), (CV20[2], 'G3 @ 20', 8700, 'bottom'),
                         (CV20[3], 'G4 @ 20', 8700, 'bottom')):
        ax2.axhline(cv, color=NARANJA, lw=0.9, ls=(0, (4, 2)))
        ax2.text(x, cv+(-3 if va == 'top' else 3), t, fontsize=5.8, color=TXT2, ha='right', va=va, bbox=caja)
    for g_, cvg in ((12, 124.94), (16, 163.49), (24, 240.59)):
        ax2.axhline(cvg, color=NARANJA, lw=0.6, ls=':')
        ax2.text(6300, cvg+(3 if g_ != 24 else -3), f'G1/G2 @ {g_}', fontsize=5.6, color=TXT2,
                 va='bottom' if g_ != 24 else 'top', bbox=caja)
    ax2.text(1500, 160, 'gas prices in USD/MMBtu', fontsize=5.6, color=TXT2, va='top')
    ax2.text(120, 446, f'WMI maximum {wmi.max():.0f} USD/MWh (off scale)', fontsize=5.8, color=TXT2, va='top')
    ax2.set_xlim(1, 8760); ax2.set_ylim(0, 450); ax2.set_xlabel('Hours of the year (sorted)')
    ax2.set_ylabel('USD/MWh'); ejes_limpios(ax2); ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    etiqueta_panel(ax2, 'b')
    guardar(fig, 'fig2_prices')


# ---------------------------------------------------------------- Fig. 3
def fig3():
    fig, ax = plt.subplots(figsize=(DOBLE, 84*MM))
    ax.set_xlim(0, 196); ax.set_ylim(0, 82); ax.axis('off')

    def caja(x, y, w, h, titulo, cuerpo, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.8', fc='white', ec=c, lw=1.3))
        ax.text(x+w/2, y+h-3, titulo, ha='center', va='top', fontsize=7, fontweight='bold')
        ax.text(x+w/2, y+h-9, cuerpo, ha='center', va='top', fontsize=6.1, color=TXT2, linespacing=1.4)

    def fl(p0, p1):
        ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='-|>', lw=1.0, color=TXT2, mutation_scale=8,
                                                          shrinkA=0, shrinkB=0))
    caja(2, 28, 36, 48, 'Inputs (2025, hourly)',
         'Demand 158.7 GWh\nTMY PV (pvlib)\nFLAT and WMI signals\nExport price\nG1–G4: minimum stable\ngeneration, heat rate,\nCVOM, start-up charge,\ndaily event limits', GRIS)
    caja(48, 55, 48, 22, 'Five heuristic rules', 'SC, DM, DMX, AP, APS\ncommon daily DP thermal manager\ncyclic SOC (fixed-point iteration)', AZUL)
    caja(48, 29, 48, 22, 'Nested annual MILP', r'OPT-B $\subset$ OPT-X $\subset$ OPT-F' + '\nPyPSA + HiGHS, perfect foresight\nsolver bounds [L, U]', NARANJA)
    caja(48, 3, 48, 22, 'Rolling-horizon MILP', 'RH-24, RH-24S, RH-48\ninherited SOC and commitment\n24 h committed per step', AQUA)
    caja(106, 43, 42, 34, 'Verification', 'physical and economic\nvalidators; membership:\nrule fixed in its\naction-matched benchmark', GRIS)
    caja(106, 3, 42, 34, 'Experimental design', '4 configurations × 2 signals\n× 5 rules; R2: gas price\n12–24 USD/MMBtu; R3: start-up\n0–1,000 USD; M01–M12 with\nunavailability and CO$_2$ price', GRIS)
    caja(156, 43, 38, 34, 'Operating metrics', 'annual cost, start-ups,\ngas use, EFC\n' + r'$\Delta C \in [C-U,\ C-L]$' + '\ngap decomposition', GRIS)
    caja(156, 3, 38, 34, '20-year economics', 'total and incremental NPV\nIRR, discounted payback\nband of max. compatible\nNPV; capture ratio', GRIS)
    for y in (66, 40, 14):
        fl((38.6, 52), (47.4, y))
        fl((96.6, y), (105.4, 60))
    fl((148.6, 60), (155.4, 60)); fl((148.6, 20), (155.4, 20)); fl((175, 42.4), (175, 37.6))
    guardar(fig, 'fig3_framework')


# ---------------------------------------------------------------- Fig. 4
def fig4():
    casos_ = ['Grid+Gas', 'PV-only', 'SC', 'DM', 'DMX', 'AP', 'APS', 'OPT-B', 'OPT-X', 'OPT-F']
    comp = pd.read_csv(RAIZ/'fuentes/factorial_comparacion.csv')
    fig = plt.figure(figsize=(DOBLE, 105*MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=0.55, wspace=0.42)
    for j, sen in enumerate(('FLAT', 'WMI')):
        ax = fig.add_subplot(gs[0, j])
        y = np.arange(len(casos_))[::-1]
        for i, c in enumerate(casos_):
            r = caso('K3', SENAL[sen], CODIGO[c])
            d = r['Carga'].sum()
            partes = {'Direct PV': r['PV_directo'].sum(), 'BESS discharge': r['Descarga_demanda'].sum(),
                      'Gas units': r['potencia'][:, :4].sum(), 'Grid purchases': r['potencia'][:, 4].sum()}
            izq = 0.
            for k, v in partes.items():
                w = 100*v/d
                ax.barh(y[i], max(w-0.4, 0), left=izq, height=0.62, color=FUENTE[k], lw=0,
                        label=k if i == 0 else None)
                izq += w
            costo = sum(componentes(r)[:2]) - componentes(r)[2]
            ax.text(103, y[i], f'{costo/1e6:.2f}', va='center', fontsize=6.5, color=TXT)
        ax.text(103, len(casos_)-0.2, 'Cost\n(M USD)', fontsize=6.3, color=TXT2, va='bottom')
        ax.set_yticks(y); ax.set_yticklabels(casos_); ax.set_xlim(0, 100)
        ax.set_xlabel('Share of annual demand supplied (%)'); ax.set_title(f'{sen} signal', loc='left', fontsize=8)
        ejes_limpios(ax, 'x'); ax.tick_params(axis='y', length=0)
        etiqueta_panel(ax, 'ab'[j], x=-0.12)
        if j == 0:
            ax.legend(ncol=4, loc='upper left', bbox_to_anchor=(0, -0.2), handlelength=1.2, columnspacing=1.2)
    # (c) distancias del bloque principal, (d) bloque de compra anticipada
    for j, (grupo, letra, escala) in enumerate(((('SC', 'DM', 'DMX'), 'c', 1e3), (('AP', 'APS'), 'd', 1e6))):
        ax = fig.add_subplot(gs[1, j])
        filas = []
        for regla in grupo:
            for sen in ('FLAT', 'WMI'):
                f = comp[(comp.K == 'K3') & (comp.Senal == SENAL[sen]) & (comp.Caso == CODIGO[regla])].iloc[0]
                filas.append((regla, sen, f.Distancia_min_USD/escala, f.Distancia_max_USD/escala))
        for i, (regla, sen, lo, hi) in enumerate(filas):
            yy = len(filas)-1-i
            ax.plot([0 if escala == 1e3 else 4.0, lo], [yy, yy], color=GRID, lw=0.8, zorder=1)
            ax.plot([lo, hi], [yy, yy], color=TXT2, lw=0.9, zorder=2)
            ax.plot((lo+hi)/2, yy, marker='o', ms=6, color=REGLA[regla], mec='white', mew=1.0, zorder=3)
            if escala == 1e3:
                t = f'{np.floor(lo*10)/10:,.1f}–{np.ceil(hi*10)/10:,.1f}'
            else:
                t = f'{np.floor(lo*1e3)/1e3:.3f}–{np.ceil(hi*1e3)/1e3:.3f}'
            ax.text(hi + (0.06 if escala == 1e6 else 14), yy, t, va='center', fontsize=6.2, color=TXT2)
        ax.set_yticks(range(len(filas))); ax.set_yticklabels([f'{r} · {s}' for r, s, *_ in filas][::-1])
        ax.tick_params(axis='y', length=0); ejes_limpios(ax, 'x')
        if escala == 1e3:
            ax.set_xlim(0, 700); ax.set_xlabel('Distance to matched benchmark (kUSD/year)')
        else:
            ax.set_xlim(4.0, 7.4); ax.set_xlabel('Distance to OPT-F (M USD/year)')
        etiqueta_panel(ax, letra, x=-0.12)
    guardar(fig, 'fig4_c280_costs_distances')


# ---------------------------------------------------------------- Fig. 5
def fig5():
    t0 = 110*24; sl = slice(t0, t0+168)
    sc, ob = caso('K3', 'T0', 'A_INT'), caso('K3', 'T0', 'OPT_AB')
    fechas = pd.DatetimeIndex(sc['Fecha'])[sl]
    x = np.arange(168)
    fig, axs = plt.subplots(4, 1, figsize=(DOBLE, 140*MM), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.15, 1.15, 0.95], hspace=0.38))
    noche = sc['PV'][sl] <= 1e-6
    for ax in axs:
        for i in np.where(noche)[0]:
            ax.axvspan(i-0.5, i+0.5, color='#efeeea', lw=0, zorder=0)
    axs[0].plot(x, sc['PV'][sl], color=AMARILLO, lw=1.6, label='Available PV')
    axs[0].plot(x, sc['Carga'][sl], color=TXT, lw=1.2, label='Demand')
    axs[0].set_ylabel('MW'); axs[0].set_ylim(0, 92)
    axs[0].legend(loc='upper left', bbox_to_anchor=(0, 1.32), ncol=2)
    for ax, r, t in ((axs[1], sc, 'SC'), (axs[2], ob, 'OPT-B')):
        capas = [r['PV_directo'][sl], r['Descarga_demanda'][sl], r['potencia'][sl, :4].sum(1), r['potencia'][sl, 4]]
        ax.stackplot(x, *capas, colors=[FUENTE[k] for k in FUENTE], labels=list(FUENTE), lw=0)
        ax.plot(x, r['Carga'][sl], color=TXT, lw=0.9)
        arr = r['arranques'][sl].sum(1)
        ix = np.where(arr > 0)[0]
        ax.plot(ix, np.full(len(ix), 24.5), ls='none', marker='v', ms=4.5, color=TXT, mec='white', mew=0.6,
                label='Unit start-up' if t == 'SC' else None)
        ax.set_title(f'{t}: {int(arr.sum())} start-ups in the week', loc='right', fontsize=6.8, color=TXT2, pad=2)
        ax.set_ylim(0, 26.5); ax.set_ylabel('MW')
    h, l = axs[1].get_legend_handles_labels()
    axs[1].legend(h, l, loc='upper left', bbox_to_anchor=(0, 1.3), ncol=5, handlelength=1.2)
    axs[3].plot(x, sc['SOC'][sl], color=AZUL, lw=1.5, label='SC')
    axs[3].plot(x, ob['SOC'][sl], color=GRIS, lw=1.5, label='OPT-B')
    axs[3].axhline(56, color=TXT2, lw=0.6, ls=':')
    axs[3].set_ylim(40, 300); axs[3].set_ylabel('SOC (MWh)')
    axs[3].legend(loc='upper left', bbox_to_anchor=(0, 1.32), ncol=2)
    for ax, letra in zip(axs, 'abcd'):
        etiqueta_panel(ax, letra, x=-0.07, y=1.0)
        ax.grid(True, axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    axs[3].set_xticks(range(0, 169, 24))
    axs[3].set_xticklabels([f.strftime('%d %b') for f in fechas[::24]] + [''])
    axs[3].set_xlim(-0.5, 167.5)
    guardar(fig, 'fig5_week')


# ---------------------------------------------------------------- Fig. 6
def fig6():
    cm_gas = LinearSegmentedColormap.from_list('gas', [SUP, '#f6b896', NARANJA, '#8c3511'])
    cm_soc = LinearSegmentedColormap.from_list('soc', ['#cde2fb', '#6da7ec', '#2a78d6', '#104281'])
    casos_ = [('SC', 'A_INT'), ('DM', 'B_INT'), ('OPT-B', 'OPT_AB')]
    fig, axs = plt.subplots(2, 3, figsize=(DOBLE, 92*MM), sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.22, wspace=0.06))
    for j, (t, c) in enumerate(casos_):
        r = caso('K3', 'T0', c)
        gas = r['potencia'][:, :4].sum(1).reshape(365, 24).T
        soc = r['SOC'].reshape(365, 24).T
        im1 = axs[0, j].imshow(gas, aspect='auto', origin='lower', cmap=cm_gas, vmin=0, vmax=25.2,
                               interpolation='nearest', extent=(0.5, 365.5, -0.5, 23.5))
        im2 = axs[1, j].imshow(soc, aspect='auto', origin='lower', cmap=cm_soc, vmin=56, vmax=280,
                               interpolation='nearest', extent=(0.5, 365.5, -0.5, 23.5))
        axs[0, j].set_title(t, loc='left')
        etiqueta_panel(axs[0, j], 'abc'[j], x=-0.03); etiqueta_panel(axs[1, j], 'def'[j], x=-0.03)
    for ax in axs.flat:
        ax.set_yticks([0, 6, 12, 18, 23]); ax.spines[:].set_visible(False)
        ax.set_xticks([1, 91, 182, 274]); ax.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
    axs[0, 0].set_ylabel('Hour'); axs[1, 0].set_ylabel('Hour')
    c1 = fig.colorbar(im1, ax=axs[0, :], fraction=0.015, pad=0.01); c1.set_label('Gas generation (MW)')
    c2 = fig.colorbar(im2, ax=axs[1, :], fraction=0.015, pad=0.01); c2.set_label('Stored energy (MWh)')
    for c in (c1, c2):
        c.outline.set_visible(False)
    guardar(fig, 'fig6_heatmaps')


# ---------------------------------------------------------------- Fig. 7
def fig7():
    r1 = pd.read_csv(REF/'R1_horizonte_rodante.csv')
    r2 = pd.read_csv(REF/'R2_precio_gas.csv'); r3 = pd.read_csv(REF/'R3_cargo_arranque.csv')
    r4 = pd.read_csv(REF/'R4_mpc_pronosticos.csv')
    fig = plt.figure(figsize=(DOBLE, 120*MM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.62, wspace=0.42)
    ax = fig.add_subplot(gs[0, :3])
    filas = []
    for regla in ('SC', 'DM', 'DMX'):
        for sen in ('FLAT', 'WMI'):
            f24 = r1[(r1.Senal == sen) & (r1.Caso == f'{regla} vs RH-24A6')].iloc[0]
            f48 = r1[(r1.Senal == sen) & (r1.Caso == f'{regla} vs RH-48')].iloc[0]
            logica = f24.Logica_kUSD
            hor = f24.Logica_kUSD*0 + (f48.Logica_kUSD - f24.Logica_kUSD)
            mas = (f48.Prevision_kUSD_min + f48.Prevision_kUSD_max)/2
            filas.append((f'{regla} · {sen}', logica, hor, mas))
    y = np.arange(len(filas))[::-1]
    colores = (AZUL, NARANJA, AQUA)
    etiquetas = ('Rule − solar-day MILP', '24-h solar day → 48-h calendar window',
                 '48-h MILP − annual optimum')
    for i, (t, lg, hz, ms) in enumerate(filas):
        izq_pos, izq_neg = 0., 0.
        for k, v in enumerate((lg, hz, ms)):
            if v >= 0:
                ax.barh(y[i], v, left=izq_pos, height=0.6, color=colores[k], ec='white', lw=0.6,
                        label=etiquetas[k] if i == 0 else None)
                izq_pos += v
            else:
                ax.barh(y[i], v, left=izq_neg, height=0.6, color=colores[k], lw=0,
                        label=etiquetas[k] if i == 0 else None)
                izq_neg += v
        ax.text(izq_pos+8, y[i], f'{lg+hz+ms:,.0f}', va='center', fontsize=6.3, color=TXT2)
    ax.axvline(0, color=TXT2, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([f[0] for f in filas]); ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Gap to the annual optimum (kUSD/year, C280)'); ax.set_xlim(-20, 650)
    ejes_limpios(ax, 'x'); ax.legend(loc='upper right', ncol=1, handlelength=1.2)
    etiqueta_panel(ax, 'a', x=-0.05)
    # (b) R4: MPC-48 re-optimizado cada hora frente a SC según la calidad del pronóstico.
    ax = fig.add_subplot(gs[0, 3])
    ax.axhline(0, color=REGLA['SC'], lw=1.0)
    ax.text(1.0, 6, 'SC', color=REGLA['SC'], fontsize=6.5, ha='right', va='bottom')
    for sen, estilo, marca in (('FLAT', '-', 'o'), ('WMI', '--', 's')):
        d = r4[(r4.Senal == sen) & (r4.Caso == 'MPC-48 horario')].sort_values('Alfa')
        ax.plot(d.Alfa, d.Ahorro_vs_SC_kUSD, color=TXT, ls=estilo, lw=1.3, marker=marca, ms=3.6, mec='white',
                mew=0.7, label=sen)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0\nperfect', '0.5', '1\npersistence'])
    ax.set_xlabel('Forecast error level α'); ax.set_ylabel('MPC-48 saving vs SC (kUSD/year)')
    ax.set_ylim(-400, 200); ejes_limpios(ax); ax.legend(loc='lower left', handlelength=2.2)
    etiqueta_panel(ax, 'b', x=-0.08)
    paneles = [(r2, 'Gas_USD_MMBtu', 'FLAT', 'Gas price (USD/MMBtu)', 'c'),
               (r2, 'Gas_USD_MMBtu', 'WMI', 'Gas price (USD/MMBtu)', 'd'),
               (r3, 'Cargo_USD', 'FLAT', 'Start-up charge (USD)', 'e'),
               (r3, 'Cargo_USD', 'WMI', 'Start-up charge (USD)', 'f')]
    for j, (df, campo, sen, xl, letra) in enumerate(paneles):
        ax = fig.add_subplot(gs[1, j])
        for regla in ('SC', 'DM', 'DMX'):
            d = df[(df.Senal == sen) & (df.Caso == regla)].dropna(subset=['Dist_pct_min']).sort_values(campo)
            if d.empty:
                continue
            mid = (d.Dist_pct_min + d.Dist_pct_max)/2
            xs = sorted(df[campo].unique())
            xv = d[campo] if campo != 'Cargo_USD' else d[campo].map({v: i for i, v in enumerate(xs)})
            ax.plot(xv, mid, color=REGLA[regla], lw=1.5, marker='o', ms=4, mec='white', mew=0.8, label=regla)
        ax.set_title(sen, loc='left', fontsize=7.5); ax.set_xlabel(xl)
        if j in (0, 2):
            ax.set_ylabel('Distance to matched\nbenchmark (%)')
        xs = sorted(df[campo].unique())
        if campo == 'Cargo_USD':
            ax.set_xticks(range(len(xs))); ax.set_xticklabels([f'{int(v):,}' for v in xs]); ax.set_ylim(0, 12)
        else:
            ax.set_xticks(xs); ax.set_ylim(0, 55)
        ejes_limpios(ax)
        etiqueta_panel(ax, letra, x=-0.08)
        if j in (0, 2):
            ax.legend(loc='upper right', handlelength=1.4)
    guardar(fig, 'fig7_gap_decomposition')


# ---------------------------------------------------------------- Fig. 8
def fig8():
    comp = pd.read_csv(RAIZ/'fuentes/factorial_comparacion.csv')
    fig = plt.figure(figsize=(DOBLE, 112*MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.75, wspace=0.18)
    reglas = ('SC', 'DM', 'DMX')
    for j, sen in enumerate(('FLAT', 'WMI')):
        ax = fig.add_subplot(gs[0, j])
        xt, xl = [], []
        for ci, (cn, k) in enumerate(CONFIG.items()):
            for ri, regla in enumerate(reglas):
                x = ci*4 + ri
                r = caso(k, SENAL[sen], CODIGO[regla])
                g_, p_, e_ = (v/1e6 for v in componentes(r))
                ref = comp[(comp.K == k) & (comp.Senal == SENAL[sen]) & (comp.Caso == CODIGO[regla])].Costo_USD.iloc[0]/1e6
                assert abs(g_+p_-e_-ref) < 1e-6, (k, sen, regla)
                ax.bar(x, g_, width=0.78, color=NARANJA, lw=0, label='Gas units (fuel, CVOM, start-ups)' if x == 0 else None)
                ax.bar(x, p_-0.03, bottom=g_+0.03, width=0.78, color=AQUA, lw=0, label='Grid purchases' if x == 0 else None)
                ax.bar(x, -e_, width=0.78, color=GRIS, lw=0, label='Export revenue' if x == 0 else None)
                ax.plot(x, g_+p_-e_, marker='D', ms=4, color=TXT, mec='white', mew=0.7, ls='none',
                        label='Net operating cost' if x == 0 else None)
                xt.append(x); xl.append(regla)
            ax.text(ci*4+1, 15.3, cn, ha='center', fontsize=7, fontweight='bold')
        ax.axhline(0, color=TXT2, lw=0.6)
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6, rotation=90); ax.set_ylim(-2.5, 16.5)
        ax.set_title(f'{sen} signal', loc='left', fontsize=8, pad=12)
        if j == 0:
            ax.set_ylabel('Annual cost (M USD)')
            ax.legend(loc='upper left', bbox_to_anchor=(0, -0.27), ncol=4, handlelength=1.2)
        ejes_limpios(ax); etiqueta_panel(ax, 'ab'[j], x=-0.05, y=1.07)
    ax = fig.add_subplot(gs[1, :])
    cols = [(cn, sen) for cn in CONFIG for sen in ('FLAT', 'WMI')]
    M = np.zeros((3, len(cols))); txt = {}
    for jc, (cn, sen) in enumerate(cols):
        k = CONFIG[cn]
        costos = {}
        for ir, regla in enumerate(reglas):
            f = comp[(comp.K == k) & (comp.Senal == SENAL[sen]) & (comp.Caso == CODIGO[regla])].iloc[0]
            lo = 100*max(0, f.Costo_USD/f.Mejor_factible_USD-1); hi = 100*max(0, f.Costo_USD/f.Cota_inferior_USD-1)
            lo, hi = np.floor(lo*100)/100, np.ceil(hi*100)/100  # hacia afuera, como la Tabla 6
            M[ir, jc] = (lo+hi)/2; txt[(ir, jc)] = f'{lo:.2f}–\n{hi:.2f}'; costos[ir] = f.Costo_USD
        txt[('best', jc)] = min(costos, key=costos.get)
    cm = LinearSegmentedColormap.from_list('dist', ['#cde2fb', '#6da7ec', '#2a78d6', '#104281'])
    im = ax.imshow(M, cmap=cm, aspect='auto', vmin=0, vmax=36)
    for (ir, jc), t in txt.items():
        if ir == 'best':
            continue
        ax.text(jc, ir, t, ha='center', va='center', fontsize=5.8, color='white' if M[ir, jc] > 14 else TXT)
    for jc in range(len(cols)):
        ax.add_patch(Rectangle((jc-0.48, txt[('best', jc)]-0.47), 0.96, 0.94, fill=False, ec=TXT, lw=1.3))
    ax.set_yticks(range(3)); ax.set_yticklabels(reglas)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f'{c}\n{s}' for c, s in cols])
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label('Distance (%, midpoint)'); cb.outline.set_visible(False)
    etiqueta_panel(ax, 'c', x=-0.03, y=1.05)
    guardar(fig, 'fig8_factorial')


# ---------------------------------------------------------------- Fig. 9
def fig9():
    vr = pd.read_csv(RAIZ/'fuentes/vpn_resumen.csv'); vi = pd.read_csv(VPN_INT)
    car = pd.read_csv(CARBONO)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOBLE, 70*MM), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.35))
    match = {'SC': 'OPT_AB', 'DM': 'OPT_AB', 'DMX': 'OPT_D', 'AP': 'OPT_CF', 'APS': 'OPT_CF'}
    filas = [(r, s) for r in ('SC', 'DM', 'DMX', 'AP', 'APS') for s in ('FLAT', 'WMI')]
    for i, (regla, sen) in enumerate(filas):
        yy = len(filas)-1-i
        v = vr[(vr.Senal == SENAL[sen]) & (vr.Caso == CODIGO[regla]) & (vr.Referencia == 'PV_SIN_BESS')].VPN_COP.iloc[0]/TRM/1e6
        b = vi[(vi.Senal == SENAL[sen]) & (vi.Benchmark == match[regla]) & (vi.Referencia == 'PV_SIN_BESS')].iloc[0]
        lo, hi = b.VPN_factible_COP/TRM/1e6, b.VPN_cota_superior_COP/TRM/1e6
        ax1.plot([0, v], [yy, yy], color=GRID, lw=0.8, zorder=1)
        ax1.add_patch(Rectangle((lo, yy-0.32), max(hi-lo, 0.12), 0.64, color=TXT2, lw=0, zorder=2))
        ax1.plot(v, yy, marker='o', ms=6, color=REGLA[regla], mec='white', mew=1.0, zorder=3)
        ax1.text(31.5, yy, f'{v:+.2f}', va='center', fontsize=6.2, color=TXT2)
    ax1.axvline(0, color=TXT2, lw=0.6)
    ax1.set_yticks(range(len(filas))); ax1.set_yticklabels([f'{r} · {s}' for r, s in filas][::-1])
    ax1.tick_params(axis='y', length=0); ax1.set_xlim(-32, 30); ejes_limpios(ax1, 'x')
    ax1.set_xlabel('Incremental BESS NPV (M USD, 20 years)')
    ax1.plot([], [], marker='o', ls='none', color=GRIS, label='Rule')
    ax1.add_patch(Rectangle((0, 0), 0, 0, color=TXT2, label='Band of max. compatible NPV'))
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.13), ncol=2, handlelength=1.0)
    etiqueta_panel(ax1, 'a', x=-0.1, y=1.13)
    esc = {'V2_p0': 0, 'V2_p10': 10, 'V2_p20': 20, 'V2_p50': 50}
    for sen, ls in (('FLAT', '-'), ('WMI', (0, (3, 2)))):
        for regla in ('SC', 'DM', 'DMX'):
            d = car[(car.Config == 'M01') & (car.Senal == SENAL[sen]) & (car.Regla == CODIGO[regla])].copy()
            d['p'] = d.Escenario_piloto.map(esc); d = d.dropna(subset=['p']).sort_values('p')
            mid = (d.Diferencia_vs_factible_pct + d.Diferencia_vs_cota_pct)/2
            ax2.plot(d.p, mid, color=REGLA[regla], ls=ls, lw=1.5, marker='o', ms=4, mec='white', mew=0.8,
                     label=f'{regla} · {sen}')
    ax2.axvspan(15, 55, color='#efeeea', lw=0, zorder=0)
    ax2.text(35, 12.9, 'gas units off under FLAT', ha='center', fontsize=6.2, color=TXT2)
    ax2.set_xticks([0, 10, 20, 50]); ax2.set_xlim(-2, 52); ax2.set_ylim(0, 13.5)
    ax2.set_xlabel('Internal carbon price (USD/t CO$_2$e)'); ax2.set_ylabel('Year-1 distance to benchmark (%)')
    ejes_limpios(ax2); ax2.legend(loc='upper left', bbox_to_anchor=(0, 1.2), ncol=3, handlelength=1.8, fontsize=6.2)
    etiqueta_panel(ax2, 'b', x=-0.1, y=1.13)
    guardar(fig, 'fig9_npv_carbon')


# ---------------------------------------------------------------- Graphical abstract
def fig_ga():
    """Resumen gráfico autónomo; cifras de C280, sin redondear las cotas como datos."""
    comp = pd.read_csv(RAIZ/'fuentes/factorial_comparacion.csv')
    r2 = pd.read_csv(REF/'R2_precio_gas.csv')
    r1 = pd.read_csv(REF/'R1_horizonte_rodante.csv')
    vpn = pd.read_csv(RAIZ/'fuentes/vpn_resumen.csv')
    bandas = pd.read_csv(VPN_INT)
    fig = plt.figure(figsize=(160*MM, 95*MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.05], wspace=0.58,
                          left=0.025, right=0.98, top=0.73, bottom=0.30)
    fig.text(0.025, 0.96, 'Industrial solar + battery + gas + grid',
             fontsize=11, fontweight='bold', va='top')
    fig.text(0.025, 0.90, 'Heuristic rules vs. MILP with the same permitted battery actions',
             fontsize=8.5, va='top')
    fig.text(0.025, 0.85, 'Colombia · 8,760 hourly steps · Base case: gas at 20 USD/MMBtu',
             fontsize=7.4, color=TXT2, va='top')
    # (1) sistema: solo los flujos básicos; las acciones añadidas se identifican debajo.
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.set_title('1  Industrial system', loc='left', fontsize=8.8, fontweight='bold', pad=11)
    cajas = [(20, 86, 'Solar\n114.4 MWp', AMARILLO),
             (20, 30, 'Battery\n45 MW\n280 MWh', AZUL),
             (80, 86, 'Gas\n4 units\n25.2 MW', NARANJA),
             (80, 30, 'Grid\npurchases', AQUA),
             (50, 58, 'Load\n158.7 GWh/yr', GRIS)]
    for x, y, t, c in cajas:
        ancho = 48 if t.startswith('Load') else 40
        ax.add_patch(FancyBboxPatch((x-ancho/2, y-11), ancho, 22,
                                   boxstyle='round,pad=0.4,rounding_size=2',
                                   fc='white', ec=c, lw=1.5))
        ax.text(x, y, t, ha='center', va='center', fontsize=7.6, linespacing=1.05)
    for (x0, y0), (x1, y1) in (((34, 75), (42, 69)), ((34, 41), (42, 47)),
                               ((66, 75), (58, 69)), ((66, 41), (58, 47)),
                               ((20, 74), (20, 43))):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', lw=0.9, color=TXT2, mutation_scale=7))
    ax.text(50, 0, 'Minimum output · start-up costs\nDaily start/stop limits',
            ha='center', va='bottom', fontsize=7.2, color=TXT2)
    # (2) las tres reglas de carga solar; barras en el punto medio de [L,U].
    ax = fig.add_subplot(gs[1])
    ax.set_title('2  Annual cost gap', loc='left', fontsize=8.8, fontweight='bold', pad=11)
    reglas = ('SC', 'DM', 'DMX'); x = np.arange(3); w = 0.36
    for j, (sen, alpha) in enumerate((('FLAT', 1.0), ('WMI', 0.50))):
        vals = []
        for regla in reglas:
            f = comp[(comp.K == 'K3') & (comp.Senal == SENAL[sen])
                     & (comp.Caso == CODIGO[regla])].iloc[0]
            vals.append(100*((f.Costo_USD/f.Mejor_factible_USD-1)
                             + (f.Costo_USD/f.Cota_inferior_USD-1))/2)
        ax.bar(x + (j-0.5)*w, vals, width=w-0.04,
               color=[REGLA[r] for r in reglas], alpha=alpha, lw=0)
        for xi, v in zip(x + (j-0.5)*w, vals):
            ax.text(xi, v+0.28, f'{v:.1f}', ha='center', fontsize=7.5, color=TXT2)
    ax.set_xticks(x)
    ax.set_xticklabels(reglas, fontsize=8)
    ax.set_ylabel('Cost above matched optimum (%)', fontsize=7.8, labelpad=3)
    ax.set_ylim(0, 16); ax.set_yticks([0, 5, 10, 15]); ax.tick_params(axis='y', labelsize=7.8)
    ejes_limpios(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRIS, label='Flat price'),
                       Patch(color=GRIS, alpha=0.50, label='Indexed price')],
              loc='upper left', fontsize=7, handlelength=1.2, labelspacing=0.3)
    # (3) esta sensibilidad es SC con señal FLAT, no el conjunto del factorial.
    ax = fig.add_subplot(gs[2])
    ax.set_title('3  Gas-price sensitivity', loc='left', fontsize=8.8, fontweight='bold', pad=11)
    ax.text(0, 1.02, 'Self-consumption · flat price', transform=ax.transAxes,
            fontsize=7.5, color=TXT2, va='bottom')
    d = r2[(r2.Senal == 'FLAT') & (r2.Caso == 'SC')].sort_values('Gas_USD_MMBtu')
    mid = (d.Dist_pct_min+d.Dist_pct_max)/2
    ax.plot(d.Gas_USD_MMBtu, mid, color=AZUL, lw=1.5, marker='o', ms=4,
            mec='white', mew=0.8)
    ax.set_xticks([12, 16, 20, 24]); ax.tick_params(labelsize=7.8)
    ax.set_xlabel('Gas price (USD/MMBtu)', fontsize=7.8, labelpad=3)
    ax.set_ylabel('Cost above matched optimum (%)', fontsize=7.8, labelpad=3)
    ax.set_ylim(0, 45); ax.set_yticks([0, 15, 30, 45]); ejes_limpios(ax)
    fin = float(mid.iloc[-1])
    ax.annotate('Gas units off\ngap < 2%', xy=(24, fin), xytext=(16, 21),
                fontsize=7.5, color=TXT2,
                arrowprops=dict(arrowstyle='-', color=TXT2, lw=0.6))
    fig.text(0.50, 0.225,
             'SC: self-consumption · DM: daily merit order · DMX: DM + battery export',
             ha='center', fontsize=7.1, color=TXT2)
    # Mensajes con alcance explícito; las capturas se calculan de las bandas financieras.
    capturas = []
    for sen in ('T0', 'T1'):
        vr = vpn[(vpn.Senal == sen) & (vpn.Caso == 'A_INT')
                 & (vpn.Referencia == 'PV_SIN_BESS')].iloc[0].VPN_COP
        vb = bandas[(bandas.Senal == sen) & (bandas.Benchmark == 'OPT_AB')
                    & (bandas.Referencia == 'PV_SIN_BESS')].iloc[0]
        capturas.extend([100*vr/vb.VPN_cota_superior_COP, 100*vr/vb.VPN_factible_COP])
    captura_min = np.floor(min(capturas)*10)/10
    captura_max = np.ceil(max(capturas)*10)/10
    rh_max = r1[r1.Caso == 'RH-48'].Dist_pct_max.max()
    # Redondeo hacia fuera a una décima: no convertir un límite de 0.46% en <0.4%.
    rh_techo = np.ceil(rh_max*10)/10
    mensajes = [('Export + grid charging', '< 1% lower optimal annual cost', NARANJA),
                ('48-h perfect foresight', f'< {rh_techo:.1f}% above annual optimum', AZUL),
                ('Self-consumption',
                 f'{captura_min:.1f}–{captura_max:.1f}% of max. battery NPV*', AQUA)]
    for xc, (titulo, dato, color) in zip((0.16, 0.50, 0.84), mensajes):
        fig.text(xc, 0.15, titulo, ha='center', va='top', fontsize=8, fontweight='bold', color=color)
        fig.text(xc, 0.105, dato, ha='center', va='top', fontsize=7.5)
    fig.text(0.025, 0.025,
             '*Incremental battery net present value over 20 years; maximum bounded by the matched benchmark.\nMILP: mixed-integer linear programming. Bars/points: solver-bound midpoints.',
             fontsize=6.7, color=TXT2, va='bottom')
    guardar(fig, 'graphical_abstract')


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8, 9: fig9, 10: fig_ga}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--solo', nargs='*', type=int)
    a = ap.parse_args()
    for n in (a.solo or FIGS):
        FIGS[n]()
