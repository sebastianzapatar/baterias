"""Regenera seis figuras de 0A/0B a partir de CSV vigentes, sin optimizar.

Revisión visual del 13/09/2026, tras la corrección N3:
- 11_1 y 11_5 indican caso, gas, tarifa y semana, y llevan la leyenda fuera del área;
- 11_3 marca la punta N3 [06:00, 21:00) y separa con etiquetas directas los costos
  variables de G1+G2 y G4, que difieren menos de 1 USD/MWh y se superponían;
- 11_6 usa coma decimal y no tiene leyenda de una sola serie.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SAL = Path('resultados_nb11'); FIG = Path('figuras_nb11'); FIG.mkdir(exist_ok=True)
colores = ['#2a78d6', '#a03060', '#0d5f4f', '#8b8a83']
nombres = ['G1+G2', 'G3', 'G4', 'Red']
SEMANA = slice(24*65, 24*72)          # 07/03/2025 00:00 a 13/03/2025 23:00
PUNTA = (6, 21)                        # punta N3: hora inicial local en [6, 21)


def es(x, dec=1):
    return f"{x:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def bloques(d):
    return np.c_[d.G1 + d.G2, d.G3, d.G4, d.Red]


def guardar(fig, nombre):
    fig.savefig(FIG/(nombre + '.png'), dpi=150, bbox_inches='tight'); plt.close(fig)


def sombrear_punta(ax, fechas):
    for dia in pd.DatetimeIndex(fechas).normalize().unique():
        ax.axvspan(dia + pd.Timedelta(hours=PUNTA[0]), dia + pd.Timedelta(hours=PUNTA[1]),
                   color='#eda100', alpha=0.10, lw=0)


d = {s: pd.read_csv(SAL/f'caso0_despacho_{s}.csv', parse_dates=['Fecha']) for s in ('T0', 'T1')}

fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, sharey=True, layout='constrained')
for a, s in zip(ax, ('T0', 'T1')):
    q = d[s].iloc[SEMANA]
    a.stackplot(q.Fecha, bloques(q).T, labels=nombres, colors=colores)
    a.plot(q.Fecha, q.Demanda_MW, color='black', ls='--', lw=1, label='Demanda')
    a.set(title=s, ylabel='MW', ylim=(0, 25))
a.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
fig.suptitle('Caso 0A · mérito sin mínimos · gas 16 USD/MMBtu · HR históricos · tarifa N3 corregida · 07–13/03/2025',
             fontsize=10)
fig.legend(*ax[0].get_legend_handles_labels(), ncol=5, fontsize=8, loc='outside lower center', frameon=False)
guardar(fig, '11_1_semana_representativa')

fig, ax = plt.subplots(figsize=(8, 4)); bottom = np.zeros(2)
for j, n in enumerate(nombres):
    values = np.array([bloques(d[s])[:, j].sum()/1000 for s in ('T0', 'T1')])
    ax.bar(['T0', 'T1'], values, bottom=bottom, color=colores[j], label=n); bottom += values
ax.set(ylabel='Energía anual [GWh]', title='Caso 0A · gas 16 USD/MMBtu · tarifa N3 corregida',
       ylim=(0, float(bottom.max())*1.22))
ax.legend(ncol=4)
guardar(fig, '11_2_energia_por_unidad')

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, layout='constrained')
cv = [(137.28, 'G1+G2', colores[0], 4), (138.2016, 'G4', colores[2], -10), (169.28, 'G3', colores[1], 4)]
for a, s in zip(ax, ('T0', 'T1')):
    f = pd.read_csv(SAL/f'fabricante_0B_gas16_{s}.csv', parse_dates=['Fecha']).iloc[SEMANA]
    sombrear_punta(a, f.Fecha)
    a.plot(f.Fecha, f.Precio_red_USD_MWh, color=colores[-1], lw=1.6, label='Red')
    for valor, n, c, dy in cv:
        a.axhline(valor, color=c, ls='--', lw=1.2)
        a.annotate(f'{n} {es(valor, 1)}', (f.Fecha.iloc[-1], valor), xytext=(4, dy), textcoords='offset points',
                   color=c, fontsize=8, va='center', annotation_clip=False)
    a.set(title=s, ylabel='USD/MWh')
    a.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax[0].annotate('franja sombreada: punta N3 06:00–21:00', (0.01, 0.97), xycoords='axes fraction', fontsize=8,
               va='top', color='#52514e')
fig.suptitle('Precio de red frente al costo variable térmico (HR históricos, gas 16 USD/MMBtu) · 07–13/03/2025',
             fontsize=10)
fig.legend(handles=[plt.Line2D([], [], color=colores[-1], lw=1.6, label='Red'),
                    plt.Line2D([], [], color='#52514e', ls='--', label='Costo variable por unidad (etiqueta a la derecha)')],
           ncol=2, fontsize=8, loc='outside lower center', frameon=False)
guardar(fig, '11_3_costos_red_vs_termico')

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout='constrained')
for a, s in zip(ax, ('T0', 'T1')):
    for values, n, c in [(d[s].Demanda_MW, 'Demanda', 'black'), (d[s][['G1', 'G2', 'G3', 'G4']].sum(axis=1), 'Térmica', colores[0]),
                         (d[s].Red, 'Red', colores[-1])]:
        a.plot(np.sort(values)[::-1], label=n, color=c)
    a.set(title=f'{s} · caso 0A', xlabel='Horas ordenadas por serie', ylabel='MW'); a.legend()
guardar(fig, '11_4_curva_duracion')

b = pd.read_csv(SAL/'caso0B_con_minimos_gas16_T1.csv', parse_dates=['Fecha'])
metric = (np.abs(b.G3 - d['T1'].G3) + np.abs(b.G4 - d['T1'].G4)).rolling(48).sum()
end = int(metric.idxmax()) + 1; start = max(0, end - 48)
fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, sharey=True, layout='constrained')
for a, df, n in zip(ax, (d['T1'], b), ('0A sin mínimos', '0B con mínimos')):
    q = df.iloc[start:end]
    a.stackplot(q.Fecha, bloques(q).T, labels=nombres, colors=colores)
    a.set(title='T1 · ' + n, ylabel='MW')
fig.suptitle('Dos días con la mayor diferencia de G3 y G4 entre 0A y 0B · T1 · gas 16 USD/MMBtu · tarifa N3 corregida',
             fontsize=10)
fig.legend(*ax[0].get_legend_handles_labels(), ncol=4, fontsize=8, loc='outside lower center', frameon=False)
guardar(fig, '11_5_caso0A_vs_caso0B')

sens = pd.read_csv(SAL/'sensibilidad_gas_resumen.csv')
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout='constrained')
for a, s in zip(ax, ('T0', 'T1')):
    q = sens[sens.Senal == s].sort_values('Gas_USD_MMBtu')
    a.scatter(q.Gas_USD_MMBtu, q.Costo_medio_USD_MWh, color=colores[0])
    for _, r in q.iterrows():
        a.annotate(f'red {es(r.Participacion_red_pct, 1)} %', (r.Gas_USD_MMBtu, r.Costo_medio_USD_MWh), xytext=(0, 8),
                   textcoords='offset points', ha='center', fontsize=8)
    a.set(title=s, xlabel='Gas [USD/MMBtu]', ylabel='Costo medio de abastecimiento [USD/MWh]'); a.margins(y=.2, x=.2)
fig.suptitle('Caso 0B con mínimos · HR históricos · tarifa N3 corregida · etiqueta: participación de la red', fontsize=10)
guardar(fig, '11_6_sensibilidad_gas')
print('Seis figuras regeneradas desde CSV corregidos')
