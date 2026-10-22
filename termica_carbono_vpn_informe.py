"""Informe de la evaluación multianual V2 (`termica_carbono_vpn.py --etapa financiero`).

Solo lee las anualidades guardadas; no resuelve despachos. Escribe en la carpeta de la corrida:
`anualidades.csv`, `vpn.csv`, `flujos_anuales.csv`, `renovaciones.csv`, `vpn_omitidos.csv`,
`verificaciones.json`, `completitud_financiero.*`, `figuras/*.png`, `manifiesto.json` y `ESTADO.txt`.
Si falta una anualidad requerida, el VPN de esa comparación no se calcula y la entrega queda INCOMPLETO.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import termica_carbono_anual as TA
import termica_carbono_vpn as V

COLOR = {'A_INT': '#2a78d6', 'B_INT': '#eb6834', 'D_INT': '#1baf7a'}
CARBONO = {'V2_p0': '0 USD/t', 'V2_p50': '50 USD/t'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def anualidades(salida):
    filas = []
    for m in sorted(Path(salida).glob('*/*_a[0-9][0-9]_meta.json')):
        d = json.loads(m.read_text(encoding='utf-8'))
        d.pop('clave', None)
        d.pop('firma', None)
        filas.append(d)
    return pd.DataFrame(filas)


def verificaciones(salida, anu):
    """Año 1 frente al piloto; año 2 con PV reducido; reutilización de la referencia; cierre y disponibilidad."""
    salida = Path(salida)
    ver = {}
    difs = []
    for n in V.ESCENARIOS_VPN:
        for s in TA.SENALES:
            for c in V.CASOS_RESUELTOS + ('RED_TERMICA',):
                with np.load(salida/n/f'{V.etiqueta(s, c, 1)}.npz') as a, np.load(V.PILOTO/n/f'{s}_{c}.npz') as b:
                    difs.append(max(float(np.abs(a[k] - b[k]).max()) for k in
                                    ('potencia', 'compromiso', 'SOC', 'Descarga_demanda', 'Export_PV', 'PV', 'Objetivo')))
    ver['anio1_vs_piloto'] = dict(casos=len(difs), diferencia_max=max(difs), reproduce=max(difs) == 0.)
    ok = anu[anu.Estado == 'OK']
    r2 = []
    for (n, s, c), g in ok[ok.Caso.isin(V.CASOS_RESUELTOS)].groupby(['Escenario_piloto', 'Senal', 'Caso']):
        g = g.set_index('Anio')
        r2.append(dict(caso=f'{n}/{s}/{c}', razon_pv=g.loc[2, 'PV_MWh']/g.loc[1, 'PV_MWh'],
                       costo_distinto=bool(g.loc[2, 'Costo_financiero_USD'] != g.loc[1, 'Costo_financiero_USD'])))
    r2 = pd.DataFrame(r2)
    ver['anio2_pv_reducido'] = dict(casos=len(r2), razon_pv_min=float(r2.razon_pv.min()), razon_pv_max=float(r2.razon_pv.max()),
                                    razon_esperada=1 - V.DEG, costos_distintos=int(r2.costo_distinto.sum()))
    assert abs(r2.razon_pv - (1 - V.DEG)).max() < 1e-12, 'el PV del año 2 no refleja la degradación'
    red = anu[anu.Caso == 'RED_TERMICA']
    ver['referencia_reutilizada'] = dict(anualidades=len(red), resueltas=int((red.Origen == 'resuelta').sum()),
                                         reutilizadas=int((red.Origen == 'reutilizada').sum()),
                                         condicion='huella de entradas idéntica al año 1 (comprobada al registrar)')
    pol = ok[ok.Caso.isin(TA.POLITICAS)]
    ver['cierre_SOC_max_MWh'] = float(pol.Cierre_SOC_MWh.abs().max())
    ver['no_convergencias'] = int((anu.Estado != 'OK').sum())
    ver['disponibilidad'] = 'validada en cada anualidad resuelta (sin compromiso en horas indisponibles)'
    return ver


def _es(v, dec=None):
    """Número con coma decimal y punto de miles (convención del proyecto)."""
    s = f'{v:,.{dec}f}' if dec is not None else f'{v:,.2f}'.rstrip('0').rstrip('.')
    return s.replace(',', '§').replace('.', ',').replace('§', '.')


def _eje_es(ax):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _es(v)))
    ax.set_xticks([1, 5, 10, 15, 20])


def _etiquetas_directas(ax, puntos, x, separacion=0.045, fusion=0.012):
    """Etiquetas al final de cada línea. Las líneas del mismo precio que coinciden a la vista (a menos de `fusion`
    del rango del eje) comparten una etiqueta; las demás se separan `separacion` verticalmente sin cruzarse."""
    ax.relim()
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    d, f = separacion*(y1 - y0), fusion*(y1 - y0)
    grupos = []
    for y, regla, precio, color in sorted(puntos):
        g = next((g for g in grupos if g['precio'] == precio and abs(g['y'][-1] - y) < f), None)
        if g is None:
            grupos.append(dict(precio=precio, y=[y], reglas=[regla], colores=[color]))
        else:
            g['y'].append(y), g['reglas'].append(regla), g['colores'].append(color)
    grupos.sort(key=lambda g: np.mean(g['y']))
    pos = []
    for g in grupos:
        p = np.mean(g['y'])
        pos.append(max(p, pos[-1] + d) if pos else p)
    exceso = pos[-1] - (y1 - 0.2*d) if pos else 0
    if exceso > 0:   # si la pila se sale por arriba, se desplaza hacia abajo conservando la separación
        pos = [p - exceso for p in pos]
    for g, p in zip(grupos, pos):
        color = g['colores'][0] if len(set(g['colores'])) == 1 else '#333333'
        ax.annotate(f"{', '.join(sorted(g['reglas']))} · {g['precio']}", (x, np.mean(g['y'])), xytext=(x + 0.4, p),
                    textcoords='data', fontsize=6.5, color=color, va='center',
                    arrowprops=dict(arrowstyle='-', color='#bbbbbb', lw=0.5, shrinkA=0, shrinkB=1))
    ax.set_ylim(y0, y1)


def figuras(salida, vpn, anu, flujos):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#deded9', 'grid.linewidth': 0.6, 'savefig.facecolor': 'white'})
    carpeta = Path(salida)/'figuras'
    carpeta.mkdir(exist_ok=True)
    reglas = list(TA.POLITICAS)
    corto = {r: r.split('_')[0] for r in reglas}
    # 1. VPN financiero y ajustado
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharey='row')
    for i, (comp, ref, titulo) in enumerate((('completo', 'RED_TERMICA', 'Proyecto completo (frente a red + térmica)'),
                                             ('incremental', 'PV_SIN_BESS', 'Incremental del BESS (frente a PV sin BESS)'))):
        for j, s in enumerate(TA.SENALES):
            ax = axs[i, j]
            x = 0
            ticks, etiquetas = [], []
            for n in V.ESCENARIOS_VPN:
                for r in reglas:
                    f = vpn[(vpn.Escenario_piloto == n) & (vpn.Senal == s) & (vpn.Proyecto == r) & (vpn.Referencia == ref)]
                    if f.empty:
                        continue
                    vf, va = f.VPN_financiero_MUSD.iloc[0], f.VPN_ajustado_MUSD.iloc[0]
                    ax.bar(x - 0.2, vf, 0.38, color=COLOR[r])
                    ax.bar(x + 0.2, va, 0.38, color=COLOR[r], alpha=0.45, hatch='//', edgecolor=COLOR[r])
                    ax.text(x - 0.2, vf, _es(vf, 1), ha='center', va='bottom', fontsize=6.3)
                    ax.text(x + 0.2, va, _es(va, 1), ha='center', va='bottom', fontsize=6.3)
                    ticks.append(x)
                    etiquetas.append(f'{r.split("_")[0]}\n{CARBONO[n]}')
                    x += 1
                x += 0.6
            ax.set_xticks(ticks, etiquetas, fontsize=8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _es(v)))
            ax.axhline(0, color='#666666', lw=0.8)
            ax.set_title(f'{titulo} — {s}', fontsize=9)
            if j == 0:
                ax.set_ylabel('VPN [MUSD]')
    fig.legend(handles=[Patch(color='#555555', label='VPN financiero'),
                        Patch(facecolor='#bbbbbb', hatch='//', edgecolor='#555555', label='VPN ajustado por carbono')]
               + [Patch(color=COLOR[r], label=r) for r in reglas], loc='lower center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(carpeta/'01_vpn_financiero_ajustado.png', dpi=150)
    plt.close(fig)
    # 2. Evolución anual de térmica, compras y emisiones
    ok = anu[(anu.Estado == 'OK') & anu.Caso.isin(reglas)]
    metricas = (('Generacion_termica_MWh', 'Generación térmica [GWh]', 1e3), ('Importacion_MWh', 'Compras de red [GWh]', 1e3),
                ('Emisiones_operativas_t', 'CO₂ térmico + CO₂e red [kt]', 1e3))
    fig, axs = plt.subplots(3, 2, figsize=(10, 8.2), sharex=True)
    for i, (col, nombre, esc) in enumerate(metricas):
        for j, s in enumerate(TA.SENALES):
            ax = axs[i, j]
            puntos = []
            for r in reglas:
                for n, ls in (('V2_p0', '-'), ('V2_p50', '--')):
                    g = ok[(ok.Caso == r) & (ok.Senal == s) & (ok.Escenario_piloto == n)].sort_values('Anio')
                    ax.plot(g.Anio, g[col]/esc, color=COLOR[r], ls=ls, lw=2)
                    puntos.append((g[col].iloc[-1]/esc, corto[r], CARBONO[n], COLOR[r]))
            ax.set_title(f'{nombre} — {s}', fontsize=9)
            ax.set_xlim(0.5, 26.5)
            _eje_es(ax)
            _etiquetas_directas(ax, puntos, 20)
            if i == 2:
                ax.set_xlabel('Año')
    fig.legend(handles=[plt.Line2D([], [], color='#555555', lw=2, ls='-', label='Carbono 0 USD/t'),
                        plt.Line2D([], [], color='#555555', lw=2, ls='--', label='Carbono 50 USD/t')]
               + [plt.Line2D([], [], color=COLOR[r], lw=2, label=r) for r in reglas], loc='lower center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(carpeta/'02_evolucion_termica_compras_emisiones.png', dpi=150)
    plt.close(fig)
    # 3. Flujos anuales y reposiciones del BESS (proyecto completo)
    fc = flujos[(flujos.Referencia == 'RED_TERMICA') & flujos.Proyecto.isin(reglas)]
    # Fila superior: flujo financiero (sin valor del carbono; el ajuste por carbono está en la figura 1).
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True)
    for j, s in enumerate(TA.SENALES):
        arriba, abajo = [], []
        for r in reglas:
            for n, ls in (('V2_p0', '-'), ('V2_p50', '--')):
                g = fc[(fc.Proyecto == r) & (fc.Senal == s) & (fc.Escenario_piloto == n)].sort_values('Anio')
                axs[0, j].plot(g.Anio, g.Flujo_financiero_COP/1e9, color=COLOR[r], ls=ls, lw=2)
                axs[1, j].plot(g.Anio, g.EFC_acumulado, color=COLOR[r], ls=ls, lw=2)
                ren = g[g.Renovaciones_en_anio > 0]
                axs[0, j].scatter(ren.Anio, ren.Flujo_financiero_COP/1e9, marker='v', s=40, color=COLOR[r], zorder=3)
                arriba.append((g.Flujo_financiero_COP.iloc[-1]/1e9, corto[r], CARBONO[n], COLOR[r]))
                abajo.append((g.EFC_acumulado.iloc[-1], corto[r], CARBONO[n], COLOR[r]))
        axs[0, j].set_title(f'Flujo financiero anual del proyecto completo — {s}', fontsize=9)
        axs[0, j].set_ylabel('Flujo [mil millones COP]')
        axs[1, j].axhline(7200, color='#8b8a83', lw=1, ls=':')
        axs[1, j].annotate('vida por ciclos: 7.200', (1, 7200), xytext=(0, -10), textcoords='offset points', fontsize=7, color='#666666')
        axs[1, j].set_title(f'Ciclos equivalentes acumulados del BESS — {s}', fontsize=9)
        axs[1, j].set_ylabel('EFC acumulados')
        axs[1, j].set_xlabel('Año')
        n_ren = int(fc[fc.Senal == s].Renovaciones_en_anio.sum())
        axs[0, j].annotate(f'reposiciones del BESS en 20 años: {n_ren}', (0.02, 0.04), xycoords='axes fraction', fontsize=7.5)
        for ax, pts in ((axs[0, j], arriba), (axs[1, j], abajo)):
            ax.set_xlim(0.5, 26.5)
            _eje_es(ax)
            _etiquetas_directas(ax, pts, 20)
    marcas = [plt.Line2D([], [], color='#555555', lw=2, ls='-', label='Carbono 0 USD/t'),
              plt.Line2D([], [], color='#555555', lw=2, ls='--', label='Carbono 50 USD/t')]
    if fc.Renovaciones_en_anio.sum() > 0:
        marcas.append(plt.Line2D([], [], color='#555555', marker='v', ls='', label='Reposición del BESS'))
    fig.legend(handles=marcas + [plt.Line2D([], [], color=COLOR[r], lw=2, label=r) for r in reglas],
               loc='lower center', ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(carpeta/'03_flujos_y_reposiciones.png', dpi=150)
    plt.close(fig)
    return sorted(str(p) for p in carpeta.glob('*.png'))


def informe(salida):
    salida = Path(salida)
    vpn, flujos, omitidos, comp = V.financiero(salida)
    completo = V.informar(salida, comp, 'financiero') and omitidos.empty
    estado = 'COMPLETO' if completo else 'INCOMPLETO'
    anu = anualidades(salida)
    anu.assign(Estado_entrega=estado).to_csv(salida/'anualidades.csv', index=False)
    vpn.assign(Estado_entrega=estado).to_csv(salida/'vpn.csv', index=False)
    flujos.assign(Estado_entrega=estado).to_csv(salida/'flujos_anuales.csv', index=False)
    omitidos.to_csv(salida/'vpn_omitidos.csv', index=False)
    if len(flujos):
        flujos[flujos.Renovaciones_en_anio > 0].to_csv(salida/'renovaciones.csv', index=False)
    ver = verificaciones(salida, anu) if not anu.empty else {}
    ver['estado_entrega'] = estado
    (salida/'verificaciones.json').write_text(json.dumps(ver, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    figs = figuras(salida, vpn, anu, flujos) if completo else []
    (salida/'ESTADO.txt').write_text(estado + chr(10), encoding='utf-8')
    archivos = sorted(p for p in salida.rglob('*') if p.is_file() and p.name != 'manifiesto.json')
    manifiesto = dict(generado=pd.Timestamp.now().isoformat(), estado_entrega=estado,
                      resultados={str(p.relative_to(salida)): sha(p) for p in archivos},
                      codigo={f: sha(f) for f in V.CODIGO + ('termica_carbono_vpn_informe.py', 'carbono_completitud_pruebas.py',
                                                             'carbono_pruebas.py')},
                      entradas={f: sha(f) for f in ('datos/load variable.csv', '02_Optimizacion_portafolio_PV_BESS.ipynb')},
                      comandos=['python termica_carbono_vpn.py --salida <carpeta> --etapa anio1 --procesos 8',
                                'python termica_carbono_vpn.py --salida <carpeta> --etapa anual --procesos 9',
                                'python termica_carbono_vpn.py --salida <carpeta> --etapa financiero',
                                'python termica_carbono_vpn.py --salida <carpeta> --etapa verificar'],
                      figuras=figs)
    (salida/'manifiesto.json').write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'informe: {estado}; {len(vpn)} VPN, {len(omitidos)} omitidos', flush=True)
    return 0 if completo else 1
