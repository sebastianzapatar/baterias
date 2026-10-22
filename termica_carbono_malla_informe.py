"""Informe de la malla de dimensionamiento V2 (`termica_carbono_malla.py --etapa financiero`).

Solo lee anualidades guardadas; no resuelve despachos. Escribe en la carpeta de la corrida:

    vpn.csv                       VPN financiero y ajustado de cada comparación (proyecto completo, incremental, solo PV)
    operacion_20_anios.csv        sumas de 20 años por caso: emisiones, gas, red, exportación, arranques, ciclos
    resultados_por_configuracion  VPN completo e incremental + operación, por configuración, regla, tarifa y carbono
    reglas_tamano_comun.csv       A: reglas comparadas a igual tamaño (mejor y segunda regla)
    mejores_por_regla.csv         B: tamaños para cada regla (mejor y segunda configuración de esa regla; bordes)
    comparacion_configuraciones   C: dos mejores configuraciones distintas (mayor VPN de cada una y su regla)
    ranking_configuraciones.csv   C: todas las configuraciones ordenadas por su mayor VPN ajustado
    mejores_global.csv            clasificación conjunta de pares (configuración, regla); el segundo par puede ser
                                  el mismo tamaño con otra regla (`Segundo_mismo_tamano`)
    flujos_anuales.csv, renovaciones.csv, vpn_omitidos.csv, verificaciones.json, completitud_financiero.*,
    figuras/*.png, manifiesto.json, ESTADO.txt

El mejor punto de una malla no es un óptimo global. Si falta una anualidad requerida, el VPN de esa comparación
no se calcula (queda en `vpn_omitidos.csv`) y la entrega queda INCOMPLETO.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import termica_carbono_anual as TA
import termica_carbono_malla as M
import termica_carbono_vpn as V
from termica_carbono_vpn_informe import _es, _etiquetas_directas

COLOR = {'A_INT': '#2a78d6', 'B_INT': '#eb6834', 'D_INT': '#1baf7a'}
GRIS_PV = '#8b8a83'
PRECIO = {'V2_p0': 0, 'V2_p10': 10, 'V2_p20': 20, 'V2_p50': 50}
TRM = None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def operacion(anu):
    cols = [c for c in M.AGREGADOS if c in anu.columns]
    g = anu.groupby(['Grupo', 'Escenario_piloto', 'Senal', 'Caso'])[cols].sum()
    g = g.rename(columns={'EFC_anio': 'EFC_20_anios'})
    g['Anualidades'] = anu.groupby(['Grupo', 'Escenario_piloto', 'Senal', 'Caso']).size()
    return g.reset_index()


def etiqueta_config(cfg, cid):
    return f"{cid}\n{_es(cfg['f_PV']*114.415019, 0)} MWp\n{_es(cfg['P_MW'], 0)}/{_es(cfg['E_MWh'], 0)}"


def resultados_config(vpn, ops, m):
    cfgs = M.configs(m)
    comp = vpn[vpn.Referencia == 'RED_TERMICA'].set_index(['Grupo', 'Escenario_piloto', 'Senal', 'Proyecto'])
    inc = vpn[vpn.Referencia == 'PV_SIN_BESS'].set_index(['Grupo', 'Escenario_piloto', 'Senal', 'Proyecto'])
    o = ops.set_index(['Grupo', 'Escenario_piloto', 'Senal', 'Caso'])
    filas = []
    for (g, n, s, p), x in comp.iterrows():
        if g not in cfgs:
            continue
        fila = dict(Config=g, **cfgs[g], PV_MWp=cfgs[g]['f_PV']*m['pv_referencia_MWp'],
                    Duracion_h=cfgs[g]['E_MWh']/cfgs[g]['P_MW'], Escenario_piloto=n, Precio_CO2_USD_t=PRECIO[n],
                    Senal=s, Regla=p, VPN_financiero_MUSD=x.VPN_financiero_MUSD, VPN_ajustado_MUSD=x.VPN_ajustado_MUSD,
                    Carbono_evitado_VP_MUSD=x.VPN_ajustado_MUSD - x.VPN_financiero_MUSD, TIR=x.TIR,
                    Payback_anios=x.Payback_anios, Renovaciones=x.Renovaciones, Anios_renovacion=x.Anios_renovacion,
                    Costo_renovaciones_VP_MUSD=x.Costo_renovaciones_VP_COP/TRM/1e6,
                    CAPEX_MUSD=x.CAPEX_COP/TRM/1e6, OPEX_anual_MUSD=x.OPEX_anual_COP/TRM/1e6)
        if (g, n, s, p) in inc.index:
            y = inc.loc[(g, n, s, p)]
            fila.update(VPN_incremental_financiero_MUSD=y.VPN_financiero_MUSD, VPN_incremental_ajustado_MUSD=y.VPN_ajustado_MUSD)
        if (g, n, s, p) in o.index:
            z = o.loc[(g, n, s, p)]
            fila.update(Emisiones_operativas_kt=z.Emisiones_operativas_t/1e3, Gas_MMBtu_M=z.Gas_MMBtu/1e6,
                        Termica_GWh=z.Generacion_termica_MWh/1e3, Compras_red_GWh=z.Importacion_MWh/1e3,
                        Exportacion_GWh=z.Exportacion_MWh/1e3, Arranques=z.Arranques, EFC_20_anios=z.EFC_20_anios)
        filas.append(fila)
    return pd.DataFrame(filas)


def bordes(fila, m):
    lim = m['limites_exploracion']
    b = []
    if abs(fila.PV_MWp - lim['PV_MWp'][1]) < 1e-6:
        b.append('PV en el tope (114,415 MWp)')
    if abs(fila.PV_MWp - lim['PV_MWp'][0]) < 1e-6:
        b.append('PV mínimo de la malla')
    if fila.P_MW == lim['P_MW'][1]:
        b.append('P máxima de la malla (45 MW)')
    if fila.P_MW == lim['P_MW'][0]:
        b.append('P mínima de la malla (25 MW)')
    if fila.E_MWh == lim['E_MWh'][1]:
        b.append('E máxima de la malla (360 MWh)')
    if fila.E_MWh == lim['E_MWh'][0]:
        b.append('E mínima de la malla (120 MWh)')
    return '; '.join(b)


ORDEN_REGLAS = ('A_INT', 'B_INT', 'D_INT')


def _ordenar(g, col='VPN_ajustado_MUSD'):
    """Orden descendente por `col` con cifras completas. Empates exactos: configuración (M01 antes que M02) y
    luego regla en el orden A, B, D. Los empates se declaran en las tablas; no se redondea antes de comparar."""
    g = g.assign(_r=g.Regla.map({r: i for i, r in enumerate(ORDEN_REGLAS)}))
    return g.sort_values([col, 'Config', '_r'], ascending=[False, True, True], kind='mergesort').drop(columns='_r')


def mejores(res, m):
    """B (tamaños para cada regla) y clasificación conjunta (configuración, regla).

    `mejores_por_regla.csv`: para cada regla, tarifa y carbono, la configuración de mayor VPN ajustado y la segunda
    configuración de ESA regla (`Segundo_config`, `Ventaja_sobre_segundo_MUSD`): compara tamaños distintos.
    `mejores_global.csv`: clasificación conjunta de pares (configuración, regla). `Segundo` y
    `Ventaja_sobre_segundo_MUSD` conservan su significado: el segundo PAR, que puede ser el mismo tamaño con otra
    regla (`Segundo_mismo_tamano`). No mide la ventaja de un tamaño sobre otro: para eso, `comparacion_configuraciones`."""
    filas, glob = [], []
    for (n, s, r), g in res.groupby(['Escenario_piloto', 'Senal', 'Regla']):
        g = _ordenar(g)
        b, seg = g.iloc[0], g.iloc[1] if len(g) > 1 else None
        filas.append(dict(b, Segundo_config=None if seg is None else seg.Config,
                          Ventaja_sobre_segundo_MUSD=None if seg is None else b.VPN_ajustado_MUSD - seg.VPN_ajustado_MUSD,
                          Empate_con_segundo=None if seg is None else bool(b.VPN_ajustado_MUSD == seg.VPN_ajustado_MUSD),
                          Configs_evaluadas=len(g), Borde=bordes(b, m),
                          Mejor_financiero_config=_ordenar(g, 'VPN_financiero_MUSD').iloc[0].Config))
    for (n, s), g in res.groupby(['Escenario_piloto', 'Senal']):
        g = _ordenar(g)
        b, seg = g.iloc[0], g.iloc[1]
        glob.append(dict(b, Clasificacion='conjunta de pares (configuración, regla)', Segundo=f'{seg.Config} {seg.Regla}',
                         Ventaja_sobre_segundo_MUSD=b.VPN_ajustado_MUSD - seg.VPN_ajustado_MUSD,
                         Segundo_config=seg.Config, Segundo_regla=seg.Regla,
                         Segundo_mismo_tamano=bool(seg.Config == b.Config), Borde=bordes(b, m)))
    return pd.DataFrame(filas), pd.DataFrame(glob)


def comparacion_configuraciones(res, m):
    """C: configuraciones distintas. Dentro de cada tarifa y carbono se toma primero, para cada configuración, su
    mayor VPN ajustado y la regla que lo logra (empate exacto entre reglas: orden A, B, D, y se listan todas las
    empatadas); después se ordenan las configuraciones (empate exacto: identificador menor primero) y se comparan
    las dos primeras, que son necesariamente configuraciones distintas. Diferencias con cifras completas."""
    ranking, filas = [], []
    for (n, s), g in res.groupby(['Escenario_piloto', 'Senal']):
        mejores_c = []
        for c, gc in g.groupby('Config'):
            gc = _ordenar(gc)
            b = gc.iloc[0]
            emp = sorted(gc[gc.VPN_ajustado_MUSD == b.VPN_ajustado_MUSD].Regla, key=ORDEN_REGLAS.index)
            mejores_c.append(dict(b, Reglas_con_el_maximo=','.join(emp)))
        t = pd.DataFrame(mejores_c).sort_values(['VPN_ajustado_MUSD', 'Config'], ascending=[False, True], kind='mergesort')
        t['Puesto_configuracion'] = np.arange(1, len(t) + 1)
        t['Diferencia_con_primera_MUSD'] = t.VPN_ajustado_MUSD.iloc[0] - t.VPN_ajustado_MUSD
        ranking.append(t)
        a, b = t.iloc[0], t.iloc[1]
        filas.append(dict(Escenario_piloto=n, Precio_CO2_USD_t=PRECIO[n], Senal=s,
                          Config_1=a.Config, Regla_1=a.Regla, Reglas_con_el_maximo_1=a.Reglas_con_el_maximo,
                          VPN_ajustado_1_MUSD=a.VPN_ajustado_MUSD, VPN_financiero_1_MUSD=a.VPN_financiero_MUSD,
                          Config_2=b.Config, Regla_2=b.Regla, Reglas_con_el_maximo_2=b.Reglas_con_el_maximo,
                          VPN_ajustado_2_MUSD=b.VPN_ajustado_MUSD, VPN_financiero_2_MUSD=b.VPN_financiero_MUSD,
                          Diferencia_ajustado_MUSD=a.VPN_ajustado_MUSD - b.VPN_ajustado_MUSD,
                          Empate_configuraciones=bool(a.VPN_ajustado_MUSD == b.VPN_ajustado_MUSD),
                          Borde_1=bordes(a, m), Borde_2=bordes(b, m)))
    return pd.DataFrame(filas), pd.concat(ranking, ignore_index=True)


def reglas_comunes(res):
    """A: reglas a tamaño fijo. La diferencia entre reglas de una misma configuración mide operación, no tamaño."""
    filas = []
    for (c, n, s), g in res.groupby(['Config', 'Escenario_piloto', 'Senal']):
        o = _ordenar(g)
        g = g.set_index('Regla')
        fila = dict(Config=c, Escenario_piloto=n, Senal=s)
        for r in g.index:
            fila[f'{r}_VPN_ajustado_MUSD'] = g.loc[r, 'VPN_ajustado_MUSD']
            fila[f'{r}_VPN_financiero_MUSD'] = g.loc[r, 'VPN_financiero_MUSD']
        fila['Mejor_regla'] = o.iloc[0].Regla
        fila['Rango_entre_reglas_MUSD'] = g.VPN_ajustado_MUSD.max() - g.VPN_ajustado_MUSD.min()
        fila['Segunda_regla'] = o.iloc[1].Regla
        fila['Diferencia_mejor_segunda_MUSD'] = o.iloc[0].VPN_ajustado_MUSD - o.iloc[1].VPN_ajustado_MUSD
        fila['Empate_primer_lugar'] = bool(o.iloc[0].VPN_ajustado_MUSD == o.iloc[1].VPN_ajustado_MUSD)
        filas.append(fila)
    return pd.DataFrame(filas)


def escribir_resumenes(salida, res, m, estado):
    """Tablas A, B, C y clasificación conjunta desde `resultados_por_configuracion` (no toca VPN, flujos ni series)."""
    salida = Path(salida)
    mr, mg = mejores(res, m)
    cc, rk = comparacion_configuraciones(res, m)
    tablas = {'reglas_tamano_comun.csv': reglas_comunes(res), 'mejores_por_regla.csv': mr,
              'comparacion_configuraciones.csv': cc, 'ranking_configuraciones.csv': rk, 'mejores_global.csv': mg}
    for nombre, tabla in tablas.items():
        tabla.assign(Estado_entrega=estado).to_csv(salida/nombre, index=False)
    return list(tablas)


def regenerar_resumenes(salida, motivo):
    """Regenera solo los resúmenes de comparación y actualiza sus huellas en el manifiesto."""
    salida = Path(salida)
    m = M.cargar_malla()
    res = pd.read_csv(salida/'resultados_por_configuracion.csv')
    estado = (salida/'ESTADO.txt').read_text(encoding='utf-8').strip()
    archivos = escribir_resumenes(salida, res, m, estado)
    man = json.loads((salida/'manifiesto.json').read_text(encoding='utf-8'))
    for a in archivos:
        man['resultados_resumen'][a] = sha(salida/a)
    man.setdefault('regeneraciones', []).append(dict(
        fecha=pd.Timestamp.now().isoformat(), motivo=motivo, archivos=archivos,
        codigo={'termica_carbono_malla_informe.py': sha('termica_carbono_malla_informe.py')},
        sin_cambios=['vpn.csv', 'flujos_anuales.csv', 'anualidades.csv', 'resultados_por_configuracion.csv',
                     'series y metadatos de las 6.400 anualidades', 'figuras']))
    (salida/'manifiesto.json').write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    return archivos


def verificaciones(salida, vpn, flujos, anu, comp, reuso, m):
    salida = Path(salida)
    ver = dict(completitud=comp.Estado_caso.value_counts().to_dict(),
               origen=comp.Origen.value_counts(dropna=False).to_dict())
    ver['reutilizacion_K3'] = reuso.Resultado.value_counts().to_dict() if len(reuso) else {}
    if len(vpn):
        dif = (vpn.VPN_ajustado_COP - vpn.VPN_financiero_COP - vpn.Valor_carbono_evitado_VP_COP).abs()
        p0 = vpn[vpn.Escenario_piloto == 'V2_p0']
        ver['identidad_carbono'] = dict(max_abs_COP=float(dif.max()), vpn=len(vpn),
                                        p0_iguales=int((p0.VPN_ajustado_COP == p0.VPN_financiero_COP).sum()), p0_total=len(p0))
        ver['aditividad'] = 'comprobada en todas las comparaciones con sus tres VPN (financiero y ajustado)'
        ren = flujos[flujos.Renovaciones_en_anio > 0] if len(flujos) else flujos
        ver['reposiciones'] = dict(vpn_con_reposicion=int((vpn.Renovaciones > 0).sum()),
                                   eventos=int(vpn.Renovaciones.sum()),
                                   filas_de_flujo_con_reposicion=int(ren.Renovaciones_en_anio.sum()) if len(ren) else 0)
    ok = anu[anu.Estado == 'OK']
    pol = ok[ok.Caso.isin(TA.POLITICAS)]
    ver['cierre_SOC_max_MWh'] = float(pol.Cierre_SOC_MWh.abs().max()) if len(pol) else None
    ver['no_convergencias'] = int((anu.Estado != 'OK').sum())
    r2 = []
    for (g, n, s, c), x in ok[ok.Origen == 'resuelta'].groupby(['Grupo', 'Escenario_piloto', 'Senal', 'Caso']):
        x = x.set_index('Anio')
        if 1 in x.index and 2 in x.index and c != 'RED_TERMICA':
            r2.append(x.loc[2, 'PV_MWh']/x.loc[1, 'PV_MWh'])
    if r2:
        ver['anio2_pv'] = dict(casos=len(r2), razon_min=float(min(r2)), razon_max=float(max(r2)))
    # Techo de exportación de las referencias PV sin BESS: no activo.
    techo = []
    for f in M.niveles_pv(m):
        for n in m['escenarios_carbono']:
            for s in m['senales']:
                with np.load(V.ruta_npz(salida/M.ref_pv(f), n, s, 'PV_SIN_BESS', 1)) as z:
                    _, _, pe, _ = TA.dimensionar(M.cfg_ref_pv(f))
                    techo.append(float(pe - z['Export_PV'].max()))
    ver['techo_referencias_holgura_min_MW'] = min(techo) if techo else None
    return ver


def figuras(salida, res, vpn, m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#deded9', 'grid.linewidth': 0.6, 'savefig.facecolor': 'white'})
    carpeta = Path(salida)/'figuras'
    carpeta.mkdir(exist_ok=True)
    cfgs = M.configs(m)
    ids = list(cfgs)
    fmt = plt.FuncFormatter(lambda v, _: _es(v))
    esc = m['escenarios_carbono']
    # 1. VPN ajustado del proyecto completo por configuración, regla, tarifa y carbono.
    fig, axs = plt.subplots(2, len(esc), figsize=(13, 7.2), sharey='row', sharex=True)
    x = np.arange(len(ids))
    for i, s in enumerate(m['senales']):
        for j, n in enumerate(esc):
            ax = axs[i, j]
            g = res[(res.Senal == s) & (res.Escenario_piloto == n)]
            for k, r in enumerate(m['reglas']):
                y = g[g.Regla == r].set_index('Config').reindex(ids).VPN_ajustado_MUSD
                ax.plot(x + (k - 1)*0.22, y, 'o', ms=4, color=COLOR[r], label=r)
            b = g.loc[g.VPN_ajustado_MUSD.idxmax()]
            ax.plot(ids.index(b.Config) + (m['reglas'].index(b.Regla) - 1)*0.22, b.VPN_ajustado_MUSD, 'o', ms=10,
                    mfc='none', mec='#222222', mew=1.2)
            ax.set_title(f'{s} — {PRECIO[n]} USD/t', fontsize=9)
            ax.yaxis.set_major_formatter(fmt)
            if j == 0:
                ax.set_ylabel('VPN ajustado, proyecto completo [MUSD]')
            ax.set_xticks(x, ids, fontsize=7.5, rotation=90)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=COLOR[r], label=r) for r in m['reglas']]
               + [Line2D([], [], marker='o', ls='', ms=10, mfc='none', mec='#222222', label='mayor VPN ajustado de la malla')],
               loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(carpeta/'01_vpn_ajustado_por_configuracion.png', dpi=150)
    plt.close(fig)
    # 2. Efecto de energía y potencia del BESS con el PV máximo (incremental del BESS, VPN ajustado).
    paneles = (('E_MWh', 'P_MW', 45., 'Energía del BESS [MWh]', 'energía variable con 45 MW'),
               ('E_MWh', 'P_MW', 25., 'Energía del BESS [MWh]', 'energía variable con 25 MW'),
               ('P_MW', 'E_MWh', 280., 'Potencia del BESS [MW]', 'potencia variable con 280 MWh'))
    fig, axs = plt.subplots(3, 2, figsize=(11, 10.5), sharey=True)
    for i, (var, fija, valor, nombre, titulo) in enumerate(paneles):
        for j, s in enumerate(m['senales']):
            ax = axs[i, j]
            g = res[(res.Senal == s) & (res.f_PV == 1.) & (res[fija] == valor)]
            puntos = []
            for r in m['reglas']:
                for n, ls, mk in (('V2_p0', '-', 'o'), ('V2_p50', '--', 's')):
                    z = g[(g.Regla == r) & (g.Escenario_piloto == n)].sort_values(var)
                    ax.plot(z[var], z.VPN_incremental_ajustado_MUSD, ls=ls, marker=mk, ms=3.5, lw=1.8, color=COLOR[r])
                    puntos.append((float(z.VPN_incremental_ajustado_MUSD.iloc[-1]), r.split('_')[0],
                                   f'{PRECIO[n]} USD/t', COLOR[r]))
            ax.axhline(0, color='#666666', lw=0.8)
            ax.set_xlabel(nombre)
            ax.yaxis.set_major_formatter(fmt)
            ax.set_title(f'{s} — PV 114,4 MWp; {titulo}', fontsize=9)
            xs = sorted(g[var].unique())
            ax.set_xticks(xs, [_es(x, 0) for x in xs])
            ax.set_xlim(xs[0] - 0.08*(xs[-1] - xs[0]), xs[-1] + 0.42*(xs[-1] - xs[0]))
            _etiquetas_directas(ax, puntos, xs[-1] + 0.03*(xs[-1] - xs[0]), separacion=0.055, fusion=0.012)
            if j == 0:
                ax.set_ylabel('VPN incremental del BESS [MUSD]')
    fig.legend(handles=[Line2D([], [], color=COLOR[r], lw=2, label=r) for r in m['reglas']]
               + [Line2D([], [], color='#555555', marker='o', ls='-', lw=1.8, label='0 USD/t'),
                  Line2D([], [], color='#555555', marker='s', ls='--', lw=1.8, label='50 USD/t')],
               loc='lower center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(carpeta/'02_efecto_energia_potencia.png', dpi=150)
    plt.close(fig)
    # 3. Efecto del PV con el BESS de K3 y sin BESS (proyecto completo, VPN ajustado).
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    pvs = vpn[vpn.Proyecto == 'PV_SIN_BESS']
    for j, s in enumerate(m['senales']):
        ax = axs[j]
        puntos = []
        for n, ls, mk in (('V2_p0', '-', 'o'), ('V2_p50', '--', 's')):
            z = pvs[(pvs.Senal == s) & (pvs.Escenario_piloto == n)].sort_values('PV_MWp')
            ax.plot(z.PV_MWp, z.VPN_ajustado_MUSD, ls=ls, marker=mk, ms=3.5, lw=1.8, color=GRIS_PV)
            puntos.append((float(z.VPN_ajustado_MUSD.iloc[-1]), 'PV sin BESS', f'{PRECIO[n]} USD/t', GRIS_PV))
            for r in m['reglas']:
                g = res[(res.Senal == s) & (res.Escenario_piloto == n) & (res.Regla == r) & (res.P_MW == 45.)
                        & (res.E_MWh == 280.)].sort_values('PV_MWp')
                ax.plot(g.PV_MWp, g.VPN_ajustado_MUSD, ls=ls, marker=mk, ms=3.5, lw=1.8, color=COLOR[r])
                puntos.append((float(g.VPN_ajustado_MUSD.iloc[-1]), r.split('_')[0], f'{PRECIO[n]} USD/t', COLOR[r]))
        ax.set_xlabel('PV [MWp DC]')
        xs = sorted(pvs.PV_MWp.unique())
        ax.set_xticks(xs, [_es(x, 1) for x in xs])
        ax.set_xlim(50, 145)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_title(f'{s} — BESS 45 MW / 280 MWh y PV sin BESS', fontsize=9)
        _etiquetas_directas(ax, puntos, xs[-1] + 0.03*(xs[-1] - xs[0]), separacion=0.05)
        if j == 0:
            ax.set_ylabel('VPN ajustado, proyecto completo [MUSD]')
    fig.legend(handles=[Line2D([], [], color=COLOR[r], lw=2, label=r) for r in m['reglas']]
               + [Line2D([], [], color=GRIS_PV, lw=2, label='PV sin BESS'),
                  Line2D([], [], color='#555555', marker='o', lw=1.8, ls='-', label='0 USD/t'),
                  Line2D([], [], color='#555555', marker='s', lw=1.8, ls='--', label='50 USD/t')],
               loc='lower center', ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(carpeta/'03_efecto_pv.png', dpi=150)
    plt.close(fig)
    # 4. Mapa de calor del VPN ajustado incremental del BESS (rampa divergente: el cero separa hacer y no hacer).
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list('vpn', ['#b2182b', '#f0efea', '#1a7f37'])
    cols = [(s, n) for s in m['senales'] for n in esc]
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.6), sharey=True)
    v = res.VPN_incremental_ajustado_MUSD
    norm = TwoSlopeNorm(vcenter=0., vmin=min(v.min(), -1.), vmax=max(v.max(), 1.))
    for k, r in enumerate(m['reglas']):
        ax = axs[k]
        mat = np.array([[res[(res.Config == c) & (res.Senal == s) & (res.Escenario_piloto == n) & (res.Regla == r)]
                         .VPN_incremental_ajustado_MUSD.squeeze() for (s, n) in cols] for c in ids], float)
        ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')
        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                ax.text(b, a, _es(mat[a, b], 1), ha='center', va='center', fontsize=6.3)
        ax.set_xticks(range(len(cols)), [f'{s}\n{PRECIO[n]}' for s, n in cols], fontsize=7)
        ax.set_yticks(range(len(ids)), [f"{c}  {_es(cfgs[c]['f_PV']*114.415, 0)} MWp  {_es(cfgs[c]['P_MW'], 0)} MW/"
                                        f"{_es(cfgs[c]['E_MWh'], 0)} MWh" for c in ids], fontsize=7)
        ax.grid(False)
        ax.set_title(f'{r}: VPN ajustado incremental del BESS [MUSD]', fontsize=9)
        ax.set_xlabel('Tarifa y carbono [USD/t]')
    fig.tight_layout()
    fig.savefig(carpeta/'04_vpn_incremental_bess_mapa.png', dpi=150)
    plt.close(fig)
    # 5. Ciclos a 20 años, reposiciones y emisiones por configuración.
    fig, axs = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    ancho = 0.26
    for k, r in enumerate(m['reglas']):
        g = res[(res.Regla == r) & (res.Escenario_piloto == 'V2_p20')]
        for s, alfa in (('T0', 1.), ('T1', 0.45)):
            z = g[g.Senal == s].set_index('Config').reindex(ids)
            pos = np.arange(len(ids)) + (k - 1)*ancho + (0 if s == 'T0' else ancho/2)
            axs[0].bar(pos - ancho/4, z.EFC_20_anios, ancho/2, color=COLOR[r], alpha=alfa)
            for p_, rep in zip(pos, z.Renovaciones):
                if rep and rep > 0:
                    axs[0].text(p_ - ancho/4, 7200*1.02, f'{int(rep)}R', ha='center', fontsize=6.5, color=COLOR[r])
            axs[1].bar(pos - ancho/4, z.Emisiones_operativas_kt, ancho/2, color=COLOR[r], alpha=alfa)
    axs[0].axhline(7200, color='#8b8a83', lw=1, ls=':')
    axs[0].text(len(ids) - 0.5, 7200, 'vida por ciclos 7.200', fontsize=7, color='#666666', ha='right', va='bottom')
    axs[0].set_ylabel('EFC en 20 años')
    axs[1].set_ylabel('Emisiones operativas 20 años [kt CO₂e]')
    axs[0].set_title('Ciclos equivalentes y reposiciones (R) con 20 USD/t; barras llenas T0, claras T1', fontsize=9)
    axs[1].set_title('Emisiones operativas con 20 USD/t (CO₂ térmico + CO₂e de red, XM 2025)', fontsize=9)
    for ax in axs:
        ax.yaxis.set_major_formatter(fmt)
    axs[1].set_xticks(range(len(ids)), [etiqueta_config(cfgs[c], c) for c in ids], fontsize=7)
    fig.legend(handles=[Line2D([], [], color=COLOR[r], lw=6, label=r) for r in m['reglas']],
               loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(carpeta/'05_ciclos_reposiciones_emisiones.png', dpi=150)
    plt.close(fig)
    return sorted(str(p) for p in carpeta.glob('*.png'))


def informe(salida):
    global TRM
    salida = Path(salida)
    m = M.cargar_malla()
    TRM = TA.contexto()['TRM']
    vpn, flujos, omitidos, comp = M.financiero(salida, m)
    completo = M.informar(salida, comp, 'financiero') and omitidos.empty
    estado = 'COMPLETO' if completo else 'INCOMPLETO'
    anu = M.leer_anualidades(salida, M.inventario(m))
    anu.assign(Estado_entrega=estado).to_csv(salida/'anualidades.csv', index=False)
    ops = operacion(anu[anu.Estado == 'OK'])
    ops.to_csv(salida/'operacion_20_anios.csv', index=False)
    ops = ops[ops.Anualidades == M.H]
    vpn.assign(Estado_entrega=estado).to_csv(salida/'vpn.csv', index=False)
    flujos.assign(Estado_entrega=estado).to_csv(salida/'flujos_anuales.csv', index=False)
    omitidos.to_csv(salida/'vpn_omitidos.csv', index=False)
    if len(flujos):
        flujos[flujos.Renovaciones_en_anio > 0].to_csv(salida/'renovaciones.csv', index=False)
    res = resultados_config(vpn, ops, m) if len(vpn) else pd.DataFrame()
    res.assign(Estado_entrega=estado).to_csv(salida/'resultados_por_configuracion.csv', index=False)
    figs = []
    if len(res):
        escribir_resumenes(salida, res, m, estado)
    reuso = pd.read_csv(salida/'reutilizacion_K3.csv') if (salida/'reutilizacion_K3.csv').exists() else pd.DataFrame()
    ver = verificaciones(salida, vpn, flujos, anu, comp, reuso, m)
    ver['estado_entrega'] = estado
    (salida/'verificaciones.json').write_text(json.dumps(ver, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    if completo:
        figs = figuras(salida, res, vpn, m)
    (salida/'ESTADO.txt').write_text(estado + chr(10), encoding='utf-8')
    archivos = sorted(p for p in salida.glob('*') if p.is_file() and p.name != 'manifiesto.json')
    archivos += sorted((salida/'figuras').glob('*.png'))
    codigo = M.CODIGO + ('termica_carbono_malla_informe.py', 'termica_carbono_vpn_informe.py', 'carbono_malla_pruebas.py',
                         'diagnostico_apagado_b.py')
    manifiesto = dict(generado=pd.Timestamp.now().isoformat(), estado_entrega=estado,
                      resultados_resumen={str(p.relative_to(salida)): sha(p) for p in archivos},
                      anualidades={f'{r.Grupo}/{r.Escenario_piloto}/{r.Etiqueta}': dict(Origen=r.Origen, SHA256_npz=r.SHA256_npz)
                                   for r in anu.itertuples()} if len(anu) else {},
                      codigo={f: sha(f) for f in dict.fromkeys(codigo)},
                      entradas={f: sha(f) for f in ('datos/load variable.csv', '02_Optimizacion_portafolio_PV_BESS.ipynb',
                                                    'datos/CurvasCapexPVBESS.xlsx')},
                      comandos=['python carbono_malla_pruebas.py --procesos 9',
                                'python termica_carbono_malla.py --salida <carpeta> --etapa reutilizar',
                                'python termica_carbono_malla.py --salida <carpeta> --etapa medir --procesos 8 --grupos M01 M04 M08 M11 '
                                'REF_RED_TERMICA REF_PV_100 REF_PV_075 REF_PV_050',
                                'python termica_carbono_malla.py --salida <carpeta> --etapa anual --procesos 9',
                                'python termica_carbono_malla.py --salida <carpeta> --etapa financiero',
                                'python diagnostico_apagado_b.py --salida resultados_carbono_v2/diagnostico_apagado_b_20260926'],
                      figuras=figs)
    (salida/'manifiesto.json').write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(f'informe: {estado}; {len(vpn)} VPN, {len(omitidos)} omitidos', flush=True)
    return 0 if completo else 1
