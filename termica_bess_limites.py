"""Límites anuales con BESS fijo. Tarifas canónicas; candidatos inmutables.

Ejecutar sin opciones completa seis casos en serie. --solo-figura no optimiza.
--caso T0 180 --segundos 1800 agrega un candidato sin borrar los anteriores.
"""
import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

SAL = Path('resultados_nb13')
CARPETA = SAL / 'limites_bess_candidatos'
RESUMEN = SAL / 'limites_bess_resumen.csv'
LIMITES = (365, 180, 60, 0)
CARGO = 100.
TOL = 1e-4


def leer(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def huella_senal(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def firma(E, P, b):
    from tarifas_termica_v2 import huella_entradas
    return huella_entradas(E, P, dict(bess=b, cargo=CARGO, minutos=0,
                                    inicial='libre', limite_diario=1))


def contar_anual(r, fechas):
    u = np.asarray(r['compromiso'])
    assert np.isin(u, [0, 1]).all()
    delta = np.diff(u, axis=0, prepend=u[:1])
    arr = (delta == 1).astype(int)
    np.testing.assert_array_equal(arr, r['arranques'])
    return pd.DataFrame(arr, index=pd.DatetimeIndex(fechas).year).groupby(level=0).sum().to_numpy()


def validar(r, E, P, b, s, limite):
    import termica_bess as TB
    assert pd.DatetimeIndex(r['Fecha']).equals(E['FECHA']), 'calendario distinto'
    np.testing.assert_allclose(r['pv_disponible'], E['PV_MW'], rtol=0, atol=1e-9)
    costo = float((r['potencia'][:, :4] @ P['cv']).sum()
                  + r['potencia'][:, 4] @ E['C_IMP_USD'][s] + CARGO*r['arranques'].sum())
    if 'objetivo' in r:
        assert abs(costo-float(r['objetivo'])) <= TOL, 'costo guardado distinto'
    v = TB.validar_bess(dict(r, objetivo=costo), E['LOAD'], E['C_IMP_USD'][s],
                        P['cv'], P['pmin'], P['pmax'], E['FECHA'], E['PV_MW'], CARGO, b)
    assert (contar_anual(r, E['FECHA']) <= limite).all(), 'límite anual excedido'
    return costo, v


def guardar_candidato(r, E, P, b, s, limite, segundos):
    """Primero guarda y relee; solo los candidatos validados se publican."""
    CARPETA.mkdir(parents=True, exist_ok=True)
    nombre = f'{s}_{limite}_{uuid4().hex}'
    temp = CARPETA / (nombre+'.pendiente.npz')
    destino = CARPETA / (nombre+'.npz')
    meta = dict(Senal=s, Limite_origen=limite, Huella_entradas=firma(E, P, b),
                Huella_senal=huella_senal(E['C_IMP_USD'][s]), Costo_USD=float(r['objetivo']),
                Cota_USD=float(r['cota_dual']), Brecha_solver=float(r['brecha']),
                Segundos=float(r['segundos']), Presupuesto_s=segundos,
                Condicion_solver=r['condicion_solver'], Fecha_UTC=datetime.now(timezone.utc).isoformat())
    claves = ('potencia', 'compromiso', 'arranques', 'paradas', 'pv_usado', 'pv_directo',
              'bess_carga', 'bess_descarga', 'soc')
    np.savez_compressed(temp, **{k: r[k] for k in claves}, Fecha=E['FECHA'].to_numpy(),
                        pv_disponible=E['PV_MW'], objetivo=r['objetivo'], metadatos=json.dumps(meta))
    costo, _ = validar(leer(temp), E, P, b, s, limite)
    assert np.isfinite(meta['Cota_USD']) and meta['Cota_USD'] <= costo+TOL
    os.replace(temp, destino)
    return destino


def referencias(E, P, b):
    """Revalida referencias; nunca llama al solver ni escribe sobre ellas."""
    import termica_limites_arranques as LA
    from tarifas_termica_v2 import huella_entradas
    sin = pd.read_csv(LA.RESUMEN, float_precision='round_trip')
    # La fórmula es la misma que usa el productor del caso sin BESS.
    firma_pv = huella_entradas(E, P, dict(cargo=LA.CARGO, minutos=0, limite_diario=1, inicial='libre'))
    originales, comparacion = [], []
    for s in ('T0', 'T1'):
        path = SAL/f'bess_anual_{s}.npz'
        meta = json.loads((SAL/f'bess_anual_{s}_solver.json').read_text(encoding='utf-8'))
        assert meta['Huella_entradas'] == firma(E, P, b), 'referencia BESS con otras entradas'
        r = leer(path)
        r['objetivo'] = meta['Objetivo_USD']
        costo, _ = validar(r, E, P, b, s, 365)
        originales.append((r, dict(Senal=s, Limite_origen=365, Costo_USD=costo,
                           Cota_USD=meta['Cota_dual_USD'], Archivo=str(path),
                           Huella_entradas=firma(E, P, b), Huella_senal=huella_senal(E['C_IMP_USD'][s]),
                           Condicion_solver=meta['Condicion_solver'], Segundos=meta['Segundos'])))
        for L in LIMITES:
            f = sin[(sin.Senal == s) & (sin.Limite_anual_por_unidad == L)].iloc[0]
            assert f.Huella_entradas == firma_pv, 'referencia PV con otras entradas'
            rr = LA.leer_npz(LA.archivo(s, L))
            costo = LA.validar_trayectoria(rr, E, P, s, L)
            assert abs(costo-f.Costo_operativo_total_USD) <= 1e-3
            comparacion.append(dict(Senal=s, Limite_anual_por_unidad=L,
                                    Costo_sin_BESS_USD=costo, Cota_sin_BESS_USD=f.Cota_dual_USD))
    return originales, pd.DataFrame(comparacion)


def cargar_pool(E, P, b, originales):
    pool = list(originales)
    for path in sorted(CARPETA.glob('*.npz')):
        if '.pendiente.' in path.name:
            continue
        r = leer(path)
        m = json.loads(str(r.pop('metadatos')))
        assert m['Huella_entradas'] == firma(E, P, b), f'caché incompatible: {path}'
        assert m['Huella_senal'] == huella_senal(E['C_IMP_USD'][m['Senal']])
        costo, _ = validar(r, E, P, b, m['Senal'], m['Limite_origen'])
        assert abs(costo-m['Costo_USD']) <= TOL
        assert np.isfinite(m['Cota_USD']) and m['Cota_USD'] <= costo+TOL
        pool.append((r, dict(m, Archivo=str(path))))
    return pool


def escoger(pool, s, L, fechas):
    del_s = [(r, m) for r, m in pool if m['Senal'] == s]
    factibles = [(r, m) for r, m in del_s if (contar_anual(r, fechas) <= L).all()]
    if not factibles:
        return None
    # Un candidato de cualquier límite sirve si cumple las restricciones destino.
    elegido = min(factibles, key=lambda x: x[1]['Costo_USD'])
    # La cota de un problema más holgado también acota al más estricto.
    cotas = [m for _, m in del_s if m['Limite_origen'] >= L]
    cota = max(cotas, key=lambda m: m['Cota_USD'])
    assert cota['Cota_USD'] <= elegido[1]['Costo_USD']+TOL, 'cotas incompatibles'
    return elegido[0], elegido[1], cota


def resumir(pool, comparacion, E, P, b):
    import termica_bess as TB
    filas, unidades, controles = [], [], []
    for s in ('T0', 'T1'):
        for L in LIMITES:
            seleccionado = escoger(pool, s, L, E['FECHA'])
            if seleccionado is None:
                continue
            r, m, c = seleccionado
            costo, v = validar(r, E, P, b, s, L)
            f = comparacion[(comparacion.Senal == s) & (comparacion.Limite_anual_por_unidad == L)].iloc[0]
            assert c['Cota_USD'] <= f.Costo_sin_BESS_USD+TOL, 'cota BESS supera factible sin BESS'
            i = TB.indicadores_bess(r, E['LOAD'], E['C_IMP_USD'][s], P, E['PV_MW'], CARGO, b)
            fila = dict(Senal=s, Limite_anual_por_unidad=L, PV_MWp=E['PV_MWp'],
                        P_BESS_MW=b['P_MW'], E_BESS_MWh=b['E_MWh'], Huella_entradas=firma(E, P, b),
                        Huella_senal=huella_senal(E['C_IMP_USD'][s]), Archivo=m['Archivo'],
                        Limite_origen_trayectoria=m['Limite_origen'], Archivo_cota=c['Archivo'],
                        Cota_dual_USD=c['Cota_USD'], Brecha=max(0., (costo-c['Cota_USD'])/abs(costo)),
                        Condicion_solver_trayectoria=m['Condicion_solver'], **i)
            fila.update(Costo_sin_BESS_USD=f.Costo_sin_BESS_USD,
                        Cota_sin_BESS_USD=f.Cota_sin_BESS_USD,
                        Ahorro_BESS_min_USD=f.Cota_sin_BESS_USD-costo,
                        Ahorro_BESS_max_USD=f.Costo_sin_BESS_USD-c['Cota_USD'])
            filas.append(fila)
            unidades += TB.tabla_unidades(r, dict(Senal=s, Limite_anual_por_unidad=L))
            controles.append(dict(Senal=s, Limite_anual_por_unidad=L, Estado='OK', **v))
    tabla = pd.DataFrame(filas)
    for s in ('T0', 'T1'):
        ref = tabla[(tabla.Senal == s) & (tabla.Limite_anual_por_unidad == 365)].iloc[0]
        mask = tabla.Senal == s
        tabla.loc[mask, 'Sobrecosto_min_USD'] = np.maximum(0., tabla.loc[mask, 'Cota_dual_USD']-ref.Costo_operativo_total_USD)
        tabla.loc[mask, 'Sobrecosto_max_USD'] = tabla.loc[mask, 'Costo_operativo_total_USD']-ref.Cota_dual_USD
        for _, a in tabla[mask].iterrows():
            for _, z in tabla[mask].iterrows():
                if a.Limite_anual_por_unidad < z.Limite_anual_por_unidad:
                    assert a.Costo_operativo_total_USD+TOL >= z.Cota_dual_USD
    tabla.loc[tabla.Limite_anual_por_unidad == 365, ['Sobrecosto_min_USD', 'Sobrecosto_max_USD']] = 0.
    tabla.to_csv(RESUMEN, index=False)
    pd.DataFrame(unidades).to_csv(SAL/'limites_bess_unidades_eoh.csv', index=False)
    pd.DataFrame(controles).to_csv(SAL/'limites_bess_validaciones.csv', index=False)
    pd.DataFrame([m for _, m in pool]).to_csv(SAL/'limites_bess_candidatos.csv', index=False)
    return tabla


def graficar(tabla):
    """Posprocesamiento puro. Intervalos deterministas, no de confianza estadística."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), layout='none', sharex=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.085, top=.88, wspace=.25, hspace=.32)
    for s, color, marker in [('T0', '#7048c0', 'o'), ('T1', '#c07a00', 's')]:
        t = tabla[tabla.Senal == s].sort_values('Limite_anual_por_unidad')
        x = t.Limite_anual_por_unidad.to_numpy(float)+(2 if s == 'T1' else -2)
        upper = t.Costo_operativo_total_USD/1e6
        axs[0, 0].errorbar(x, upper, yerr=[np.maximum(0, upper-t.Cota_dual_USD/1e6), np.zeros(len(t))],
                          fmt=marker, color=color, label=s+' con BESS', capsize=4)
        # El contraste sin BESS se conserva en las tablas; este panel muestra la variación con BESS.
        for ax, col in [(axs[0, 1], 'Arranques'), (axs[1, 0], 'Energia_termica_MWh'), (axs[1, 1], 'Vertimiento_pct')]:
            ax.scatter(x, t[col]/(1000 if col == 'Energia_termica_MWh' else 1), color=color, marker=marker, label=s)
    for ax, ylabel in zip(axs.flat, ['Costo operativo [MUSD/año]', 'Arranques totales', 'Generación térmica [GWh/año]', 'Vertimiento del PV [%]']):
        ax.set_ylabel(ylabel); ax.set_xlabel('Límite por unidad y año')
        ax.set_xticks([0, 60, 180, 365], ['0', '60', '180', '365¹']); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'.replace('.', ',')))
    axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('PV fijo 114,415 MWp · BESS 45 MW / 280 MWh · gas 16 · cargo hipotético 100 USD/arranque\n'
                 'Barras: cota inferior–costo factible. ¹ Sin límite anual adicional al diario; estado inicial libre.', fontsize=10)
    out = Path('figuras_nb13/13_3_limites_arranques_bess.png')
    out.parent.mkdir(exist_ok=True)
    fig.canvas.draw()  # Calcula el espacio del título antes de guardar bajo el backend del notebook.
    fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solo-figura', action='store_true')
    parser.add_argument('--caso', nargs=2, metavar=('SENAL', 'LIMITE'))
    parser.add_argument('--segundos', type=int, default=600)
    parser.add_argument('--solo-resumen', action='store_true')
    args = parser.parse_args()
    if args.solo_figura:
        graficar(pd.read_csv(RESUMEN)); return
    if args.segundos <= 0:
        parser.error('segundos debe ser positivo')
    import termica_bess as TB
    import termica_pv as TP
    from termica_pypsa import resolver_pypsa
    for nombre in ('pypsa', 'linopy'):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    E, P, b = TP.cargar_entradas_pv(), TP.parametros_termicos(), TB.parametros_bess()
    assert abs(E['PV_MWp']-114.415019) < 1e-5 and b['P_MW'] == 45 and b['E_MWh'] == 280
    originales, comparacion = referencias(E, P, b)
    pool = cargar_pool(E, P, b, originales)
    casos = [(s, L) for s in ('T0', 'T1') for L in (180, 60, 0)]
    if args.caso:
        caso = (args.caso[0], int(args.caso[1]))
        if caso not in casos:
            parser.error('caso fuera de la malla')
        casos = [caso]
    for s, L in ([] if args.solo_resumen else casos):
        if not args.caso and any(m['Senal'] == s and m['Limite_origen'] == L for _, m in pool):
            print('Reutilizado', s, L, flush=True); continue
        print(f'INICIO {s}/{L}: presupuesto {args.segundos} s; tarifa {huella_senal(E["C_IMP_USD"][s])}', flush=True)
        r = resolver_pypsa(E['LOAD'], E['C_IMP_USD'][s], P['cv'], P['pmin'], P['pmax'], E['FECHA'], CARGO,
                          limite_segundos=args.segundos, brecha=1e-6, pv_disponible=E['PV_MW'], bess=b,
                          aceptar_limite_tiempo=True, bess_exclusion='relajada', max_arranques_anuales=L)
        r = TB.reparar_simultaneidad(r, b, E['PV_MW'])
        guardar_candidato(r, E, P, b, s, L, args.segundos)
        pool = cargar_pool(E, P, b, originales)
        resumir(pool, comparacion, E, P, b)
        print(f'COMPLETADO {s}/{L}: costo {r["objetivo"]:.2f}; cota {r["cota_dual"]:.2f}; '
              f'brecha {r["brecha"]:.6%}; arranques {r["arranques"].sum(axis=0).tolist()}', flush=True)
    tabla = resumir(pool, comparacion, E, P, b)
    graficar(tabla)
    print(f'Resumen validado: {len(tabla)}/8 escenarios. Referencias originales intactas.', flush=True)


if __name__ == '__main__':
    main()
