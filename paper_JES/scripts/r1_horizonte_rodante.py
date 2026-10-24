"""Refuerzo R1: MILP de horizonte rodante (RH-24 y RH-48) sobre C280 (K3 en el repositorio).

Cada día calendario d se resuelve un MILP con previsión perfecta solo dentro de la ventana
[24(d-1), min(8760, 24(d-1)+H)), con la misma topología y restricciones del bloque `solar` de
`termica_pypsa.resolver_pypsa` (prioridad PV impuesta fuera, mínimos técnicos, cargo de arranque,
un arranque y una parada por unidad y día calendario, techo de exportación, máscara de exportación
del BESS en horas con déficit y exclusión binaria carga/exportación). Se conservan las primeras 24 h.

Fronteras:
- SOC inicial de cada ventana = SOC heredado; día 1: e0 = SOC inicial del benchmark anual cerrado.
- Compromiso inicial heredado: alpha - beta = u[t0] - u[t0-1]; día 1 libre sin evento (como el anual).
- SOC final libre en [0,20E, E], salvo en las ventanas que terminan en la hora 8760: e[8760] = e0.
  Si esa condición resulta infactible, la ventana se amplía hacia atrás un día por vez (hasta 7) y
  se resuelve de una vez hasta el 31 de diciembre; el retroceso queda registrado.

Las trayectorias concatenadas se validan con `termica_reglas.validar_integrado` y se fijan en el
MILP anual compatible (pertenencia). Como son factibles para el anual, L(OPT) <= costo(RH).

Uso:  OperacionBESS/.venv/bin/python paper_JES/scripts/r1_horizonte_rodante.py --caso 24 B T0
"""
from __future__ import annotations

import argparse
import logging
import time
import traceback

import comun as C
from comun import V, REG, np, pd

import pypsa  # noqa: E402
import termica_pypsa as TPY  # noqa: E402
import xarray as xr  # noqa: E402

UNIDADES = ['G1', 'G2', 'G3', 'G4']
SEG_VENTANA, BRECHA_VENTANA = 30., 1e-6
MAX_RETROCESO = 7


class Infactible(RuntimeError):
    pass


def ventana(fechas, carga, exc, ce, red, p, b, pe, cargo, exporta, soc_ini, u_prev, soc_final, usados=None):
    """MILP de una ventana. Devuelve las series de la ventana completa.

    `usados` (opcional): arranques y paradas por unidad ya comprometidos en el primer día calendario de
    la ventana antes de su inicio (ventanas que no empiezan a las 00:00); se descuentan del límite diario."""
    fechas = pd.DatetimeIndex(fechas, name='snapshot')
    Pb, Eb, eta_c, eta_d = b['P_MW'], b['E_MWh'], b['eta_c'], b['eta_d']
    deficit = carga > TPY.TOL_DEFICIT_MW
    n = pypsa.Network()
    n.set_snapshots(fechas)
    n.add('Bus', 'Demanda')
    n.add('Load', 'Consumo', bus='Demanda', p_set=carga)
    for j, g in enumerate(UNIDADES):
        n.add('Generator', g, bus='Demanda', p_nom=p['pmax'][j], committable=True,
              p_min_pu=p['pmin'][j]/p['pmax'][j], marginal_cost=p['cv'][j], start_up_cost=cargo,
              min_up_time=0, min_down_time=0, up_time_before=0, down_time_before=0)
    n.add('Generator', 'Red', bus='Demanda', p_nom=max(float(carga.max()), 1.), marginal_cost=red)
    p_nom_exc = max(float(exc.max()), 1.)
    n.add('Bus', 'Excedente')
    n.add('Bus', 'BESS')
    n.add('Generator', 'PV_excedente', bus='Excedente', p_nom=p_nom_exc, p_max_pu=exc/p_nom_exc, marginal_cost=0.)
    n.add('Generator', 'Exportacion_PV', bus='Excedente', sign=-1, p_nom=pe,
          marginal_cost=pd.Series(-ce, index=fechas))
    n.add('Link', 'BESS_carga', bus0='Excedente', bus1='BESS', p_nom=Pb, efficiency=eta_c, p_min_pu=0., p_max_pu=1.)
    n.add('Link', 'BESS_descarga', bus0='BESS', bus1='Demanda', p_nom=Pb/eta_d, efficiency=eta_d,
          p_min_pu=0., p_max_pu=1.)
    if exporta:
        n.add('Bus', 'ExportBESS')
        n.add('Link', 'BESS_a_export', bus0='BESS', bus1='ExportBESS', p_nom=Pb/eta_d, efficiency=eta_d,
              p_min_pu=0., p_max_pu=pd.Series(np.where(deficit, 0., 1.), index=fechas))
        n.add('Generator', 'Exportacion_BESS', bus='ExportBESS', sign=-1, p_nom=pe,
              marginal_cost=pd.Series(-ce, index=fechas))
    n.add('Store', 'BESS', bus='BESS', e_nom=Eb, e_min_pu=b['soc_min_pu'], e_max_pu=b['soc_max_pu'],
          e_cyclic=False, e_initial=float(soc_ini), standing_loss=0.)
    m = n.optimize.create_model()
    u, y, z = [m.variables['Generator-'+a] for a in ('status', 'start_up', 'shut_down')]
    m.remove_constraints(['Generator-com-transition-start-up', 'Generator-com-transition-shut-down'])
    if u_prev is None:
        m.add_constraints(y.isel(snapshot=0) == 0, name='Inicial-sin-arranque')
        m.add_constraints(z.isel(snapshot=0) == 0, name='Inicial-sin-parada')
    else:
        previo = xr.DataArray(np.asarray(u_prev, float), coords={'name': UNIDADES}, dims='name')
        u0, y0, z0 = (v.isel(snapshot=0).sel(name=UNIDADES) for v in (u, y, z))
        m.add_constraints(u0 - y0 + z0 == previo, name='Transicion-heredada')
    if len(fechas) > 1:
        m.add_constraints((u - u.shift(snapshot=1) - y + z).isel(snapshot=slice(1, None)) == 0,
                          name='Transicion-exacta')
    m.add_constraints(y + z <= 1, name='Eventos-excluyentes')
    dias = xr.DataArray(fechas.normalize().to_numpy(), coords={'snapshot': fechas}, dims='snapshot', name='dia')
    if usados is None:
        m.add_constraints(y.groupby(dias).sum() <= 1, name='Un-arranque-diario')
        m.add_constraints(z.groupby(dias).sum() <= 1, name='Una-parada-diaria')
    else:
        dias_u = pd.DatetimeIndex(fechas.normalize().unique())
        for nombre, var, previo in (('Un-arranque-diario', y, usados[0]), ('Una-parada-diaria', z, usados[1])):
            tope = np.ones((len(dias_u), 4))
            tope[0] = 1. - np.asarray(previo, float)
            assert (tope >= 0).all(), 'eventos previos por encima del límite diario'
            rhs = xr.DataArray(tope, coords={'dia': dias_u.to_numpy(), 'name': UNIDADES}, dims=('dia', 'name'))
            suma = var.sel(name=UNIDADES).groupby(dias).sum()
            m.add_constraints(suma <= rhs, name=nombre)
    if usados is not None and fechas[-1].hour != 23:
        # Cierre seguro de ventanas desfasadas: una unidad no termina encendida si ya usó su parada
        # en el último día calendario de la ventana; si no, la ventana siguiente podría quedar infactible
        # (unidad obligada a seguir encendida con demanda residual nula).
        ult = fechas[fechas.normalize() == fechas[-1].normalize()]
        m.add_constraints(u.isel(snapshot=-1).sel(name=UNIDADES)
                          + z.sel(snapshot=ult, name=UNIDADES).sum('snapshot') <= 1, name='Cierre-seguro')
    lps, gps = m.variables['Link-p'], m.variables['Generator-p']
    if soc_final is not None:
        m.add_constraints(m.variables['Store-e'].sel(name='BESS').isel(snapshot=-1) == float(soc_final),
                          name='Cierre-anual')
    if exporta:
        m.add_constraints(lps.sel(name='BESS_descarga') + lps.sel(name='BESS_a_export') <= Pb/eta_d,
                          name='BESS-descarga-total')
        m.add_constraints(gps.sel(name='Exportacion_PV') + gps.sel(name='Exportacion_BESS') <= pe,
                          name='PCC-exportacion-total')
        horas = fechas[exc > 0]
        if len(horas):
            horas = pd.Index(horas, name='snapshot')
            modo = m.add_variables(binary=True, coords=[horas], name='BESS-modo-carga')
            m.add_constraints(lps.sel(name='BESS_carga', snapshot=horas) - Pb*modo <= 0, name='Carga-si-modo')
            m.add_constraints(lps.sel(name='BESS_a_export', snapshot=horas) + (Pb/eta_d)*modo <= Pb/eta_d,
                              name='Exporta-si-no-modo')
    status, cond = n.optimize.solve_model(solver_name='highs', solver_options={
        'time_limit': SEG_VENTANA, 'mip_rel_gap': BRECHA_VENTANA, 'mip_feasibility_tolerance': 1e-9,
        'primal_feasibility_tolerance': 1e-9, 'threads': 1, 'log_to_console': False})
    info = m.solver_model.getInfo()
    if cond in ('infeasible', 'infeasible_or_unbounded') or (status != 'ok'):
        raise Infactible(f'{status}/{cond}')
    if cond != 'optimal' and not (cond == 'time_limit' and info.primal_solution_status == 2):
        raise RuntimeError(f'Ventana sin solución factible certificada: {status}/{cond}')
    cero = np.zeros(len(fechas))
    pot = n.generators_t.p[UNIDADES + ['Red']].to_numpy()
    bruto = n.generators_t.status[UNIDADES].to_numpy()
    comp = np.rint(bruto).astype(int)
    np.testing.assert_allclose(bruto, comp, rtol=0, atol=1e-7)
    carga_b = n.links_t.p0['BESS_carga'].to_numpy()
    desc_dem = -n.links_t.p1['BESS_descarga'].to_numpy()
    desc_exp = -n.links_t.p1['BESS_a_export'].to_numpy() if exporta else cero
    exp_pv = n.generators_t.p['Exportacion_PV'].to_numpy()
    exc_usado = n.generators_t.p['PV_excedente'].to_numpy()
    soc = n.stores_t.e['BESS'].to_numpy()
    return dict(potencia=np.maximum(pot, 0.), compromiso=comp, carga=np.maximum(carga_b, 0.),
                desc_dem=np.maximum(desc_dem, 0.), desc_exp=np.maximum(desc_exp, 0.),
                exp_pv=np.maximum(exp_pv, 0.), vert=np.maximum(exc - exc_usado, 0.), soc=soc,
                condicion=f'{status}/{cond}', brecha=float(info.mip_gap))


def horizonte_rodante(H, accion, senal, desfase=0):
    caso_bench = 'OPT_D' if accion == 'X' else 'OPT_AB'
    datos, m = C.contexto_c280(senal)
    f, pv, load, ci, ce = datos
    p, b, pe, cargo = m['parametros_termicos'], m['bess'], m['P_EXP_MAX'], m['cargo_arranque_USD']
    x = np.minimum(pv, load)
    carga, exc = load - x, pv - x
    e0 = C.soc_inicial_cerrado(senal, caso_bench)
    T = len(f)
    S = {k: np.zeros(T) for k in ('carga', 'desc_dem', 'desc_exp', 'exp_pv', 'vert', 'soc')}
    S['potencia'], S['compromiso'] = np.zeros((T, 5)), np.zeros((T, 4), int)
    soc_prev, u_prev = e0, None
    registro, retrocesos = [], []
    # Fronteras de compromiso: cada 24 h desde 00:00 (desfase 0) o desde la hora `desfase`
    # (primer tramo corto [0, desfase)); cada ventana dura H h desde su frontera.
    fronteras = ([0] if desfase else []) + list(range(desfase, T, 24)) + [T]
    fronteras = sorted(set(fronteras))

    def eventos_previos(t0):
        # Arranques/paradas ya comprometidos entre las 00:00 del día de t0 y t0 (solo si desfase > 0).
        if not desfase:
            return None
        if t0 % 24 == 0:
            return np.zeros(4), np.zeros(4)
        ini = t0 - t0 % 24
        comp = S['compromiso'][:t0]
        delta = np.diff(np.vstack([comp[0], comp]), axis=0)[ini:t0]
        return (delta == 1).sum(axis=0), (delta == -1).sum(axis=0)

    i = 0
    while i < len(fronteras) - 1:
        t0 = fronteras[i]
        t1 = min(T, t0 + H)
        cierre = e0 if t1 == T else None
        try:
            w = ventana(f[t0:t1], carga[t0:t1], exc[t0:t1], ce[t0:t1], ci[t0:t1], p, b, pe, cargo,
                        accion == 'X', soc_prev, u_prev, cierre, usados=eventos_previos(t0))
            fin = fronteras[i + 1]
        except Infactible:
            if cierre is None:
                raise
            # Retroceso: se resuelve de una vez desde fronteras anteriores hasta el cierre anual.
            for k in range(1, MAX_RETROCESO + 1):
                if i - k < 0:
                    break
                t0b = fronteras[i - k]
                soc_b = e0 if t0b == 0 else S['soc'][t0b - 1]
                u_b = None if t0b == 0 else S['compromiso'][t0b - 1]
                try:
                    w = ventana(f[t0b:T], carga[t0b:T], exc[t0b:T], ce[t0b:T], ci[t0b:T], p, b, pe, cargo,
                                accion == 'X', soc_b, u_b, e0, usados=eventos_previos(t0b))
                    t0, fin = t0b, T
                    retrocesos.append(dict(frontera_infactible=fronteras[i], tramos_retroceso=k))
                    break
                except Infactible:
                    continue
            else:
                raise RuntimeError(f'Cierre anual infactible aun con {MAX_RETROCESO} tramos de retroceso')
            if fin != T:
                raise RuntimeError('Cierre anual infactible en el primer tramo')
        n_keep = fin - t0
        for k in ('carga', 'desc_dem', 'desc_exp', 'exp_pv', 'vert', 'soc'):
            S[k][t0:fin] = w[k][:n_keep]
        S['potencia'][t0:fin] = w['potencia'][:n_keep]
        S['compromiso'][t0:fin] = w['compromiso'][:n_keep]
        registro.append(dict(tramo=i + 1, t0=t0, horas=len(w['soc']), condicion=w['condicion'], brecha=w['brecha']))
        soc_prev, u_prev = float(S['soc'][fin - 1]), S['compromiso'][fin - 1].copy()
        i = fronteras.index(fin)
    comp = S['compromiso']
    delta = np.diff(np.vstack([comp[0], comp]), axis=0)
    arr, par = (delta == 1).astype(int), (delta == -1).astype(int)
    r = dict(Caso=f'RH{H}_{accion}', Tipo='rolling_horizon', Fechas=pd.DatetimeIndex(f), PV=pv, Carga=load,
             PV_directo=x, Carga_BESS=S['carga'], Descarga_demanda=S['desc_dem'], Descarga_export=S['desc_exp'],
             Export_PV=S['exp_pv'], Vertimiento=S['vert'], SOC=S['soc'], SOC_inicial=e0, potencia=S['potencia'],
             compromiso=comp, arranques=arr, paradas=par, C_imp_USD=ci, C_exp_USD=ce, BESS=b, Cargo=cargo)
    ind = C.indicadores(r, p, cargo)
    r['Objetivo'] = ind['Costo_operativo_USD']
    r['Cota'] = r['Objetivo']
    return r, ind, registro, retrocesos, caso_bench, datos, m


def ejecutar(H, accion, senal, desfase=0):
    carpeta = C.SALIDA / 'R1'
    nombre_rh = f'RH{H}' + (f'A{desfase}' if desfase else '')
    etiqueta = f'R1_{nombre_rh}_{accion}_{senal}'
    js = carpeta / 'casos' / f'{etiqueta}.json'
    if js.exists():
        return C.leer_json(js)
    for nombre in ('pypsa', 'linopy'):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    tic = time.perf_counter()
    r, ind, registro, retrocesos, caso_bench, datos, m = horizonte_rodante(H, accion, senal, desfase)
    r['Caso'] = f'{nombre_rh}_{accion}'
    p = m['parametros_termicos']
    val = REG.validar_integrado(r, p, accion == 'X')
    V.contabilidad(r, p)
    assert (r['Export_PV'] + r['Descarga_export'] <= m['P_EXP_MAX'] + 1e-6).all()
    seg = time.perf_counter() - tic
    cond = pd.Series([x['condicion'] for x in registro]).value_counts().to_dict()
    fila = dict(Refuerzo='R1', Escenario=nombre_rh, Accion=accion, Senal=senal, Caso=f'{nombre_rh}_{accion}',
                Benchmark=caso_bench, Estado='OK', Horizonte_h=H, Desfase_h=desfase, Ventanas=len(registro),
                Condiciones_ventanas=cond, Brecha_ventana_max=float(max(x['brecha'] for x in registro)),
                Retrocesos=retrocesos, Segundos=seg, Segundos_limite_ventana=SEG_VENTANA,
                Brecha_solicitada_ventana=BRECHA_VENTANA, Validacion=val, Objetivo_USD=r['Objetivo'], **ind)
    fila.update(C.pertenencia(r, caso_bench, datos, m, p))
    fila = C.guardar_caso(carpeta, etiqueta, r, fila)
    C.json_guardar(carpeta / 'casos' / f'{etiqueta}_ventanas.json', registro)
    return fila


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--caso', nargs=3, metavar=('H', 'ACCION', 'SENAL'), required=True)
    ap.add_argument('--desfase', type=int, default=0, help='hora de inicio de las ventanas (0 = 00:00; 6 = día solar)')
    a = ap.parse_args()
    H, accion, senal = int(a.caso[0]), a.caso[1], a.caso[2]
    assert H in (24, 48) and accion in ('B', 'X') and senal in ('T0', 'T1')
    try:
        fila = ejecutar(H, accion, senal, a.desfase)
        print('OK', fila['Caso'], senal, round(fila['Costo_operativo_USD'], 2), flush=True)
    except Exception as exc:
        (C.SALIDA / 'R1' / 'errores').mkdir(parents=True, exist_ok=True)
        C.json_guardar(C.SALIDA / 'R1' / 'errores' / (f'R1_RH{H}' + (f'A{a.desfase}' if a.desfase else '') + f'_{accion}_{senal}.json'),
                       dict(caso=a.caso, error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
