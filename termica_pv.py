# -*- coding: utf-8 -*-
"""Caso 1 de la integración térmica: red + cuatro unidades a gas + PV de tamaño fijo, sin BESS.

PyPSA (`termica_pypsa.resolver_pypsa` con `pv_disponible`) es el modelo
principal; la programación dinámica de `termica.py` queda como referencia del
caso sin PV. Este módulo reúne las entradas, el validador independiente, los
indicadores y la referencia económica del PV; `termica_pv_pruebas.py` y
`termica_pv_anual.py` los ejecutan y escriben `resultados_nb12/`.

Supuestos vigentes del caso (las salidas anteriores conservan su escenario):

    unidades G1–G4 con HR de fabricante, mínimos del 30 % de la potencia disponible
    gas 20 USD/MMBtu; CVOM G1/G2 = 9,28 y G3/G4 = 15,32 USD/MWh (costos)
    como máximo un arranque y una parada por unidad y día
    estado inicial libre, sin parada final obligatoria
    cargo de 100 USD por arranque, HIPOTÉTICO y no calibrado; 0 minutos por evento
    señales T0 y T1 del cuaderno 02; sin BESS ni exportación

PV: recurso de `solar_tmy.perfil_pv` (TMY NSRDB, región Oriente, año 2025) con
el tamaño de referencia vigente, PV_MAX_ALCANCE = demanda anual / yield
(INFORME §0.3). El tamaño se expresa en MWp DC; la potencia AC es
MWp / DC_AC_RATIO. Convención económica vigente del cuaderno 02 y del factorial:
**PV propio**, sin costo variable, con CAPEX de `costos.capex_pv` y OPEX anual del
2 % del CAPEX. El vertimiento no se paga porque no hay PPA.
"""

import numpy as np
import pandas as pd

import costos
import solar_tmy as ST
import termica as T

REGION_SOLAR = 'Oriente'
GAS_USD_MMBTU = costos.GAS_TERMICO_USD_MMBTU
CARGO_ARRANQUE_USD = 100.     # hipotético, no calibrado; permite comparar con la referencia anual
MINUTOS_EVENTO = 0            # los descuentos de 10 o 20 minutos siguen disponibles en resolver_pypsa
TOL_MW = 1e-6
TOL_USD = 1e-4


def parametros_termicos(gas=GAS_USD_MMBTU, *, cvom=None):
    """Parámetros integrados con HR de referencia y CVOM vigente de `costos`.

    Desde el 18/09/2026: gas 20 USD/MMBtu y CVOM por unidad:
    G1/G2 = 9,28 y G3/G4 = 15,32 USD/MWh, en ese orden.
    `parametros_termicos(gas=16, cvom=9.28)` reproduce los parámetros de las
    corridas integradas previas; sus salidas no se actualizan al cambiar esta función.
    `cvom=11` reconstruye el supuesto uniforme de la corrida pausada del 17/09.
    `cvom` admite un escalar o cuatro valores no negativos (G1, G2, G3, G4).
    La tabla histórica del cuaderno 11 conserva explícitamente su supuesto anterior.
    """
    gen = T.unidades_historicas()
    hr = T.HR_FABRICANTE_BTU_KWH/1000
    cvom = np.broadcast_to(np.asarray(costos.CVOM_TERMICO_USD_MWH if cvom is None else cvom,
                                     dtype=float), (4,)).copy()
    if not np.isfinite(cvom).all() or (cvom < 0).any():
        raise ValueError('CVOM debe contener valores finitos y no negativos')
    return dict(pmin=gen.P_min_MW.to_numpy(), pmax=gen.P_disp_MW.to_numpy(),
                hr=hr, cvom=cvom, gas=gas, cv=hr*gas+cvom)


def tamano_pv_referencia(carga, region=REGION_SOLAR):
    """PV_MAX_ALCANCE [MWp DC]: la fórmula de `.build/trm_sizing_scope_industrial.py`."""
    return float(np.sum(carga)) / float(ST.perfil_pv(region, 1.0).PV_MW.sum())


def cargar_entradas_pv(pv_MWp=None, region=REGION_SOLAR, version="corregida"):
    """Entradas térmicas + perfil PV AC alineado por fecha, no por posición."""
    E = T.cargar_entradas(version=version)
    if pv_MWp is None:
        pv_MWp = tamano_pv_referencia(E['LOAD'], region)
    perfil = ST.perfil_pv(region, pv_MWp)
    demanda = pd.DataFrame({'Fecha': E['FECHA'], 'Demanda_MW': E['LOAD']})
    unido = demanda.merge(perfil[['Fecha', 'PV_MW']], on='Fecha', how='left', validate='one_to_one')
    if len(unido) != 8760 or unido.PV_MW.isna().any():
        raise ValueError('El calendario del perfil PV no coincide con el de la demanda')
    fechas = pd.DatetimeIndex(unido.Fecha)
    assert (fechas == E['FECHA']).all()
    assert fechas.year.unique().tolist() == [ST.ANIO_SIMULACION]
    assert ((fechas[1:]-fechas[:-1]) == pd.Timedelta(hours=1)).all()
    E.update(PV_MW=unido.PV_MW.to_numpy(), PV_MWp=float(pv_MWp),
             PV_MWac=float(pv_MWp)/ST.DC_AC_RATIO, REGION=region)
    return E


def validar_pv(r, carga, red, cv, pmin, pmax, fechas, pv, costo_arranque, minutos=MINUTOS_EVENTO):
    """Validación independiente de una trayectoria anual o corta; no usa PyPSA."""
    p, u = np.asarray(r['potencia'], float), np.asarray(r['compromiso'], int)
    pv = np.asarray(pv, float)
    pv_usado = np.asarray(r['pv_usado'], float)
    vertido = pv - pv_usado
    n = len(carga)
    assert p.shape == (n, 5) and u.shape == (n, 4) and np.isin(u, [0, 1]).all()
    residuo_balance = float(np.abs(p.sum(axis=1) + pv_usado - carga).max())
    residuo_pv = float(np.abs(pv_usado + vertido - pv).max())
    assert residuo_balance <= TOL_MW, ('balance', residuo_balance)
    assert (pv_usado >= -TOL_MW).all() and (vertido >= -TOL_MW).all(), 'PV utilizado fuera de [0, disponible]'
    # Red y térmico no negativos y balance exacto: no hay exportación posible.
    assert (p >= -TOL_MW).all(), 'potencia negativa'
    assert (p[:, :4] >= u*pmin - TOL_MW).all() and (p[:, :4] <= u*pmax + TOL_MW).all()
    delta = np.diff(np.vstack([u[0], u]), axis=0)
    arranques, paradas = (delta == 1).astype(int), (delta == -1).astype(int)
    assert (arranques == np.asarray(r['arranques'])).all() and (paradas == np.asarray(r['paradas'])).all()
    assert not arranques[0].any() and not paradas[0].any() and ((arranques + paradas) <= 1).all()
    diario = pd.DataFrame(np.c_[arranques, paradas], index=pd.DatetimeIndex(fechas).normalize()) \
               .groupby(level=0).sum().to_numpy()
    assert (diario <= 1).all(), 'más de un arranque o parada por unidad y día'
    tau = minutos/60.
    capacidad = pmax*(u - tau*arranques - tau*np.vstack([paradas[1:], np.zeros(4, int)]))
    assert (p[:, :4] <= capacidad + TOL_MW).all()
    cargos = np.broadcast_to(np.asarray(costo_arranque, float), (4,))
    total = float((p[:, :4]@cv + p[:, 4]*red).sum() + (arranques@cargos).sum())
    error = abs(total - float(r['objetivo']))
    assert error <= TOL_USD, ('costo', error)
    return dict(Residuo_balance_MW=residuo_balance, Residuo_PV_MW=residuo_pv,
                Eventos_dia_max=int(diario.max()), Error_reconstruccion_costo_USD=error)


def indicadores_pv(r, carga, red, P, pv, costo_arranque):
    """Energías, gas, costos desagregados y arranques por bloque."""
    p, u = np.asarray(r['potencia'], float), np.asarray(r['compromiso'], int)
    arr, par = np.asarray(r['arranques'], int), np.asarray(r['paradas'], int)
    pv, pv_usado = np.asarray(pv, float), np.asarray(r['pv_usado'], float)
    energia = p[:, :4].sum(axis=0)
    combustible = energia*P['hr']*P['gas']
    cvom = energia*P['cvom']
    cargos = arr@np.broadcast_to(np.asarray(costo_arranque, float), (4,))
    costo_red = float((p[:, 4]*red).sum())
    termico = float((p[:, :4]@P['cv']).sum())
    assert abs(termico - float(combustible.sum() + cvom.sum())) <= TOL_USD
    disponible = float(pv.sum())
    out = dict(
        Demanda_MWh=float(np.sum(carga)),
        PV_disponible_MWh=disponible, PV_utilizado_MWh=float(pv_usado.sum()),
        PV_vertido_MWh=float((pv - pv_usado).sum()),
        Vertimiento_pct=100*float((pv - pv_usado).sum())/disponible if disponible > 0 else 0.,
        Participacion_PV_demanda_pct=100*float(pv_usado.sum())/float(np.sum(carga)),
        Horas_con_vertimiento=int(((pv - pv_usado) > TOL_MW).sum()),
        Energia_termica_MWh=float(energia.sum()), Energia_G1_G2_MWh=float(energia[:2].sum()),
        Energia_G3_MWh=float(energia[2]), Energia_G4_MWh=float(energia[3]),
        Compras_red_MWh=float(p[:, 4].sum()), Gas_MMBtu=float(energia@P['hr']),
        Costo_combustible_USD=float(combustible.sum()), Costo_CVOM_USD=float(cvom.sum()),
        Costo_red_USD=costo_red, Costo_arranques_USD=float(cargos.sum()), Costo_PV_variable_USD=0.,
    )
    out['Costo_operativo_total_USD'] = (out['Costo_combustible_USD'] + out['Costo_CVOM_USD']
                                        + costo_red + out['Costo_arranques_USD'])
    out.update(Arranques=int(arr.sum()), Arranques_G1_G2=int(arr[:, :2].sum()),
               Arranques_G3=int(arr[:, 2].sum()), Arranques_G4=int(arr[:, 3].sum()), Paradas=int(par.sum()),
               Horas_unidad_encendida_G1_G2=int(u[:, :2].sum()), Horas_encendida_G3=int(u[:, 2].sum()),
               Horas_encendida_G4=int(u[:, 3].sum()))
    return out


def referencia_economica_pv(pv_MWp):
    """CAPEX y OPEX del PV propio, en USD, con la convención vigente. No es un VPN."""
    trm = costos.TRM_COP_USD
    return dict(PV_MWp=pv_MWp, PV_MWac=pv_MWp/ST.DC_AC_RATIO,
                CAPEX_PV_USD=costos.capex_pv(pv_MWp)/trm,
                OPEX_PV_anual_USD=costos.opex_pv_anual(pv_MWp)/trm,
                OPEX_fraccion_CAPEX=costos.OPEX_PV_FRACCION_CAPEX, TRM_COP_USD=trm,
                Fuente='costos.capex_pv (CurvasCapexPVBESS.xlsx) y costos.opex_pv_anual')
