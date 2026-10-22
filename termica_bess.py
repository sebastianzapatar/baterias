# -*- coding: utf-8 -*-
"""Caso 2 de la integración térmica: red + G1–G4 + PV fijo + BESS de capacidad fija.

Primera comparación controlada contra el caso 1 (`termica_pv.py`, sin BESS). Se
mantienen exactamente sus supuestos: demanda y señales T0/T1, PV de referencia
(114,415019 MWp DC en Oriente), G1–G4 con HR de fabricante y mínimos del 30 %,
gas y CVOM vigentes de `costos` (20 USD/MMBtu; G1/G2: 9,28 y G3/G4: 15,32 USD/MWh),
cargo **hipotético** de 100 USD/arranque, 0 minutos por
evento, estado inicial libre y como máximo un arranque y una parada por unidad
y día. No se agregan límites anuales de arranques.

BESS de referencia: **K3 = 45 MW / 280 MWh (6,22 h)**, la configuración
representativa vigente (INFORME §0.3 y §0.4). No se dimensiona aquí: esa
configuración se obtuvo para PV+BESS **con exportación y sin térmica**, así que
es un punto de prueba, no un tamaño óptimo para este sistema.

Parámetros técnicos del cuaderno 02, sin modificar su fuente:
DoD = 0,8 → SOC entre 20 % y 100 % de la energía nominal; RTE = 0,9 con
eta_c = eta_d = √RTE. Supuestos propios de este caso, declarados:
- carga **solo desde PV**; sin exportación;
- SOC inicial = SOC final = 60 % (punto medio del rango permitido);
- sin costo de degradación ni de mantenimiento en el despacho. El `c_deg` del
  cuaderno 02 es un precio sombra de otra formulación y no se traslada;
- carga y descarga simultáneas excluidas de forma exacta: con binarias
  (`bess_exclusion='binaria'`) o, en el año completo, resolviendo la relajación
  sin binarias y aplicando `reparar_simultaneidad`, que conserva SOC y costo.

Convención económica: PV propio, como el caso 1. CAPEX del BESS desde
`costos.capex_bess` (CurvasCapexPVBESS.xlsx) y OPEX anual del 1,5 % del CAPEX,
convertidos a USD con la TRM vigente. Se reportan aparte del costo operativo.
"""

import numpy as np
import pandas as pd

import costos
import termica_eoh as EOH
import termica_pv as TP

P_REFERENCIA_MW = 45.
E_REFERENCIA_MWh = 280.
DOD_NB02 = 0.8
ETA_RT_NB02 = 0.9
SOC0_PU = 0.6
FACTORES_EOH = EOH.FACTORES
TOL_MW = 1e-6
TOL_MWh = 1e-5


def parametros_bess(P_MW=P_REFERENCIA_MW, E_MWh=E_REFERENCIA_MWh, dod=DOD_NB02, eta_rt=ETA_RT_NB02, soc0_pu=SOC0_PU):
    eta = float(np.sqrt(eta_rt))
    return dict(P_MW=float(P_MW), E_MWh=float(E_MWh), eta_c=eta, eta_d=eta,
                soc_min_pu=1.-dod, soc_max_pu=1., soc0_pu=float(soc0_pu), DoD=dod, RTE=eta_rt)


def verificar_parametros_nb02(ns):
    """El BESS usa los parámetros técnicos del cuaderno 02; si cambian allí, se detiene."""
    assert abs(ns['DoD'] - DOD_NB02) < 1e-12 and abs(ns['ETA_RT'] - ETA_RT_NB02) < 1e-12
    assert abs(ns['ETA_C'] - np.sqrt(ETA_RT_NB02)) < 1e-12 and abs(ns['ETA_D'] - np.sqrt(ETA_RT_NB02)) < 1e-12


def reparar_simultaneidad(r, b, pv, tol=1e-9):
    """Elimina carga y descarga simultáneas de una solución relajada, sin cambiar SOC, térmica, red ni costo.

    En una hora con carga c y descarga d a la vez (k = η_c·η_d):
    - si η_c·c ≥ d/η_d: la carga pasa a c − d/k y la descarga a 0; la demanda recibe d más de PV directo;
    - si no: la descarga pasa a d − k·c y la carga a 0; la demanda recibe k·c más de PV directo.
    El cambio de SOC de la hora es idéntico. El PV generado baja o se mantiene (k ≤ 1), así que cabe en
    el PV disponible, y las potencias no aumentan. Por eso la solución reparada es factible para el
    modelo con binarias y conserva el costo de la solución factible. El óptimo de la relajación
    y su cota dual acotan por debajo el modelo con binarias; un costo factible obtenido por tiempo
    no es una cota inferior. La reparación conserva la validez de la cota dual y la brecha."""
    pv = np.asarray(pv, float)
    c, d, x = (np.asarray(r[k], float).copy() for k in ('bess_carga', 'bess_descarga', 'pv_directo'))
    k = b['eta_c']*b['eta_d']
    ambas = (c > tol) & (d > tol)
    uno = ambas & (b['eta_c']*c >= d/b['eta_d'])
    dos = ambas & ~uno
    x[uno] += d[uno]; c[uno] -= d[uno]/k; d[uno] = 0.
    x[dos] += k*c[dos]; d[dos] -= k*c[dos]; c[dos] = 0.
    c, d = np.maximum(c, 0.), np.maximum(d, 0.)
    previo = np.r_[b['E_MWh']*b['soc0_pu'], np.asarray(r['soc'], float)[:-1]]
    assert np.abs(np.asarray(r['soc']) - (previo + b['eta_c']*c - d/b['eta_d'])).max() <= TOL_MWh
    assert (x + c <= pv + TOL_MW).all()
    return dict(r, pv_directo=x, bess_carga=c, bess_descarga=d, pv_usado=x + c, pv_vertido=pv - x - c,
                bess_horas_simultaneas=int(((c > TOL_MW) & (d > TOL_MW)).sum()), bess_horas_reparadas=int(ambas.sum()))


def validar_bess(r, carga, red, cv, pmin, pmax, fechas, pv, costo_arranque, b):
    """Validación independiente; no usa PyPSA. Reutiliza `termica_pv.validar_pv` para
    balance, térmica, transiciones, eventos diarios y costo, entregando como
    suministro no térmico el PV directo más la descarga del BESS."""
    pv = np.asarray(pv, float)
    d, c, s = (np.asarray(r[k], float) for k in ('pv_directo', 'bess_carga', 'bess_descarga'))
    soc = np.asarray(r['soc'], float)
    E, P = b['E_MWh'], b['P_MW']
    r2 = dict(r, pv_usado=d + s)
    base = TP.validar_pv(r2, carga, red, cv, pmin, pmax, fechas, d + s, costo_arranque)
    assert (d >= -TOL_MW).all() and (c >= -TOL_MW).all() and (s >= -TOL_MW).all(), 'flujo negativo'
    # Carga solo desde PV: PV directo + carga ≤ PV disponible; sin PV no hay carga.
    exceso_pv = float((d + c - pv).max())
    assert exceso_pv <= TOL_MW, ('carga fuera del PV', exceso_pv)
    assert (c[pv <= 0] <= TOL_MW).all(), 'carga sin PV disponible'
    assert (c <= P + TOL_MW).all() and (s <= P + TOL_MW).all(), 'potencia del BESS'
    assert (soc >= E*b['soc_min_pu'] - TOL_MWh).all() and (soc <= E*b['soc_max_pu'] + TOL_MWh).all(), 'SOC'
    previo = np.r_[E*b['soc0_pu'], soc[:-1]]
    residuo_soc = float(np.abs(soc - (previo + b['eta_c']*c - s/b['eta_d'])).max())
    assert residuo_soc <= TOL_MWh, ('dinámica del SOC', residuo_soc)
    cierre = abs(float(soc[-1]) - E*b['soc0_pu'])
    assert cierre <= TOL_MWh, ('SOC final', cierre)
    simultaneas = int(((c > TOL_MW) & (s > TOL_MW)).sum())
    assert simultaneas == 0, ('carga y descarga simultáneas', simultaneas)
    # Sin exportación: red y térmica no negativas (validar_pv) y balance exacto.
    perdidas = float(c.sum() - s.sum() - (soc[-1] - E*b['soc0_pu']))
    esperadas = float(c.sum()*(1 - b['eta_c']*b['eta_d']))  # con ΔSOC = 0
    assert abs(perdidas - esperadas) <= 1e-4*max(1., c.sum()), ('pérdidas', perdidas, esperadas)
    base.update(Residuo_SOC_MWh=residuo_soc, Cierre_SOC_MWh=cierre, Horas_simultaneas=simultaneas,
                Exceso_carga_sobre_PV_MW=max(exceso_pv, 0.))
    return base


def indicadores_bess(r, carga, red, P, pv, costo_arranque, b):
    """Costos, energías y conteos. La energía cargada y luego descargada no se cuenta dos veces:
    el PV entregado a la demanda es directo + descarga; la diferencia con lo generado son pérdidas."""
    pv = np.asarray(pv, float)
    tiene = 'pv_directo' in r
    d = np.asarray(r['pv_directo'] if tiene else r['pv_usado'], float)
    c = np.asarray(r['bess_carga'], float) if tiene else np.zeros_like(pv)
    s = np.asarray(r['bess_descarga'], float) if tiene else np.zeros_like(pv)
    ent = d + s
    base = TP.indicadores_pv(dict(r, pv_usado=ent), carga, red, P, ent, costo_arranque)
    for k in ('PV_disponible_MWh', 'PV_utilizado_MWh', 'PV_vertido_MWh', 'Vertimiento_pct',
              'Participacion_PV_demanda_pct', 'Horas_con_vertimiento'):
        base.pop(k)
    generado = d + c
    vertido = pv - generado
    E = b['E_MWh'] if tiene else 0.
    util = E*(b['soc_max_pu'] - b['soc_min_pu']) if tiene else 0.
    dsoc = float(np.asarray(r['soc'])[-1] - E*b['soc0_pu']) if tiene else 0.
    out = dict(PV_disponible_MWh=float(pv.sum()), PV_generado_MWh=float(generado.sum()),
               PV_directo_demanda_MWh=float(d.sum()), PV_a_BESS_MWh=float(c.sum()),
               PV_vertido_MWh=float(vertido.sum()),
               Vertimiento_pct=100*float(vertido.sum())/float(pv.sum()),
               Horas_con_vertimiento=int((vertido > TOL_MW).sum()),
               BESS_descarga_demanda_MWh=float(s.sum()),
               BESS_perdidas_MWh=float(c.sum() - s.sum() - dsoc), BESS_delta_SOC_MWh=dsoc,
               PV_entregado_demanda_MWh=float(ent.sum()),
               Participacion_PV_demanda_pct=100*float(ent.sum())/float(np.sum(carga)),
               BESS_capacidad_util_MWh=util,
               # Ciclos equivalentes: energía descargada entregada (lado AC) / capacidad útil E·DoD,
               # la convención de factorial_operativo. También sobre la energía nominal.
               BESS_ciclos_equivalentes_util=float(s.sum())/util if util > 0 else 0.,
               BESS_ciclos_equivalentes_nominal=float(s.sum())/E if E > 0 else 0.,
               BESS_horas_carga=int((c > TOL_MW).sum()), BESS_horas_descarga=int((s > TOL_MW).sum()))
    out.update(base)
    return out


def tabla_unidades(r, etiquetas):
    """Horas encendida, arranques, paradas, energía y EOH por unidad, G1+G2 y total."""
    rec = EOH.reconstruir(r['compromiso'], r['potencia'])
    assert np.array_equal(rec['arranques'], np.asarray(r['arranques'])) and np.array_equal(rec['paradas'], np.asarray(r['paradas']))
    filas = []
    for nombre, idx in EOH.BLOQUES.items():
        H, N = float(rec['H'][idx].sum()), int(rec['N'][idx].sum())
        fila = dict(etiquetas, Unidad=nombre, Horas_encendida_h=H, Arranques=N, Paradas=int(rec['Z'][idx].sum()),
                    Energia_MWh=float(rec['E'][idx].sum()))
        fila.update({f'EOH_f{int(f)}_h': float(EOH.eoh(H, N, f)) for f in FACTORES_EOH})
        filas.append(fila)
    return filas


def referencia_economica_bess(b):
    trm = costos.TRM_COP_USD
    capex = costos.capex_bess(b['P_MW'], b['E_MWh'])/trm if b['E_MWh'] > 0 else 0.
    opex = costos.opex_bess_anual(b['E_MWh'], power_MW=b['P_MW'])/trm if b['E_MWh'] > 0 else 0.
    return dict(P_BESS_MW=b['P_MW'], E_BESS_MWh=b['E_MWh'], Duracion_h=b['E_MWh']/b['P_MW'] if b['P_MW'] else 0.,
                CAPEX_BESS_USD=capex, OPEX_BESS_anual_USD=opex, OPEX_fraccion_CAPEX=costos.OPEX_BESS_FRACCION_CAPEX,
                TRM_COP_USD=trm, Fuente='costos.capex_bess (CurvasCapexPVBESS.xlsx) y costos.opex_bess_anual')
