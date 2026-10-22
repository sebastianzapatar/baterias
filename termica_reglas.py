# -*- coding: utf-8 -*-
"""Bloque principal A/B/D con red + térmica (`REGLAS_TERMICA.md`).

    .venv/Scripts/python.exe -B termica_reglas.py                         # año completo: SIN_BESS, A, B, D, OPT_AB, OPT_D × T0/T1
    .venv/Scripts/python.exe -B termica_reglas.py --inicio "2025-04-10 12:00" --horas 48 --salida <carpeta>

Condiciones del bloque:
1. el PV atiende primero la demanda (`PV_directo = min(PV, demanda)`);
2. el BESS carga solo con el excedente solar tras ese autoconsumo;
3. red y térmica solo atienden la demanda residual: no exportan ni cargan el BESS;
4. A, B y D pueden exportar el PV sobrante con el mismo precio y límite;
5. D exporta además energía almacenada (solar en este bloque); A y B descargan solo a demanda.

Reglas: se ejecutan las funciones originales `despachar` y `validar` del cuaderno 02
(`benchmark_optimo.cargar_entorno_nb02`), sin reescribir su lógica. Para ventanas cortas
solo se reconstruye el calendario de días (`DIA_IDX`, `DIA_KEY`) y se vuelven a ejecutar
esas mismas definiciones del cuaderno. Sus flujos solares y del BESS quedan fijos; la
demanda residual `demanda − PV_directo − descarga_BESS_a_demanda` se despacha con red +
térmica mediante `termica_pypsa.resolver_pypsa`, con los mismos eventos diarios, mínimos,
frontera libre y cargo de arranque que el caso térmico vigente.

Contabilidad (USD): combustible + CVOM + compras reales de red (salida de PyPSA) +
cargos de arranque − ingreso por exportación (`c_exp` del cuaderno 02 / TRM). `Import_MW`
de `despachar` es demanda pendiente antes de la térmica y no se cobra. `c_deg = 0`, igual
que en el cuaderno 13; no hay precio sombra del BESS en el costo.

Benchmarks: `OPT_AB` (acciones de A/B) y `OPT_D` (añade exportación del BESS solo en horas
sin déficit tras el PV directo, como la D original), con previsión perfecta, prioridad solar
y el mismo límite de exportación. Frontera de SOC común: primero se calculan las
trayectorias originales A/B/D; si comparten SOC inicial/final dentro de `TOL_CIERRE_MWh`,
ese valor derivado se impone como SOC inicial y final fijo de los benchmarks
(`frontera_soc`). Si no coinciden, el contexto se detiene con diagnóstico. Toda trayectoria
de A/B es factible en OPT_AB y la de D en OPT_D: la cota del benchmark no puede superar su costo.

Límite de exportación: `P_EXP_MAX = máx(PV) + P_BESS`, el mismo en reglas y benchmarks.
No es vinculante (las reglas exportan a lo sumo el excedente más la potencia del BESS);
reemplaza el `inf` por defecto de `despachar` solo para que PyPSA reciba un valor finito.
"""
import argparse
import ast
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import benchmark_optimo as BO
import disponibilidad_termica as DT
import termica_bess as TB
import termica_pv as TP
from termica_pypsa import TOL_DEFICIT_MW, resolver_pypsa

REGLAS = {'A': 'A Autoconsumo', 'B': 'B Merito diario', 'D': 'D Merito+Export'}
CASOS = ('SIN_BESS', 'A', 'B', 'D', 'OPT_AB', 'OPT_D')
EXPORTA_BESS = {'SIN_BESS': False, 'A': False, 'B': False, 'D': True, 'OPT_AB': False, 'OPT_D': True}
SAL = Path('resultados_reglas_termica')
C_DEG = 0.
CARGO = TP.CARGO_ARRANQUE_USD
TOL_MW = 1e-6
TOL_MWh = 1e-5
TOL_CIERRE_MWh = 1e-3   # tolerancia de cierre cíclico de `validar` del cuaderno 02
TOL_USD = 1e-3
TOL_REDONDEO_MWh = 1e-9  # solo para acotar al límite de SOC una frontera que difiere por redondeo


def entradas():
    """Cuaderno 02, entradas canónicas térmicas, BESS K3 y señales en COP y USD."""
    ns = BO.cargar_entorno_nb02()
    E = TP.cargar_entradas_pv()
    P = TP.parametros_termicos()
    b = TB.parametros_bess()
    TB.verificar_parametros_nb02(ns)
    np.testing.assert_array_equal(ns['LOAD'], E['LOAD'])
    trm = float(ns['costos'].TRM_COP_USD)
    assert abs(trm - float(E['TRM'])) < 1e-12
    c_imp = {}
    for s in ('T0', 'T1'):
        ci = np.asarray(next(v for k, v in ns['TARIFAS'].items() if k.startswith(s)), float)
        np.testing.assert_allclose(ci/trm, E['C_IMP_USD'][s], rtol=0, atol=1e-9)
        c_imp[s] = ci
    return dict(ns=ns, E=E, P=P, b=b, TRM=trm, C_IMP_COP=c_imp, C_EXP_COP=np.asarray(ns['c_exp'], float),
                P_EXP_MAX=float(np.max(E['PV_MW']) + b['P_MW']))


def dias_calendario(fechas):
    codigos, _ = pd.factorize(pd.DatetimeIndex(fechas).normalize())
    cortes = np.flatnonzero(np.r_[True, codigos[1:] != codigos[:-1]])
    fin = np.r_[cortes[1:], len(codigos)]
    return [(int(a), int(z)) for a, z in zip(cortes, fin)], codigos


def entorno(ns, fechas):
    """Mismo código de `despachar` y `umbral_volumetrico`; solo cambia el calendario de días."""
    idx, clave = dias_calendario(fechas)
    if len(fechas) == len(ns['LOAD']) and idx == list(ns['DIA_IDX']) and np.array_equal(clave, ns['DIA_KEY']):
        return ns
    copia = dict(ns)
    copia['DIA_IDX'], copia['DIA_KEY'] = idx, clave
    nb = json.loads(BO.CUADERNO.read_text(encoding='utf-8'))
    for celda in nb['cells']:
        if celda['cell_type'] != 'code':
            continue
        for nodo in ast.parse(''.join(celda['source'])).body:
            if isinstance(nodo, ast.FunctionDef) and nodo.name in ('despachar', 'umbral_volumetrico'):
                exec(compile(ast.Module(body=[nodo], type_ignores=[]), str(BO.CUADERNO), 'exec'), copia)
    assert copia['despachar'] is not ns['despachar']
    return copia


def bess_de(caso, b):
    return TB.parametros_bess(0., 0.) if caso == 'SIN_BESS' else b


def despachar_regla(env, pv, load, c_imp_cop, c_exp_cop, caso, b, p_exp_max):
    """Regla original del cuaderno 02; SIN_BESS es A con potencia y energía nulas."""
    bb = bess_de(caso, b)
    regla = REGLAS['A'] if caso == 'SIN_BESS' else REGLAS[caso]
    d = env['despachar'](np.asarray(pv, float), np.asarray(load, float), np.asarray(c_imp_cop, float),
                         np.asarray(c_exp_cop, float), regla, power_MW=bb['P_MW'], cap_MWh=bb['E_MWh'],
                         c_deg=C_DEG, p_exp_max=p_exp_max)
    fallos = env['validar'](d, bb['P_MW'], bb['E_MWh'])
    assert not fallos, (caso, fallos)
    assert float(d.Carga_Red_MW.abs().max()) <= TOL_MW, 'carga desde red en el bloque principal'
    if not EXPORTA_BESS[caso]:
        assert float(d.Descarga_Red_MW.abs().max()) <= TOL_MW, 'A/B no exportan desde el BESS'
    return d


class FronteraSOCError(ValueError):
    """Las reglas A/B/D no comparten SOC inicial/final: no hay frontera común para los benchmarks."""


def frontera_soc(trayectorias, contexto=''):
    """SOC inicial = final común de las trayectorias originales A, B y D (MWh).

    Comprueba, con la tolerancia de cierre cíclico vigente, que cada regla cierra su ciclo y
    que las tres parten del mismo SOC. Devuelve la media de los SOC iniciales (coinciden dentro
    de la tolerancia). No modifica ninguna trayectoria ni ajusta la última hora."""
    faltan = [k for k in ('A', 'B', 'D') if k not in trayectorias]
    if faltan:
        raise FronteraSOCError(f'{contexto}: faltan trayectorias {faltan} para derivar la frontera')
    ini = {k: float(trayectorias[k].attrs['SOC_inicial_MWh']) for k in ('A', 'B', 'D')}
    fin = {k: float(trayectorias[k].SOC_MWh.iloc[-1]) for k in ('A', 'B', 'D')}
    cierre = {k: abs(fin[k] - ini[k]) for k in ini}
    dispersion = max(ini.values()) - min(ini.values())
    if max(cierre.values()) > TOL_CIERRE_MWh or dispersion > TOL_CIERRE_MWh:
        raise FronteraSOCError(
            f'{contexto}: sin frontera de SOC común (tolerancia {TOL_CIERRE_MWh} MWh). '
            f'SOC inicial {ini}; SOC final {fin}; cierre {cierre}; dispersión inicial {dispersion:.6g} MWh. '
            'No se ajusta la última hora: revisar las reglas o la ventana antes de comparar.')
    return float(np.mean(list(ini.values())))


def acoplar(caso, d, fechas, c_imp_usd, c_exp_usd, P, b, cargo=CARGO, segundos=600, brecha=1e-6,
            aceptar_limite_tiempo=False):
    """Fija los flujos de la regla y despacha red + térmica sobre la demanda residual."""
    pv_directo = d.PV_a_Load_MW.to_numpy(float)
    dd = d.Descarga_Demanda_MW.to_numpy(float)
    de = d.Descarga_Red_MW.to_numpy(float)
    load = d.Load_MW.to_numpy(float)
    residual = load - pv_directo - dd
    assert residual.min() >= -TOL_MW
    residual = np.maximum(residual, 0.)
    r = resolver_pypsa(residual, c_imp_usd, P['cv'], P['pmin'], P['pmax'], fechas, cargo,
                       limite_segundos=segundos, brecha=brecha, aceptar_limite_tiempo=aceptar_limite_tiempo)
    export_total = d.Export_MW.to_numpy(float)
    ingreso = float(export_total @ c_exp_usd)
    return dict(Caso=caso, Tipo='regla', Fechas=pd.DatetimeIndex(fechas), PV=d.PV_MW.to_numpy(float), Carga=load,
                PV_directo=pv_directo, Carga_BESS=d.Carga_PV_MW.to_numpy(float), Descarga_demanda=dd,
                Descarga_export=de, Export_PV=export_total - de, Vertimiento=d.Vertimiento_MW.to_numpy(float),
                SOC=d.SOC_MWh.to_numpy(float), SOC_inicial=float(d.attrs['SOC_inicial_MWh']),
                Import_regla_MW=d.Import_MW.to_numpy(float),
                potencia=r['potencia'], compromiso=r['compromiso'], arranques=r['arranques'], paradas=r['paradas'],
                C_imp_USD=np.asarray(c_imp_usd, float), C_exp_USD=np.asarray(c_exp_usd, float),
                BESS=bess_de(caso, b), Cargo=float(cargo),
                Objetivo=float(r['objetivo']) - ingreso, Cota=float(r['cota_dual']) - ingreso,
                Brecha_solver=float(r['brecha']), Condicion=f"{r['estado_solver']}/{r['condicion_solver']}",
                Segundos=float(r['segundos']))


def reparar_exportacion(r, b, tol=1e-9):
    """Quita carga y exportación del BESS simultáneas de una solución relajada de OPT_D.

    Con k = η_c·η_d, en cada hora con carga c y exportación del BESS d:
    - si η_c·c ≥ d/η_d: la carga pasa a c − d/k y la exportación del BESS a 0; se exporta d desde el
      PV y se vierten d/k − d;
    - si no: la exportación del BESS pasa a d − k·c y la carga a 0; se exportan k·c desde el PV y se
      vierten c − k·c.
    El SOC, la exportación total, el ingreso y el costo no cambian; la potencia usada no aumenta."""
    c = np.asarray(r['solar_carga'], float).copy()
    d = np.asarray(r['solar_descarga_export'], float).copy()
    x = np.asarray(r['solar_export_pv'], float).copy()
    v = np.asarray(r['solar_vertido'], float).copy()
    k = b['eta_c']*b['eta_d']
    ambas = (c > tol) & (d > tol)
    uno = ambas & (b['eta_c']*c >= d/b['eta_d'])
    dos = ambas & ~uno
    x[uno] += d[uno]; v[uno] += d[uno]/k - d[uno]; c[uno] -= d[uno]/k; d[uno] = 0.
    x[dos] += k*c[dos]; v[dos] += c[dos] - k*c[dos]; d[dos] -= k*c[dos]; c[dos] = 0.
    c, d, v = np.maximum(c, 0.), np.maximum(d, 0.), np.maximum(v, 0.)
    antes = np.asarray(r['solar_export_pv']) + np.asarray(r['solar_descarga_export'])
    np.testing.assert_allclose(x + d, antes, rtol=0, atol=1e-7)
    return dict(r, solar_carga=c, solar_descarga_export=d, solar_export_pv=x, solar_vertido=v,
                solar_horas_reparadas=int(ambas.sum()),
                solar_horas_simultaneas=int(((c > TOL_MW) & ((np.asarray(r['solar_descarga_demanda']) + d) > TOL_MW)).sum()))


def optimo(caso, fechas, pv, load, c_imp_usd, c_exp_usd, P, b, p_exp_max, *, soc0_MWh=None, frontera_soc='fija',
           cargo=CARGO, segundos=600, brecha=1e-6, exclusion='binaria', aceptar_limite_tiempo=False, fijar=None,
           disponible=None):
    """OPT_AB u OPT_D: previsión perfecta en el mismo espacio de acciones que las reglas.

    `frontera_soc`:
    - 'fija' (predeterminada, comportamiento anterior): SOC inicial y final iguales a `soc0_MWh`, la frontera
      común de las reglas (`frontera_soc`).
    - 'ciclica' (experimento de comparación cíclica): SOC final = SOC inicial, con el SOC inicial libre dentro de
      los límites del BESS; `soc0_MWh` no se admite. Solo afecta al BESS: el estado térmico inicial sigue libre
      sin evento en la primera hora y el compromiso final libre.
    `fijar` se pasa a `resolver_pypsa` para comprobar la pertenencia de una trayectoria externa."""
    pv, load = np.asarray(pv, float), np.asarray(load, float)
    pv_directo = np.minimum(pv, load)
    exc = pv - pv_directo
    if frontera_soc == 'fija':
        if soc0_MWh is None:
            raise ValueError("frontera_soc='fija' requiere soc0_MWh")
        # Una frontera en el límite del SOC (p. ej. cap·(1−DoD)) puede quedar un ulp fuera al pasar a p.u.
        # Solo se acota ese redondeo; una diferencia mayor es un error.
        lo, hi = b['E_MWh']*b['soc_min_pu'], b['E_MWh']*b['soc_max_pu']
        if not (lo - TOL_REDONDEO_MWh <= soc0_MWh <= hi + TOL_REDONDEO_MWh):
            raise ValueError(f'frontera de SOC {soc0_MWh} MWh fuera de [{lo}, {hi}]')
        soc0_pu = min(max(float(soc0_MWh)/b['E_MWh'], b['soc_min_pu']), b['soc_max_pu'])
        bess_solar, modo = dict(b, soc0_pu=soc0_pu), 'fijo'
    elif frontera_soc == 'ciclica':
        if soc0_MWh is not None:
            raise ValueError("frontera_soc='ciclica' deja libre el SOC inicial: no se admite soc0_MWh")
        bess_solar, modo = dict(b), 'ciclico'
    else:
        raise ValueError("frontera_soc: 'fija' o 'ciclica'")
    solar = dict(excedente=exc, c_exp=np.asarray(c_exp_usd, float), p_exp_max=p_exp_max,
                 exportar_bess=EXPORTA_BESS[caso], bess=bess_solar, soc=modo, exclusion=exclusion)
    r = resolver_pypsa(load - pv_directo, c_imp_usd, P['cv'], P['pmin'], P['pmax'], fechas, cargo,
                       limite_segundos=segundos, brecha=brecha, aceptar_limite_tiempo=aceptar_limite_tiempo, solar=solar,
                       fijar=fijar, disponible=DT.alinear(disponible, fechas))
    horas_reparadas = 0
    if r['solar_horas_simultaneas']:
        assert exclusion == 'relajada', 'simultaneidad con exclusión binaria'
        r = reparar_exportacion(r, b)
        horas_reparadas = r['solar_horas_reparadas']
    return dict(Caso=caso, Tipo='benchmark', Fechas=pd.DatetimeIndex(fechas), PV=pv, Carga=load, PV_directo=pv_directo,
                Carga_BESS=r['solar_carga'], Descarga_demanda=r['solar_descarga_demanda'],
                Descarga_export=r['solar_descarga_export'], Export_PV=r['solar_export_pv'],
                Vertimiento=r['solar_vertido'], SOC=r['soc'], SOC_inicial=float(r['soc_inicial']),
                potencia=r['potencia'], compromiso=r['compromiso'], arranques=r['arranques'], paradas=r['paradas'],
                C_imp_USD=np.asarray(c_imp_usd, float), C_exp_USD=np.asarray(c_exp_usd, float), BESS=b,
                Cargo=float(cargo), Objetivo=float(r['objetivo']), Cota=float(r['cota_dual']),
                Brecha_solver=float(r['brecha']), Condicion=f"{r['estado_solver']}/{r['condicion_solver']}",
                Segundos=float(r['segundos']), Binarias=int(r['solar_binarias']), Exclusion=exclusion,
                Horas_reparadas=horas_reparadas, Horas_export_bloqueada=int(r['solar_horas_export_bloqueada']),
                Frontera_SOC_modo=frontera_soc, SOC_final=float(r['solar_soc_final']),
                Residuo_cierre_SOC_MWh=float(r['solar_residuo_cierre']),
                Residuo_primera_hora_SOC_MWh=float(r['solar_residuo_primera_hora']),
                SOC_fin_primera_hora=float(r['solar_soc_fin_primera_hora']), Fijado=r['solar_fijado'],
                Disponible=r.get('disponible'))


def validar_integrado(res, P, exportar_bess, frontera_MWh=None):
    """Validación independiente del resultado integrado; no usa PyPSA.

    `frontera_MWh` (opcional): SOC inicial y final deben coincidir con la frontera común."""
    b = res['BESS']
    pv, load = res['PV'], res['Carga']
    c, dd, de = res['Carga_BESS'], res['Descarga_demanda'], res['Descarga_export']
    xpv, vert, soc = res['Export_PV'], res['Vertimiento'], res['SOC']
    p = np.asarray(res['potencia'], float)
    n = len(load)
    assert all(len(x) == n for x in (pv, c, dd, de, xpv, vert, soc)) and p.shape == (n, 5)
    np.testing.assert_allclose(res['PV_directo'], np.minimum(pv, load), rtol=0, atol=TOL_MW,
                               err_msg='prioridad solar: PV directo distinto de min(PV, demanda)')
    exc = pv - res['PV_directo']
    for nombre, x in (('carga', c), ('descarga demanda', dd), ('descarga export', de), ('export PV', xpv),
                      ('vertimiento', vert), ('red y térmica', p)):
        assert (np.asarray(x) >= -TOL_MW).all(), f'{nombre} negativo'
    assert (c <= exc + TOL_MW).all(), 'el BESS carga más que el excedente solar'
    assert (c <= b['P_MW'] + TOL_MW).all() and (dd + de <= b['P_MW'] + TOL_MW).all(), 'potencia del BESS'
    if not exportar_bess:
        assert (de <= TOL_MW).all(), 'exportación del BESS no permitida'
    simultaneas = int(((c > TOL_MW) & ((dd + de) > TOL_MW)).sum())
    assert simultaneas == 0, ('carga y descarga simultáneas', simultaneas)
    residuo_pv = float(np.abs(exc - c - xpv - vert).max())
    assert residuo_pv <= TOL_MW, ('balance del excedente PV', residuo_pv)
    residuo_dem = float(np.abs(res['PV_directo'] + dd + p.sum(axis=1) - load).max())
    assert residuo_dem <= TOL_MW, ('balance de demanda', residuo_dem)
    termica, red = p[:, :4].sum(axis=1), p[:, 4]
    exporta = (xpv + de) > TOL_MW
    assert not (exporta & (red > TOL_MW)).any(), 'importación y exportación simultáneas'
    assert not (exporta & (termica > TOL_MW)).any(), 'térmica generando en hora de exportación'
    # Máscara de D/OPT_D: exportación del BESS solo en horas sin déficit tras el PV directo.
    deficit = (load - res['PV_directo']) > TOL_DEFICIT_MW
    horas_exp_deficit = int((deficit & (de > TOL_MW)).sum())
    assert horas_exp_deficit == 0, ('exportación del BESS en horas con déficit', horas_exp_deficit)
    previo = np.r_[res['SOC_inicial'], soc[:-1]]
    residuo_soc = float(np.abs(soc - (previo + b['eta_c']*c - (dd + de)/b['eta_d'])).max()) if n else 0.
    assert residuo_soc <= TOL_MWh, ('dinámica del SOC', residuo_soc)
    assert (soc >= b['E_MWh']*b['soc_min_pu'] - TOL_MWh).all() and (soc <= b['E_MWh']*b['soc_max_pu'] + TOL_MWh).all()
    cierre = abs(float(soc[-1]) - res['SOC_inicial'])
    assert cierre <= TOL_CIERRE_MWh, ('cierre cíclico del SOC', cierre)
    error_frontera = np.nan
    if frontera_MWh is not None:
        error_frontera = max(abs(res['SOC_inicial'] - frontera_MWh), abs(float(soc[-1]) - frontera_MWh))
        assert error_frontera <= TOL_CIERRE_MWh, ('frontera de SOC común', res['Caso'], error_frontera)
    # Red y térmica: compromiso, eventos diarios, mínimos y costo, sobre la demanda residual.
    residual = load - res['PV_directo'] - dd
    costo_red_termica = float((p[:, :4] @ P['cv']).sum() + red @ res['C_imp_USD'] + res['Cargo']*res['arranques'].sum())
    v = TP.validar_pv(dict(potencia=p, compromiso=res['compromiso'], arranques=res['arranques'], paradas=res['paradas'],
                           pv_usado=np.zeros(n), objetivo=costo_red_termica),
                      residual, res['C_imp_USD'], P['cv'], P['pmin'], P['pmax'], res['Fechas'], np.zeros(n), res['Cargo'])
    ingreso = float((xpv + de) @ res['C_exp_USD'])
    error = abs(costo_red_termica - ingreso - res['Objetivo'])
    assert error <= TOL_USD, ('costo reconstruido', error)
    assert res['Cota'] <= res['Objetivo'] + TOL_USD, 'cota por encima del costo'
    return dict(Caso=res['Caso'], Residuo_PV_MW=residuo_pv, Residuo_demanda_MW=residuo_dem, Residuo_SOC_MWh=residuo_soc,
                Cierre_SOC_MWh=cierre, Error_frontera_SOC_MWh=error_frontera, Horas_simultaneas=simultaneas,
                Horas_export_BESS_con_deficit=horas_exp_deficit, Ingreso_reconstruido_USD=ingreso, Eventos_dia_max=v['Eventos_dia_max'],
                Error_costo_USD=error, Estado='OK')


def indicadores(res, P):
    p = np.asarray(res['potencia'], float)
    b = res['BESS']
    energia = p[:, :4].sum(axis=0)
    red = p[:, 4]
    exp_total = res['Export_PV'] + res['Descarga_export']
    ingreso = float(exp_total @ res['C_exp_USD'])
    costo_termico = float((p[:, :4] @ P['cv']).sum())
    costo_red = float(red @ res['C_imp_USD'])
    costo_arr = float(res['Cargo']*res['arranques'].sum())
    util = b['E_MWh']*(b['soc_max_pu'] - b['soc_min_pu'])
    pv_total = float(res['PV'].sum())
    arr_u = np.asarray(res['arranques']).sum(axis=0)
    return dict(Caso=res['Caso'], Tipo=res['Tipo'], Horas=len(res['Carga']),
                Estado_solucion='PRECISION_SOLICITADA' if res['Condicion'].endswith('/optimal') else 'FACTIBLE_LIMITE_TIEMPO',
                Intervalo_USD=res['Objetivo'] - res['Cota'], Horas_reparadas=int(res.get('Horas_reparadas', 0)),
                SOC_inicial_MWh=res['SOC_inicial'], SOC_final_MWh=float(res['SOC'][-1]),
                Costo_combustible_CVOM_USD=costo_termico, Costo_red_USD=costo_red, Costo_arranques_USD=costo_arr,
                Ingreso_exportacion_USD=ingreso, Costo_operativo_neto_USD=costo_termico + costo_red + costo_arr - ingreso,
                Objetivo_USD=res['Objetivo'], Cota_USD=res['Cota'], Brecha_solver=res['Brecha_solver'],
                Condicion=res['Condicion'], Segundos=res['Segundos'],
                PV_MWh=pv_total, PV_directo_MWh=float(res['PV_directo'].sum()), Carga_BESS_MWh=float(res['Carga_BESS'].sum()),
                Descarga_demanda_MWh=float(res['Descarga_demanda'].sum()), Descarga_export_MWh=float(res['Descarga_export'].sum()),
                Export_PV_MWh=float(res['Export_PV'].sum()), Vertimiento_MWh=float(res['Vertimiento'].sum()),
                Aprovechamiento_PV_pct=100*(pv_total - float(res['Vertimiento'].sum()))/pv_total if pv_total > 0 else 0.,
                Compras_red_MWh=float(red.sum()), Energia_termica_MWh=float(energia.sum()),
                Energia_G1_G2_MWh=float(energia[:2].sum()), Energia_G3_MWh=float(energia[2]), Energia_G4_MWh=float(energia[3]),
                Arranques=int(res['arranques'].sum()), Arranques_G1_G2=int(arr_u[:2].sum()), Arranques_G3=int(arr_u[2]),
                Arranques_G4=int(arr_u[3]), Horas_unidad_encendida=int(res['compromiso'].sum()),
                EFC_util=float((res['Descarga_demanda'] + res['Descarga_export']).sum())/util if util > 0 else 0.)


def huella_codigo():
    """Rama, HEAD, estado y SHA-256 de los archivos que determinan el resultado."""
    archivos = ['termica_reglas.py', 'termica_pypsa.py', 'termica_pv.py', 'termica_bess.py', 'termica.py',
                'benchmark_optimo.py', Path(BO.CUADERNO).name]

    def git(*c):
        try:
            return subprocess.run(['git', *c], capture_output=True, text=True, check=True).stdout.strip()
        except Exception as e:  # sin git el resultado sigue identificado por los SHA-256
            return f'no disponible: {e}'
    return dict(git_rama=git('branch', '--show-current'), git_head=git('rev-parse', 'HEAD'),
                git_estado_archivos=git('status', '--short', '--', *archivos),
                sha256={a: hashlib.sha256(Path(a).read_bytes()).hexdigest() for a in archivos})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--senales', nargs='+', default=['T0', 'T1'], choices=['T0', 'T1'])
    parser.add_argument('--casos', nargs='+', default=list(CASOS), choices=list(CASOS))
    parser.add_argument('--segundos', type=int, default=600, help='límite de tiempo por resolución')
    parser.add_argument('--brecha', type=float, default=1e-6)
    parser.add_argument('--exclusion', choices=('binaria', 'relajada'), default='relajada',
                        help='exclusión carga/exportación del BESS en OPT_D')
    parser.add_argument('--salida', type=Path, default=SAL)
    parser.add_argument('--inicio', help='inicio de una ventana corta, p. ej. "2025-04-10 12:00"')
    parser.add_argument('--horas', type=int, help='horas de la ventana corta')
    args = parser.parse_args()
    for nombre in ('pypsa', 'linopy'):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    ctx = entradas()
    ns, E, P, b = ctx['ns'], ctx['E'], ctx['P'], ctx['b']
    if args.inicio:
        a = E['FECHA'].get_loc(pd.Timestamp(args.inicio))
        sl = slice(a, a + int(args.horas))
    else:
        sl = slice(0, len(E['LOAD']))
    fechas = E['FECHA'][sl]
    env = entorno(ns, fechas)
    args.salida.mkdir(parents=True, exist_ok=True)
    resumen_csv, validaciones_csv = args.salida/'resumen.csv', args.salida/'validaciones.csv'
    meta_json = args.salida/'ejecucion.json'
    if resumen_csv.exists() or validaciones_csv.exists() or meta_json.exists():
        raise SystemExit(f'{args.salida} ya tiene resultados; no se sobrescriben. Usar otra --salida.')
    filas, validaciones = [], []
    pv, load = np.asarray(E['PV_MW'])[sl], np.asarray(E['LOAD'])[sl]
    # Frontera común: trayectorias originales A/B/D de todas las señales antes de resolver nada,
    # también cuando solo se piden benchmarks. Si alguna no coincide, se detiene sin escribir.
    trayectorias, fronteras = {}, {}
    for s in args.senales:
        ci_cop, ce_cop = ctx['C_IMP_COP'][s][sl], ctx['C_EXP_COP'][sl]
        trayectorias[s] = {k: despachar_regla(env, pv, load, ci_cop, ce_cop, k, b, ctx['P_EXP_MAX']) for k in ('A', 'B', 'D')}
        try:
            fronteras[s] = frontera_soc(trayectorias[s], contexto=f'{s} {fechas[0]}–{fechas[-1]}')
        except FronteraSOCError as e:
            raise SystemExit(f'Contexto detenido. {e}')
        print(f'{s}: frontera de SOC común {fronteras[s]:.9f} MWh (A/B/D)', flush=True)
    import pypsa
    meta = dict(inicio=pd.Timestamp.now().isoformat(), argumentos={k: str(v) for k, v in vars(args).items()},
                python=sys.version, pypsa=pypsa.__version__, horas=len(fechas), fecha_inicial=str(fechas[0]),
                fecha_final=str(fechas[-1]), codigo=huella_codigo(), fronteras_SOC_MWh=fronteras,
                parametros=dict(termica={k: np.asarray(v).tolist() for k, v in P.items()}, cargo_arranque_USD=CARGO,
                                minutos_eventos=0, limite_anual_arranques=None,
                                limite_diario='un arranque y una parada por unidad y día', bess=b,
                                PV_max_MW=float(pv.max()), PV_MWh=float(pv.sum()), demanda_MWh=float(load.sum()),
                                P_EXP_MAX_MW=ctx['P_EXP_MAX'], c_deg=C_DEG, TRM=ctx['TRM'],
                                exclusion_OPT_D=args.exclusion, tolerancias=dict(TOL_MW=TOL_MW, TOL_MWh=TOL_MWh,
                                TOL_CIERRE_MWh=TOL_CIERRE_MWh, TOL_USD=TOL_USD, TOL_DEFICIT_MW=TOL_DEFICIT_MW)))
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    total, hechos, reloj = len(args.senales)*len(args.casos), 0, time.perf_counter()
    tiempos = {'regla': [], 'benchmark': []}
    for s in args.senales:
        ci_cop, ce_cop = ctx['C_IMP_COP'][s][sl], ctx['C_EXP_COP'][sl]
        ci_usd, ce_usd = ci_cop/ctx['TRM'], ce_cop/ctx['TRM']
        resultados = {}
        for caso in args.casos:
            destino = args.salida/f'{s}_{caso}.npz'
            if destino.exists():
                raise SystemExit(f'{destino} ya existe; no se sobrescribe.')
            print(f'{s} {caso}: resolviendo ...', flush=True)
            if caso.startswith('OPT'):
                res = optimo(caso, fechas, pv, load, ci_usd, ce_usd, P, b, ctx['P_EXP_MAX'], soc0_MWh=fronteras[s],
                             segundos=args.segundos, brecha=args.brecha, exclusion=args.exclusion,
                             aceptar_limite_tiempo=True)
            else:
                d = trayectorias[s][caso] if caso in trayectorias[s] else \
                    despachar_regla(env, pv, load, ci_cop, ce_cop, caso, b, ctx['P_EXP_MAX'])
                res = acoplar(caso, d, fechas, ci_usd, ce_usd, P, b, segundos=args.segundos, brecha=args.brecha,
                              aceptar_limite_tiempo=True)
            # Se guarda antes de validar: si la validación falla, la trayectoria queda para diagnóstico.
            np.savez_compressed(destino, Fecha=fechas.to_numpy(), **{k: np.asarray(res[k]) for k in (
                'PV', 'Carga', 'PV_directo', 'Carga_BESS', 'Descarga_demanda', 'Descarga_export', 'Export_PV',
                'Vertimiento', 'SOC', 'potencia', 'compromiso', 'arranques', 'paradas', 'C_imp_USD', 'C_exp_USD')},
                SOC_inicial=res['SOC_inicial'], SOC_frontera=fronteras[s], Objetivo=res['Objetivo'], Cota=res['Cota'],
                Brecha_solver=res['Brecha_solver'], Condicion=res['Condicion'], Segundos=res['Segundos'],
                Horas_reparadas=res.get('Horas_reparadas', 0))
            v = validar_integrado(res, P, EXPORTA_BESS[caso], None if caso == 'SIN_BESS' else fronteras[s])
            resultados[caso] = res
            ind = indicadores(res, P)
            filas.append(dict(Senal=s, SOC_frontera_MWh=fronteras[s], **ind))
            validaciones.append(dict(Senal=s, **v))
            pd.DataFrame(filas).to_csv(args.salida/'resumen_parcial.csv', index=False)
            pd.DataFrame(validaciones).to_csv(args.salida/'validaciones_parcial.csv', index=False)
            hechos += 1
            tiempos[res['Tipo']].append(res['Segundos'])
            pendientes = [c for c in args.casos[args.casos.index(caso) + 1:]] + \
                [c for s2 in args.senales[args.senales.index(s) + 1:] for c in args.casos]
            media = {t: (np.mean(x) if x else float(args.segundos)) for t, x in tiempos.items()}
            restante = sum(media['benchmark' if c.startswith('OPT') else 'regla'] for c in pendientes)/60
            print(f"[{hechos}/{total}] {s} {caso}: {ind['Estado_solucion']} ({res['Condicion']}); {res['Segundos']:.0f} s; "
                  f"costo {res['Objetivo']:,.2f} USD; cota {res['Cota']:,.2f}; brecha solver {res['Brecha_solver']:.2e}; "
                  f"intervalo {ind['Intervalo_USD']:,.2f} USD; reparadas {ind['Horas_reparadas']}; validación {v['Estado']}; "
                  f"transcurrido {(time.perf_counter() - reloj)/60:.1f} min; restante estimado ~{restante:.0f} min (no garantizado)",
                  flush=True)
        # Cotas: A/B son factibles en OPT_AB y D en OPT_D; OPT_AB es factible en OPT_D.
        for bench, reglas in (('OPT_AB', ('A', 'B')), ('OPT_D', ('A', 'B', 'D', 'OPT_AB'))):
            for regla in reglas:
                if bench in resultados and regla in resultados:
                    assert resultados[bench]['Cota'] <= resultados[regla]['Objetivo'] + TOL_USD, (s, bench, regla)
    pd.DataFrame(filas).to_csv(resumen_csv, index=False)
    pd.DataFrame(validaciones).to_csv(validaciones_csv, index=False)
    meta.update(fin=pd.Timestamp.now().isoformat(), minutos=(time.perf_counter() - reloj)/60)
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f'Escritos {resumen_csv} y {validaciones_csv}', flush=True)


if __name__ == '__main__':
    main()
