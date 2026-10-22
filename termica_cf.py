# -*- coding: utf-8 -*-
"""C_INT/F_INT: mérito diario del sistema con compra de red y reserva solar.

Un mismo motor y candidatos; F añade únicamente el presupuesto diario de compras
en unidades de SOC. El evaluador térmico recibe solo demanda, nunca carga del BESS.
Las políticas hacen una pasada finita; OPT_CF optimiza conjuntamente en PyPSA.
"""
import math
import time

import numpy as np
import pandas as pd

import termica_politicas as POL
import disponibilidad_termica as DT

TOL = POL.TOL_FLUJO_MW
TOL_USD = POL.TOL_OBJETIVO_USD
FRACCIONES = (.25, .5, .75, 1.)


def entradas(fechas, pv, load, ci, ce, b, p_exp):
    f = POL._validar_calendario(fechas)
    pv, load, ci, ce = (np.asarray(x, float) for x in (pv, load, ci, ce))
    if any(x.shape != (len(f),) or not np.isfinite(x).all() for x in (pv, load, ci, ce)):
        raise ValueError('Series horarias incompatibles o no finitas')
    if (pv < 0).any() or (load < 0).any() or not np.isfinite(p_exp) or p_exp < 0:
        raise ValueError('PV, demanda o límite de exportación inválidos')
    if not all(np.isfinite(b[k]) for k in ('P_MW', 'E_MWh', 'eta_c', 'eta_d', 'soc_min_pu', 'soc_max_pu')):
        raise ValueError('Parámetros BESS no finitos')
    if not (b['P_MW'] > 0 and b['E_MWh'] > 0 and 0 < b['eta_c'] <= 1 and 0 < b['eta_d'] <= 1
            and 0 <= b['soc_min_pu'] <= b['soc_max_pu'] <= 1):
        raise ValueError('Parámetros BESS inválidos')
    x = np.minimum(pv, load)
    return f, pv, load, ci, ce, x, load-x, pv-x


def presupuesto_f(eps, b):
    """Definición original de F: compras acumuladas de SOC, sin reponer cupo al descargar."""
    util = b['E_MWh']*(b['soc_max_pu']-b['soc_min_pu'])
    return max(0., util-min(util, b['eta_c']*float(np.sum(eps))))


def trayectoria(q, de, cr, eps, delta, ce, e0, b, p_exp):
    """Carga PV voraz y cronología completa. Puede devolver SOC negativo para diagnosticar
    la primera falta de energía al construir un paquete; `factible` siempre la rechaza."""
    n = len(q)
    cp, soc = np.zeros(n), np.zeros(n)
    e = float(e0)
    lo, hi = b['E_MWh']*b['soc_min_pu'], b['E_MWh']*b['soc_max_pu']
    for h in range(n):
        cp[h] = min(eps[h], b['P_MW'], max(0., hi-e)/b['eta_c'])
        e += b['eta_c']*(cp[h]+cr[h])-(q[h]+de[h])/b['eta_d']
        soc[h] = e
    xp = np.where(ce > 0, np.minimum(np.maximum(eps-cp, 0.), p_exp), 0.)
    falta = np.flatnonzero(soc < lo-POL.TOL_SOC_MWh)
    factible = (not len(falta) and np.max(soc) <= hi+POL.TOL_SOC_MWh
                and np.min(np.r_[q, de, cr]) >= -TOL and (q <= delta+TOL).all()
                and (cp+cr <= b['P_MW']+TOL).all() and (q+de <= b['P_MW']+TOL).all()
                and not ((cp+cr > TOL) & (q+de > TOL)).any()
                and not ((cr > TOL) & (eps > POL.TOL_DEFICIT_MW)).any()
                and not ((de > TOL) & ((delta > POL.TOL_DEFICIT_MW) | (ce <= 0))).any()
                and (xp+de <= p_exp+TOL).all())
    return dict(cp=cp, cr=cr.copy(), q=q.copy(), de=de.copy(), soc=soc, xp=xp,
                vert=np.maximum(eps-cp-xp, 0.), factible=bool(factible),
                primera_falta=None if not len(falta) else int(falta[0]))


def completar_compras(q, de, cr, eps, delta, ci, ce, e0, b, p_exp, cupo, q25):
    """Paquete cronológico: cada hora barata anterior a la primera falta se visita una vez.
    Solo se compra la energía faltante dentro de los límites. No es una compra óptima."""
    cr = cr.copy()
    visitas = set()
    lo, hi = b['E_MWh']*b['soc_min_pu'], b['E_MWh']*b['soc_max_pu']
    for _ in range(len(q)+1):
        tr = trayectoria(q, de, cr, eps, delta, ce, e0, b, p_exp)
        h = tr['primera_falta']
        if h is None:
            return tr if tr['factible'] else None
        anteriores = [k for k in range(h) if k not in visitas and eps[k] <= POL.TOL_DEFICIT_MW
                      and q[k]+de[k] <= TOL and ci[k] <= q25]
        if not anteriores:
            return None
        k = min(anteriores, key=lambda k: (ci[k], k))
        visitas.add(k)
        previo = e0 if k == 0 else tr['soc'][k-1]
        margen = max(0., (hi-previo)/b['eta_c']-cr[k])
        presupuesto = max(0., cupo-b['eta_c']*cr.sum())/b['eta_c']
        falta = max(0., lo-tr['soc'][h])/b['eta_c']
        compra = min(falta, max(0., b['P_MW']-cr[k]), margen, presupuesto)
        if compra > TOL:
            cr[k] += compra
    return None


def planificar_dia(f, delta, eps, ci, ce, b, p_exp, P, cargo, estado, e0, reserva, disponible=None):
    """Sondeo económico → filtro por cuartil → una pasada de candidatos → despacho diario."""
    n = len(f)
    cv, pmin, pmax = (np.asarray(P[k], float) for k in ('cv', 'pmin', 'pmax'))
    cargos = np.broadcast_to(np.asarray(cargo, float), (4,))
    cache = {}
    def termico(q):
        residual = np.maximum(delta-q, 0.)
        key = residual.tobytes()
        if key not in cache:
            try:
                cache[key] = POL.evaluador_dia(residual, ci, cv, pmin, pmax, f, cargo, estado, disponible=disponible)
            except ValueError as error:
                if 'No existe trayectoria térmica factible' not in str(error):
                    raise
                cache[key] = None
        return cache[key]

    cero = np.zeros(n)
    referencia = termico(cero)
    if referencia is None:
        raise ValueError('Referencia térmica infactible con el estado heredado')
    cupo = presupuesto_f(eps, b) if reserva else np.inf
    q25 = float(np.quantile(ci, .25))
    senal = np.zeros(n)
    for h in range(n):
        if delta[h] > POL.TOL_DEFICIT_MW:
            for fr in FRACCIONES:
                q = cero.copy()
                q[h] = fr*min(b['P_MW'], delta[h])
                r = termico(q)
                if r is not None:
                    senal[h] = max(senal[h], (referencia['objetivo']-r['objetivo'])/q[h])
        elif p_exp > 0:
            senal[h] = max(0., ce[h])
    q75 = float(np.quantile(senal, .75))
    candidatos = POL.candidatos_sistema(delta, ci, cv, pmin, pmax, cargos, referencia, b)
    for c in candidatos:
        c['valor_medio'] = c['valor_USD']/sum(c['q']) if c['tipo'] == 'U' else c['costo']
    for h in range(n):
        if delta[h] <= POL.TOL_DEFICIT_MW and ce[h] > 0 and p_exp > 0:
            candidatos.append(dict(tipo='X', hora=h, fuente='BESS', valor_medio=float(ce[h]),
                                   mw=min(b['P_MW'], p_exp)))
    candidatos = sorted((c for c in candidatos if c['valor_medio'] > 0 and c['valor_medio']+TOL_USD >= q75),
                        key=lambda c: (-c['valor_medio'], c['hora'], c['tipo'], c['fuente']))
    actual = trayectoria(cero, cero, cero, eps, delta, ce, e0, b, p_exp)
    assert actual['factible']
    def valorar(tr):
        if tr is None or not tr['factible']:
            return None
        if b['eta_c']*tr['cr'].sum() > cupo+POL.TOL_SOC_MWh:
            return None
        r = termico(tr['q'])
        if r is None:
            return None
        imp = r['potencia'][:, 4]+tr['cr']
        if ((imp > TOL) & (tr['xp']+tr['de'] > TOL)).any():
            return None
        objetivo = r['objetivo'] + float(ci@tr['cr'])-float(ce@(tr['xp']+tr['de']))
        return dict(tr, termica=r, objetivo=objetivo)
    actual = valorar(actual)
    inicial = actual['objetivo']
    aceptados, evaluados = [], 0
    for ic, c in enumerate(candidatos):
        mejor = actual
        for fr in ((1.,) if c['tipo'] == 'U' else FRACCIONES):
            q, de = actual['q'].copy(), actual['de'].copy()
            if c['tipo'] == 'U':
                horas = c['horas']
                # El bloque mantiene la descarga completa, sin duplicarla con segmentos ya aceptados.
                if (q[horas]+de[horas] > TOL).any():
                    continue
                q[horas] = c['q']
            elif c['tipo'] == 'S':
                h = c['hora']
                q[h] += fr*min(c['mw'], max(0., min(delta[h], b['P_MW'])-q[h]))
            else:
                h = c['hora']
                de[h] += fr*min(c['mw'], max(0., b['P_MW']-de[h]), max(0., p_exp-actual['xp'][h]-de[h]))
            base = trayectoria(q, de, actual['cr'], eps, delta, ce, e0, b, p_exp)
            opciones = [base]
            if base['primera_falta'] is not None:
                opciones.append(completar_compras(q, de, actual['cr'], eps, delta, ci, ce, e0, b, p_exp, cupo, q25))
            for opcion in opciones:
                candidato = valorar(opcion)
                evaluados += 1
                if candidato is None or candidato['objetivo'] >= actual['objetivo']-TOL_USD:
                    continue
                diferencia = candidato['objetivo']-mejor['objetivo']
                empate = abs(diferencia) <= TOL_USD
                desempate = lambda tr: (float(tr['cr'].sum()), int(tr['termica']['arranques'].sum()), tuple(tr['cr']))
                if diferencia < -TOL_USD or (empate and desempate(candidato) < desempate(mejor)):
                    mejor = candidato
        if mejor is not actual:
            aceptados.append(dict(candidato=ic, tipo=c['tipo'], hora=c['hora'], fuente=c['fuente'],
                                  costo_antes=actual['objetivo'], costo_despues=mejor['objetivo'],
                                  beneficio=actual['objetivo']-mejor['objetivo'],
                                  compra_red_MWh=float(mejor['cr'].sum()-actual['cr'].sum())))
            actual = mejor
    actual['diagnostico'] = dict(fecha=f[0], reserva=reserva, cupo_red_SOC_MWh=float(cupo),
                                 compra_SOC_MWh=float(b['eta_c']*actual['cr'].sum()),
                                 q25=q25, q75=q75, senal=senal, candidatos=len(candidatos),
                                 evaluaciones_termicas=len(cache), evaluaciones_planes=evaluados,
                                 costo_inicial=inicial, costo_final=actual['objetivo'], aceptados=aceptados)
    return actual


def simular_cf(fechas, pv, load, ci, ce, b, p_exp, P, cargo, estado_termico, soc_inicial, *, reserva=False,
               disponible=None):
    """Simulación con frontera explícita; el cierre cíclico es un procedimiento externo."""
    t0 = time.perf_counter()
    f, pv, load, ci, ce, x, delta, eps = entradas(fechas, pv, load, ci, ce, b, p_exp)
    lo, hi = b['E_MWh']*b['soc_min_pu'], b['E_MWh']*b['soc_max_pu']
    if not lo-1e-9 <= soc_inicial <= hi+1e-9:
        raise ValueError('SOC inicial fuera de límites')
    disp = DT.alinear(disponible, f)
    e, estado, planes, segmentos = float(soc_inicial), estado_termico, [], []
    for a, z in POL.segmentos_dias(f):
        p = planificar_dia(f[a:z], delta[a:z], eps[a:z], ci[a:z], ce[a:z], b, p_exp, P, cargo, estado, e, reserva,
                           disponible=None if disp is None else disp[a:z])
        planes.append(p)
        segmentos.append(dict(inicio=a, fin=z, estado_previo=estado, resultado=p['termica']))
        estado, e = p['termica']['estado_final'], float(p['soc'][-1])
    unir = lambda k: np.concatenate([p[k] for p in planes])
    termica = lambda k: np.concatenate([p['termica'][k] for p in planes])
    cp, cr = unir('cp'), unir('cr')
    return dict(Caso='F_INT' if reserva else 'C_INT', Tipo='politica', Fechas=f, PV=pv, Carga=load,
                PV_directo=x, Carga_PV=cp, Carga_Red=cr, Carga_BESS=cp+cr, Descarga_demanda=unir('q'),
                Descarga_export=unir('de'), Export_PV=unir('xp'), Vertimiento=unir('vert'), SOC=unir('soc'),
                SOC_inicial=float(soc_inicial), BESS=dict(b), Cargo=cargo, P_EXP_MAX=p_exp,
                C_imp_USD=ci, C_exp_USD=ce, potencia=termica('potencia'), compromiso=termica('compromiso'),
                arranques=termica('arranques'), paradas=termica('paradas'),
                Objetivo=math.fsum(p['objetivo'] for p in planes), segmentos=segmentos,
                dias_cf=[p['diagnostico'] for p in planes], estado_termico_inicial=estado_termico,
                estado_termico_final=estado, Segundos=time.perf_counter()-t0, Disponible=disp)


def ciclo_cf(*args, reserva=False, max_iter=25, tol=1e-4, disponible=None):
    """Mismo punto fijo y diagnósticos que B/D; solo el SOC es cíclico."""
    return POL.ciclo_soc_politica(lambda e: simular_cf(*args, e, reserva=reserva, disponible=disponible), args[5],
                                  max_iter=max_iter, tol=tol)


def optimo_cf(fechas, pv, load, ci, ce, b, p_exp, P, cargo=100., *, cargar_red=True,
              soc0=None, fijar=None, segundos=120, brecha=1e-6, disponible=None):
    """Optimización conjunta PyPSA; sin cuartiles ni presupuesto heurístico de F."""
    from termica_pypsa import resolver_pypsa
    f, pv, load, ci, ce, x, delta, eps = entradas(fechas, pv, load, ci, ce, b, p_exp)
    if f[0].month != 1 or f[0].day != 1 or f[0].hour != 0:
        raise ValueError('Esta interfaz de benchmark requiere iniciar en enero con estado térmico libre')
    bs = dict(b)
    if soc0 is not None:
        lo, hi = b['E_MWh']*b['soc_min_pu'], b['E_MWh']*b['soc_max_pu']
        if not lo-1e-9 <= soc0 <= hi+1e-9:
            raise ValueError('SOC fijo fuera de límites')
        bs['soc0_pu'] = min(b['soc_max_pu'], max(b['soc_min_pu'], soc0/b['E_MWh']))
    solar = dict(excedente=eps, c_exp=ce, p_exp_max=p_exp, bess=bs, soc='ciclico' if soc0 is None else 'fijo',
                 exportar_bess=True, exclusion='binaria', cargar_red=cargar_red)
    r = resolver_pypsa(delta, ci, P['cv'], P['pmin'], P['pmax'], f, cargo, limite_segundos=segundos,
                       brecha=brecha, aceptar_limite_tiempo=True, solar=solar, fijar=fijar,
                       disponible=DT.alinear(disponible, f))
    cr = r.get('solar_carga_red', np.zeros(len(f)))
    cp = r['solar_carga']
    return dict(Caso='OPT_CF' if cargar_red else 'OPT_D_regresion', Tipo='benchmark', Fechas=f, PV=pv, Carga=load,
                PV_directo=x, Carga_PV=cp, Carga_Red=cr, Carga_BESS=cp+cr, Descarga_demanda=r['solar_descarga_demanda'],
                Descarga_export=r['solar_descarga_export'], Export_PV=r['solar_export_pv'], Vertimiento=r['solar_vertido'],
                SOC=r['soc'], SOC_inicial=r['soc_inicial'], BESS=dict(b), Cargo=cargo, P_EXP_MAX=p_exp,
                C_imp_USD=ci, C_exp_USD=ce, potencia=r['potencia'], compromiso=r['compromiso'], arranques=r['arranques'],
                paradas=r['paradas'], Objetivo=r['objetivo'], Cota=r['cota_dual'], Brecha_solver=r['brecha'],
                Condicion=f"{r['estado_solver']}/{r['condicion_solver']}", Segundos=r['segundos'],
                estado_termico_inicial='libre', Fijado=r['solar_fijado'], Disponible=r.get('disponible'))


def validar_cf(r, P, *, exigir_cierre=True):
    """Auditoría independiente por rutas, calendario, eventos, presupuesto y reconstrucción económica."""
    b = r['BESS']
    f, pv, load, ci, ce, x, delta, eps = entradas(r['Fechas'], r['PV'], r['Carga'], r['C_imp_USD'], r['C_exp_USD'], b, r['P_EXP_MAX'])
    cp, cr, q, de, xp, vert, soc = [np.asarray(r[k], float) for k in
                                    ('Carga_PV', 'Carga_Red', 'Descarga_demanda', 'Descarga_export', 'Export_PV', 'Vertimiento', 'SOC')]
    n = len(f)
    assert all(v.shape == (n,) and np.isfinite(v).all() for v in (cp, cr, q, de, xp, vert, soc))
    assert all((v >= -TOL).all() for v in (cp, cr, q, de, xp, vert))
    p = np.asarray(r['potencia'], float)
    assert p.shape == (n, 5) and np.isfinite(p).all() and (p >= -TOL).all()
    u, arr, par = [POL._binario(r[k], k, (n, 4)) for k in ('compromiso', 'arranques', 'paradas')]
    np.testing.assert_allclose(r['PV_directo'], x, rtol=0, atol=TOL)
    np.testing.assert_allclose(r['Carga_BESS'], cp+cr, rtol=0, atol=TOL)
    balance_pv = float(np.max(np.abs(eps-cp-xp-vert)))
    balance_dem = float(np.max(np.abs(delta-q-p.sum(axis=1))))
    assert balance_pv <= TOL and balance_dem <= TOL, 'balance por origen/destino'
    assert (cp+cr <= b['P_MW']+TOL).all() and (q+de <= b['P_MW']+TOL).all()
    assert not ((cp+cr > TOL) & (q+de > TOL)).any(), 'carga y descarga simultáneas'
    assert not ((p[:,4]+cr > TOL) & (xp+de > TOL)).any(), 'importación y exportación simultáneas'
    assert not ((de > TOL) & (delta > POL.TOL_DEFICIT_MW)).any(), 'exportación durante déficit'
    assert (xp+de <= r['P_EXP_MAX']+TOL).all()
    previo = np.r_[r['SOC_inicial'], soc[:-1]]
    error_soc = float(np.max(np.abs(soc-previo-b['eta_c']*(cp+cr)+(q+de)/b['eta_d'])))
    assert error_soc <= POL.TOL_SOC_MWh
    assert (soc >= b['E_MWh']*b['soc_min_pu']-POL.TOL_SOC_MWh).all() and (soc <= b['E_MWh']*b['soc_max_pu']+POL.TOL_SOC_MWh).all()
    cierre = float(soc[-1]-r['SOC_inicial'])
    if exigir_cierre:
        assert abs(cierre) <= 1e-3, 'cierre del SOC'
    libre, u0, a0, b0, _ = POL._estado_previo(r['estado_termico_inicial'], f)
    cambio = np.diff(np.vstack([u[0] if libre else u0, u]), axis=0)
    np.testing.assert_array_equal(arr, (cambio == 1).astype(int))
    np.testing.assert_array_equal(par, (cambio == -1).astype(int))
    assert (p[:,:4] >= u*P['pmin']-TOL).all() and (p[:,:4] <= u*P['pmax']+TOL).all()
    max_eventos = 0
    for i, (a, z) in enumerate(POL.segmentos_dias(f)):
        na, np_ = arr[a:z].sum(axis=0)+(a0 if i == 0 else 0), par[a:z].sum(axis=0)+(b0 if i == 0 else 0)
        assert (na <= 1).all() and (np_ <= 1).all()
        max_eventos = max(max_eventos, int(max(na.max(), np_.max())))
        if r['Caso'] == 'F_INT':
            assert b['eta_c']*cr[a:z].sum() <= presupuesto_f(eps[a:z], b)+POL.TOL_SOC_MWh
    if r['Tipo'] == 'politica':
        assert not ((cr > TOL) & (eps > POL.TOL_DEFICIT_MW)).any()
        assert (ce[de > TOL] > 0).all()
        for seg, dia in zip(r['segmentos'], r['dias_cf']):
            a, z = seg['inicio'], seg['fin']
            disp = r.get('Disponible')
            POL.validar_evaluador(seg['resultado'], delta[a:z]-q[a:z], ci[a:z], P['cv'], P['pmin'], P['pmax'],
                                 f[a:z], r['Cargo'], seg['estado_previo'],
                                 disponible=None if disp is None else np.asarray(disp)[a:z])
            for acc in dia['aceptados']:
                assert acc['beneficio'] > TOL_USD
            assert abs(dia['compra_SOC_MWh']-b['eta_c']*cr[a:z].sum()) <= POL.TOL_SOC_MWh
    cargos = np.broadcast_to(np.asarray(r['Cargo'], float), (4,))
    costo = math.fsum(np.r_[p[:,:4]@P['cv'], (p[:,4]+cr)*ci, arr@cargos, -(xp+de)*ce])
    assert abs(costo-r['Objetivo']) <= 1e-5, ('reconstrucción de costo', costo-r['Objetivo'])
    return dict(Estado='OK', Residuo_PV_MW=balance_pv, Residuo_demanda_MW=balance_dem, Residuo_SOC_MWh=error_soc,
                Cierre_SOC_MWh=cierre, Error_costo_USD=costo-r['Objetivo'], Eventos_dia_max=max_eventos)


def fijacion(r):
    return dict(carga=r['Carga_PV'], carga_red=r['Carga_Red'], descarga_demanda=r['Descarga_demanda'],
                descarga_export=r['Descarga_export'], export_pv=r['Export_PV'], soc=r['SOC'],
                potencia=r['potencia'], compromiso=r['compromiso'])


def indicadores(r, P):
    p, ci, ce, b = r['potencia'], r['C_imp_USD'], r['C_exp_USD'], r['BESS']
    return dict(Caso=r['Caso'], Horas=len(r['Fechas']), Costo_USD=r['Objetivo'],
                Costo_termico_USD=float((p[:,:4]@P['cv']).sum()),
                Costo_red_demanda_USD=float(p[:,4]@ci), Costo_red_BESS_USD=float(r['Carga_Red']@ci),
                Costo_arranques_USD=float((r['arranques']@np.broadcast_to(r['Cargo'], (4,))).sum()),
                Ingreso_exportacion_USD=float((r['Export_PV']+r['Descarga_export'])@ce),
                Arranques=int(r['arranques'].sum()), Termica_MWh=float(p[:,:4].sum()), Red_demanda_MWh=float(p[:,4].sum()),
                Carga_red_MWh=float(r['Carga_Red'].sum()), Carga_PV_MWh=float(r['Carga_PV'].sum()),
                Descarga_demanda_MWh=float(r['Descarga_demanda'].sum()), Descarga_export_MWh=float(r['Descarga_export'].sum()),
                Export_PV_MWh=float(r['Export_PV'].sum()), Vertimiento_MWh=float(r['Vertimiento'].sum()),
                SOC_inicial_MWh=r['SOC_inicial'], SOC_final_MWh=float(r['SOC'][-1]),
                Cota_USD=r.get('Cota', np.nan), Brecha_solver=r.get('Brecha_solver', np.nan),
                Condicion=r.get('Condicion', 'politica'), Segundos=r['Segundos'])
