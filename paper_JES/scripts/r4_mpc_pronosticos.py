"""Experimento R4: control predictivo de 48 h (MPC-48) con pronósticos, sobre C280 (K3 en el repositorio).

Variante diaria (`--ejecucion plan` u `holgura`):

Cada día calendario d, a las 00:00, se resuelve el MILP de ventana de R1 (48 h, conjunto de acciones B: carga
solo desde PV, sin exportación del BESS) con PRONÓSTICOS de demanda, PV y precios para d y d+1, el SOC real y el
compromiso térmico real al cierre de d−1. Del plan solo se toma la descarga a demanda de las 24 h de d.

Ejecución sobre los valores reales, con la misma estructura que B_INT:
- PV directo primero y carga voraz del excedente solar real (`trayectoria_soc` de las reglas);
- descarga q_t = min(plan_t, déficit real_t, potencia, energía disponible sobre el mínimo);
- térmica y red por `evaluador_dia` (gestor térmico común a todas las reglas) sobre el residual real del día;
- el excedente PV no almacenado se exporta con precio positivo y dentro del techo, o se vierte.
Así el MPC se evalúa igual que las reglas: solo cambian la información y la lógica de las decisiones del BESS.
Con `holgura`, la batería cubre el déficit real que exceda el gas y la red planificados, en vez de la descarga planificada.

Variante horaria (`--ejecucion horario`, la del artículo): la ventana se re-optimiza cada hora con persistencia
rodante, se aplica solo la primera hora y la batería actúa como holgura del suministro planificado
(`simular_mpc_horario`).

Pronósticos (familia de calidad α): f = real + α·(persistencia − real), donde la persistencia es el valor real de
la misma hora del día d−1 (el último día completo observado), usado para d y para d+1.
α = 0 es previsión perfecta y α = 1 persistencia pura. La señal FLAT es una tarifa diaria fija, así que su
persistencia coincide con el valor real; con WMI se pronostica el precio de bolsa, y con ambas el de exportación.

El año se trata como cíclico, igual que los benchmarks anuales: la ventana del 31/12 continúa con el 1/1 y el
pronóstico del 1/1 persiste el 31/12. El SOC inicial es el punto fijo cíclico (`ciclo_soc_politica`, como B_INT)
y la trayectoria final se certifica en OPT-B (pertenencia).

Uso:  OperacionBESS/.venv/bin/python paper_JES/scripts/r4_mpc_pronosticos.py --caso 1.0 T0 [--ejecucion horario]
"""
from __future__ import annotations

import argparse
import logging
import math
import time
import traceback
from collections import Counter

import comun as C
from comun import V, REG, np, pd

import r1_horizonte_rodante as R1  # noqa: E402
import termica_politicas as POL  # noqa: E402

H = 48
CASO_BENCH = 'OPT_AB'
SERIES_PRONOSTICO = ('Carga', 'PV', 'C_imp', 'C_exp')
# Ejecución del plan: descarga planificada ('plan') o batería como holgura del suministro planificado ('holgura').
PREFIJO = {'plan': 'MPC48', 'holgura': 'MPC48S', 'horario': 'MPC48H'}


def pronostico(x, t0, alfa, T, horas=H):
    """Valores real y pronosticado de la serie anual `x` en la ventana [t0, t0+horas) (índices cíclicos).

    Persistencia: hora h de la ventana → valor real de la misma hora del día anterior a t0."""
    idx = (t0 + np.arange(horas)) % T
    pers = (t0 - 24 + np.arange(horas) % 24) % T
    real = x[idx]
    return real, real + alfa*(x[pers] - real)


def fechas_ventana(f, t0, horas=H):
    """Fechas continuas de la ventana; las horas que pasan del 31/12 se rotulan en el año siguiente."""
    T = len(f)
    idx = t0 + np.arange(horas)
    return pd.DatetimeIndex([f[i] if i < T else f[i - T] + pd.DateOffset(years=1) for i in idx])


def ejecutar_dia(plan, delta, eps, e, b, suministro=None):
    """Ejecuta el plan del día sobre el déficit y el excedente reales.

    Sin `suministro`, la batería entrega la descarga planificada (recortada si no hay déficit o energía).
    Con `suministro` (gas + red planificados), la batería actúa como holgura: cubre lo que el déficit real
    exceda ese suministro, dentro de su potencia y energía."""
    e_min, e_max = b['E_MWh']*b['soc_min_pu'], b['E_MWh']*b['soc_max_pu']
    n = len(delta)
    q, c, soc = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        if eps[t] > 0:
            c[t] = min(eps[t], b['P_MW'], max(e_max - e, 0.)/b['eta_c'])
        disponible = max(e + b['eta_c']*c[t] - e_min, 0.)*b['eta_d']
        objetivo = max(plan[t], 0.) if suministro is None else max(delta[t] - suministro[t], 0.)
        q[t] = min(objetivo, delta[t], b['P_MW'], disponible)
        e = e + b['eta_c']*c[t] - q[t]/b['eta_d']
        soc[t] = e
    return q, c, soc


def simular_mpc(alfa, datos, m, soc_inicial, ejecucion='plan'):
    f, pv, load, ci, ce = datos
    p, b, pe, cargo = m['parametros_termicos'], m['bess'], m['P_EXP_MAX'], m['cargo_arranque_USD']
    fechas = POL._validar_calendario(f)
    T = len(f)
    cv, pmin, pmax = (np.asarray(p[k], float) for k in ('cv', 'pmin', 'pmax'))
    cargos = np.broadcast_to(np.asarray(cargo, float), (4,))
    x = np.minimum(pv, load)
    delta, eps = load - x, pv - x
    q, c, soc, plan_q = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    segmentos, registro = [], []
    estado, e, u_prev = 'libre', float(soc_inicial), None
    inicio = time.perf_counter()
    for a, z in POL.segmentos_dias(fechas):
        tic = time.perf_counter()
        _, load_f = pronostico(load, a, alfa, T)
        _, pv_f = pronostico(pv, a, alfa, T)
        _, ci_f = pronostico(ci, a, alfa, T)
        _, ce_f = pronostico(ce, a, alfa, T)
        load_f, pv_f = np.maximum(load_f, 0.), np.maximum(pv_f, 0.)
        x_f = np.minimum(pv_f, load_f)
        w = R1.ventana(fechas_ventana(f, a), load_f - x_f, pv_f - x_f, ce_f, ci_f, p, b, pe, cargo, False,
                       e, u_prev, None)
        plan_q[a:z] = w['desc_dem'][:z - a]
        suministro = None
        if ejecucion == 'holgura':
            suministro = np.maximum((load_f - x_f)[:z - a] - plan_q[a:z], 0.)
        q[a:z], c[a:z], soc[a:z] = ejecutar_dia(plan_q[a:z], delta[a:z], eps[a:z], e, b, suministro)
        e = float(soc[z - 1])
        residual = np.maximum(delta[a:z] - q[a:z], 0.)
        ev = POL.evaluador_dia(residual, ci[a:z], cv, pmin, pmax, fechas[a:z], cargos, estado)
        segmentos.append(dict(inicio=a, fin=z, estado_previo=estado, resultado=ev, segundos=time.perf_counter() - tic))
        registro.append(dict(dia=len(registro) + 1, t0=a, condicion=w['condicion'], brecha=w['brecha'],
                             descarga_plan_MWh=float(plan_q[a:z].sum()), descarga_real_MWh=float(q[a:z].sum())))
        estado = ev['estado_final']
        u_prev = np.asarray(ev['compromiso'][-1], int)
    sobrante = eps - c
    xpv = np.where(ce > 0, np.minimum(sobrante, pe), 0.)
    vert = sobrante - xpv
    pot = np.concatenate([s['resultado']['potencia'] for s in segmentos])
    arr = np.concatenate([s['resultado']['arranques'] for s in segmentos])
    costo_termica = math.fsum(s['resultado']['objetivo'] for s in segmentos)
    ingreso = math.fsum((xpv*ce).tolist())
    nombre = f'{PREFIJO[ejecucion]}_a{alfa:g}'
    return dict(Caso=nombre, Tipo='politica', Fechas=fechas, PV=pv, Carga=load, PV_directo=x, Carga_BESS=c,
                Descarga_demanda=q, Descarga_export=np.zeros(T), Export_PV=xpv, Vertimiento=vert, SOC=soc,
                SOC_inicial=float(soc_inicial), Frontera_SOC=dict(tipo='ciclica_mpc', valor=float(soc_inicial)), BESS=b,
                Residual=np.maximum(delta - q, 0.), potencia=pot,
                compromiso=np.concatenate([s['resultado']['compromiso'] for s in segmentos]), arranques=arr,
                paradas=np.concatenate([s['resultado']['paradas'] for s in segmentos]), C_imp_USD=ci, C_exp_USD=ce,
                Cargo=cargo, P_EXP_MAX=pe,
                Costo_combustible_CVOM_USD=math.fsum((pot[:, :4]@cv).tolist()),
                Costo_red_USD=math.fsum((pot[:, 4]*ci).tolist()), Costo_arranques_USD=math.fsum((arr@cargos).tolist()),
                Costo_termica_red_arranques_USD=costo_termica, Ingreso_exportacion_USD=ingreso,
                Objetivo=costo_termica - ingreso, segmentos=segmentos, estado_termico_inicial='libre',
                estado_termico_final=estado, Disponible=None, Plan_descarga=plan_q, registro=registro,
                Segundos_simulacion=time.perf_counter() - inicio)


def pronostico_rodante(x, t, alfa, T, horas=H):
    """Persistencia rodante: cada hora de la ventana toma el último valor observado de la misma hora del día
    (24 h antes, o 48 h si la hora de 24 h antes aún no ocurrió). A las 00:00 coincide con `pronostico`."""
    i = t + np.arange(horas)
    pers = (i - 24*((i - t)//24 + 1)) % T
    real = x[i % T]
    return real, real + alfa*(x[pers] - real)


def simular_mpc_horario(alfa, datos, m, soc_inicial, previo=None):
    """MPC-48 re-optimizado cada hora con persistencia rodante; batería como holgura del suministro planificado.

    El modelo de planificación hereda dentro del día el compromiso y los eventos que planificó para la hora
    anterior, y a medianoche el compromiso realizado por el gestor térmico. Si una ventana resulta infactible con
    ese estado heredado, se resuelve con compromiso inicial libre (respaldo registrado).
    `previo` (iteración anterior del punto fijo): si al empezar un día el SOC y el estado térmico coinciden con los
    de esa iteración, el resto del año es idéntico y se reutiliza."""
    f, pv, load, ci, ce = datos
    p, b, pe, cargo = m['parametros_termicos'], m['bess'], m['P_EXP_MAX'], m['cargo_arranque_USD']
    fechas = POL._validar_calendario(f)
    T = len(f)
    cv, pmin, pmax = (np.asarray(p[k], float) for k in ('cv', 'pmin', 'pmax'))
    cargos = np.broadcast_to(np.asarray(cargo, float), (4,))
    x = np.minimum(pv, load)
    delta, eps = load - x, pv - x
    q, c, soc, plan_q = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    segmentos, registro, inicios = [], [], []
    estado, e, u_real = 'libre', float(soc_inicial), None
    inicio = time.perf_counter()
    dias = POL.segmentos_dias(fechas)
    for n_dia, (a, z) in enumerate(dias):
        inicios.append(dict(soc=e, estado=estado))
        if previo is not None:
            pi = previo['Inicios'][n_dia]
            if abs(pi['soc'] - e) <= 1e-9 and POL._mismo_estado(pi['estado'], estado):
                for k, v in (('q', q), ('c', c), ('soc', soc), ('plan_q', plan_q)):
                    v[a:] = previo['Series_MPC'][k][a:]
                segmentos += previo['segmentos'][n_dia:]
                registro += previo['registro'][n_dia:]
                inicios += previo['Inicios'][n_dia + 1:]
                estado = previo['estado_termico_final']
                break
        tic = time.perf_counter()
        u_plan, usados = u_real, None
        condiciones, brecha, respaldos = Counter(), 0., 0
        for t in range(a, z):
            _, load_f = pronostico_rodante(load, t, alfa, T)
            _, pv_f = pronostico_rodante(pv, t, alfa, T)
            _, ci_f = pronostico_rodante(ci, t, alfa, T)
            _, ce_f = pronostico_rodante(ce, t, alfa, T)
            load_f, pv_f = np.maximum(load_f, 0.), np.maximum(pv_f, 0.)
            x_f = np.minimum(pv_f, load_f)
            args = (fechas_ventana(f, t), load_f - x_f, pv_f - x_f, ce_f, ci_f, p, b, pe, cargo, False, e)
            try:
                w = R1.ventana(*args, u_plan, None, usados=usados)
            except R1.Infactible:
                w = R1.ventana(*args, None, None)
                respaldos += 1
            condiciones[w['condicion']] += 1
            brecha = max(brecha, w['brecha'])
            plan_q[t] = w['desc_dem'][0]
            suministro = np.array([max((load_f - x_f)[0] - plan_q[t], 0.)])
            qt, ct, st = ejecutar_dia(plan_q[t:t+1], delta[t:t+1], eps[t:t+1], e, b, suministro)
            q[t], c[t], soc[t] = qt[0], ct[0], st[0]
            e = float(soc[t])
            u_t = np.asarray(w['compromiso'][0], int)
            previo_u = u_t if u_plan is None else u_plan
            ev_dia = np.zeros((2, 4), int) if usados is None else np.array(usados, int)
            # Tras un respaldo el plan puede superar el límite diario: los contadores se acotan a 1.
            usados = (np.minimum(ev_dia[0] + (u_t > previo_u), 1), np.minimum(ev_dia[1] + (u_t < previo_u), 1))
            u_plan = u_t
        residual = np.maximum(delta[a:z] - q[a:z], 0.)
        ev = POL.evaluador_dia(residual, ci[a:z], cv, pmin, pmax, fechas[a:z], cargos, estado)
        segmentos.append(dict(inicio=a, fin=z, estado_previo=estado, resultado=ev, segundos=time.perf_counter() - tic))
        registro.append(dict(dia=n_dia + 1, t0=a, condiciones=dict(condiciones), brecha=brecha, respaldos=respaldos,
                             descarga_plan_MWh=float(plan_q[a:z].sum()), descarga_real_MWh=float(q[a:z].sum())))
        estado = ev['estado_final']
        u_real = np.asarray(ev['compromiso'][-1], int)
    r = cerrar_trayectoria(f'MPC48H_a{alfa:g}', datos, m, soc_inicial, q, c, soc, plan_q, segmentos, estado, inicio)
    r.update(registro=registro, Inicios=inicios, Series_MPC=dict(q=q, c=c, soc=soc, plan_q=plan_q))
    return r


def cerrar_trayectoria(nombre, datos, m, soc_inicial, q, c, soc, plan_q, segmentos, estado, inicio):
    """Exportación PV, costos y diccionario de resultado con el formato de las políticas (B_INT)."""
    f, pv, load, ci, ce = datos
    p, b, pe, cargo = m['parametros_termicos'], m['bess'], m['P_EXP_MAX'], m['cargo_arranque_USD']
    cv = np.asarray(p['cv'], float)
    cargos = np.broadcast_to(np.asarray(cargo, float), (4,))
    T = len(f)
    x = np.minimum(pv, load)
    delta, eps = load - x, pv - x
    sobrante = eps - c
    xpv = np.where(ce > 0, np.minimum(sobrante, pe), 0.)
    vert = sobrante - xpv
    pot = np.concatenate([s['resultado']['potencia'] for s in segmentos])
    arr = np.concatenate([s['resultado']['arranques'] for s in segmentos])
    costo_termica = math.fsum(s['resultado']['objetivo'] for s in segmentos)
    ingreso = math.fsum((xpv*ce).tolist())
    return dict(Caso=nombre, Tipo='politica', Fechas=pd.DatetimeIndex(f), PV=pv, Carga=load, PV_directo=x,
                Carga_BESS=c, Descarga_demanda=q, Descarga_export=np.zeros(T), Export_PV=xpv, Vertimiento=vert,
                SOC=soc, SOC_inicial=float(soc_inicial), Frontera_SOC=dict(tipo='ciclica_mpc', valor=float(soc_inicial)),
                BESS=b, Residual=np.maximum(delta - q, 0.), potencia=pot,
                compromiso=np.concatenate([s['resultado']['compromiso'] for s in segmentos]), arranques=arr,
                paradas=np.concatenate([s['resultado']['paradas'] for s in segmentos]), C_imp_USD=ci, C_exp_USD=ce,
                Cargo=cargo, P_EXP_MAX=pe,
                Costo_combustible_CVOM_USD=math.fsum((pot[:, :4]@cv).tolist()),
                Costo_red_USD=math.fsum((pot[:, 4]*ci).tolist()), Costo_arranques_USD=math.fsum((arr@cargos).tolist()),
                Costo_termica_red_arranques_USD=costo_termica, Ingreso_exportacion_USD=ingreso,
                Objetivo=costo_termica - ingreso, segmentos=segmentos, estado_termico_inicial='libre',
                estado_termico_final=estado, Disponible=None, Plan_descarga=plan_q,
                Segundos_simulacion=time.perf_counter() - inicio)


def errores_pronostico(datos, alfa):
    """nMAE del pronóstico (media |f − real| / media real) para las 24 h ejecutadas y para la ventana de 48 h."""
    f, pv, load, ci, ce = datos
    T = len(f)
    salida = {}
    for nombre, x in zip(SERIES_PRONOSTICO, (load, pv, ci, ce)):
        r48, f48 = [], []
        for t0 in range(0, T, 24):
            real, pr = pronostico(x, t0, alfa, T)
            r48.append(real)
            f48.append(pr)
        r48, f48 = np.array(r48), np.array(f48)
        for clave, sl in (('24h', slice(0, 24)), ('48h', slice(0, 48))):
            err = np.abs(f48[:, sl] - r48[:, sl]).mean()
            salida[f'nMAE_{nombre}_{clave}'] = float(err/np.abs(r48[:, sl]).mean())
    return salida


def ejecutar(alfa, senal, ejecucion='plan'):
    carpeta = C.SALIDA / 'R4'
    etiqueta = f'R4_{PREFIJO[ejecucion]}_a{alfa:g}_{senal}'
    js = carpeta / 'casos' / f'{etiqueta}.json'
    if js.exists():
        return C.leer_json(js)
    for nombre in ('pypsa', 'linopy'):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    tic = time.perf_counter()
    datos, m = C.contexto_c280(senal)
    p, b = m['parametros_termicos'], m['bess']
    if ejecucion == 'horario':
        anterior = []

        def simular(e0):
            r = simular_mpc_horario(alfa, datos, m, e0, previo=anterior[-1] if anterior else None)
            anterior.append(r)
            return r
    else:
        def simular(e0):
            return simular_mpc(alfa, datos, m, e0, ejecucion)
    ciclo = POL.ciclo_soc_politica(simular, b, max_iter=25, tol=1e-4)
    if ciclo['estado'] != 'CONVERGE':
        raise RuntimeError(f'MPC-48 α={alfa}: SOC {ciclo["estado"]}')
    r = ciclo['resultado']
    val = POL.validar_politica(r, p)
    r['Cota'] = r['Objetivo']
    val_int = REG.validar_integrado(r, p, False)
    V.contabilidad(r, p)
    ind = C.indicadores(r, p, m['cargo_arranque_USD'])
    assert abs(ind['Costo_operativo_USD'] - r['Objetivo']) <= 1e-6, 'costo de indicadores distinto del objetivo'
    reg = r['registro']
    if ejecucion == 'horario':
        cond = sum((Counter(x['condiciones']) for x in reg), Counter())
        cond = dict(cond, respaldos=sum(x['respaldos'] for x in reg))
    else:
        cond = pd.Series([x['condicion'] for x in reg]).value_counts().to_dict()
    plan, real = r['Plan_descarga'], r['Descarga_demanda']
    fila = dict(Refuerzo='R4', Escenario=PREFIJO[ejecucion], Ejecucion=ejecucion, Alfa=alfa, Accion='B', Senal=senal, Caso=r['Caso'],
                Benchmark=CASO_BENCH, Estado='OK', Horizonte_h=H,
                Ventanas=len(reg) if ejecucion != 'horario' else sum(v for k, v in cond.items() if k != 'respaldos'), Condiciones_ventanas=cond,
                Brecha_ventana_max=float(max(x['brecha'] for x in reg)), Segundos=time.perf_counter() - tic,
                Segundos_limite_ventana=R1.SEG_VENTANA, Brecha_solicitada_ventana=R1.BRECHA_VENTANA,
                Iteraciones_ciclo=ciclo['iteraciones'], Residuo_ciclo_MWh=ciclo['residuo'],
                Descarga_plan_MWh=float(plan.sum()), Descarga_ejecutada_MWh=float(real.sum()),
                Descarga_recortada_MWh=float(np.maximum(plan - real, 0.).sum()),
                Horas_recorte=int((plan - real > 1e-6).sum()),
                Validacion=val, Validacion_integrada=val_int, Objetivo_USD=r['Objetivo'], **ind,
                **errores_pronostico(datos, alfa))
    fila.update(C.pertenencia(r, CASO_BENCH, datos, m, p))
    fila = C.guardar_caso(carpeta, etiqueta, r, fila)
    C.json_guardar(carpeta / 'casos' / f'{etiqueta}_ventanas.json', reg)
    return fila


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--caso', nargs=2, metavar=('ALFA', 'SENAL'), required=True)
    ap.add_argument('--ejecucion', choices=tuple(PREFIJO), default='plan')
    a = ap.parse_args()
    alfa, senal = float(a.caso[0]), a.caso[1]
    assert 0. <= alfa <= 1. and senal in ('T0', 'T1')
    try:
        fila = ejecutar(alfa, senal, a.ejecucion)
        print('OK', fila['Caso'], senal, round(fila['Costo_operativo_USD'], 2), flush=True)
    except Exception as exc:
        (C.SALIDA / 'R4' / 'errores').mkdir(parents=True, exist_ok=True)
        C.json_guardar(C.SALIDA / 'R4' / 'errores' / f'R4_{PREFIJO[a.ejecucion]}_a{alfa:g}_{senal}.json',
                       dict(caso=a.caso, error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
