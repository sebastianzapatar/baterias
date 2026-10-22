"""Diagnóstico acotado: apagado anticipado de unidades térmicas heredadas del día anterior en B_INT (K3, año 1).

No modifica B_INT ni D_INT ni propone todavía una regla. Estudia una limitación conocida de B_INT: sus candidatos
(S) sustituyen generación por encima de los mínimos y sus bloques (U) solo retiran unidades que ARRANCAN en el día;
una unidad que llega encendida desde la víspera no tiene un candidato explícito para apagarse antes.

1. Frecuencia. En las trayectorias anuales guardadas de B_INT (K3, año 1) se cuentan los pares (día, unidad) con:
   N1 unidad heredada encendida a las 00:00 (encendida a las 23:00 del día anterior y a las 00:00);
   N2 además en su mínimo o cerca (≤ 1,05·Pmin) en la primera hora del día;
   N3 además con potencia libre del BESS y SOC por encima del mínimo en esa hora;
   N4 además con descarga adicional factible (SOC cronológico, eficiencias y recarga solar voraz) que cubra toda la
      generación de la unidad durante sus horas iniciales cerca del mínimo, sin reasignar las descargas ya planeadas.
2. Ventanas. Seis situaciones representativas, T0 y T1, carbono 0 e intermedio. Regla fija: la situación de
   energía mediana de cada clase: N4 y N3 sin N4 (requiere reasignar energía) en T0 y T1 con 0 USD/t; N4 en T0 con
   10 USD/t y en T1 con 20 USD/t (bajo T0 con 20 USD/t la térmica ya no opera).
3. Alternativas para cada ventana, con el mismo `evaluador_dia` y el mismo estado heredado:
   ALT1 apagado forzado de la unidad desde la hora h_s hasta el final de su bloque heredado, con descarga adicional
        del BESS min(P − q, p_unidad) limitada por el SOC cronológico sin reasignar el plan; se elige el mejor h_s;
   ALT2 un candidato de apagado heredado (sin crédito de arranque: el arranque ya ocurrió) añadido a los candidatos
        de B y seleccionado con la misma regla de densidad de B (reasigna energía); despacho final libre, como B.
   Los días siguientes se simulan con B_INT sin cambios desde el estado que deja cada alternativa, hasta que SOC y
   compromiso térmico al final de un día coinciden con la trayectoria original (frontera común) o hasta 7 días.
   Si no se reencuentran, la diferencia se informa como NO comparable, con el SOC y el estado que quedan distintos.

Costos por ventana: ajustado (el que optimiza la regla), financiero y emisiones (CO₂ térmico + CO₂e de red, XM
2025), con compras de red, generación de las demás unidades, arranques e ingreso de exportación PV. No se extrapola
a un ahorro anual.

Uso, desde la raíz: `python diagnostico_apagado_b.py --salida resultados_carbono_v2/diagnostico_apagado_b_20260926`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import carbono as C
import disponibilidad_termica as DT
import termica_carbono_anual as TA
import termica_politicas as POL
import termica_reglas as REG
from termica import despacho_capacidad_eventos

K3_VPN = Path('resultados_carbono_v2/vpn_K3_20260926')
MALLA = Path('resultados_carbono_v2/malla_20260926/M01')
ESCENARIOS = (('V2_p0', 'T0'), ('V2_p0', 'T1'), ('V2_p10', 'T0'), ('V2_p10', 'T1'), ('V2_p20', 'T0'),
              ('V2_p20', 'T1'), ('V2_p50', 'T1'))
CERCA_MIN = 1.05
MAX_DIAS = 7
TOL_SOC = 1e-6


def fuente(n, s):
    base = K3_VPN if n in ('V2_p0', 'V2_p50') else MALLA
    return base/n/f'{s}_B_INT_a01.npz'


def preparar(n, s):
    """Entradas del escenario (mismas que el ejecutor) y trayectoria anual guardada de B_INT."""
    ctx = TA.contexto()
    esc, con_disp = TA.escenario(n)
    P = C.parametros_ajustados(ctx['P'], esc)
    pv, b, pe, _ = TA.dimensionar(None)
    trm = ctx['TRM']
    load = np.asarray(ctx['E']['LOAD'], float)
    f = ctx['fechas']
    ci_fin = np.asarray(ctx['C_IMP_COP'][s], float)/trm
    ci, ce = C.senal_ajustada(ci_fin, esc), np.asarray(ctx['C_EXP_COP'], float)/trm
    disp = DT.alinear(ctx['disp_v2'].to_numpy() if con_disp else None, f)
    x = np.minimum(pv, load)
    ruta = fuente(n, s)
    with np.load(ruta) as z:
        tray = {k: z[k] for k in ('potencia', 'compromiso', 'SOC', 'SOC_inicial', 'Descarga_demanda', 'Carga_BESS',
                                  'Export_PV', 'PV')}
    assert np.array_equal(tray['PV'], pv), 'la trayectoria no es de K3 en el año 1'
    return dict(n=n, s=s, esc=esc, P=P, b=b, pe=pe, f=f, load=load, pv=pv, ci=ci, ci_fin=ci_fin, ce=ce, disp=disp,
                delta=load - x, eps=pv - x, cv=np.asarray(P['cv'], float), cv_fin=np.asarray(P['cv_financiero'], float),
                pmin=np.asarray(P['pmin'], float), pmax=np.asarray(P['pmax'], float),
                cargos=np.broadcast_to(np.asarray(REG.CARGO, float), (4,)).copy(),
                intensidad=C.intensidad_termica_t_MWh(P, esc), fe_red=esc.fe_red_t_MWh, tray=tray, ruta=str(ruta),
                sha=hashlib.sha256(ruta.read_bytes()).hexdigest())


def estado_inicio(c, d):
    a = 24*d
    return dict(u=c['tray']['compromiso'][a - 1].astype(int).copy(), arranques_usados=np.zeros(4, int),
                paradas_usadas=np.zeros(4, int), fecha=c['f'][a - 1])


def soc_inicio(c, d):
    return float(c['tray']['SOC'][24*d - 1])


def _extra_factible(q, eps, soc0, b, objetivo):
    """Descarga adicional hora a hora, en orden cronológico, lo mayor posible hasta `objetivo` con SOC factible."""
    x = np.zeros(len(q))
    for t in np.flatnonzero(objetivo > 0):
        def ok(v):
            y = x.copy()
            y[t] = v
            return POL._soc_factible(q + y, eps, soc0, b)
        if ok(objetivo[t]):
            x[t] = objetivo[t]
            continue
        lo, hi = 0., float(objetivo[t])
        for _ in range(40):
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        x[t] = lo
    return x


# ------------------------------------------------------------------------------------ 1. frecuencia
def escanear(c):
    t = c['tray']
    p, u, q, soc = t['potencia'], t['compromiso'], t['Descarga_demanda'], t['SOC']
    b, pmin = c['b'], c['pmin']
    e_min = b['E_MWh']*b['soc_min_pu']
    filas = []
    for d in range(1, 365):
        a, z = 24*d, 24*d + 24
        for j in range(4):
            if not (u[a - 1, j] and u[a, j]):
                continue
            h2 = a
            while h2 + 1 < z and u[h2 + 1, j]:
                h2 += 1
            cerca = p[a:h2 + 1, j] <= CERCA_MIN*pmin[j] + 1e-9
            pref = int(np.argmin(cerca)) if not cerca.all() else len(cerca)
            fila = dict(Escenario=c['n'], Senal=c['s'], Dia=d, Fecha=str(c['f'][a].date()), Unidad=f'G{j + 1}',
                        Horas_bloque_heredado=h2 - a + 1, Parada_original_hora=(h2 + 1 - a) if h2 + 1 < z else None,
                        Horas_cerca_min_inicio=pref, Carga_00_MW=float(p[a, j]), Pmin_MW=float(pmin[j]),
                        SOC_inicio_MWh=float(soc[a - 1]), Holgura_P_00_MW=float(b['P_MW'] - q[a]),
                        N1=True, N2=bool(cerca[0]))
            fila['N3'] = fila['N2'] and fila['Holgura_P_00_MW'] > 1e-6 and fila['SOC_inicio_MWh'] > e_min + 1e-6
            fila['Energia_unidad_prefijo_MWh'] = float(p[a:a + pref, j].sum())
            fila['N4'] = False
            fila['Satura_mismo_dia'] = bool(soc[a:z].max() >= b['E_MWh']*b['soc_max_pu'] - 1e-6)
            if fila['N3'] and pref > 0:
                obj = np.zeros(24)
                obj[:pref] = np.minimum(b['P_MW'] - q[a:z][:pref], p[a:a + pref, j])
                x = _extra_factible(q[a:z], c['eps'][a:z], float(soc[a - 1]), b, obj)
                fila['Cobertura_sin_reasignar'] = float(x.sum()/max(p[a:a + pref, j].sum(), 1e-12))
                fila['N4'] = bool(x.sum() >= p[a:a + pref, j].sum() - 1e-6)
            filas.append(fila)
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------ 2-3. ventanas
def costo_dia(c, a, z, q, soc0, final):
    """Costos y emisiones del día con la descarga `q` ya ejecutada y el despacho térmico `final`."""
    b = c['b']
    carga, soc = POL.trayectoria_soc(q, c['eps'][a:z], soc0, b)
    sobrante = c['eps'][a:z] - carga
    xpv = np.where(c['ce'][a:z] > 0, np.minimum(sobrante, c['pe']), 0.)
    ingreso = float(xpv@c['ce'][a:z])
    p = final['potencia']
    arr = final['arranques']
    fin = float(p[:, :4].sum(axis=0)@c['cv_fin'] + p[:, 4]@c['ci_fin'][a:z] + arr.sum(axis=0)@c['cargos']) - ingreso
    return dict(ajustado=final['objetivo'] - ingreso, financiero=fin,
                emisiones=float(p[:, :4].sum(axis=0)@c['intensidad'] + p[:, 4].sum()*c['fe_red']),
                red_MWh=float(p[:, 4].sum()), termica_MWh=float(p[:, :4].sum()), arranques=int(arr.sum()),
                descarga_MWh=float(q.sum()), soc_fin=float(soc[-1]), u_fin=final['compromiso'][-1].astype(int).copy(),
                estado_final=final['estado_final'])


def dia_original(c, d):
    """Día de la trayectoria original recalculado con B_INT (en caché: no depende de la alternativa)."""
    cache = c.setdefault('_orig', {})
    if d not in cache:
        cache[d] = dia_b(c, d, estado_inicio(c, d), soc_inicio(c, d))
    return cache[d]


def dia_b(c, d, estado, soc0):
    """Un día de B_INT sin cambios, desde un estado y un SOC dados."""
    a, z = 24*d, 24*d + 24
    plan = POL.planificar_dia_b(c['delta'][a:z], c['eps'][a:z], c['ci'][a:z], c['f'][a:z], c['cv'], c['pmin'],
                                c['pmax'], c['cargos'], estado, soc0, c['b'],
                                disponible=None if c['disp'] is None else c['disp'][a:z])
    return plan, costo_dia(c, a, z, plan['q'], soc0, plan['final'])


def candidato_heredado(c, a, z, referencia, j, h_s, h2):
    """Candidato de apagado de la unidad heredada en [h_s, h2] (horas del día), sin crédito de arranque."""
    b = c['b']
    p0, u0 = referencia['potencia'], referencia['compromiso']
    if not u0[h_s, j]:
        return None
    fin = h_s
    while fin + 1 <= h2 and u0[fin + 1, j]:
        fin += 1
    horas = np.arange(h_s, fin + 1)          # bloque contiguo de la referencia desde h_s
    q = np.minimum(b['P_MW'], p0[horas, j])
    u1 = u0[horas].copy()
    u1[:, j] = 0
    delta, precio = c['delta'][a:z], c['ci'][a:z]
    p1, c1 = despacho_capacidad_eventos(delta[horas] - q, precio[horas], c['cv'], c['pmin'], c['pmax']*u1, u1)
    if not np.isfinite(c1).all() or q.sum() <= 0:
        return None
    valor = math.fsum((referencia['costo_energia'][horas] - c1).tolist())
    energia = float(q.sum())/b['eta_d']
    if valor <= 0:
        return None
    return dict(tipo='U', hora=int(horas[0]), horas=[int(h) for h in horas], fuente=POL.FUENTES[j], unidad=j, rango=j,
                version=0, q=q, valor_USD=valor, energia_MWh=energia, densidad=valor/energia, p_tras=p1, u_tras=u1)


def alternativa(c, d, j, modo, h_s):
    """Día d con la alternativa; devuelve (costos del día, plan/diagnóstico) o None si es infactible."""
    a, z = 24*d, 24*d + 24
    estado, soc0 = estado_inicio(c, d), soc_inicio(c, d)
    dsp = None if c['disp'] is None else c['disp'][a:z]
    plan = POL.planificar_dia_b(c['delta'][a:z], c['eps'][a:z], c['ci'][a:z], c['f'][a:z], c['cv'], c['pmin'],
                                c['pmax'], c['cargos'], estado, soc0, c['b'], disponible=dsp)
    ref = plan['referencia']
    u0 = ref['compromiso']
    h2 = 0
    while h2 + 1 < 24 and plan['final']['compromiso'][h2 + 1, j]:
        h2 += 1
    q = plan['q']
    if modo == 'ALT1':
        p_plan = plan['final']['potencia'][:, j]
        obj = np.zeros(24)
        obj[h_s:h2 + 1] = np.minimum(c['b']['P_MW'] - q[h_s:h2 + 1], p_plan[h_s:h2 + 1])
        x = _extra_factible(q, c['eps'][a:z], soc0, c['b'], obj)
        mascara = np.ones((24, 4), bool)
        if dsp is not None:
            dd = np.asarray(dsp, bool)
            mascara &= dd if dd.ndim == 2 else np.repeat(dd[:, None], 4, axis=1)
        mascara[h_s:h2 + 1, j] = False
        qa = q + x
        try:
            final = POL.evaluador_dia(np.maximum(c['delta'][a:z] - qa, 0.), c['ci'][a:z], c['cv'], c['pmin'], c['pmax'],
                                      c['f'][a:z], c['cargos'], estado, disponible=mascara)
        except ValueError:
            return None
        info = dict(Extra_MWh=float(x.sum()), Unidad_MWh_original=float(p_plan[h_s:h2 + 1].sum()), h2=h2)
    else:
        cand = candidato_heredado(c, a, z, ref, j, h_s, 23)
        if cand is None:
            return None
        cands = POL.candidatos_sistema(c['delta'][a:z], c['ci'][a:z], c['cv'], c['pmin'], c['pmax'], c['cargos'],
                                       ref, c['b']) + [cand]
        qa, aceptados, rechazos = POL.seleccionar_plan(cands, c['delta'][a:z], c['eps'][a:z], soc0, c['b'],
                                                       c['ci'][a:z], c['cv'], c['pmin'])
        try:
            final = POL.evaluador_dia(np.maximum(c['delta'][a:z] - qa, 0.), c['ci'][a:z], c['cv'], c['pmin'], c['pmax'],
                                      c['f'][a:z], c['cargos'], estado, disponible=dsp)
        except ValueError:
            return None
        info = dict(Candidato_aceptado=any(x['tipo'] == 'U' and x['horas'] == cand['horas'] and x['fuente'] == cand['fuente']
                                           for x in aceptados),
                    Densidad_candidato=cand['densidad'], Umbral_plan_original=min(
                        [x['densidad'] for x in plan['aceptados']], default=0.),
                    Horas_candidato=len(cand['horas']), Energia_candidato_MWh=cand['energia_MWh'],
                    Rechazos_U=json.dumps({k: v for k, v in dict(rechazos).items() if k.startswith('U_')}))
    info['Unidad_apagada_en_bloque'] = bool((final['compromiso'][h_s:h2 + 1, j] == 0).all())
    return costo_dia(c, a, z, qa, soc0, final), info


def ventana(c, d, j, modo, h_s):
    """Alternativa en d y B_INT sin cambios después, hasta reencontrar la trayectoria original o MAX_DIAS."""
    t = c['tray']
    r = alternativa(c, d, j, modo, h_s)
    if r is None:
        return dict(Factible=False)
    dia0, info = r
    orig0 = dia_original(c, d)[1]
    alt, ori = [dia0], [orig0]
    estado, soc = dia0['estado_final'], dia0['soc_fin']
    reencuentro = None
    for k in range(1, MAX_DIAS + 1):
        z_prev = 24*(d + k) - 1
        if abs(soc - float(t['SOC'][z_prev])) <= TOL_SOC and np.array_equal(estado['u'], t['compromiso'][z_prev].astype(int)):
            reencuentro = k
            break
        if d + k >= 365:
            break
        alt.append(dia_b(c, d + k, estado, soc)[1])
        ori.append(dia_original(c, d + k)[1])
        estado, soc = alt[-1]['estado_final'], alt[-1]['soc_fin']
    fila = dict(Factible=True, Dias_ventana=len(alt), Frontera_comun=reencuentro is not None,
                Delta_SOC_final_MWh=None if reencuentro else soc - float(t['SOC'][24*(d + len(alt)) - 1]),
                Delta_u_final=None if reencuentro else (estado['u'] - t['compromiso'][24*(d + len(alt)) - 1]).tolist(),
                **info)
    for clave in ('ajustado', 'financiero', 'emisiones', 'red_MWh', 'termica_MWh', 'arranques', 'descarga_MWh'):
        fila[f'Delta_{clave}'] = sum(x[clave] for x in alt) - sum(x[clave] for x in ori)
    fila['Costo_ajustado_original_USD'] = sum(x['ajustado'] for x in ori)
    return fila


def reproducir(c, d):
    """El día recalculado con B_INT sin cambios reproduce la trayectoria guardada (condición del diagnóstico)."""
    a, z = 24*d, 24*d + 24
    plan, _ = dia_original(c, d)
    t = c['tray']
    return float(max(np.abs(plan['q'] - t['Descarga_demanda'][a:z]).max(),
                     np.abs(plan['final']['potencia'] - t['potencia'][a:z]).max()))


SELECCION = (('V2_p0', 'T0', 'N4'), ('V2_p0', 'T0', 'N3_sin_N4'), ('V2_p0', 'T1', 'N4'), ('V2_p0', 'T1', 'N3_sin_N4'),
             ('V2_p10', 'T0', 'N4'), ('V2_p20', 'T1', 'N4'))


def seleccionar(freq):
    """Situación de energía mediana (desempate: día más temprano) de cada clase y escenario de `SELECCION`."""
    elegidas = []
    for n, s, clase in SELECCION:
        g = freq[(freq.Escenario == n) & (freq.Senal == s)]
        base = g[g.N4] if clase == 'N4' else g[g.N3 & ~g.N4]
        if base.empty:
            continue
        base = base.sort_values(['Energia_unidad_prefijo_MWh', 'Dia'])
        elegidas.append(base.iloc[(len(base) - 1)//2].copy())
        elegidas[-1]['Clase'] = clase
    return pd.DataFrame(elegidas)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--salida', type=Path, required=True)
    a = p.parse_args(argv)
    a.salida.mkdir(parents=True, exist_ok=True)
    TA.inicializar(True)
    t0 = time.perf_counter()
    ctxs, freqs, fuentes = {}, [], {}
    for n, s in ESCENARIOS:
        if not fuente(n, s).exists():
            print('sin trayectoria', n, s, flush=True)
            continue
        c = preparar(n, s)
        ctxs[(n, s)] = c
        fuentes[f'{n}_{s}'] = dict(ruta=c['ruta'], sha256=c['sha'])
        freqs.append(escanear(c))
        print('escaneado', n, s, len(freqs[-1]), f'{time.perf_counter() - t0:.0f}s', flush=True)
    freq = pd.concat(freqs, ignore_index=True)
    freq.to_csv(a.salida/'situaciones_candidatas.csv', index=False)
    resumen = freq.groupby(['Escenario', 'Senal'], sort=False).agg(
        N1_unidad_heredada=('N1', 'sum'), N2_cerca_del_minimo=('N2', 'sum'), N3_con_BESS_disponible=('N3', 'sum'),
        N4_sustitucion_sin_reasignar=('N4', 'sum'),
        N4_y_BESS_satura_ese_dia=('Satura_mismo_dia', lambda x: int((x & freq.loc[x.index, 'N4']).sum())),
        Dias_distintos_N3=('Dia', lambda x: x[freq.loc[x.index, 'N3']].nunique()),
        Energia_mediana_N3_MWh=('Energia_unidad_prefijo_MWh', lambda x: x[freq.loc[x.index, 'N3']].median()),
        Unidades=('Unidad', lambda x: ','.join(sorted(set(x)))))
    resumen.to_csv(a.salida/'frecuencia.csv')
    print(resumen, flush=True)
    elegidas = seleccionar(freq)
    filas = []
    for e in elegidas.itertuples(index=False):
        c = ctxs[(e.Escenario, e.Senal)]
        d, j = int(e.Dia), int(e.Unidad[1]) - 1
        err = reproducir(c, d)
        assert err <= 1e-6, f'el día {d} no se reproduce ({err})'
        mejor = None
        horas = range(0, int(e.Parada_original_hora) if e.Parada_original_hora == e.Parada_original_hora and
                       e.Parada_original_hora is not None else 24)
        for h_s in horas:
            r = ventana(c, d, j, 'ALT1', h_s)
            if r.get('Factible') and (mejor is None or r['Delta_ajustado'] < mejor['Delta_ajustado']):
                mejor = dict(r, h_s=h_s)
        base = dict(Escenario=e.Escenario, Senal=e.Senal, Clase=e.Clase, Fecha=e.Fecha, Dia=d, Unidad=e.Unidad,
                    Horas_cerca_min_inicio=e.Horas_cerca_min_inicio, Parada_original_hora=e.Parada_original_hora,
                    Satura_mismo_dia=e.Satura_mismo_dia,
                    Carga_00_MW=e.Carga_00_MW, Pmin_MW=e.Pmin_MW, SOC_inicio_MWh=e.SOC_inicio_MWh,
                    N4=e.N4, Reproduccion_error=err)
        filas.append(dict(base, Alternativa='ALT1_apagado_forzado_mejor_hora', **(mejor or dict(Factible=False))))
        r2 = ventana(c, d, j, 'ALT2', 0)
        filas.append(dict(base, Alternativa='ALT2_candidato_heredado_en_B', h_s=0, **r2))
        print('ventana', e.Escenario, e.Senal, e.Fecha, e.Unidad, f'{time.perf_counter() - t0:.0f}s', flush=True)
    res = pd.DataFrame(filas)
    res.to_csv(a.salida/'ventanas.csv', index=False)
    (a.salida/'manifiesto.json').write_text(json.dumps(dict(
        generado=pd.Timestamp.now().isoformat(), trayectorias=fuentes, cerca_del_minimo=CERCA_MIN, max_dias=MAX_DIAS,
        codigo={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                ('diagnostico_apagado_b.py', 'termica_politicas.py', 'termica_carbono_anual.py', 'carbono.py')},
        comando='python diagnostico_apagado_b.py --salida ' + str(a.salida)), ensure_ascii=False, indent=1),
        encoding='utf-8')
    print(res.to_string(), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
