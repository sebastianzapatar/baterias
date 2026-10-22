# -*- coding: utf-8 -*-
"""Diagnóstico exploratorio de horas equivalentes de operación (EOH) del caso 1.

Posprocesamiento puro de trayectorias anuales ya guardadas: **no optimiza**, no
llama a PyPSA ni a la programación dinámica y no modifica ningún resultado previo.
Escribe CSV nuevos con prefijo ``eoh_`` en ``resultados_nb12/``.

Casos comparables (gas 16 USD/MMBtu, 100 USD/arranque hipotético, 0 minutos por
evento, estado inicial libre, señales T0 y T1):

    sin_PV       red + térmico              resultados_nb11/pypsa_anual_{T0,T1}.npz
    con_PV       red + térmico + PV fijo    resultados_nb12/pv_anual_{T0,T1}.npz
                 (es también el límite 365: el límite diario ya lo implica; no es otro experimento)
    limite_180   ídem con ≤ 180 arranques por unidad y año   resultados_nb12/limites_{T0,T1}_180.npz
    limite_60    ídem con ≤ 60                               resultados_nb12/limites_{T0,T1}_60.npz
    limite_0     ídem con 0                                  resultados_nb12/limites_{T0,T1}_0.npz

Aproximación exploratoria, única que se usa:

    EOH_g(f) = H_g + f · N_g

H son horas encendida (compromiso × paso), N arranques (transiciones de apagada
a encendida) y f horas equivalentes por arranque, con f ∈ {0, 10, 20, 50}. Esos
valores son **ilustrativos**: no provienen de fabricantes ni son un rango
certificado o probable. Un mismo f para las cuatro unidades es un supuesto
simplificador. EOH no es el contador real de mantenimiento: no incorpora carga
parcial, temperatura, tipo de arranque, disparos ni historial previo, y compara
acumulación en el año modelado, no vida remanente ni fechas de overhaul. La suma
entre unidades es un indicador agregado del escenario, no la vida de un equipo.

Comparación caso − referencia:

    ΔH = H_caso − H_ref,  ΔN = N_caso − N_ref,  ΔEOH(f) = ΔH + f · ΔN
    f* = −ΔH / ΔN   si ΔN ≠ 0

El umbral de un bloque se calcula con ΔH y ΔN agregados, nunca promediando
umbrales individuales, y queda condicionado al factor común.
"""

import matplotlib
matplotlib.use('Agg')     # el cargador del cuaderno 02 ejecuta plt.show()

from pathlib import Path

import numpy as np
import pandas as pd

NB11 = Path('resultados_nb11')
NB12 = Path('resultados_nb12')
UNIDADES = ('G1', 'G2', 'G3', 'G4')
BLOQUES = {'G1': [0], 'G2': [1], 'G3': [2], 'G4': [3], 'G1+G2': [0, 1], 'Total': [0, 1, 2, 3]}
FACTORES = (0., 10., 20., 50.)
GAS_USD_MMBTU = 16.
CARGO_ARRANQUE_USD = 100.
MINUTOS_EVENTO = 0
LIMITES = (180, 60, 0)
CLAVES = ('Fecha', 'potencia', 'compromiso', 'arranques', 'paradas', 'costo_energia', 'costo_arranques')
TOL_MWH = 1e-6
TOL_USD = 1e-4
NOTA_TIEMPO = ('Métricas de la trayectoria factible guardada. La brecha acota el costo, '
               'no las horas, los arranques ni la EOH.')


# ---------------------------------------------------------------- cálculo puro
def reconstruir(compromiso, potencia, paso_h=1.):
    """Horas encendida, arranques, paradas y energía por unidad desde el compromiso.

    Estado inicial libre: estar encendida en la primera hora no cuenta como
    arranque. No se agrega una parada terminal fuera del horizonte.
    """
    u = np.asarray(compromiso, int)
    p = np.asarray(potencia, float)
    if u.ndim != 2 or not np.isin(u, [0, 1]).all():
        raise ValueError('El compromiso debe ser una matriz binaria horas × unidades')
    delta = np.diff(u, axis=0)
    arranques = np.vstack([np.zeros((1, u.shape[1]), int), (delta == 1).astype(int)])
    paradas = np.vstack([np.zeros((1, u.shape[1]), int), (delta == -1).astype(int)])
    return dict(H=u.sum(axis=0)*paso_h, N=arranques.sum(axis=0), Z=paradas.sum(axis=0),
                E=p[:, :u.shape[1]].sum(axis=0)*paso_h, arranques=arranques, paradas=paradas)


def eoh(H, N, f):
    """EOH = H + f·N, elemento a elemento."""
    return np.asarray(H, float) + f*np.asarray(N, float)


def _signo(x):
    return 'menor' if x < 0 else ('mayor' if x > 0 else 'igual')


def equilibrio(dH, dN):
    """Umbral f* = −ΔH/ΔN y lectura del signo de ΔEOH para f ≥ 0.

    Devuelve (f*, tipo, lectura). f* es NaN cuando ΔN = 0. Tipos:
    sin_cambio, sin_cambio_en_arranques, sin_cruce, umbral_cero, cruce.
    """
    dH, dN = float(dH), float(dN)
    if dN == 0:
        if dH == 0:
            return np.nan, 'sin_cambio', 'ΔH = ΔN = 0: la EOH es igual para todo f.'
        return np.nan, 'sin_cambio_en_arranques', (
            f'ΔN = 0: la EOH del caso es {_signo(dH)} que la de la referencia para todo f; no hay umbral.')
    fe = -dH/dN
    if fe < 0:
        return fe, 'sin_cruce', (
            f'f* = {fe:.4g} < 0: ΔH y ΔN tienen el mismo signo; la EOH del caso es {_signo(dN)} '
            'que la de la referencia para todo f ≥ 0.')
    if fe == 0:
        return 0., 'umbral_cero', (
            f'ΔH = 0: la EOH coincide en f = 0 y es {_signo(dN)} que la de la referencia para todo f > 0.')
    return fe, 'cruce', (
        f'Para 0 ≤ f < {fe:.4g} la EOH del caso es {_signo(dH)} que la de la referencia; '
        f'en f* coinciden; para f > {fe:.4g} es {_signo(dN)}.')


def signo_esperado(dH, dN, f):
    """Signo de ΔEOH(f) según la clasificación, sin evaluar ΔH + fΔN. Para las pruebas."""
    fe, tipo, _ = equilibrio(dH, dN)
    s = lambda x: int(np.sign(x))
    if tipo == 'sin_cambio':
        return 0
    if tipo == 'sin_cambio_en_arranques':
        return s(dH)
    if tipo == 'sin_cruce':
        return s(dN)
    if tipo == 'umbral_cero':
        return 0 if f == 0 else s(dN)
    return s(dH) if f < fe else (0 if f == fe else s(dN))


# ---------------------------------------------------------------- lectura
def leer_trayectoria(ruta):
    """Lee un .npz verificando claves, forma, calendario horario 2025 y binarios."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(ruta)
    with np.load(ruta, allow_pickle=False) as z:
        faltan = [k for k in CLAVES if k not in z.files]
        if faltan:
            raise ValueError(f'{ruta}: faltan {faltan}')
        t = {k: z[k] for k in z.files}
    fechas = pd.DatetimeIndex(t['Fecha'])
    n = len(fechas)
    if n != 8760 or fechas.year.unique().tolist() != [2025]:
        raise ValueError(f'{ruta}: se esperaban las 8.760 horas de 2025')
    pasos = (fechas[1:] - fechas[:-1]) / pd.Timedelta(hours=1)
    if not (pasos == 1).all():
        raise ValueError(f'{ruta}: calendario no horario o no continuo')
    formas = dict(potencia=(n, 5), compromiso=(n, 4), arranques=(n, 4), paradas=(n, 4),
                  costo_energia=(n,), costo_arranques=(n,))
    for k, forma in formas.items():
        if t[k].shape != forma:
            raise ValueError(f'{ruta}: {k} con forma {t[k].shape}, se esperaba {forma}')
    if not np.isin(t['compromiso'], [0, 1]).all():
        raise ValueError(f'{ruta}: compromiso no binario')
    t['fechas'] = fechas
    t['paso_h'] = 1.
    return t


def _fila(tabla, **filtros):
    sel = tabla
    for k, v in filtros.items():
        sel = sel[sel[k] == v]
    if len(sel) != 1:
        raise ValueError(f'Se esperaba una fila con {filtros}; hay {len(sel)}')
    return sel.iloc[0]


def definir_casos():
    """Casos comparables con su archivo, fila de resumen y estado de la solución."""
    leer = lambda p: pd.read_csv(p, float_precision='round_trip')
    r11, rpv, rlim = leer(NB11/'pypsa_anual_resumen.csv'), leer(NB12/'pv_anual_resumen.csv'), \
        leer(NB12/'limites_arranques_resumen.csv')
    for tabla, cols in [(r11, ['Senal', 'Gas_USD_MMBtu', 'Cargo_USD_arranque', 'Horas', 'Brecha', 'Estado']),
                        (rpv, ['Senal', 'Caso', 'Estado_solver', 'Brecha', 'Arranques_G1_G2', 'Horas_encendida_G3']),
                        (rlim, ['Senal', 'Limite_anual_por_unidad', 'Estado', 'Brecha', 'Reutilizado',
                                'Arranques_G1', 'Arranques_G2'])]:
        faltan = [c for c in cols if c not in tabla.columns]
        if faltan:
            raise ValueError(f'Columnas ausentes en un resumen: {faltan}')
    casos = []
    for s in ('T0', 'T1'):
        a = _fila(r11, Senal=s)
        assert a.Gas_USD_MMBtu == GAS_USD_MMBTU and a.Cargo_USD_arranque == CARGO_ARRANQUE_USD
        # Brecha 0 solicitada; HiGHS puede reportar residuos del orden de 1e−15.
        assert a.Horas == 8760 and a.Estado == 'OK' and a.Brecha <= 1e-9
        sin = _fila(rpv, Senal=s, Caso='sin_PV')
        con = _fila(rpv, Senal=s, Caso='con_PV')
        assert con.Estado_solver == 'ok/optimal' and con.Brecha <= 1e-9
        l365 = _fila(rlim, Senal=s, Limite_anual_por_unidad=365)
        assert bool(l365.Reutilizado), 'el límite 365 debe reutilizar el caso PV'
        casos.append(dict(Senal=s, Caso='sin_PV', Limite_anual_por_unidad=np.nan,
                          Archivo=NB11/f'pypsa_anual_{s}.npz', Resumen=sin, Estado_solucion='PRECISION_SOLICITADA',
                          Brecha_costo=float(sin.Brecha), Reutilizado=False,
                          Nota='trayectoria PyPSA anual del cuaderno 11, 0 minutos (no pypsa_eventos_)'))
        casos.append(dict(Senal=s, Caso='con_PV', Limite_anual_por_unidad=365,
                          Archivo=NB12/f'pv_anual_{s}.npz', Resumen=con, Resumen_365=l365,
                          Estado_solucion='PRECISION_SOLICITADA', Brecha_costo=float(con.Brecha), Reutilizado=False,
                          Nota='caso PV sin límite anual adicional; el límite 365 lo reutiliza, no es otro experimento'))
        for L in LIMITES:
            f = _fila(rlim, Senal=s, Limite_anual_por_unidad=L)
            if f.Estado == 'OK':
                assert f.Brecha <= 1e-6 + 1e-12, (s, L, f.Brecha)
                estado, nota = 'PRECISION_SOLICITADA', 'brecha relativa ≤ 1e−6 solicitada'
            elif f.Estado == 'FACTIBLE_LIMITE_TIEMPO':
                estado, nota = 'FACTIBLE_LIMITE_TIEMPO', NOTA_TIEMPO
            else:
                raise ValueError(f'Estado desconocido {f.Estado}')
            casos.append(dict(Senal=s, Caso=f'limite_{L}', Limite_anual_por_unidad=L,
                              Archivo=NB12/f'limites_{s}_{L}.npz', Resumen=f, Estado_solucion=estado,
                              Brecha_costo=float(f.Brecha), Reutilizado=False, Nota=nota))
    return casos


# ---------------------------------------------------------------- verificación de supuestos y conteos
def verificar_trayectoria(t, caso, E, P):
    """Supuestos (gas, cargo, 0 min, frontera libre) y conteos frente al resumen guardado."""
    s, R = caso['Senal'], caso['Resumen']
    p, u = t['potencia'], t['compromiso']
    rec = reconstruir(u, p, t['paso_h'])
    assert (pd.DatetimeIndex(E['FECHA']) == t['fechas']).all(), 'calendario distinto al de las entradas'
    # Eventos: reconstrucción independiente frente a los arrays guardados; primera hora sin eventos.
    assert np.array_equal(rec['arranques'], t['arranques']) and np.array_equal(rec['paradas'], t['paradas'])
    # Cargo uniforme de 100 USD y costo de energía con gas 16; 0 minutos: capacidad = Pdisp·u.
    assert np.abs(t['costo_arranques'] - CARGO_ARRANQUE_USD*t['arranques'].sum(axis=1)).max() <= 1e-9
    energia = p[:, :4]@P['cv'] + p[:, 4]*E['C_IMP_USD'][s]
    err_costo = float(np.abs(energia - t['costo_energia']).max())
    assert err_costo <= 1e-6, ('costo horario', caso['Caso'], err_costo)
    assert (p[:, :4] <= u*P['pmax'] + 1e-6).all() and (p[:, :4] >= u*P['pmin'] - 1e-6).all()
    total = float(t['costo_energia'].sum() + t['costo_arranques'].sum())
    ref_total = R['Costo_objetivo_USD'] if 'Costo_objetivo_USD' in R.index and caso['Caso'] in ('sin_PV', 'con_PV') \
        else R['Costo_operativo_total_USD']
    assert abs(total - ref_total) <= TOL_USD*10, ('costo total', caso['Caso'], total - ref_total)
    if caso['Caso'] != 'sin_PV':
        assert abs(t['pv_disponible'].sum() - R['PV_disponible_MWh']) <= TOL_MWH*10
    # Conteos frente al resumen guardado.
    H, N, Z, En = rec['H'], rec['N'], rec['Z'], rec['E']
    esperado = dict(Arranques=(N.sum(), R.Arranques), Arranques_G1_G2=(N[:2].sum(), R.Arranques_G1_G2),
                    Arranques_G3=(N[2], R.Arranques_G3), Arranques_G4=(N[3], R.Arranques_G4),
                    Paradas=(Z.sum(), R.Paradas),
                    Horas_unidad_encendida_G1_G2=(H[:2].sum(), R.Horas_unidad_encendida_G1_G2),
                    Horas_encendida_G3=(H[2], R.Horas_encendida_G3), Horas_encendida_G4=(H[3], R.Horas_encendida_G4))
    if 'Arranques_G1' in R.index:
        esperado.update(Arranques_G1=(N[0], R.Arranques_G1), Arranques_G2=(N[1], R.Arranques_G2))
    for k, (a, b) in esperado.items():
        assert int(a) == int(b), (caso['Senal'], caso['Caso'], k, a, b)
    for k, a in [('Energia_G1_G2_MWh', En[:2].sum()), ('Energia_G3_MWh', En[2]), ('Energia_G4_MWh', En[3]),
                 ('Energia_termica_MWh', En.sum())]:
        assert abs(a - R[k]) <= TOL_MWH, (caso['Senal'], caso['Caso'], k, a, R[k])
    if 'Resumen_365' in caso:
        L = caso['Resumen_365']
        for k in ('Arranques', 'Arranques_G1_G2', 'Arranques_G3', 'Arranques_G4', 'Horas_unidad_encendida_G1_G2',
                  'Horas_encendida_G3', 'Horas_encendida_G4'):
            assert int(L[k]) == int(R[k]), ('límite 365 ≠ caso PV', k)
        assert int(L.Arranques_G1) == int(N[0]) and int(L.Arranques_G2) == int(N[1])
    return rec, dict(Error_costo_horario_USD=err_costo, Error_costo_total_USD=abs(total - ref_total),
                     Unidades_encendidas_hora_0=int(u[0].sum()))


# ---------------------------------------------------------------- tablas
def tabla_unidades(registros):
    filas = []
    for c, rec in registros:
        for nombre, idx in BLOQUES.items():
            H, N = float(rec['H'][idx].sum()), int(rec['N'][idx].sum())
            fila = dict(Senal=c['Senal'], Caso=c['Caso'], Limite_anual_por_unidad=c['Limite_anual_por_unidad'],
                        Estado_solucion=c['Estado_solucion'], Unidad=nombre, Horas_encendida_h=H, Arranques=N,
                        Paradas=int(rec['Z'][idx].sum()), Energia_MWh=float(rec['E'][idx].sum()))
            fila.update({f'EOH_f{int(f)}_h': float(eoh(H, N, f)) for f in FACTORES})
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_equilibrio(U):
    comparaciones = [('PV_vs_sin_PV', 'con_PV', 'sin_PV')] + \
                    [(f'limite_{L}_vs_PV', f'limite_{L}', 'con_PV') for L in LIMITES]
    filas = []
    for s in ('T0', 'T1'):
        for nombre, caso, ref in comparaciones:
            for unidad in BLOQUES:
                a = _fila(U, Senal=s, Caso=caso, Unidad=unidad)
                b = _fila(U, Senal=s, Caso=ref, Unidad=unidad)
                dH, dN = a.Horas_encendida_h - b.Horas_encendida_h, int(a.Arranques - b.Arranques)
                fe, tipo, lectura = equilibrio(dH, dN)
                fila = dict(Senal=s, Comparacion=nombre, Caso=caso, Referencia=ref, Unidad=unidad,
                            Delta_H_h=dH, Delta_N=dN)
                # ΔEOH por diferencia directa de EOH, para contrastar la identidad ΔH + fΔN.
                fila.update({f'Delta_EOH_f{int(f)}_h': a[f'EOH_f{int(f)}_h'] - b[f'EOH_f{int(f)}_h'] for f in FACTORES})
                fila.update(f_equilibrio_h_por_arranque=fe, Tipo_umbral=tipo, Lectura=lectura,
                            Estado_caso=a.Estado_solucion, Estado_referencia=b.Estado_solucion,
                            Nota=NOTA_TIEMPO if 'FACTIBLE_LIMITE_TIEMPO' in (a.Estado_solucion, b.Estado_solucion) else '',
                            Condicion='umbral condicionado a un mismo f para las unidades del bloque'
                            if unidad in ('G1+G2', 'Total') else '')
                filas.append(fila)
    return pd.DataFrame(filas)


def pruebas_reales(U, Q):
    """Controles sobre los resultados: f=0, bloques, identidad, igualdad en f* y signos."""
    out = []
    ok = lambda nombre, cond, detalle='': out.append(dict(Tipo='resultados', Prueba=nombre,
                                                          Estado='OK' if cond else 'FALLA', Detalle=detalle))
    ok('EOH con f = 0 igual a horas encendida', np.allclose(U.EOH_f0_h, U.Horas_encendida_h, rtol=0, atol=0),
       f'{len(U)} filas')
    err_bloque = 0.
    for (s, caso), g in U.groupby(['Senal', 'Caso']):
        g = g.set_index('Unidad')
        for col in ['Horas_encendida_h', 'Arranques', 'Paradas', 'Energia_MWh'] + [f'EOH_f{int(f)}_h' for f in FACTORES]:
            err_bloque = max(err_bloque, abs(g.loc['G1+G2', col] - g.loc[['G1', 'G2'], col].sum()),
                             abs(g.loc['Total', col] - g.loc[list(UNIDADES), col].sum()))
    ok('Bloques G1+G2 y Total iguales a la suma de unidades', err_bloque <= 1e-9, f'error máximo {err_bloque:.1e}')
    err_id = max(float(np.abs(Q[f'Delta_EOH_f{int(f)}_h'] - (Q.Delta_H_h + f*Q.Delta_N)).max()) for f in FACTORES)
    ok('Identidad ΔEOH = ΔH + f·ΔN', err_id <= 1e-9, f'error máximo {err_id:.1e} h')
    err_eq, fallas_signo, n_umbral = 0., 0, 0
    for _, q in Q.iterrows():
        fe = q.f_equilibrio_h_por_arranque
        if q.Tipo_umbral == 'cruce':
            n_umbral += 1
            err_eq = max(err_eq, abs(q.Delta_H_h + fe*q.Delta_N))
        pruebas_f = [0., 1e-3] + list(FACTORES) + ([fe/2, fe, fe*2 + 1] if q.Tipo_umbral == 'cruce' else [])
        for f in pruebas_f:
            valor = q.Delta_H_h + f*q.Delta_N
            real = 0 if abs(valor) <= 1e-9*max(1., abs(q.Delta_H_h)) else int(np.sign(valor))
            fallas_signo += real != signo_esperado(q.Delta_H_h, q.Delta_N, f)
    ok('ΔEOH(f*) = 0 cuando hay cruce', err_eq <= 1e-9, f'{n_umbral} umbrales; residuo máximo {err_eq:.1e} h')
    ok('Signo de ΔEOH antes, en y después del umbral', fallas_signo == 0, f'{fallas_signo} discrepancias')
    # El umbral del bloque sale de ΔH y ΔN agregados.
    err_ag = 0.
    for (s, comp), g in Q.groupby(['Senal', 'Comparacion']):
        g = g.set_index('Unidad')
        for col in ('Delta_H_h', 'Delta_N'):
            err_ag = max(err_ag, abs(g.loc['G1+G2', col] - g.loc[['G1', 'G2'], col].sum()),
                         abs(g.loc['Total', col] - g.loc[list(UNIDADES), col].sum()))
        for b, idx in (('G1+G2', ['G1', 'G2']), ('Total', list(UNIDADES))):
            if g.loc[b, 'Delta_N'] != 0:
                esperado = -g.loc[idx, 'Delta_H_h'].sum()/g.loc[idx, 'Delta_N'].sum()
                err_ag = max(err_ag, abs(g.loc[b, 'f_equilibrio_h_por_arranque'] - esperado))
    ok('Umbral de bloque desde ΔH y ΔN agregados', err_ag <= 1e-9, f'error máximo {err_ag:.1e}')
    return out


def main():
    import logging
    for nombre in ('pypsa', 'linopy'):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    import termica as T
    import termica_pv as TP
    import termica_eoh_pruebas as PR

    pruebas = PR.pruebas_sinteticas()
    assert all(p['Estado'] == 'OK' for p in pruebas), pruebas
    print(f'{len(pruebas)} pruebas sintéticas OK', flush=True)

    E = T.cargar_entradas()          # solo entradas (demanda, señales); no resuelve nada
    P = TP.parametros_termicos(GAS_USD_MMBTU)
    casos = definir_casos()
    registros, filas_casos = [], []
    for c in casos:
        t = leer_trayectoria(c['Archivo'])
        rec, control = verificar_trayectoria(t, c, E, P)
        registros.append((c, rec))
        filas_casos.append(dict(Senal=c['Senal'], Caso=c['Caso'], Limite_anual_por_unidad=c['Limite_anual_por_unidad'],
                                Archivo=str(c['Archivo']).replace('\\', '/'), Estado_solucion=c['Estado_solucion'],
                                Brecha_costo=c['Brecha_costo'], Gas_USD_MMBtu=GAS_USD_MMBTU,
                                Cargo_USD_arranque=CARGO_ARRANQUE_USD, Minutos_evento=MINUTOS_EVENTO,
                                Estado_inicial='libre', Nota=c['Nota'], **control))
        pruebas.append(dict(Tipo='resultados', Prueba=f'Conteos, energía y costo {c["Senal"]} {c["Caso"]}',
                            Estado='OK', Detalle='arranques, paradas, horas y energía iguales al resumen; '
                            f'error de costo horario {control["Error_costo_horario_USD"]:.1e} USD'))
        print('verificado', c['Senal'], c['Caso'], c['Estado_solucion'], flush=True)
    U = tabla_unidades(registros)
    Q = tabla_equilibrio(U)
    reales = pruebas_reales(U, Q)
    pruebas += reales
    assert all(p['Estado'] == 'OK' for p in pruebas), [p for p in pruebas if p['Estado'] != 'OK']
    pd.DataFrame(filas_casos).to_csv(NB12/'eoh_casos.csv', index=False)
    U.to_csv(NB12/'eoh_unidades.csv', index=False)
    Q.to_csv(NB12/'eoh_equilibrio.csv', index=False)
    pd.DataFrame(pruebas).to_csv(NB12/'eoh_pruebas.csv', index=False)
    print(f'{len(pruebas)} pruebas OK; escritos eoh_casos, eoh_unidades, eoh_equilibrio y eoh_pruebas', flush=True)
    vista = Q[Q.Unidad.isin(['G1+G2', 'Total'])][['Senal', 'Comparacion', 'Unidad', 'Delta_H_h', 'Delta_N',
                                                  'f_equilibrio_h_por_arranque', 'Tipo_umbral', 'Estado_caso']]
    with pd.option_context('display.width', 200):
        print(vista.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
