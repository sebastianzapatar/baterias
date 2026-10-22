"""Malla exploratoria de dimensionamiento V2: PV + BESS con la flota térmica fija, 20 anualidades explícitas.

Configuraciones en `malla_carbono_v2.json` (hasta doce, K3 incluida). Para cada configuración, tarifa (T0, T1) y
escenario de carbono (0, 10, 20 y 50 USD/t, `gas_y_red`, XM 2025, disponibilidad V2) se resuelven A_INT, B_INT y
D_INT con el ejecutor multianual V2 (`termica_carbono_vpn.tarea_anual`), sin cambiar las reglas. Referencias:

    REF_RED_TERMICA     red + térmica, sin inversión nueva; no depende del tamaño
    REF_PV_<f>          PV sin BESS por nivel de PV (f = 100, 075, 050); no depende del BESS

Las referencias sin BESS usan el techo de exportación de K3, que no se activa: su exportación nunca supera el PV
máximo (se comprueba). Así, la referencia con PV de K3 conserva exactamente la clave de la corrida de K3.

Reutilización. Las anualidades de K3 con 0 y 50 USD/t (`resultados_carbono_v2/vpn_K3_20260926/`) se registran como
reutilizadas solo si su clave coincide con la nueva en todo salvo las huellas de código, y el `.npz` original
conserva su huella; el metadato apunta al archivo original. La red + térmica de los años 2-20 se reutiliza del
año 1 de su misma carpeta (entradas idénticas). Nada se extrapola ni interpola.

VPN (carbono.vpn_financiero_y_ajustado sobre el motor multianual existente): proyecto completo frente a red +
térmica (CAPEX y OPEX de PV y BESS), incremental del BESS frente a PV sin BESS del mismo PV (CAPEX y OPEX del BESS)
y solo PV (control de aditividad). El VPN financiero es el de una operación elegida con el carbono del escenario.

Etapas, desde la raíz (Numba aislado opcional con PYTHONPATH=.venv/numba):

    python termica_carbono_malla.py --salida resultados_carbono_v2/malla_20260926 --etapa reutilizar
    python termica_carbono_malla.py --salida ... --etapa medir --procesos N      # año 1 de algunos casos
    python termica_carbono_malla.py --salida ... --etapa anual --procesos N
    python termica_carbono_malla.py --salida ... --etapa financiero
    python termica_carbono_malla.py --salida ... --etapa verificar

Cada etapa devuelve 1 si algún caso esperado no quedó completo. Un caso que no converge queda NO_CONVERGE y su
comparación se registra como no evaluada económicamente (sin VPN).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import termica_carbono_anual as TA
import termica_carbono_vpn as V
import termica_politicas as POL

MALLA = Path('malla_carbono_v2.json')
K3_VPN = Path('resultados_carbono_v2/vpn_K3_20260926')
REUTILIZABLES = ('V2_p0', 'V2_p50')
REF_RED = 'REF_RED_TERMICA'
H = V.H
CODIGO = V.CODIGO + ('termica_carbono_malla.py', 'malla_carbono_v2.json')
ESTADOS = ('COMPLETADO', 'FALLIDO', 'FALTANTE', 'NO_CONVERGE', 'IDENTIDAD_INCOHERENTE')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cargar_malla(ruta=MALLA):
    m = json.loads(Path(ruta).read_text(encoding='utf-8'))
    cfg = m['configuraciones']
    ids = [c['id'] for c in cfg]
    assert len(cfg) <= 12 and len(set(ids)) == len(ids), 'hasta doce configuraciones con identificador único'
    assert any((c['f_PV'], c['P_MW'], c['E_MWh']) == (1., 45., 280.) for c in cfg), 'la malla debe incluir K3'
    for c in cfg:
        assert 0 < c['f_PV'] <= 1 and c['P_MW'] > 0 and c['E_MWh'] > 0, c
    assert set(m['escenarios_carbono']) <= set(TA.ESCENARIOS) | set(TA.ESCENARIOS_MALLA)
    return m


def configs(m=None):
    m = m or cargar_malla()
    return {c['id']: dict(f_PV=float(c['f_PV']), P_MW=float(c['P_MW']), E_MWh=float(c['E_MWh']))
            for c in m['configuraciones']}


def es_k3(cfg):
    return cfg is not None and (cfg['f_PV'], cfg['P_MW'], cfg['E_MWh']) == (1., 45., 280.)


def ref_pv(f):
    return f'REF_PV_{int(round(f*100)):03d}'


def cfg_ref_pv(f):
    """Referencia PV sin BESS: techo de exportación de K3 (no activo sin BESS)."""
    return dict(f_PV=float(f), P_MW=45., E_MWh=280.)


def niveles_pv(m=None):
    return sorted({c['f_PV'] for c in configs(m).values()}, reverse=True)


# ------------------------------------------------------------------------------------ inventario
def inventario(m=None):
    m = m or cargar_malla()
    filas = []
    for n in m['escenarios_carbono']:
        for s in m['senales']:
            for a in range(1, H + 1):
                for cid in configs(m):
                    for c in m['reglas']:
                        filas.append(dict(Grupo=cid, Escenario_piloto=n, Senal=s, Caso=c, Anio=a))
                filas.append(dict(Grupo=REF_RED, Escenario_piloto=n, Senal=s, Caso='RED_TERMICA', Anio=a))
                for f in niveles_pv(m):
                    filas.append(dict(Grupo=ref_pv(f), Escenario_piloto=n, Senal=s, Caso='PV_SIN_BESS', Anio=a))
    inv = pd.DataFrame(filas)
    inv['Etiqueta'] = [V.etiqueta(s, c, a) for s, c, a in zip(inv.Senal, inv.Caso, inv.Anio)]
    assert not inv.duplicated(['Grupo', 'Escenario_piloto', 'Etiqueta']).any()
    return inv


def config_grupo(grupo, m=None):
    if grupo == REF_RED:
        return None                       # contexto de K3: la red + térmica no usa PV ni BESS
    if grupo.startswith('REF_PV_'):
        return cfg_ref_pv(int(grupo[-3:])/100)
    return configs(m)[grupo]


def evaluar_completitud(salida, inv):
    salida = Path(salida)
    filas = []
    for e in inv.itertuples(index=False):
        carpeta = salida/e.Grupo/e.Escenario_piloto
        meta, error = carpeta/f'{e.Etiqueta}_meta.json', carpeta/f'{e.Etiqueta}_error.json'
        fila = e._asdict()
        if not meta.exists():
            fila['Estado_caso'] = 'FALLIDO' if error.exists() else 'FALTANTE'
            if error.exists():
                fila['Detalle'] = json.loads(error.read_text(encoding='utf-8')).get('Mensaje', '')[:300]
        else:
            d = json.loads(meta.read_text(encoding='utf-8'))
            k = d.get('clave', {})
            if (k.get('escenario'), k.get('senal'), k.get('caso'), k.get('anio'), d.get('Etiqueta')) != \
                    (e.Escenario_piloto, e.Senal, e.Caso, int(e.Anio), e.Etiqueta):
                fila['Estado_caso'] = 'IDENTIDAD_INCOHERENTE'
            elif d.get('Estado') != 'OK':
                fila['Estado_caso'] = 'NO_CONVERGE'
            elif e.Caso in TA.POLITICAS and abs(float(d.get('Cierre_SOC_MWh', np.inf))) > POL.TOL_CIERRE_MWh:
                fila['Estado_caso'] = 'NO_CONVERGE'
            else:
                fila['Estado_caso'] = 'COMPLETADO'
            fila['Origen'] = d.get('Origen')
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    assert set(tabla.Estado_caso) <= set(ESTADOS)
    return tabla


def informar(salida, tabla, nombre):
    tabla.to_csv(Path(salida)/f'completitud_{nombre}.csv', index=False)
    conteo = tabla.Estado_caso.value_counts().reindex(ESTADOS, fill_value=0).to_dict()
    completo = conteo['COMPLETADO'] == len(tabla)
    (Path(salida)/f'completitud_{nombre}.json').write_text(json.dumps(dict(
        Estado_entrega='COMPLETO' if completo else 'INCOMPLETO', Casos_esperados=len(tabla), **conteo,
        Origen=tabla.Origen.value_counts().to_dict() if 'Origen' in tabla else {},
        Pendientes=tabla[tabla.Estado_caso != 'COMPLETADO'][['Grupo', 'Escenario_piloto', 'Etiqueta', 'Estado_caso']]
        .head(200).to_dict('records')), ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"completitud {nombre}: {'COMPLETO' if completo else 'INCOMPLETO'} {conteo}", flush=True)
    return completo


# ------------------------------------------------------------------------------------ reutilización de K3
def _sin_codigo(k):
    return {c: v for c, v in k.items() if c not in ('codigo', 'config')}


def registrar_k3(salida, fuente=K3_VPN, m=None):
    """Registra las anualidades de K3 con 0 y 50 USD/t como reutilizadas si sus entradas son idénticas.

    Compara la clave completa (escenario, carbono, disponibilidad, BESS, techo, parámetros, huellas de las series,
    anualidad y huella de entradas) salvo las huellas de código, y comprueba la huella del `.npz` original."""
    m = m or cargar_malla()
    salida, fuente = Path(salida), Path(fuente)
    k3 = [cid for cid, c in configs(m).items() if es_k3(c)][0]
    registro = []
    destinos = [(k3, c) for c in m['reglas']] + [(REF_RED, 'RED_TERMICA'), (ref_pv(1.), 'PV_SIN_BESS')]
    for n in REUTILIZABLES:
        if n not in m['escenarios_carbono']:
            continue
        for s in m['senales']:
            for grupo, caso in destinos:
                cfg = config_grupo(grupo, m)
                for a in range(1, H + 1):
                    et = V.etiqueta(s, caso, a)
                    origen = fuente/n/f'{et}_meta.json'
                    destino = salida/grupo/n/f'{et}_meta.json'
                    fila = dict(Grupo=grupo, Escenario_piloto=n, Etiqueta=et, Fuente=str(origen))
                    if destino.exists():
                        fila['Resultado'] = 'ya_registrada'
                        registro.append(fila)
                        continue
                    if not origen.exists():
                        fila['Resultado'] = 'fuente_ausente'
                        registro.append(fila)
                        continue
                    d = json.loads(origen.read_text(encoding='utf-8'))
                    k = V.clave_anual(n, s, caso, a, cfg)
                    if d.get('Ruta_npz'):
                        npz_origen = Path(d['Ruta_npz'])
                    elif d.get('Origen') == 'reutilizada':      # red + térmica de los años 2-20: series del año 1
                        npz_origen = origen.with_name(f"{d['Fuente']}.npz")
                    else:
                        npz_origen = origen.with_name(f'{et}.npz')
                    iguales = json.dumps(_sin_codigo(d['clave']), sort_keys=True, default=str) == \
                        json.dumps(_sin_codigo(k), sort_keys=True, default=str)
                    npz_ok = npz_origen.exists() and d.get('SHA256_npz') is not None and sha(npz_origen) == d['SHA256_npz']
                    if d.get('Estado') != 'OK' or not iguales or not npz_ok:
                        fila['Resultado'] = 'rechazada: ' + ('estado' if d.get('Estado') != 'OK' else
                                                             'clave distinta' if not iguales else 'npz')
                        registro.append(fila)
                        continue
                    nuevo = {c: v for c, v in d.items() if c not in ('clave', 'firma')}
                    nuevo.update(Origen='reutilizada_K3', Origen_fuente=d.get('Origen'), Fuente=str(origen),
                                 Fuente_firma=d['firma'], Ruta_npz=str(npz_origen), firma=TA.firma(k))
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    destino.write_text(json.dumps(dict(nuevo, clave=k), ensure_ascii=False, indent=1, default=str),
                                       encoding='utf-8')
                    fila['Resultado'] = 'reutilizada'
                    registro.append(fila)
    tabla = pd.DataFrame(registro)
    tabla.to_csv(salida/'reutilizacion_K3.csv', index=False)
    print('reutilización K3:', tabla.Resultado.value_counts().to_dict(), flush=True)
    return tabla


# ------------------------------------------------------------------------------------ ejecución
def tarea(salida, grupo, n, s, caso, a):
    cfg = config_grupo(grupo)
    return V.tarea_anual(Path(salida)/grupo, n, s, caso, a, config=cfg)


def pendientes(salida, anios, m=None, grupos=None, escenarios=None):
    m = m or cargar_malla()
    inv = inventario(m)
    inv = inv[inv.Anio.isin(anios)]
    if grupos is not None:
        inv = inv[inv.Grupo.isin(grupos)]
    if escenarios is not None:
        inv = inv[inv.Escenario_piloto.isin(escenarios)]
    # La red + térmica de los años 2-20 se registra desde su año 1 fuera de los procesos.
    inv = inv[(inv.Caso != 'RED_TERMICA') | (inv.Anio == 1)]
    salida = Path(salida)
    faltan = [e for e in inv.itertuples(index=False)
              if not (salida/e.Grupo/e.Escenario_piloto/f'{e.Etiqueta}_meta.json').exists()]
    orden = {'D_INT': 0, 'B_INT': 1, 'A_INT': 2, 'RED_TERMICA': 3, 'PV_SIN_BESS': 4}
    cfgs = configs(m)
    # Las tareas largas primero (D con poca energía por MW de PV tarda más): la cola final queda con tareas cortas.
    energia = {g: cfgs[g]['E_MWh']/cfgs[g]['f_PV'] if g in cfgs else 0. for g in {e.Grupo for e in faltan}}
    faltan.sort(key=lambda e: (orden[e.Caso], energia[e.Grupo], e.Anio, e.Grupo, e.Escenario_piloto, e.Senal))
    return [(e.Grupo, e.Escenario_piloto, e.Senal, e.Caso, int(e.Anio)) for e in faltan]


def ejecutar(lista, salida, procesos, acelerar=True):
    salida = Path(salida)
    progreso = salida/'progreso.log'
    fallos, hechos, total = 0, 0, len(lista)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=procesos, initializer=TA.inicializar, initargs=(acelerar,)) as pool:
        futuros = {pool.submit(tarea, salida, *t): t for t in lista}
        for fut in as_completed(futuros):
            t = futuros[fut]
            hechos += 1
            try:
                f = fut.result()
                linea = (f"{pd.Timestamp.now():%H:%M:%S} [{hechos}/{total}] {t[0]} {t[1]} {t[2]} {t[3]} a{t[4]:02d} "
                         f"{f['Estado']} {f.get('Origen', '')} {f.get('Segundos', 0):.1f}s {f.get('RAM_pico_MB')}MB")
            except Exception as e:   # se registra y se sigue: un fallo no oculta los demás casos
                fallos += 1
                carpeta = salida/t[0]/t[1]
                carpeta.mkdir(parents=True, exist_ok=True)
                (carpeta/f'{V.etiqueta(t[2], t[3], t[4])}_error.json').write_text(json.dumps(dict(
                    Estado='FALLIDO', Tipo=type(e).__name__, Mensaje=str(e)[:2000], Fecha=pd.Timestamp.now().isoformat()),
                    ensure_ascii=False, indent=1), encoding='utf-8')
                linea = (f"{pd.Timestamp.now():%H:%M:%S} [{hechos}/{total}] {t[0]} {t[1]} {t[2]} {t[3]} a{t[4]:02d} "
                         f"ERROR {type(e).__name__}: {str(e)[:200]}")
            with open(progreso, 'a', encoding='utf-8') as fh:
                fh.write(linea + '\n')
            if hechos % 25 == 0 or hechos == total or 'ERROR' in linea:
                print(linea, f'| {time.perf_counter() - t0:.0f}s', flush=True)
    return fallos


def registrar_red(salida, m=None):
    """Años 2-20 de la red + térmica desde el año 1 de su carpeta (sin procesos; exige entradas idénticas)."""
    m = m or cargar_malla()
    for n in m['escenarios_carbono']:
        for s in m['senales']:
            if not (Path(salida)/REF_RED/n/f"{V.etiqueta(s, 'RED_TERMICA', 1)}_meta.json").exists():
                continue
            for a in range(2, H + 1):
                tarea(salida, REF_RED, n, s, 'RED_TERMICA', a)


# ------------------------------------------------------------------------------------ financiero
AGREGADOS = ('Costo_financiero_USD', 'Cargo_carbono_USD', 'Costo_ajustado_USD', 'Gas_MMBtu', 'Generacion_termica_MWh',
             'Importacion_MWh', 'Exportacion_MWh', 'Arranques', 'CO2_termico_t', 'CO2e_red_t', 'Emisiones_operativas_t',
             'Emisiones_cobradas_t', 'Descarga_MWh', 'EFC_anio', 'PV_MWh')


def leer_anualidades(salida, inv):
    filas = []
    for e in inv.itertuples(index=False):
        p = Path(salida)/e.Grupo/e.Escenario_piloto/f'{e.Etiqueta}_meta.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        d.pop('clave', None)
        d.pop('firma', None)
        filas.append(dict(Grupo=e.Grupo, **d))
    return pd.DataFrame(filas)


def financiero(salida, m=None):
    m = m or cargar_malla()
    salida = Path(salida)
    inv = inventario(m)
    comp = evaluar_completitud(salida, inv)
    cfgs = configs(m)
    resumenes, flujos, omitidos = [], [], []

    def evaluar(grupo, n, s, proyecto, grupo_ref, referencia, cfg):
        tab = comp[comp.Grupo.isin([grupo, grupo_ref])]
        try:
            res, motivo = V.evaluar_vpn(salida/grupo, n, s, proyecto, referencia, tab, config=cfg,
                                        salida_ref=salida/grupo_ref)
        except AssertionError as e:
            res, motivo = None, f'verificación fallida: {e}'
        base = dict(Config=grupo, Escenario_piloto=n, Senal=s, Proyecto=proyecto, Referencia=referencia)
        if res is None:
            omitidos.append(dict(base, Motivo=motivo))
            return
        resumenes.append(dict(res[0], Config=grupo))
        flujos.append(res[1].assign(Config=grupo))

    for n in m['escenarios_carbono']:
        for s in m['senales']:
            for f in niveles_pv(m):
                evaluar(ref_pv(f), n, s, 'PV_SIN_BESS', REF_RED, 'RED_TERMICA', cfg_ref_pv(f))
            for cid, cfg in cfgs.items():
                for regla in m['reglas']:
                    evaluar(cid, n, s, regla, REF_RED, 'RED_TERMICA', cfg)
                    evaluar(cid, n, s, regla, ref_pv(cfg['f_PV']), 'PV_SIN_BESS', cfg)
    vpn = pd.DataFrame(resumenes)
    if len(vpn):
        vpn = vpn.rename(columns={'Config': 'Grupo'})
        for c in ('f_PV', 'P_MW', 'E_MWh'):
            vpn[c] = [cfgs[g][c] if g in cfgs else (int(g[-3:])/100 if c == 'f_PV' else 0.) for g in vpn.Grupo]
        vpn['PV_MWp'] = vpn.f_PV*m['pv_referencia_MWp']
        # Aditividad: proyecto completo = solo PV + incremental del BESS (financiero y ajustado)
        pv = vpn[vpn.Proyecto == 'PV_SIN_BESS'].set_index(['Escenario_piloto', 'Senal', 'f_PV'])
        for (g, n, s, p), x in vpn[vpn.Proyecto.isin(m['reglas'])].groupby(['Grupo', 'Escenario_piloto', 'Senal', 'Proyecto']):
            x = x.set_index('Referencia')
            if {'RED_TERMICA', 'PV_SIN_BESS'} <= set(x.index) and (n, s, cfgs[g]['f_PV']) in pv.index:
                for col in ('VPN_financiero_COP', 'VPN_ajustado_COP'):
                    dif = x.loc['RED_TERMICA', col] - pv.loc[(n, s, cfgs[g]['f_PV']), col] - x.loc['PV_SIN_BESS', col]
                    assert abs(dif) <= 1e-6*abs(x.loc['RED_TERMICA', col]) + 1., (g, n, s, p, col, dif)
        vpn['Aditividad_verificada'] = True
    return vpn, (pd.concat(flujos, ignore_index=True) if flujos else pd.DataFrame()), pd.DataFrame(omitidos), comp


# ------------------------------------------------------------------------------------ contraste de finalistas
def finalistas(salida):
    """Contraste operativo compatible de los finalistas (año 1): A/B frente a OPT_AB y D frente a OPT_D.

    Finalista: la configuración de mayor VPN ajustado de cada regla, tarifa y carbono (`mejores_por_regla.csv`).
    Es una referencia operativa anual con previsión perfecta, no un VPN óptimo multianual."""
    salida = Path(salida)
    mr = pd.read_csv(salida/'mejores_por_regla.csv')
    filas = []
    for (c, s, n, bench), g in mr.assign(Benchmark=mr.Regla.map(TA.BENCHMARKS)).groupby(
            ['Config', 'Senal', 'Escenario_piloto', 'Benchmark']):
        filas.append(dict(Config=c, Senal=s, Escenario_piloto=n, Benchmark=bench, Reglas=','.join(sorted(g.Regla)),
                          K3_ya_calculado=bool(c == 'M01' and n == TA.ESCENARIO_BENCHMARK)))
    t = pd.DataFrame(filas)
    t.to_csv(salida/'finalistas_benchmarks.csv', index=False)
    return t


PILOTO = Path('resultados_carbono_v2/piloto_K3_20260926')
REGLAS_COMPATIBLES = {'OPT_AB': ('A_INT', 'B_INT'), 'OPT_D': ('D_INT',)}
TOL_SOC_BENCH_MWh = 1e-6
CAMPOS_COMPATIBILIDAD = ('escenario', 'carbono', 'senal', 'disponibilidad', 'bess', 'P_EXP_MAX', 'TRM', 'cargo',
                         'parametros_termicos', 'entradas')


def _contexto_caso(n, s):
    import carbono as C
    ctx = TA.contexto()
    esc, con_disp = TA.escenario(n)
    return (esc, C.parametros_ajustados(ctx['P'], esc), np.asarray(ctx['C_IMP_COP'][s], float)/ctx['TRM'],
            ctx['disp_v2'].to_numpy() if con_disp else None)


def revalidar(carpeta, etiqueta, n, s, meta):
    """Relee una trayectoria guardada y la revalida con los validadores del repositorio: balances, SOC, compromiso,
    mínimos y máximos, eventos (`termica_vpn.auditar`), disponibilidad, cierre cíclico del SOC y reconstrucción de
    costos y emisiones (`carbono.contabilidad`, que exige costo ajustado = objetivo) frente a su metadato."""
    import carbono as C
    import termica_politicas_anual as AN
    import termica_vpn as TV
    r = AN.leer_caso(Path(carpeta), etiqueta)
    esc, P, ci_fin, d = _contexto_caso(n, s)
    TV.auditar(r, P)
    if d is not None:
        assert not np.asarray(r['compromiso']).astype(bool)[~d].any(), 'compromiso en hora indisponible'
    cierre = float(np.asarray(r['SOC'], float)[-1] - float(r['SOC_inicial']))
    assert abs(cierre) <= TOL_SOC_BENCH_MWh, f'frontera cíclica del SOC no cerrada: {cierre}'
    k = C.contabilidad(r, P, esc, ci_fin)
    for c in ('Costo_ajustado_USD', 'Costo_financiero_USD', 'Emisiones_operativas_t', 'Emisiones_cobradas_t'):
        assert abs(k[c] - meta[c]) <= 1e-6*max(1., abs(meta[c])), (c, k[c], meta[c])
    return dict(Validaciones='OK', Cierre_SOC_MWh=cierre, Costo_ajustado_recalculado_USD=k['Costo_ajustado_USD'])


def reutilizar_benchmark_piloto(salida, cid, n, s, caso):
    """Registra un benchmark de K3 del piloto solo si sus entradas son idénticas y se revalida sin error.

    Comprueba: archivos, huella del `.npz`, estado, clave completa salvo huellas de código (escenario, carbono,
    disponibilidad, tamaños, techo, parámetros, cargo, series y procedimiento: 600 s y brecha 1e-6) y revalidación."""
    cfg = configs()[cid]
    if not es_k3(cfg):
        return dict(Resultado='no aplica: la configuración no es K3')
    et = f'{s}_{caso}'
    origen = PILOTO/n/f'{et}_meta.json'
    npz = origen.with_name(f'{et}.npz')
    if not (origen.exists() and npz.exists() and origen.with_name(f'{et}.json.gz').exists()):
        return dict(Resultado='rechazada: faltan archivos')
    d = json.loads(origen.read_text(encoding='utf-8'))
    if d.get('Estado') != 'OK':
        return dict(Resultado=f"rechazada: estado {d.get('Estado')}")
    if sha(npz) != d.get('SHA256_npz'):
        return dict(Resultado='rechazada: huella del npz distinta')
    k = TA.clave(n, s, caso, 600, 1e-6, config=cfg)
    if json.dumps(_sin_codigo(d['clave']), sort_keys=True, default=str) != json.dumps(_sin_codigo(k), sort_keys=True, default=str):
        return dict(Resultado='rechazada: clave distinta')
    try:
        rev = revalidar(PILOTO/n, et, n, s, d)
    except AssertionError as e:
        return dict(Resultado=f'rechazada: revalidación ({e})')
    destino = Path(salida)/'BENCHMARKS'/cid/n/f'{et}_meta.json'
    destino.parent.mkdir(parents=True, exist_ok=True)
    nuevo = {c: v for c, v in d.items() if c not in ('clave', 'firma')}
    nuevo.update(Config=cid, Caso=caso, Origen='reutilizada_piloto', Fuente=str(origen), Fuente_firma=d['firma'],
                 Ruta_npz=str(npz), Carpeta_series=str(PILOTO/n), Revalidacion=rev, firma=TA.firma(k))
    destino.write_text(json.dumps(dict(nuevo, clave=k), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    return dict(Resultado='reutilizada', **rev)


def tarea_benchmark(salida, cid, n, s, caso, segundos=600, brecha=1e-6):
    """Benchmark anual (año 1) de una configuración finalista; se guarda una vez con su clave.

    `TA.resolver` valida la solución (balances, SOC, compromiso, eventos, disponibilidad) y reconstruye costos y
    emisiones antes de guardarla. Con límite de tiempo, la solución es factible con cota, no un óptimo probado."""
    cfg = configs()[cid]
    carpeta = Path(salida)/'BENCHMARKS'/cid/n
    carpeta.mkdir(parents=True, exist_ok=True)
    et = f'{s}_{caso}'
    meta = carpeta/f'{et}_meta.json'
    k = TA.clave(n, s, caso, segundos, brecha, config=cfg)
    if meta.exists():
        d = json.loads(meta.read_text(encoding='utf-8'))
        if d['firma'] != TA.firma(k):
            raise RuntimeError(f'{cid}/{n}/{et}: clave distinta de la guardada')
        return d
    import termica_politicas_anual as AN
    out = TA.resolver(n, s, caso, segundos, brecha, config=cfg)
    r = out['resultado']
    AN.guardar_caso(carpeta, et, r)
    cont = dict(out['contabilidad'])
    fila = dict(Config=cid, Escenario_piloto=n, Senal=s, Caso=caso, Estado='OK', Origen='ejecutada',
                Escenario_carbono=cont.pop('Escenario'), **cont,
                Condicion=r.get('Condicion'), Objetivo_motor_USD=r['Objetivo'], Cota_USD=r.get('Cota'),
                Brecha_solver=r.get('Brecha_solver'), Segundos=out['segundos'], RAM_pico_MB=V.ram_pico_MB(),
                SOC_inicial_MWh=float(r['SOC_inicial']), Cierre_SOC_MWh=float(r['SOC'][-1] - r['SOC_inicial']),
                Carpeta_series=str(carpeta), SHA256_npz=TA.sha_archivo(carpeta/f'{et}.npz'), firma=TA.firma(k))
    meta.write_text(json.dumps(dict(fila, clave=k), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    return fila


def contraste_benchmarks(salida):
    """Año 1: cada regla compatible frente a su benchmark (A y B frente a OPT_AB; D frente a OPT_D).

    Exige la misma clave de entradas (escenario, carbono, disponibilidad, tamaños, techo, parámetros, cargo y
    series) en la regla (año 1, escala PV 1) y en el benchmark. El óptimo del año 1, C*, está entre la cota y la
    mejor solución factible; la distancia de la regla a C* está en [C_regla − factible, C_regla − cota]. No es una
    diferencia de VPN a veinte años: estos benchmarks no optimizan el VPN multianual."""
    salida = Path(salida)
    fin = pd.read_csv(salida/'finalistas_benchmarks.csv')
    filas = []
    for e in fin.itertuples(index=False):
        et = f'{e.Senal}_{e.Benchmark}'
        mb = salida/'BENCHMARKS'/e.Config/e.Escenario_piloto/f'{et}_meta.json'
        base = dict(Config=e.Config, Senal=e.Senal, Escenario_piloto=e.Escenario_piloto, Benchmark=e.Benchmark,
                    Reglas_finalistas=e.Reglas)
        if not mb.exists():
            filas.append(dict(base, Estado_benchmark='PENDIENTE'))
            continue
        b = json.loads(mb.read_text(encoding='utf-8'))
        rev = revalidar(b['Carpeta_series'], et, e.Escenario_piloto, e.Senal, b)
        cond = str(b.get('Condicion'))
        brecha = b.get('Brecha_solver')
        certificado = cond.endswith('optimal') and brecha is not None and float(brecha) <= 1e-6
        for regla in REGLAS_COMPATIBLES[e.Benchmark]:
            mr = json.loads((salida/e.Config/e.Escenario_piloto/f'{V.etiqueta(e.Senal, regla, 1)}_meta.json')
                            .read_text(encoding='utf-8'))
            kr, kb = mr['clave'], b['clave']
            compatibles = all(json.dumps(kr.get(c), sort_keys=True, default=str) == json.dumps(kb.get(c), sort_keys=True, default=str)
                              for c in CAMPOS_COMPATIBILIDAD) and kr.get('escala_pv') == 1.
            cr, cf, cota = mr['Costo_ajustado_USD'], b['Costo_ajustado_USD'], b.get('Cota_USD')
            filas.append(dict(base, Regla=regla, Regla_es_finalista=regla in e.Reglas.split(','),
                              Estado_benchmark='COMPLETADO', Origen_benchmark=b.get('Origen', 'ejecutada'),
                              Condicion=cond, Certificado_brecha_1e6=certificado,
                              Terminacion=('certificado dentro de la tolerancia relativa 1e-6' if certificado
                                           else 'límite de tiempo: factible con cota (sin certificar)'),
                              Mejor_factible_USD=cf, Cota_USD=cota, Brecha_solver=brecha, Segundos_solver=b.get('Segundos'),
                              RAM_pico_MB=b.get('RAM_pico_MB'), Validaciones=rev['Validaciones'],
                              Cierre_SOC_benchmark_MWh=rev['Cierre_SOC_MWh'], Entradas_compatibles=compatibles,
                              Costo_regla_ajustado_USD=cr, Costo_regla_financiero_USD=mr['Costo_financiero_USD'],
                              Costo_benchmark_financiero_USD=b['Costo_financiero_USD'],
                              Emisiones_regla_t=mr['Emisiones_operativas_t'], Emisiones_benchmark_t=b['Emisiones_operativas_t'],
                              Diferencia_vs_factible_USD=cr - cf, Diferencia_vs_cota_USD=None if cota is None else cr - cota,
                              Diferencia_vs_factible_pct=100*(cr - cf)/cf,
                              Diferencia_vs_cota_pct=None if cota is None else 100*(cr - cota)/cota,
                              Diferencia_vs_factible_pct_del_costo_regla=100*(cr - cf)/cr,
                              Regla_mas_barata_que_factible=bool(cr < cf)))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(salida/'benchmarks_finalistas.csv', index=False)
    return tabla


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--salida', type=Path, required=True)
    p.add_argument('--etapa', required=True,
                   choices=('reutilizar', 'medir', 'anual', 'financiero', 'verificar', 'finalistas', 'benchmarks',
                            'contraste'))
    p.add_argument('--procesos', type=int, default=4)
    p.add_argument('--grupos', nargs='*', default=None)
    p.add_argument('--escenarios', nargs='*', default=None)
    p.add_argument('--sin-numba', action='store_true')
    a = p.parse_args(argv)
    a.salida.mkdir(parents=True, exist_ok=True)
    m = cargar_malla()
    if a.etapa == 'reutilizar':
        (a.salida/'malla_carbono_v2.json').write_bytes(MALLA.read_bytes())
        registrar_k3(a.salida, m=m)
        return 0
    if a.etapa in ('medir', 'anual'):
        anios = [1] if a.etapa == 'medir' else list(range(1, H + 1))
        lista = pendientes(a.salida, anios, m, a.grupos, a.escenarios)
        print(f'{len(lista)} anualidades por resolver', flush=True)
        ejecutar(lista, a.salida, a.procesos, not a.sin_numba)
        registrar_red(a.salida, m)
        inv = inventario(m)
        inv = inv[inv.Anio.isin(anios)]
        if a.grupos is not None:
            inv = inv[inv.Grupo.isin(a.grupos)]
        if a.escenarios is not None:
            inv = inv[inv.Escenario_piloto.isin(a.escenarios)]
        return 0 if informar(a.salida, evaluar_completitud(a.salida, inv), a.etapa) else 1
    if a.etapa == 'verificar':
        return 0 if informar(a.salida, evaluar_completitud(a.salida, inventario(m)), 'total') else 1
    if a.etapa == 'finalistas':
        print(finalistas(a.salida).to_string(), flush=True)
        return 0
    if a.etapa == 'contraste':
        print(contraste_benchmarks(a.salida).to_string(), flush=True)
        return 0
    if a.etapa == 'benchmarks':   # PyPSA anual por finalista (600 s, brecha 1e-6); como máximo dos procesos
        t = pd.read_csv(a.salida/'finalistas_benchmarks.csv')
        if a.grupos is not None:
            t = t[t.Config.isin(a.grupos)]
        if a.escenarios is not None:
            t = t[t.Escenario_piloto.isin(a.escenarios)]
        # La marca K3_ya_calculado es solo una etiqueta: la reutilización exige la verificación completa.
        reuso = []
        for e in t[t.K3_ya_calculado].itertuples(index=False):
            destino = a.salida/'BENCHMARKS'/e.Config/e.Escenario_piloto/f'{e.Senal}_{e.Benchmark}_meta.json'
            r = (dict(Resultado='ya_registrada') if destino.exists()
                 else reutilizar_benchmark_piloto(a.salida, e.Config, e.Escenario_piloto, e.Senal, e.Benchmark))
            reuso.append(dict(Config=e.Config, Senal=e.Senal, Escenario_piloto=e.Escenario_piloto, Benchmark=e.Benchmark, **r))
            print('reutilización', e.Config, e.Escenario_piloto, e.Senal, e.Benchmark, r['Resultado'], flush=True)
        if reuso:
            pd.DataFrame(reuso).to_csv(a.salida/'reutilizacion_benchmarks.csv', index=False)
        t = t[[not (a.salida/'BENCHMARKS'/e.Config/e.Escenario_piloto/f'{e.Senal}_{e.Benchmark}_meta.json').exists()
               for e in t.itertuples(index=False)]]
        print(f'{len(t)} benchmarks por resolver', flush=True)
        fallos = 0
        with ProcessPoolExecutor(max_workers=a.procesos, initializer=TA.inicializar, initargs=(not a.sin_numba,)) as pool:
            futuros = {pool.submit(tarea_benchmark, a.salida, e.Config, e.Escenario_piloto, e.Senal, e.Benchmark): e
                       for e in t.itertuples(index=False)}
            for fut in as_completed(futuros):
                e = futuros[fut]
                try:
                    f = fut.result()
                    linea = (f"{pd.Timestamp.now():%H:%M:%S} {e.Config} {e.Escenario_piloto} {e.Senal} {e.Benchmark} "
                             f"{f.get('Condicion')} brecha={f.get('Brecha_solver')} {f.get('Segundos', 0):.0f}s "
                             f"{f.get('RAM_pico_MB')}MB")
                except Exception as ex:
                    fallos += 1
                    carpeta = a.salida/'BENCHMARKS'/e.Config/e.Escenario_piloto
                    carpeta.mkdir(parents=True, exist_ok=True)
                    (carpeta/f'{e.Senal}_{e.Benchmark}_error.json').write_text(json.dumps(dict(
                        Estado='FALLIDO', Tipo=type(ex).__name__, Mensaje=str(ex)[:2000],
                        Fecha=pd.Timestamp.now().isoformat()), ensure_ascii=False, indent=1), encoding='utf-8')
                    linea = f"{pd.Timestamp.now():%H:%M:%S} {e.Config} {e.Escenario_piloto} {e.Senal} {e.Benchmark} ERROR {ex}"
                print(linea, flush=True)
                with open(a.salida/'benchmarks_progreso.log', 'a', encoding='utf-8') as fh:
                    fh.write(linea + chr(10))
        contraste_benchmarks(a.salida)
        return 0 if fallos == 0 else 1
    import termica_carbono_malla_informe as INF
    return INF.informe(a.salida)


if __name__ == '__main__':
    sys.exit(main())
