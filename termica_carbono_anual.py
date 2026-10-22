"""Piloto anual de la versión 2: precio interno de carbono y disponibilidad térmica con K3 a su tamaño vigente.

Ejecutor anual de la versión 2 para A_INT, B_INT y D_INT, sus benchmarks compatibles (OPT_AB, OPT_D) y las
referencias red + térmica y PV sin BESS. Reutiliza sin cambios los motores y validadores del repositorio y el
procedimiento de `termica_politicas_anual.py` (A con su ciclo original; B y D con punto fijo cíclico del SOC;
benchmarks con SOC cíclico, límite de tiempo y cotas). No redimensiona ni ejecuta C/F.

Escenarios (parámetros explícitos, sin mutaciones globales):

    C1_v1_p0          carbono 0, sin disponibilidad        → debe reproducir la versión 1 vigente
    V2_p0             carbono 0, calendario V2 (437 h)      → referencia V2
    V2_p26, V2_p50    gas_y_red, factor XM 2025, V2
    V2_p50_solo_gas   50 USD/t solo a la térmica, V2
    V2_p50_UPME2024   50 USD/t gas_y_red con factor UPME 2024, V2

C1 y V2_p0 no deben coincidir: su diferencia es la disponibilidad, no el carbono.

Uso, desde la raíz, con Numba aislado si se desea (ver REPRODUCCION.md):

    python termica_carbono_anual.py --salida resultados_carbono_v2/piloto_K3_20260926 --etapa reglas --procesos 14
    python termica_carbono_anual.py --salida ... --etapa benchmarks --procesos 4
    python termica_carbono_anual.py --salida ... --etapa pertenencia --procesos 4
    python termica_carbono_anual.py --salida ... --etapa consolidar [--destino carpeta_nueva]
    python termica_carbono_anual.py --salida ... --etapa verificar [--destino carpeta_nueva]

Cada caso se guarda una sola vez con su clave (escenario, factores, disponibilidad, tamaños, código y
entradas). Si la clave cambia, el ejecutor se detiene: usar otra carpeta. Los resultados previos no se tocan.

Completitud (revisión del 26/09/2026): cada etapa compara lo guardado con su inventario de casos esperados y
clasifica cada uno como COMPLETADO, FALLIDO (error registrado en `<etiqueta>_error.json`), FALTANTE,
NO_CONVERGE o IDENTIDAD_INCOHERENTE. El proceso devuelve código 1 si algún caso requerido no quedó completado;
los errores no detienen a los demás trabajos y los casos válidos se reanudan desde su caché. Los metadatos
nuevos guardan `Escenario_piloto` y `Escenario_carbono`; los del piloto del 26/09/2026 (campo `Escenario`
sobrescrito por la contabilidad) se leen de forma compatible desde `clave`.

Tamaños (malla de dimensionamiento, 26/09/2026): `resolver` y `clave` aceptan `config` = {f_PV, P_MW, E_MWh}.
Sin `config` se usa K3 exactamente como antes. Con `config`, el perfil PV se escala por `f_PV`, el BESS toma
`P_MW` y `E_MWh` (mismos DoD, RTE y SOC) y el techo de exportación sigue la convención del factorial,
máx(PV) + P_BESS. Los escenarios de la malla (`ESCENARIOS_MALLA`, 10 y 20 USD/t) no entran en el inventario
del piloto.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import carbono as C
import costos
import disponibilidad_termica as DT
import termica_politicas as POL
import termica_politicas_anual as AN
import termica_reglas as REG
import termica_vpn as V

SENALES = ('T0', 'T1')
POLITICAS = ('A_INT', 'B_INT', 'D_INT')
BASES = ('RED_TERMICA', 'PV_SIN_BESS')
BENCHMARKS = {'A_INT': 'OPT_AB', 'B_INT': 'OPT_AB', 'D_INT': 'OPT_D'}
ESCENARIOS = {
    'C1_v1_p0': (C.EscenarioCarbono(0.), False),
    'V2_p0': (C.EscenarioCarbono(0.), True),
    'V2_p26': (C.EscenarioCarbono(26.), True),
    'V2_p50': (C.EscenarioCarbono(50.), True),
    'V2_p50_solo_gas': (C.EscenarioCarbono(50., 'solo_gas'), True),
    'V2_p50_UPME2024': (C.escenario_upme_2024(50.), True),
}
ESCENARIOS_MALLA = {
    'V2_p10': (C.EscenarioCarbono(10.), True),
    'V2_p20': (C.EscenarioCarbono(20.), True),
}
ESCENARIO_BENCHMARK = 'V2_p50'
V1_CASOS = Path('resultados_correccion_numerica/20260920/factorial/casos')
CODIGO = ('termica_carbono_anual.py', 'carbono.py', 'costos.py', 'costos_curvas.py', 'disponibilidad_termica.py',
          'termica_politicas.py', 'termica_politicas_anual.py', 'termica_reglas.py', 'termica_pypsa.py', 'termica_vpn.py',
          'termica_pv.py', 'termica_bess.py', 'termica.py', 'termica_dp_acelerada.py', 'termica_cf.py',
          'motor_multianual.py', 'benchmark_optimo.py', 'tarifas_termica_v2.py', 'demanda.py', 'solar_tmy.py')
MAX_ITER, TOL_CICLO = 25, 1e-4
_CTX = None


def sha_archivo(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_arreglo(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def contexto():
    """Entradas canónicas K3 (una vez por proceso) y calendario V2 sobre la demanda procesada del caso."""
    global _CTX
    if _CTX is None:
        ctx = REG.entradas()
        E = ctx['E']
        ctx['fechas'] = pd.DatetimeIndex(E['FECHA'])
        assert ctx['fechas'].equals(pd.date_range('2025-01-01', periods=8760, freq='h'))
        ctx['disp_v2'] = DT.serie_disponibilidad(ctx['fechas'], E['LOAD'])
        res = DT.resumen(ctx['disp_v2'], E['LOAD'])
        assert (res['horas_parada'], res['horas_mantenimiento'], res['horas_indisponibles']) == (101, 336, 437), res
        _CTX = ctx
    return _CTX


def escenario(nombre):
    """(EscenarioCarbono, con_disponibilidad) del piloto o de la malla."""
    return ESCENARIOS[nombre] if nombre in ESCENARIOS else ESCENARIOS_MALLA[nombre]


def dimensionar(config=None):
    """(PV del año 1, BESS, techo de exportación, MWp DC) de K3 (`config` None) o de una configuración."""
    ctx = contexto()
    E = ctx['E']
    if config is None:
        return np.asarray(E['PV_MW'], float), ctx['b'], ctx['P_EXP_MAX'], float(E['PV_MWp'])
    f = float(config['f_PV'])
    pv = np.asarray(E['PV_MW'], float)*f
    b = dict(ctx['b'], P_MW=float(config['P_MW']), E_MWh=float(config['E_MWh']))
    return pv, b, float(np.max(pv) + b['P_MW']), float(E['PV_MWp'])*f


def pv_anualidad(config, escala_pv):
    """Perfil PV de una anualidad; una sola definición para el despacho y para las huellas."""
    return dimensionar(config)[0]*float(escala_pv)


def clave(nombre, senal, caso, segundos, brecha, config=None):
    ctx = contexto()
    esc, con_disp = escenario(nombre)
    E = ctx['E']
    pv, b, pe, pv_MWp = dimensionar(config)
    k = dict(escenario=nombre, carbono=esc.clave(), senal=senal, caso=caso,
             disponibilidad=('V2', sha_arreglo(ctx['disp_v2'].to_numpy())) if con_disp else None,
             bess=b, P_EXP_MAX=pe, TRM=ctx['TRM'], cargo=REG.CARGO,
             parametros_termicos={k: np.asarray(v).tolist() if np.ndim(v) else v for k, v in ctx['P'].items()},
             entradas=dict(carga=sha_arreglo(E['LOAD']), pv=sha_arreglo(pv), pv_MWp=pv_MWp,
                           c_imp=sha_arreglo(ctx['C_IMP_COP'][senal]), c_exp=sha_arreglo(ctx['C_EXP_COP'])),
             procedimiento=dict(max_iter=MAX_ITER, tol_ciclo=TOL_CICLO, segundos=segundos, brecha=brecha),
             codigo={f: sha_archivo(f) for f in CODIGO})
    if config is not None:
        k['config'] = {c: float(config[c]) for c in ('f_PV', 'P_MW', 'E_MWh')}
    return k


def firma(k):
    return hashlib.sha256(json.dumps(k, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def resolver(nombre, senal, caso, segundos, brecha, fijar_de=None, escala_pv=1., config=None):
    """Resuelve un caso del piloto con parámetros y señal ajustados; valida y contabiliza.

    `escala_pv` multiplica el perfil solar (degradación de una anualidad); 1 reproduce el año 1. La
    disponibilidad se construye siempre sobre la demanda procesada, no sobre el residual. `config` fija PV,
    potencia y energía del BESS (None: K3)."""
    ctx = contexto()
    esc, con_disp = escenario(nombre)
    P = C.parametros_ajustados(ctx['P'], esc)
    _, b, pe, _ = dimensionar(config)
    trm = ctx['TRM']
    f = ctx['fechas']
    pv, load = pv_anualidad(config, escala_pv), np.asarray(ctx['E']['LOAD'], float)
    ci_fin = np.asarray(ctx['C_IMP_COP'][senal], float)/trm
    ci, ce = C.senal_ajustada(ci_fin, esc), np.asarray(ctx['C_EXP_COP'], float)/trm
    d = ctx['disp_v2'].to_numpy() if con_disp else None
    ciclo = None
    t0 = time.perf_counter()
    if caso in BASES:
        m = dict(parametros_termicos=P, cargo_arranque_USD=REG.CARGO, P_EXP_MAX=pe, bess=b)
        r = V.resolver_base(caso, (f, pv, load, ci, ce), m, disponible=d)
    elif caso == 'A_INT':
        env = REG.entorno(ctx['ns'], f)
        r = POL.simular_a_int(caso, env, f, pv, load, ci*trm, ctx['C_EXP_COP'], trm, b, pe, P, REG.CARGO, 'libre',
                              disponible=d)
        residuo = float(r['SOC'][-1] - r['SOC_inicial'])
        ciclo = dict(estado='CONVERGE' if abs(residuo) <= POL.TOL_CIERRE_MWh else 'NO_CONVERGE', iteraciones=None,
                     residuo=residuo, procedimiento='ciclo original de A del cuaderno 02')
    elif caso in ('B_INT', 'D_INT'):
        motor = POL.simular_b_int if caso == 'B_INT' else POL.simular_d_int
        c = POL.ciclo_soc_politica(lambda e: motor(f, pv, load, ci, ce, b, pe, P, REG.CARGO, 'libre', e,
                                                   frontera='ciclica_anual', disponible=d),
                                   b, max_iter=MAX_ITER, tol=TOL_CICLO)
        r = c.pop('resultado')
        ciclo = {k: v for k, v in c.items() if k != 'historial'} | dict(historial=c['historial'])
        if r is None:
            return dict(estado=c['estado'], ciclo=ciclo, segundos=time.perf_counter() - t0)
    elif caso in ('OPT_AB', 'OPT_D'):
        fx = None
        if fijar_de is not None:
            import termica_politicas_ciclica as CIC
            fx = CIC.fijar_desde(fijar_de)
            fx['descarga_export'] = fijar_de['Descarga_export']
        r = REG.optimo(caso, f, pv, load, ci, ce, P, b, pe, frontera_soc='ciclica', cargo=REG.CARGO, segundos=segundos,
                       brecha=brecha, exclusion='binaria', aceptar_limite_tiempo=True, fijar=fx, disponible=d)
    else:
        raise ValueError(caso)
    segundos_caso = time.perf_counter() - t0
    V.auditar(r, P)
    if d is not None:
        assert not np.asarray(r['compromiso']).astype(bool)[~d].any(), 'compromiso en hora indisponible'
    k = C.contabilidad(r, P, esc, ci_fin)
    return dict(estado='OK', resultado=r, ciclo=ciclo, contabilidad=k, segundos=segundos_caso)


def etiqueta_de(senal, caso, fijar_de=None):
    return f'{fijar_de}_fijado_en_{caso}' if fijar_de else f'{senal}_{caso}'


def tarea(salida, nombre, senal, caso, segundos, brecha, etiqueta=None, fijar_de=None):
    """Resuelve un caso o devuelve el guardado si su clave coincide (reanudación sin repetir).

    Los metadatos guardan `Escenario_piloto` (carpeta del piloto) y `Escenario_carbono` (nombre del escenario
    de carbono) por separado: dos escenarios del piloto pueden compartir precio y factor."""
    salida = Path(salida)
    carpeta = salida/nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    etiqueta = etiqueta or etiqueta_de(senal, caso, fijar_de)
    meta = carpeta/f'{etiqueta}_meta.json'
    k = clave(nombre, senal, caso, segundos, brecha)
    if fijar_de is not None:
        k['fijado_desde'] = fijar_de
    fk = firma(k)
    if meta.exists():
        guardado = json.loads(meta.read_text(encoding='utf-8'))
        if guardado['firma'] != fk:
            raise RuntimeError(f'{nombre}/{etiqueta}: clave distinta de la guardada; usar otra carpeta')
        return guardado
    politica = None
    if fijar_de is not None:
        politica = AN.leer_caso(carpeta, fijar_de)
    out = resolver(nombre, senal, caso, segundos, brecha, fijar_de=politica)
    fila = dict(Escenario_piloto=nombre, Senal=senal, Caso=caso, Etiqueta=etiqueta, Estado=out['estado'],
                Segundos=out['segundos'], firma=fk)
    if out['estado'] == 'OK':
        r = out['resultado']
        AN.guardar_caso(carpeta, etiqueta, r)
        cont = dict(out['contabilidad'])
        fila['Escenario_carbono'] = cont.pop('Escenario')
        fila.update(cont, Condicion=r.get('Condicion', 'politica/CONVERGE'),
                    Cota_USD=r.get('Cota'), Brecha_solver=r.get('Brecha_solver'), Objetivo_motor_USD=r['Objetivo'],
                    SOC_inicial_MWh=float(r['SOC_inicial']), Cierre_SOC_MWh=float(r['SOC'][-1] - r['SOC_inicial']),
                    EFC=float(np.sum(np.asarray(r['Descarga_demanda']) + np.asarray(r['Descarga_export']))
                              / (ctx_bess_util() or 1.)),
                    SHA256_npz=sha_archivo(carpeta/f'{etiqueta}.npz'))
    else:
        fila['Escenario_carbono'] = escenario(nombre)[0].nombre()
    if out.get('ciclo') is not None:
        (carpeta/f'{etiqueta}_ciclo.json').write_text(json.dumps(out['ciclo'], ensure_ascii=False, indent=1,
                                                                  default=AN.codificar), encoding='utf-8')
        fila['Ciclo'] = out['ciclo']['estado']
    meta.write_text(json.dumps(dict(fila, clave=k), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    error = carpeta/f'{etiqueta}_error.json'
    if error.exists():   # un fallo anterior quedó resuelto al reanudar: se conserva como antecedente
        error.replace(carpeta/f'{etiqueta}_error_resuelto.json')
    return dict(fila, clave=k)


def ctx_bess_util():
    b = contexto()['b']
    return b['E_MWh']*b['DoD'] if b.get('DoD') else None


def inicializar(acelerar):
    if acelerar:
        try:
            from termica_dp_acelerada import activar
            activar()
        except ImportError:
            pass
    for n in ('pypsa', 'linopy'):
        logging.getLogger(n).setLevel(logging.ERROR)


def registrar_error(salida, trabajo, e):
    """Error de un caso: archivo propio (no es caché; el caso se reintenta al reanudar)."""
    nombre, senal, caso = trabajo[:3]
    etiqueta = trabajo[5] if len(trabajo) > 5 and trabajo[5] else etiqueta_de(senal, caso, trabajo[6] if len(trabajo) > 6 else None)
    carpeta = Path(salida)/nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta/f'{etiqueta}_error.json').write_text(json.dumps(dict(
        Escenario_piloto=nombre, Senal=senal, Caso=caso, Etiqueta=etiqueta, Estado='FALLIDO',
        Tipo=type(e).__name__, Mensaje=str(e)[:2000], Fecha=pd.Timestamp.now().isoformat()),
        ensure_ascii=False, indent=1), encoding='utf-8')


def ejecutar(trabajos, salida, procesos, acelerar):
    """Ejecuta los trabajos independientes; un fallo se registra y no detiene a los demás."""
    progreso = Path(salida)/'progreso.log'
    fallos = 0
    with ProcessPoolExecutor(max_workers=procesos, initializer=inicializar, initargs=(acelerar,)) as pool:
        futuros = {pool.submit(tarea, salida, *t): t for t in trabajos}
        for fut in as_completed(futuros):
            t = futuros[fut]
            try:
                fila = fut.result()
                linea = (f"{pd.Timestamp.now():%H:%M:%S} {t[0]} {t[1]} {t[2]} {fila['Estado']} "
                         f"{fila.get('Condicion', '')} {fila.get('Segundos', 0):.0f}s")
            except Exception as e:  # se registra y se sigue: un fallo no oculta los demás casos
                fallos += 1
                registrar_error(salida, t, e)
                linea = f"{pd.Timestamp.now():%H:%M:%S} {t[0]} {t[1]} {t[2]} ERROR {type(e).__name__}: {e}"
            print(linea, flush=True)
            with open(progreso, 'a', encoding='utf-8') as fh:
                fh.write(linea + '\n')
    return fallos


# ------------------------------------------------------------------------------------ completitud
ESTADOS = ('COMPLETADO', 'FALLIDO', 'FALTANTE', 'NO_CONVERGE', 'IDENTIDAD_INCOHERENTE')


def trabajos_de(etapa, segundos=600, brecha=1e-6):
    if etapa == 'reglas':
        t = [(n, s, c, segundos, brecha) for n in ESCENARIOS for s in SENALES for c in POLITICAS + BASES]
        return sorted(t, key=lambda x: {'D_INT': 0, 'B_INT': 1}.get(x[2], 2))
    if etapa == 'benchmarks':
        return [(ESCENARIO_BENCHMARK, s, c, segundos, brecha) for s in SENALES for c in ('OPT_AB', 'OPT_D')]
    if etapa == 'pertenencia':
        return [(ESCENARIO_BENCHMARK, s, BENCHMARKS[pol], 120, 0., etiqueta_de(s, BENCHMARKS[pol], f'{s}_{pol}'),
                 f'{s}_{pol}') for s in SENALES for pol in POLITICAS]
    raise ValueError(etapa)


def inventario(etapas=('reglas', 'benchmarks', 'pertenencia')):
    """Casos esperados: identidad (escenario, señal, caso, etiqueta) única por etapa y en conjunto."""
    filas = []
    for etapa in etapas:
        for t in trabajos_de(etapa):
            nombre, senal, caso = t[:3]
            fijar = t[6] if len(t) > 6 else None
            filas.append(dict(Etapa=etapa, Escenario_piloto=nombre, Senal=senal, Caso=caso,
                              Etiqueta=etiqueta_de(senal, caso, fijar), Fijado_desde=fijar))
    inv = pd.DataFrame(filas)
    if inv.duplicated(['Escenario_piloto', 'Etiqueta']).any():
        raise AssertionError('inventario con identidades duplicadas')
    return inv


def evaluar_completitud(salida, inv):
    """Estado de cada caso esperado a partir de lo guardado; no resuelve nada."""
    salida = Path(salida)
    filas = []
    for _, e in inv.iterrows():
        carpeta = salida/e.Escenario_piloto
        meta, error = carpeta/f'{e.Etiqueta}_meta.json', carpeta/f'{e.Etiqueta}_error.json'
        fila = dict(e)
        if not meta.exists():
            fila['Estado_caso'] = 'FALLIDO' if error.exists() else 'FALTANTE'
            if error.exists():
                fila['Detalle'] = json.loads(error.read_text(encoding='utf-8')).get('Mensaje', '')[:300]
            filas.append(fila)
            continue
        d = json.loads(meta.read_text(encoding='utf-8'))
        k = d.get('clave', {})
        coherente = (k.get('escenario') == e.Escenario_piloto and k.get('senal') == e.Senal and k.get('caso') == e.Caso
                     and d.get('Etiqueta') == e.Etiqueta and (pd.isna(e.Fijado_desde) or k.get('fijado_desde') == e.Fijado_desde))
        if not coherente:
            fila['Estado_caso'], fila['Detalle'] = 'IDENTIDAD_INCOHERENTE', 'clave o etiqueta no corresponden al caso esperado'
        elif d.get('Estado') != 'OK':
            fila['Estado_caso'], fila['Detalle'] = 'NO_CONVERGE', f"estado guardado {d.get('Estado')}"
        elif e.Caso in POLITICAS and abs(float(d.get('Cierre_SOC_MWh', np.inf))) > POL.TOL_CIERRE_MWh:
            fila['Estado_caso'], fila['Detalle'] = 'NO_CONVERGE', f"cierre del SOC {d.get('Cierre_SOC_MWh')}"
        else:
            fila['Estado_caso'] = 'COMPLETADO'
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    assert set(tabla.Estado_caso) <= set(ESTADOS)
    return tabla


def informar_completitud(salida, tabla, nombre):
    tabla.to_csv(Path(salida)/f'completitud_{nombre}.csv', index=False)
    conteo = tabla.Estado_caso.value_counts().reindex(ESTADOS, fill_value=0).to_dict()
    completo = conteo['COMPLETADO'] == len(tabla)
    resumen = dict(Estado_entrega='COMPLETO' if completo else 'INCOMPLETO', Casos_esperados=len(tabla), **conteo,
                   Pendientes=tabla[tabla.Estado_caso != 'COMPLETADO'][['Escenario_piloto', 'Etiqueta', 'Estado_caso']]
                   .to_dict('records'))
    (Path(salida)/f'completitud_{nombre}.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=1),
                                                           encoding='utf-8')
    print(f"completitud {nombre}: {resumen['Estado_entrega']} {conteo}", flush=True)
    return completo


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--salida', type=Path, required=True)
    p.add_argument('--etapa', required=True, choices=('reglas', 'benchmarks', 'pertenencia', 'consolidar', 'verificar'))
    p.add_argument('--procesos', type=int, default=4)
    p.add_argument('--segundos', type=int, default=600)
    p.add_argument('--brecha', type=float, default=1e-6)
    p.add_argument('--sin-numba', action='store_true')
    p.add_argument('--destino', type=Path, default=None,
                   help='consolidar: carpeta nueva para las tablas (la fuente no se modifica)')
    a = p.parse_args(argv)
    if a.etapa in ('reglas', 'benchmarks', 'pertenencia'):
        a.salida.mkdir(parents=True, exist_ok=True)
        ejecutar(trabajos_de(a.etapa, a.segundos, a.brecha), a.salida, a.procesos, not a.sin_numba)
        return 0 if informar_completitud(a.salida, evaluar_completitud(a.salida, inventario((a.etapa,))), a.etapa) else 1
    if a.etapa == 'verificar':
        destino = a.destino or a.salida
        destino.mkdir(parents=True, exist_ok=True)
        return 0 if informar_completitud(destino, evaluar_completitud(a.salida, inventario()), 'total') else 1
    import termica_carbono_informe as INF
    return 0 if INF.consolidar(a.salida, destino=a.destino, inventario=inventario())['completo'] else 1


if __name__ == '__main__':
    sys.exit(main())
