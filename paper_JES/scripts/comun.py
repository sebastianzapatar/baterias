"""Utilidades comunes de los refuerzos R1–R3 del artículo JES.

Reutiliza los motores vigentes de `OperacionBESS` sin modificarlos. Todo se ejecuta
desde la raíz de `OperacionBESS` (los módulos resuelven rutas contra `Path.cwd()`),
pero las salidas se escriben solo en `paper_JES/resultados_refuerzos/`.
Las carpetas cerradas de la tesis se leen, nunca se sobrescriben.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

RAIZ_JES = Path(__file__).resolve().parents[1]
OPBESS = RAIZ_JES.parent / 'OperacionBESS'
# En el repositorio público, paper_JES/ está dentro de la raíz de OperacionBESS.
if not OPBESS.is_dir():
    OPBESS = RAIZ_JES.parent
os.chdir(OPBESS)
sys.path.insert(0, str(OPBESS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import termica_factorial as F  # noqa: E402
import termica_vpn as V  # noqa: E402
import termica_reglas as REG  # noqa: E402
import termica_politicas_ciclica as CIC  # noqa: E402

SALIDA = RAIZ_JES / 'resultados_refuerzos'
CERRADO = OPBESS / 'resultados_correccion_numerica/20260920/factorial/casos'
CODIGO_MOTOR = ('termica_vpn.py', 'termica_politicas.py', 'termica_reglas.py', 'termica_pypsa.py',
                'termica_cf.py', 'termica_dp_acelerada.py', 'termica.py', 'termica_factorial.py',
                'termica_politicas_ciclica.py', 'costos.py')
BENCH_DE = {'A_INT': 'OPT_AB', 'B_INT': 'OPT_AB', 'D_INT': 'OPT_D'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def huella_codigo(extra=()):
    archivos = {f: sha(OPBESS / f) for f in CODIGO_MOTOR}
    archivos.update({str(Path(f).name): sha(f) for f in extra})
    return archivos


def json_guardar(path, valor):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(valor, ensure_ascii=False, indent=2, default=V.codificar), encoding='utf-8')
    tmp.replace(path)


def leer_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def contexto_c280(senal, gas=None, cargo=None):
    """Entradas de K3 (= C280 en el artículo) con, opcionalmente, otro precio del gas o cargo de arranque.

    Solo cambian `parametros_termicos['gas']`, `parametros_termicos['cv']` y `cargo_arranque_USD`;
    el resto (demanda, PV, señales, BESS, techo de exportación) es idéntico al de la corrida cerrada."""
    datos, m = F.contexto('K3', senal)
    m = deepcopy(m)
    p = m['parametros_termicos']
    if gas is not None:
        p['gas'] = float(gas)
        p['cv'] = np.asarray(p['hr'], float) * float(gas) + np.asarray(p['cvom'], float)
    if cargo is not None:
        m['cargo_arranque_USD'] = float(cargo)
    return datos, m


def caso_cerrado(senal, caso):
    """Resumen JSON y series de la corrida cerrada de K3 (gas 20, cargo 100)."""
    js = CERRADO / f'K3_{senal}_{caso}.json'
    fila = leer_json(js)
    assert sha(js.with_suffix('.npz')) == fila['SHA256_npz'], f'Hash alterado en {js}'
    return fila


def soc_inicial_cerrado(senal, caso):
    with np.load(CERRADO / f'K3_{senal}_{caso}.npz', allow_pickle=False) as z:
        return float(z['SOC_inicial'])


def indicadores(r, p, cargo):
    """Contabilidad completa (misma fórmula que `termica_vpn.contabilidad`) más indicadores físicos."""
    pot = np.asarray(r['potencia'], float)
    cr = np.asarray(r.get('Carga_Red', np.zeros(len(pot))), float)
    energias = pot[:, :4].sum(axis=0)
    gas = float(energias @ (np.asarray(p['hr']) * p['gas']))
    cvom = float(energias @ np.asarray(p['cvom']))
    compras = float((pot[:, 4] + cr) @ r['C_imp_USD'])
    arranques = float(np.asarray(r['arranques']).sum() * cargo)
    ventas = float((r['Export_PV'] + r['Descarga_export']) @ r['C_exp_USD'])
    util = r['BESS']['E_MWh'] * r['BESS']['DoD'] if r['BESS']['E_MWh'] > 0 else np.nan
    descarga = float(np.sum(r['Descarga_demanda'] + r['Descarga_export']))
    return dict(Costo_operativo_USD=gas + cvom + compras + arranques - ventas, Gas_USD=gas, CVOM_USD=cvom,
                Compras_red_USD=compras, Arranques_USD=arranques, Ingreso_exportacion_USD=ventas,
                Arranques=int(np.asarray(r['arranques']).sum()),
                Arranques_por_unidad=np.asarray(r['arranques']).sum(axis=0).astype(int).tolist(),
                Gas_MMBtu=float(energias @ np.asarray(p['hr'])), Termica_MWh=float(energias.sum()),
                Horas_con_termica=int((pot[:, :4].sum(axis=1) > 1e-6).sum()),
                Red_MWh=float((pot[:, 4] + cr).sum()), Descarga_MWh=descarga,
                Descarga_demanda_MWh=float(np.sum(r['Descarga_demanda'])),
                Export_BESS_MWh=float(np.sum(r['Descarga_export'])), Export_PV_MWh=float(np.sum(r['Export_PV'])),
                Vertimiento_MWh=float(np.sum(r['Vertimiento'])), EFC=descarga / util if util == util else 0.,
                SOC_inicial_MWh=float(r['SOC_inicial']), Cierre_SOC_MWh=float(r['SOC'][-1] - r['SOC_inicial']))


def guardar_caso(carpeta, etiqueta, r, fila):
    carpeta = Path(carpeta)
    (carpeta / 'casos').mkdir(parents=True, exist_ok=True)
    npz = carpeta / 'casos' / f'{etiqueta}.npz'
    js = npz.with_suffix('.json')
    assert not js.exists(), f'No se sobrescribe {etiqueta}'
    series = {k: np.asarray(r[k]) for k in V.SERIES}
    series.update(Fecha=pd.DatetimeIndex(r['Fechas']).to_numpy(), SOC_inicial=np.asarray(r['SOC_inicial']),
                  Carga_Red=np.asarray(r.get('Carga_Red', np.zeros(len(r['Fechas'])))))
    tmp = npz.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **series)
    tmp.replace(npz)
    fila = dict(fila, SHA256_npz=sha(npz))
    json_guardar(js, fila)
    return fila


def pertenencia(r, caso_bench, datos, m, p):
    """Fija la trayectoria en el MILP anual compatible (OPT_AB u OPT_D) con los mismos parámetros.

    Replica `.build/validar_pertenencia_politicas_anual.py`: comprueba que el modelo anual reproduce
    flujos, SOC, compromiso y objetivo dentro de la banda de fijación."""
    import math
    fx = CIC.fijar_desde(r)
    fx['descarga_export'] = r['Descarga_export']
    f, pv, load, ci, ce = datos
    with F.solver_un_hilo():
        fij = REG.optimo(caso_bench, f, pv, load, ci, ce, p, m['bess'], m['P_EXP_MAX'], frontera_soc='ciclica',
                         cargo=m['cargo_arranque_USD'], segundos=300, brecha=0., exclusion='binaria', fijar=fx)
    REG.validar_integrado(fij, p, caso_bench == 'OPT_D')
    assert fij['Condicion'].endswith('/optimal'), fij['Condicion']
    error = 0.
    for k in ('Carga_BESS', 'Descarga_demanda', 'Descarga_export', 'Export_PV', 'SOC', 'potencia'):
        dif = np.abs(np.asarray(fij[k]) - np.asarray(r[k]))
        escala = np.maximum(np.abs(fij[k]), np.abs(r[k]))
        assert np.all(dif <= fx['tol'] + 1e-9 + 2*np.spacing(escala)), (k, float(dif.max()))
        error = max(error, float(dif.max()))
    for k in ('compromiso', 'arranques', 'paradas'):
        np.testing.assert_array_equal(fij[k], r[k], err_msg=k)
    tol = fx['tol'] * math.fsum((np.sum(p['cv']) + ci + np.maximum(ce, 0.) + 1.).tolist()) + 1e-6
    dif_obj = abs(fij['Objetivo'] - r['Objetivo'])
    assert dif_obj <= tol, (dif_obj, tol)
    return dict(Pertenencia='OK', Pertenencia_benchmark=caso_bench, Pertenencia_dif_objetivo_USD=dif_obj,
                Pertenencia_tolerancia_USD=tol, Pertenencia_error_flujos=error, Pertenencia_banda=fx['tol'],
                Pertenencia_segundos=float(fij['Segundos']))
