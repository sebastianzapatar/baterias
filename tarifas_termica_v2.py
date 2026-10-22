"""Señales corregidas para integración térmica; no modifica la fuente histórica nb02."""
import numpy as np
import pandas as pd

VERSION = "N3_punta_diurna_v2"
PUNTA_COP_KWH = 659.9496
FUERA_PUNTA_COP_KWH = 653.2247


def construir(fechas, demanda, bolsa_cop_mwh):
    """Intervalos etiquetados por hora inicial local: punta [06:00, 21:00)."""
    fechas = pd.DatetimeIndex(fechas)
    demanda = np.asarray(demanda, dtype=float)
    bolsa = np.asarray(bolsa_cop_mwh, dtype=float)
    if demanda.shape != (len(fechas),) or bolsa.shape != demanda.shape:
        raise ValueError("Calendario, demanda y bolsa deben tener igual longitud")
    if not np.isfinite(demanda).all() or not np.isfinite(bolsa).all() or (demanda < 0).any() or demanda.sum() <= 0:
        raise ValueError("Entradas no finitas o demanda inválida")
    punta = (fechas.hour >= 6) & (fechas.hour < 21)
    t0 = np.where(punta, PUNTA_COP_KWH, FUERA_PUNTA_COP_KWH)*1000
    margen = float(np.dot(t0-bolsa, demanda)/demanda.sum())
    t1 = bolsa + margen
    np.testing.assert_allclose(t0@demanda, t1@demanda, rtol=1e-12, atol=1e-3)
    return dict(T0=t0, T1=t1, Margen_COP_MWh=margen, Punta=punta)


def cargar_entradas_pv():
    """Alias explícito del cargador vigente; una sola construcción de señales."""
    import termica_pv as TP
    e = TP.cargar_entradas_pv(version="corregida")
    antigua=pd.read_csv('resultados_tarifas_v2/senales.csv')
    assert pd.DatetimeIndex(pd.to_datetime(antigua.Fecha)).equals(e['FECHA'])
    e['C_IMP_USD_HISTORICO']={s:antigua[f'{s}_historica_COP_kWh'].to_numpy()*1000/e['TRM'] for s in ('T0','T1')}
    return e


def huella_entradas(e, parametros, extras=None):
    """Evita reutilizar trayectorias cuando cambian tarifas, demanda, solar o parámetros."""
    import hashlib
    import json
    h=hashlib.sha256()
    for a in (e['FECHA'].asi8,e['LOAD'],e.get('PV_MW',[]),e['C_IMP_USD']['T0'],e['C_IMP_USD']['T1']):
        h.update(np.ascontiguousarray(a,dtype=np.float64).tobytes())
    h.update(json.dumps(parametros,sort_keys=True,default=lambda a:np.asarray(a).tolist()).encode())
    h.update(json.dumps(extras,sort_keys=True).encode())
    return h.hexdigest()
