# -*- coding: utf-8 -*-
"""Lectura del libro único de datos de entrada: ``datos/Datos_entrada.xlsx``.

Los cuadernos 01, 01B y 02 y el módulo ``costos`` toman de aquí todo lo que antes estaba en CSV, en otros Excel o
escrito en el código: demanda, precio de bolsa, tarifas, batería, parámetros financieros y tamaños a evaluar.

Cada hoja de parámetros tiene una columna ``Clave`` (columna A) que el código busca por nombre; el usuario solo
cambia la columna ``Valor``. Las filas pueden moverse sin romper la lectura.

El libro también trae las hojas de CAPEX (Parametros_PV, Parametros_BESS, Suministro_BESS, Trayectorias), que lee
``costos_curvas``. El recurso solar sigue en los ``datos/*_tmy-2025.csv`` de NSRDB; en el libro se elige la región.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import os

# La variable de entorno DATOS_ENTRADA permite apuntar a otro libro (por ejemplo, para pruebas).
LIBRO = Path(os.environ.get("DATOS_ENTRADA", "datos/Datos_entrada.xlsx"))
MODOS_DEMANDA = ("plana", "perfil_24h", "serie_anual")
MODOS_TAMANO = ("factores", "fijos")


def existe() -> bool:
    return LIBRO.exists()


@lru_cache(maxsize=1)
def _hojas():
    if not LIBRO.exists():
        raise FileNotFoundError(f"No se encontró {LIBRO}. Abra JupyterLab desde la carpeta de la entrega (la que contiene datos/).")
    return pd.read_excel(LIBRO, sheet_name=None, header=None)


def recargar():
    """Vuelve a leer el libro (después de editarlo con el cuaderno abierto)."""
    _hojas.cache_clear()
    _parametros_hoja.cache_clear()


@lru_cache(maxsize=None)
def _parametros_hoja(hoja: str) -> dict:
    """Pares Clave → Valor de una hoja de parámetros (columna A = clave, columna C = valor)."""
    df = _hojas()[hoja]
    out = {}
    for _, fila in df.iterrows():
        clave = fila.iloc[0]
        if isinstance(clave, str) and clave.strip() and not clave.startswith("#") and len(fila) > 2:
            out[clave.strip()] = fila.iloc[2]
    return out


def _valor(hoja: str, clave: str, tipo=float):
    p = _parametros_hoja(hoja)
    if clave not in p or (isinstance(p[clave], float) and np.isnan(p[clave])):
        raise ValueError(f"Datos_entrada.xlsx, hoja «{hoja}»: falta el valor de «{clave}».")
    v = p[clave]
    if tipo is str:
        return str(v).strip()
    try:
        return tipo(v)
    except (TypeError, ValueError):
        raise ValueError(f"Datos_entrada.xlsx, hoja «{hoja}», «{clave}»: se esperaba un número y se encontró {v!r}.")


def parametro(clave: str, defecto):
    """Parámetro de la hoja «Costos» para ``costos.py``. Sin libro, devuelve el valor por defecto del código."""
    if not LIBRO.exists():
        return defecto
    p = _parametros_hoja("Costos")
    v = p.get(clave)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return defecto
    return type(defecto)(v)


def _tabla(hoja: str, encabezado: str, ncols: int | None = None) -> pd.DataFrame:
    """Tabla que empieza en la fila cuyo primer valor es ``encabezado``.

    Termina en la primera fila con las ``ncols`` primeras columnas vacías (por defecto, todas las columnas con
    encabezado). Las columnas calculadas de la derecha no cuentan para decidir dónde termina."""
    df = _hojas()[hoja]
    filas = df.index[df.iloc[:, 0].astype(str).str.strip() == encabezado]
    if len(filas) == 0:
        raise ValueError(f"Datos_entrada.xlsx, hoja «{hoja}»: no se encontró la tabla «{encabezado}».")
    i = filas[0]
    cab = [str(c).strip() for c in df.iloc[i]]
    cuerpo = df.iloc[i + 1:].copy()
    cuerpo.columns = cab
    n = ncols or sum(1 for c in cab if c not in ("nan", "None", ""))
    fin = cuerpo.index[cuerpo.iloc[:, :n].isna().all(axis=1)]
    if len(fin):
        cuerpo = cuerpo.loc[:fin[0] - 1]
    return cuerpo.reset_index(drop=True)


# ------------------------------------------------------------------ General
def general() -> dict:
    return dict(anio=_valor("General", "anio", int), region=_valor("General", "region", str),
                caso=_valor("General", "nombre_caso", str))


def _fechas(anio: int) -> pd.DatetimeIndex:
    f = pd.date_range(f"{anio}-01-01", f"{anio}-12-31 23:00", freq="h")
    if len(f) != 8760:
        raise ValueError(f"El año {anio} es bisiesto ({len(f)} horas). Los modelos trabajan con 8.760 horas: use un año no bisiesto.")
    return f


# ------------------------------------------------------------------ Demanda
def demanda_config() -> dict:
    modo = _valor("Demanda", "modo", str)
    if modo not in MODOS_DEMANDA:
        raise ValueError(f"Datos_entrada.xlsx, hoja «Demanda»: el modo debe ser uno de {MODOS_DEMANDA}; se encontró «{modo}».")
    perfil = _tabla("Demanda", "Hora", 3)
    perfil = perfil.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
    perfil.columns = ["Hora", "Laborable_MW", "Fin_de_semana_MW"]
    if len(perfil) != 24 or perfil.isna().any().any() or list(perfil.Hora.astype(int)) != list(range(24)):
        raise ValueError("Datos_entrada.xlsx, hoja «Demanda»: el perfil debe tener las 24 horas (0 a 23) con valor en las dos columnas.")
    return dict(modo=modo, plana_MW=_valor("Demanda", "demanda_plana_MW"), perfil=perfil)


def serie_demanda(anio: int | None = None) -> pd.DataFrame:
    """Demanda horaria del año: columnas ``Fecha`` y ``Load_MW``.

    - ``plana``: el mismo valor en las 8.760 horas.
    - ``perfil_24h``: un perfil de 24 h para lunes a viernes y otro para sábado y domingo.
    - ``serie_anual``: la serie horaria de la hoja «Demanda_anual», con la limpieza de cortes de medidor de
      ``demanda.limpiar`` (imputa rachas de horas exactamente en 0 MW).
    """
    import demanda as DEM
    anio = general()["anio"] if anio is None else anio
    cfg = demanda_config()
    fechas = _fechas(anio)
    if cfg["modo"] == "plana":
        df = pd.DataFrame({"Fecha": fechas, "Load_MW": float(cfg["plana_MW"])})
    elif cfg["modo"] == "perfil_24h":
        p = cfg["perfil"]
        lab, fds = p.Laborable_MW.to_numpy(float), p.Fin_de_semana_MW.to_numpy(float)
        h, finde = fechas.hour.to_numpy(), fechas.dayofweek.to_numpy() >= 5
        df = pd.DataFrame({"Fecha": fechas, "Load_MW": np.where(finde, fds[h], lab[h])})
    else:
        crudo = _tabla("Demanda_anual", "Fecha", 2).iloc[:, :2]
        crudo.columns = ["Fecha", "Load_MW"]
        crudo["Fecha"] = pd.to_datetime(crudo["Fecha"], errors="coerce")
        crudo["Load_MW"] = pd.to_numeric(crudo["Load_MW"], errors="coerce")
        crudo = crudo.dropna()
        crudo = crudo[crudo.Fecha.dt.year == anio].sort_values("Fecha").reset_index(drop=True)
        if len(crudo) != 8760 or not (crudo.Fecha.to_numpy() == fechas.to_numpy()).all():
            raise ValueError(f"Datos_entrada.xlsx, hoja «Demanda_anual»: se esperaban las 8.760 horas de {anio} "
                             f"en orden y se encontraron {len(crudo)}.")
        return DEM.limpiar(crudo)
    if (df.Load_MW < 0).any():
        raise ValueError("Datos_entrada.xlsx, hoja «Demanda»: la demanda no puede ser negativa.")
    e = float(df.Load_MW.sum())
    df.attrs["limpieza_demanda"] = {"horas_imputadas": 0, "horas_en_cero": 0, "horas_de_recuperacion": 0, "rachas": 0,
                                    "energia_antes_MWh": e, "energia_despues_MWh": e, "incremento_%": 0.0,
                                    "horas_bajas_no_tocadas": 0, "modo": cfg["modo"]}
    return df


def serie_demanda_cruda(anio: int | None = None) -> pd.DataFrame:
    """Serie antes de la limpieza (solo cambia en el modo ``serie_anual``)."""
    anio = general()["anio"] if anio is None else anio
    if demanda_config()["modo"] != "serie_anual":
        return serie_demanda(anio)
    crudo = _tabla("Demanda_anual", "Fecha", 2).iloc[:, :2]
    crudo.columns = ["Fecha", "Load_MW"]
    crudo["Fecha"] = pd.to_datetime(crudo["Fecha"], errors="coerce")
    crudo["Load_MW"] = pd.to_numeric(crudo["Load_MW"], errors="coerce")
    crudo = crudo.dropna()
    return crudo[crudo.Fecha.dt.year == anio].sort_values("Fecha").reset_index(drop=True)


# ------------------------------------------------------------------ Precio de bolsa
def precio_bolsa(anio: int | None = None) -> pd.DataFrame:
    """Precio de bolsa horario: ``Fecha`` y ``Bolsa_COP_kWh``. La hoja tiene una fila por día y columnas 0–23 (formato XM)."""
    anio = general()["anio"] if anio is None else anio
    t = _tabla("Precio_bolsa", "Fecha", 25)
    t["Fecha"] = pd.to_datetime(t["Fecha"], errors="coerce")
    t = t.dropna(subset=["Fecha"])
    horas = [c for c in t.columns if c.replace(".0", "").isdigit()]
    largo = t.melt(id_vars=["Fecha"], value_vars=horas, var_name="Hora", value_name="Bolsa_COP_kWh")
    largo["Hora"] = largo["Hora"].str.replace(".0", "", regex=False).astype(int)
    largo["Fecha"] = largo["Fecha"].dt.normalize() + pd.to_timedelta(largo["Hora"], unit="h")
    largo["Bolsa_COP_kWh"] = pd.to_numeric(largo["Bolsa_COP_kWh"], errors="coerce")
    largo = (largo.loc[largo.Fecha.dt.year == anio, ["Fecha", "Bolsa_COP_kWh"]]
             .sort_values("Fecha").reset_index(drop=True))
    if len(largo) != 8760 or largo.Bolsa_COP_kWh.isna().any():
        raise ValueError(f"Datos_entrada.xlsx, hoja «Precio_bolsa»: se esperaban 365 días × 24 horas de {anio} "
                         f"sin celdas vacías; hay {len(largo)} horas y {int(largo.Bolsa_COP_kWh.isna().sum())} vacías.")
    return largo


# ------------------------------------------------------------------ Tarifas y batería
def _horas(inicio: int, fin: int) -> set:
    """Horas [inicio, fin) del día; si fin < inicio, el tramo cruza la medianoche."""
    return set(range(inicio, fin)) if inicio <= fin else set(range(inicio, 24)) | set(range(0, fin))


def tarifas() -> dict:
    h = "Tarifas"
    opcion = _valor(h, "opcion_tarifaria", str)
    if opcion not in ("opciones_horarias", "monomia"):
        raise ValueError("Datos_entrada.xlsx, hoja «Tarifas»: la opción tarifaria debe ser «opciones_horarias» o «monomia».")
    return dict(
        nivel_tension=_valor(h, "nivel_tension", int), opcion_tarifaria=opcion,
        cara_COP_kWh=_valor(h, "tarifa_cara_COP_kWh"), barata_COP_kWh=_valor(h, "tarifa_barata_COP_kWh"),
        horas_caras=_horas(_valor(h, "hora_inicio_cara", int), _valor(h, "hora_fin_cara", int)),
        monomia_COP_kWh=_valor(h, "tarifa_monomia_COP_kWh"),
        ppa_solar_COP_kWh=_valor(h, "ppa_solar_COP_kWh"), cere_COP_MWh=_valor(h, "cere_COP_MWh"),
        fazni_COP_kWh=_valor(h, "fazni_COP_kWh"), arranque_parada_COP_kWh=_valor(h, "arranque_parada_COP_kWh"),
        cnd_asic_COP_kW_mes=_valor(h, "cnd_asic_COP_kW_mes"),
    )


def bateria() -> dict:
    return dict(DoD=_valor("Bateria", "DoD"), RTE=_valor("Bateria", "RTE"))


# ------------------------------------------------------------------ Tamaños
def _lista(t: pd.DataFrame, col: str) -> list[float]:
    v = pd.to_numeric(t[col], errors="coerce").dropna()
    return [float(x) for x in v]


@lru_cache(maxsize=8)
def produccion_especifica(region: str, anio: int) -> float:
    """MWh que produce 1 MWp DC en el año en la región (TMY + pvlib, el mismo cálculo de los modelos)."""
    import solar_tmy as ST
    return float(ST.perfil_pv(region, 1.0, anio=anio).PV_MW.sum())


def tamanos(load_MW=None) -> dict:
    """Tamaños a evaluar en MWp, MW y MWh.

    Modo ``factores``: PV = factor × demanda anual ÷ producción específica de la región; energía = % × demanda
    diaria media; potencia = factor × demanda pico. Modo ``fijos``: los valores escritos en la hoja.
    """
    g = general()
    load = np.asarray(serie_demanda(g["anio"]).Load_MW if load_MW is None else load_MW, float)
    anual, pico = float(load.sum()), float(load.max())
    diaria = anual / (len(load) / 24)
    yld = produccion_especifica(g["region"], g["anio"])
    h = "Dimensionamiento"
    modo = _valor(h, "modo_tamanos", str)
    if modo not in MODOS_TAMANO:
        raise ValueError(f"Datos_entrada.xlsx, hoja «Dimensionamiento»: el modo debe ser uno de {MODOS_TAMANO}.")
    dur = (_valor(h, "duracion_min_h"), _valor(h, "duracion_max_h"))
    if modo == "factores":
        t = _tabla(h, "PV cuaderno 02 [veces demanda anual]", 4)
        pv = lambda f: round(f * anual / yld, 1)
        e = lambda f: round(f / 100 * diaria, 1)
        p = lambda f: round(f * pico, 1)
        base = dict(pv=pv(_valor(h, "base_pv_factor")), e=e(_valor(h, "base_energia_pct")), p=p(_valor(h, "base_potencia_factor")))
        listas = dict(pv02=[pv(x) for x in _lista(t, t.columns[0]) if x > 0], pv01b=[pv(x) for x in _lista(t, t.columns[1]) if x > 0],
                      e=[e(x) for x in _lista(t, t.columns[2]) if x > 0], p=[p(x) for x in _lista(t, t.columns[3]) if x > 0])
    else:
        t = _tabla(h, "PV cuaderno 02 [MWp]", 4)
        base = dict(pv=_valor(h, "base_pv_MWp"), e=_valor(h, "base_energia_MWh"), p=_valor(h, "base_potencia_MW"))
        listas = dict(pv02=[x for x in _lista(t, t.columns[0]) if x > 0], pv01b=[x for x in _lista(t, t.columns[1]) if x > 0],
                      e=[x for x in _lista(t, t.columns[2]) if x > 0], p=[x for x in _lista(t, t.columns[3]) if x > 0])
    ordenar = lambda xs: sorted(set(xs))
    listas = {k: ordenar(v) for k, v in listas.items()}
    pares = [(pp, ee) for pp in listas["p"] for ee in listas["e"] if dur[0] <= ee / pp <= dur[1]]
    fuera = [(pp, ee) for pp in listas["p"] for ee in listas["e"] if not dur[0] <= ee / pp <= dur[1]]
    return dict(modo=modo, demanda_anual_MWh=anual, demanda_diaria_MWh=diaria, demanda_pico_MW=pico,
                produccion_especifica_MWh_MWp=yld, base=base, pv_02=[0.0] + listas["pv02"], pv_01b=listas["pv01b"],
                energias_MWh=listas["e"], potencias_MW=listas["p"], pares_MW_MWh=pares, pares_fuera=fuera,
                duracion_h=dur)


def resumen(t: dict | None = None) -> str:
    """Texto corto con la demanda y los tamaños que resultan, para imprimir en los cuadernos."""
    t = tamanos() if t is None else t
    b = t["base"]
    return "\n".join([
        f"Tamaños en modo «{t['modo']}» (Datos_entrada.xlsx, hoja Dimensionamiento)",
        f"  Demanda anual {t['demanda_anual_MWh']:,.0f} MWh · diaria {t['demanda_diaria_MWh']:,.1f} MWh · pico {t['demanda_pico_MW']:,.2f} MW",
        f"  Producción específica PV: {t['produccion_especifica_MWh_MWp']:,.0f} MWh/MWp-año",
        f"  Caso base: PV {b['pv']:,.1f} MWp · batería {b['p']:,.1f} MW / {b['e']:,.1f} MWh",
        f"  PV del cuaderno 02 [MWp]: {', '.join(f'{x:,.1f}' for x in t['pv_02'])}",
        f"  Energías [MWh]: {', '.join(f'{x:,.1f}' for x in t['energias_MWh'])}",
        f"  Potencias [MW]: {', '.join(f'{x:,.1f}' for x in t['potencias_MW'])}",
        f"  Pares potencia/energía con duración entre {t['duracion_h'][0]:g} y {t['duracion_h'][1]:g} h: {len(t['pares_MW_MWh'])}"
        + (f" (se excluyen {len(t['pares_fuera'])} fuera de ese rango)" if t["pares_fuera"] else ""),
    ])
