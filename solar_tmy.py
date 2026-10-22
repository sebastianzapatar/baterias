"""Generación solar horaria simulada desde irradiancia NSRDB con pvlib.

Sustituye los perfiles precalculados de `<Sitio>.xlsx`, que llegaban ya
convertidos a potencia y sin trazabilidad de sus supuestos. Aquí la cadena es
explícita y auditable:

    TMY NSRDB -> posición solar -> POA (Perez) -> IAM -> temperatura de celda
              -> potencia DC (PVWatts) -> pérdidas de sistema -> inversor -> AC

Convenciones de este proyecto
-----------------------------
* Los cinco puntos de estudio son **regiones UPME**, no plantas con nombre.
* El tamaño se sigue expresando en **MWp (DC)**, que es la unidad en la que
  `costos_curvas.py` cotiza el CAPEX solar (USD/kWp). La potencia AC se deriva
  como ``MWp / DC_AC_RATIO``; invertir la convención subestimaría el CAPEX en
  todo el barrido.
* El perfil se calcula una sola vez por región para 1 MWac y se escala: con
  DC/AC, orientación y pérdidas constantes la respuesta es proporcional.

Basado en `solar_modelo_tesis.py`, con dos correcciones sin las cuales no
corría: el nombre del argumento en la llamada a `cargar_tmy_nsrdb`, y la
conversión a `DatetimeIndex` antes de `tz_localize`.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pvlib


# ============================================================
# Puntos de estudio: regiones UPME
# ============================================================

REGIONES: Dict[str, Path] = {
    "Caribe": Path("datos/Caribe_tmy-2025.csv"),
    "Antioquia": Path("datos/Antioquia_tmy-2025.csv"),
    "Suroccidente": Path("datos/Suroccidental_tmy-2025.csv"),
    "Oriente": Path("datos/Oriental_tmy-2025.csv"),
    "Nordeste": Path("datos/Nordeste_tmy-2025.csv"),
}


# ============================================================
# Pérdidas de sistema: valores por defecto de PVWatts V5
# ============================================================
# Se declaran por categoría y NO se suman: PVWatts las combina de forma
# multiplicativa. La diferencia no es cosmética -- la suma lineal daría 15,0 %
# y el producto da 14,08 %, que es el 14 % que documenta el manual.

PERDIDAS_PVWATTS_PCT: Dict[str, float] = {
    "suciedad": 2.0,
    "sombreado": 3.0,
    "nieve": 0.0,
    "mismatch": 2.0,
    "cableado": 2.0,
    "conexiones": 0.5,
    "degradacion_inducida_por_luz": 1.5,
    "tolerancia_de_placa": 1.0,
    "envejecimiento": 0.0,
    "disponibilidad": 3.0,
}


def perdidas_totales_pct(categorias: Dict[str, float] | None = None) -> float:
    """Combina las categorías como lo hace PVWatts, de forma multiplicativa.

        L_total = 100 * [ 1 - producto_i (1 - L_i / 100) ]
    """
    categorias = PERDIDAS_PVWATTS_PCT if categorias is None else categorias
    producto = math.prod(1.0 - v / 100.0 for v in categorias.values())
    return 100.0 * (1.0 - producto)


# ============================================================
# Parámetros de planta adoptados
# ============================================================

DC_AC_RATIO = 1.20
TILT_DEG = 10.0
AZIMUTH_DEG = 180.0
PERDIDAS_SISTEMA_PCT = perdidas_totales_pct()
ANIO_SIMULACION = 2025
ZONA_HORARIA = "America/Bogota"

REFERENCIAS = {
    "perdidas_sistema": {
        "valor": f"{PERDIDAS_SISTEMA_PCT:.2f} % (valor por defecto de PVWatts V5)",
        "detalle": (
            "Suciedad 2 %, sombreado 3 %, nieve 0 %, mismatch 2 %, cableado 2 %, "
            "conexiones 0,5 %, degradación inducida por luz 1,5 %, tolerancia de "
            "placa 1 %, envejecimiento 0 %, disponibilidad 3 %. Se combinan de "
            "forma multiplicativa y no aditiva: la suma daría 15,0 %."
        ),
        "excluye": (
            "Temperatura de celda, IAM, eficiencia del inversor y recorte por "
            "potencia AC, porque el modelo ya los calcula explícitamente."
        ),
        "cita": "Dobos, A. P. (2014). PVWatts Version 5 Manual. NREL/TP-6A20-62641.",
        "url": "https://docs.nrel.gov/docs/fy14osti/62641.pdf",
        "consultado": "2026-08-30",
    },
    "recurso_solar": {
        "valor": "TMY horario NSRDB, 8.760 pasos, punto por región UPME",
        "cita": "National Solar Radiation Database (NSRDB), NREL.",
        "url": "https://nsrdb.nrel.gov/",
        "consultado": "2026-08-30",
    },
}


# ============================================================
# 1. Lectura del TMY NSRDB
# ============================================================

def cargar_tmy_nsrdb(
    ruta_csv: str | Path,
    anio_sintetico: int = ANIO_SIMULACION,
    timezone_local: str = ZONA_HORARIA,
) -> Tuple[pd.DataFrame, dict]:
    """Lee un TMY de NSRDB y devuelve la meteorología en hora local.

    El archivo trae dos filas de metadatos, una de encabezados y 8.760
    registros. Las marcas de tiempo vienen en UTC (``Time Zone = 0``) y a la
    media hora, porque NSRDB entrega valores instantáneos centrados en el
    intervalo. El TMY combina meses de años distintos, así que la columna
    ``Year`` no sirve como calendario: se reconstruye un año sintético continuo.
    """
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

    metadata = pd.read_csv(ruta_csv, nrows=1).iloc[0].to_dict()
    raw = pd.read_csv(ruta_csv, skiprows=2)

    requeridas = [
        "Month", "Day", "Hour", "Minute", "GHI", "DNI", "DHI",
        "Temperature", "Wind Speed", "Surface Albedo", "Pressure",
    ]
    faltantes = [c for c in requeridas if c not in raw.columns]
    if faltantes:
        raise ValueError(
            "El archivo no contiene las columnas requeridas: " + ", ".join(faltantes)
        )
    if len(raw) != 8_760:
        raise ValueError(
            f"Se esperaban 8.760 registros horarios y se encontraron {len(raw):,}."
        )

    idx_utc = pd.DatetimeIndex(pd.to_datetime(
        {
            "year": np.full(len(raw), anio_sintetico, dtype=int),
            "month": raw["Month"].astype(int).to_numpy(),
            "day": raw["Day"].astype(int).to_numpy(),
            "hour": raw["Hour"].astype(int).to_numpy(),
            "minute": raw["Minute"].astype(int).to_numpy(),
        },
        utc=True,
    ))

    local = idx_utc.tz_convert(timezone_local)

    # Al pasar de UTC a UTC-5 las primeras horas caen en el 31 de diciembre
    # anterior. Como el TMY es un año sintético, se reconstruye el calendario
    # local con el mismo año y se ordena: el resultado es un año continuo.
    idx_local = pd.DatetimeIndex(pd.to_datetime(
        {
            "year": np.full(len(raw), anio_sintetico, dtype=int),
            "month": local.month,
            "day": local.day,
            "hour": local.hour,
            "minute": local.minute,
        }
    )).tz_localize(timezone_local)

    weather = pd.DataFrame(
        {
            "ghi": pd.to_numeric(raw["GHI"], errors="coerce").to_numpy(),
            "dni": pd.to_numeric(raw["DNI"], errors="coerce").to_numpy(),
            "dhi": pd.to_numeric(raw["DHI"], errors="coerce").to_numpy(),
            "temp_air": pd.to_numeric(raw["Temperature"], errors="coerce").to_numpy(),
            "wind_speed": pd.to_numeric(raw["Wind Speed"], errors="coerce").to_numpy(),
            "albedo": pd.to_numeric(raw["Surface Albedo"], errors="coerce").to_numpy(),
            "pressure": pd.to_numeric(raw["Pressure"], errors="coerce").to_numpy(),
        },
        index=idx_local,
    ).sort_index()
    weather.index.name = "fecha"

    if weather.index.has_duplicates:
        raise ValueError("El calendario TMY reconstruido tiene fechas duplicadas.")
    if weather.isna().any().any():
        columnas = weather.columns[weather.isna().any()].tolist()
        raise ValueError("Valores faltantes o no numéricos en: " + ", ".join(columnas))

    for col in ["ghi", "dni", "dhi", "wind_speed", "albedo"]:
        weather[col] = weather[col].clip(lower=0)

    meta = {
        "source": metadata.get("Source", "NSRDB"),
        "location_id": metadata.get("Location ID"),
        "latitude": float(metadata["Latitude"]),
        "longitude": float(metadata["Longitude"]),
        "elevation_m": float(metadata["Elevation"]),
        "timezone_local": timezone_local,
        "version": metadata.get("Version"),
    }
    return weather, meta


# ============================================================
# 2. Modelo físico de la planta
# ============================================================

def simular_pv(
    ruta_tmy: str | Path,
    pv_mw_ac: float,
    dc_ac_ratio: float = DC_AC_RATIO,
    tilt_deg: float = TILT_DEG,
    azimuth_deg: float = AZIMUTH_DEG,
    losses_sistema_pct: float = PERDIDAS_SISTEMA_PCT,
    gamma_pdc: float = -0.0035,
    eta_inv_nom: float = 0.96,
    module_efficiency: float = 0.20,
    u_c: float = 29.0,
    u_v: float = 0.0,
    alpha_absorption: float = 0.90,
    aplicar_iam: bool = True,
    anio_sintetico: int = ANIO_SIMULACION,
    timezone_local: str = ZONA_HORARIA,
) -> pd.DataFrame:
    """Simula una planta PV fija de eje fijo a partir del TMY.

    `losses_sistema_pct` es el agregado ya combinado de PVWatts. No debe
    incluir temperatura, IAM, eficiencia del inversor ni recorte: el modelo
    calcula esos cuatro efectos por separado.
    """
    if pv_mw_ac < 0:
        raise ValueError("pv_mw_ac no puede ser negativo.")
    if dc_ac_ratio <= 0:
        raise ValueError("dc_ac_ratio debe ser mayor que cero.")
    if not 0 <= losses_sistema_pct < 100:
        raise ValueError("losses_sistema_pct debe estar entre 0 y 100.")
    if not 0 < eta_inv_nom <= 1:
        raise ValueError("eta_inv_nom debe estar entre 0 y 1.")
    if not 0 < module_efficiency <= 1:
        raise ValueError("module_efficiency debe estar entre 0 y 1.")

    weather, meta = cargar_tmy_nsrdb(
        ruta_csv=ruta_tmy,
        anio_sintetico=anio_sintetico,
        timezone_local=timezone_local,
    )

    pv_mw_dc = pv_mw_ac * dc_ac_ratio

    solar_position = pvlib.solarposition.get_solarposition(
        time=weather.index,
        latitude=meta["latitude"],
        longitude=meta["longitude"],
        altitude=meta["elevation_m"],
        pressure=weather["pressure"] * 100.0,   # mbar -> Pa
        temperature=weather["temp_air"],
    )
    solar_zenith = solar_position["apparent_zenith"]
    solar_azimuth = solar_position["azimuth"]

    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
    airmass = pvlib.atmosphere.get_relative_airmass(zenith=solar_zenith)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=weather["dni"],
        ghi=weather["ghi"],
        dhi=weather["dhi"],
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=weather["albedo"],
        model="perez",
    ).fillna(0.0)
    for col in poa.columns:
        poa[col] = poa[col].clip(lower=0.0)

    if aplicar_iam:
        aoi = pvlib.irradiance.aoi(
            surface_tilt=tilt_deg,
            surface_azimuth=azimuth_deg,
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
        )
        iam = pd.Series(pvlib.iam.physical(aoi), index=weather.index)
        iam = iam.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        efectiva = poa["poa_direct"] * iam + poa["poa_diffuse"]
    else:
        efectiva = poa["poa_global"].copy()
    efectiva = efectiva.clip(lower=0.0)

    temp_cell = pd.Series(
        pvlib.temperature.pvsyst_cell(
            poa_global=poa["poa_global"],
            temp_air=weather["temp_air"],
            wind_speed=weather["wind_speed"],
            u_c=u_c,
            u_v=u_v,
            module_efficiency=module_efficiency,
            alpha_absorption=alpha_absorption,
        ),
        index=weather.index,
    )

    p_dc_w = pd.Series(
        pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=efectiva,
            temp_cell=temp_cell,
            pdc0=pv_mw_dc * 1e6,
            gamma_pdc=gamma_pdc,
        ),
        index=weather.index,
    ).clip(lower=0.0)

    p_dc_w = p_dc_w * (1.0 - losses_sistema_pct / 100.0)

    pac0_w = pv_mw_ac * 1e6
    p_ac_w = pd.Series(
        pvlib.inverter.pvwatts(
            pdc=p_dc_w, pdc0=pac0_w / eta_inv_nom, eta_inv_nom=eta_inv_nom
        ),
        index=weather.index,
    ).clip(lower=0.0, upper=pac0_w)

    dt_h = weather.index.to_series().diff().dt.total_seconds().median() / 3600.0
    if not np.isclose(dt_h, 1.0):
        raise ValueError(f"Se esperaba resolución de 1 h y se obtuvo {dt_h:.4f} h.")

    resultado = pd.DataFrame(index=weather.index)
    resultado["ghi_wm2"] = weather["ghi"]
    resultado["poa_global_wm2"] = poa["poa_global"]
    resultado["effective_irradiance_wm2"] = efectiva
    resultado["temp_air_c"] = weather["temp_air"]
    resultado["temp_cell_c"] = temp_cell
    resultado["p_dc_mw"] = p_dc_w / 1e6
    resultado["p_ac_mw"] = p_ac_w / 1e6
    resultado["energia_mwh"] = resultado["p_ac_mw"] * dt_h
    resultado["cf_ac"] = resultado["p_ac_mw"] / pv_mw_ac if pv_mw_ac > 0 else 0.0

    energia = float(resultado["energia_mwh"].sum())
    resultado.attrs.update(
        {
            "pv_mw_ac": pv_mw_ac,
            "pv_mw_dc": pv_mw_dc,
            "dc_ac_ratio": dc_ac_ratio,
            "energia_anual_mwh": energia,
            "factor_capacidad_ac": energia / (pv_mw_ac * 8760.0) if pv_mw_ac > 0 else 0.0,
            "specific_yield_kwh_kwp": energia / pv_mw_dc if pv_mw_dc > 0 else 0.0,
            "tilt_deg": tilt_deg,
            "azimuth_deg": azimuth_deg,
            "losses_sistema_pct": losses_sistema_pct,
            "horas_en_recorte": int((p_ac_w >= pac0_w * 0.999).sum()),
            **meta,
        }
    )
    return resultado


# ============================================================
# 3. Adaptador al contrato del proyecto
# ============================================================

@lru_cache(maxsize=len(REGIONES))
def _perfil_normalizado(region: str) -> pd.DataFrame:
    """Perfil físico de 1 MWac para una región. Se calcula una sola vez."""
    if region not in REGIONES:
        raise KeyError(
            f"Región desconocida: {region!r}. Disponibles: {sorted(REGIONES)}"
        )
    return simular_pv(ruta_tmy=REGIONES[region], pv_mw_ac=1.0)


def perfil_pv(region: str, pv_MWp: float, anio: int = ANIO_SIMULACION) -> pd.DataFrame:
    """Devuelve el perfil en el contrato del proyecto: ``Fecha`` y ``PV_MW``.

    `pv_MWp` es potencia **DC**, igual que en `costos_curvas.py`. La potencia
    AC de la planta es ``pv_MWp / DC_AC_RATIO``, que es donde recorta el
    inversor.

    NSRDB centra cada valor en la media hora; el resto del proyecto etiqueta
    cada hora por su inicio, así que la marca se traslada a ``HH:00``.
    """
    if pv_MWp < 0:
        raise ValueError("pv_MWp no puede ser negativo.")

    perfil = _perfil_normalizado(region)
    pv_mw_ac = pv_MWp / DC_AC_RATIO

    df = pd.DataFrame(
        {
            "Fecha": perfil.index.tz_localize(None).floor("h"),
            "PV_MW": perfil["cf_ac"].to_numpy() * pv_mw_ac,
        }
    ).sort_values("Fecha").reset_index(drop=True)

    df["PV_MW"] = df["PV_MW"].clip(lower=0.0)

    if len(df) != 8_760:
        raise ValueError(f"{region}: se obtuvieron {len(df):,} horas y se esperaban 8.760.")
    if df["Fecha"].duplicated().any():
        raise ValueError(f"{region}: hay fechas duplicadas tras trasladar a HH:00.")
    if df["Fecha"].dt.year.nunique() != 1 or df["Fecha"].dt.year.iloc[0] != anio:
        raise ValueError(f"{region}: el calendario no corresponde al año {anio}.")
    if df["PV_MW"].isna().any():
        raise ValueError(f"{region}: hay potencias no numéricas.")
    if df["PV_MW"].max() > pv_MWp * 1.10:
        raise ValueError(f"{region}: la potencia PV supera el nominal DC de forma inesperada.")

    df.attrs.update(perfil.attrs)
    df.attrs["pv_MWp"] = pv_MWp
    df.attrs["pv_mw_ac"] = pv_mw_ac
    df.attrs["region"] = region
    return df


# Generación anual [MWh] de cada región a 100 MWp con los parámetros adoptados.
# Sirve de guarda de regresión: si cambia pvlib, el TMY o un supuesto de planta,
# esta comprobación lo detecta en vez de dejarlo pasar en silencio.
GENERACION_REFERENCIA_100MWP: Dict[str, float] = {
    "Caribe": 157_429.18,
    "Antioquia": 135_361.65,
    "Suroccidente": 150_361.65,
    "Oriente": 138_742.77,
    "Nordeste": 144_011.59,
}


def validar_regiones(tolerancia_rel: float = 1e-4) -> Dict[str, float]:
    """Comprueba que las cinco regiones reproducen su generación de referencia."""
    obtenido = {}
    for region, esperado in GENERACION_REFERENCIA_100MWP.items():
        energia = float(perfil_pv(region, 100.0)["PV_MW"].sum())
        if abs(energia - esperado) > tolerancia_rel * esperado:
            raise AssertionError(
                f"{region}: se esperaban {esperado:,.2f} MWh y se obtuvieron "
                f"{energia:,.2f} MWh."
            )
        obtenido[region] = energia
    return obtenido


def resumen_regiones(pv_MWp: float = 100.0) -> pd.DataFrame:
    """Tabla comparativa de las cinco regiones para un tamaño dado."""
    filas = []
    for region in REGIONES:
        df = perfil_pv(region, pv_MWp)
        a = df.attrs
        energia = float(df["PV_MW"].sum())
        filas.append(
            {
                "Region": region,
                "Latitud": a["latitude"],
                "Longitud": a["longitude"],
                "Elevacion_m": a["elevation_m"],
                "PV_MWp": pv_MWp,
                "PV_MWac": a["pv_mw_ac"],
                "Generacion_MWh": energia,
                "Factor_planta_%": energia / (pv_MWp * 8760.0) * 100.0,
                "Yield_kWh_kWp": energia / pv_MWp,
                "Pico_MW": float(df["PV_MW"].max()),
                "Horas_en_recorte": a["horas_en_recorte"],
            }
        )
    return pd.DataFrame(filas)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(f"Pérdidas de sistema adoptadas: {PERDIDAS_SISTEMA_PCT:.4f} %")
    print(f"  (suma lineal de las categorías: {sum(PERDIDAS_PVWATTS_PCT.values()):.1f} %)")
    print(f"  {REFERENCIAS['perdidas_sistema']['cita']}")
    print()
    print(resumen_regiones(100.0).round(2).to_string(index=False))
