"""Disponibilidad horaria de la flota térmica G1–G4 (versión 2 de parámetros).

Criterio adoptado el 24/09/2026 del artículo del proyecto (`generador_gas.disponibilidad`)
para que tesis y artículo compartan el mismo supuesto:

* **Paradas de planta.** La flota no está disponible en las horas cuya demanda procesada
  es menor que el 25 % de la mediana anual. Sobre la serie de 2025 son 101 horas, todas
  dentro de las 115 horas bajas que `demanda.limpiar()` conserva sin imputar. Tratarlas
  como paradas es un **supuesto**: desde el archivo no se puede distinguir una parada real
  de un problema de medición (ver `demanda.py`).
* **Mantenimiento mayor.** Catorce días continuos desde el 1 de septiembre, para las cuatro
  unidades a la vez: es mantenimiento de planta, no escalonado por unidad. La fecha es un
  supuesto sin dato operativo; el resultado depende de la estación solar de esa ventana.

Sobre 2025 suman 437 horas indisponibles (disponibilidad del 95,01 %), sin solapamiento.

Cómo entra en el despacho: una hora no disponible solo admite el compromiso con las cuatro
unidades apagadas. Una unidad encendida al comenzar la ventana se apaga con una parada
ordinaria, que cuenta en el límite diario de paradas y no tiene cargo; el rearranque
posterior es un arranque ordinario y paga el cargo por arranque. La red no tiene límite de
importación, así que siempre existe una trayectoria factible.

`None` como disponibilidad equivale a flota siempre disponible y reproduce los resultados
de la versión 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRACCION_PARADA = 0.25
MANTENIMIENTO_INICIO = "09-01"
MANTENIMIENTO_DIAS = 14

REFERENCIAS = {
    "disponibilidad_flota_v2": {
        "valor": "Paradas: demanda < 25 % de la mediana anual; mantenimiento de planta 1–14 sep",
        "origen": "Criterio del artículo del proyecto, `generador_gas.disponibilidad` (commit a9e2911 "
                  "y siguientes de Sebastian Zapata), adoptado por la usuaria el 24/09/2026.",
        "resultado_2025": "101 h de parada + 336 h de mantenimiento = 437 h; disponibilidad 95,01 %",
        "sensibilidad_umbral": "0,15–0,40 de la mediana da 95–115 h de parada",
        "limites": "Las paradas son un supuesto; la fecha del mantenimiento no tiene dato operativo.",
    }
}


def serie_disponibilidad(fechas, carga, fraccion_parada=FRACCION_PARADA,
                         mantenimiento_inicio=MANTENIMIENTO_INICIO,
                         mantenimiento_dias=MANTENIMIENTO_DIAS) -> pd.Series:
    """Serie booleana horaria: True si la flota está disponible."""
    fechas = pd.DatetimeIndex(fechas)
    carga = np.asarray(carga, dtype=float)
    if carga.shape != (len(fechas),) or not np.isfinite(carga).all():
        raise ValueError("La carga debe ser una serie finita alineada con las fechas")
    if fechas.has_duplicates or not fechas.is_monotonic_increasing:
        raise ValueError("Las fechas deben ser crecientes y sin duplicados")
    disponible = carga >= fraccion_parada * float(np.median(carga))
    if mantenimiento_dias:
        for anio in np.unique(fechas.year):
            inicio = pd.Timestamp(f"{anio}-{mantenimiento_inicio} 00:00")
            fin = inicio + pd.Timedelta(days=mantenimiento_dias)
            disponible &= ~((fechas >= inicio) & (fechas < fin))
    return pd.Series(disponible, index=fechas, name="disponible")


def alinear(disponibilidad, fechas):
    """Disponibilidad booleana de la ventana `fechas`, o None si la flota está siempre disponible.

    Acepta None; una serie (flota) o una tabla de cuatro columnas (G1–G4) indexadas por fecha, que se
    reindexan; o un arreglo ya alineado de forma (n,) o (n, 4). Una fecha sin dato es un error: nunca
    se asume disponibilidad por omisión."""
    if disponibilidad is None:
        return None
    fechas = pd.DatetimeIndex(fechas)
    if isinstance(disponibilidad, (pd.Series, pd.DataFrame)):
        faltan = fechas.difference(disponibilidad.index)
        if len(faltan):
            raise ValueError(f"Disponibilidad sin dato en {len(faltan)} horas, p. ej. {faltan[0]}")
        if isinstance(disponibilidad, pd.DataFrame) and disponibilidad.shape[1] != 4:
            raise ValueError("La tabla de disponibilidad debe tener una columna por unidad G1–G4")
        return disponibilidad.reindex(fechas).to_numpy(dtype=bool)
    d = np.asarray(disponibilidad, dtype=bool)
    if d.shape not in ((len(fechas),), (len(fechas), 4)):
        raise ValueError("La disponibilidad debe tener forma (n,) o (n, 4) para la ventana")
    return d


def resumen(disponibilidad: pd.Series, carga, fraccion_parada=FRACCION_PARADA) -> dict:
    """Conteos para declarar el supuesto en cada corrida."""
    carga = np.asarray(carga, dtype=float)
    d = disponibilidad.to_numpy(dtype=bool)
    parada = carga < fraccion_parada * float(np.median(carga))
    return dict(horas=len(d), horas_indisponibles=int((~d).sum()),
                horas_parada=int(parada.sum()), horas_mantenimiento=int((~d & ~parada).sum()),
                disponibilidad_pct=100 * float(d.mean()),
                umbral_parada_MW=fraccion_parada * float(np.median(carga)))
