"""Limpieza declarada de la serie de demanda.

El perfil `Porvenir` de `load variable.csv` tiene **71 horas exactamente en
0 MW**, repartidas en 11 rachas de 1 a 22 horas. No son consumo: son cortes de
medidor. El patrón no deja lugar a dudas — demanda normal antes, demanda normal
después, cero limpio en medio, y las rachas largas arrancan de madrugada y se
extienden todo el día:

    2025-03-05 06:00 .. 17:00 (12 h)  antes 19,3 20,4 20,6  ->  después 21,3 21,3
    2025-11-14 06:00 .. 03:00 (22 h)  antes 19,2 20,3 19,5  ->  después 13,6 16,2

En varias rachas la recuperación del medidor es **gradual** (3,1 · 6,3 · 19,7),
así que la primera hora o dos tras el corte también están corruptas y se tratan
igual.

Qué NO se toca
--------------
En la serie evaluada se conservan **115 horas con 0 < demanda < 8 MW** que
no fueron marcadas para imputación por `marcar_invalidas`. Este conteo no
aplica un filtro de distancia de 12 horas a los cortes ni un límite inferior
de 0,7 MW. Incluye una lectura inferior a 0,7 MW y un bloque continuo de
51 horas, por lo que no todas son horas dispersas. Podrían ser paradas reales
de planta o problemas de medición; desde el archivo no se puede distinguir.
**No se imputan**: rellenarlas sería fabricar consumo donde quizá la planta
estaba parada. Quedan declaradas como incertidumbre conocida. El conteo es
reproducible con `manuscrito/metodologia_completa_20260920/`
`evidencia_horas_bajas/conteo_horas_bajas.py`.

Método
------
Mediana de la **misma hora del día** en los días válidos más cercanos, dentro de
una ventana de ±7 días. Se prefiere a promediar las horas anteriores porque esto
último se rompe en las rachas largas: en la de 22 horas dejaría un valor plano
durante un ciclo diario completo, borrando la variación día/noche, que es
justamente la información que necesita un estudio de despacho.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Una hora por debajo de este valor, dentro de la ventana posterior a un corte,
# se considera lectura parcial de recuperación y no consumo real.
UMBRAL_RECUPERACION_MW = 8.0
VENTANA_RECUPERACION_H = 3

# Días a cada lado de los que se toma la referencia para imputar.
DIAS_VECINOS = 7

REFERENCIAS = {
    "limpieza_demanda": {
        "valor": "Imputación de 71 horas en cero y sus horas de recuperación",
        "metodo": (
            "Mediana de la misma hora del día en los días válidos dentro de "
            "±7 días."
        ),
        "no_incluye": (
            "Las 115 horas con 0 < demanda < 8 MW no marcadas para imputación "
            "por marcar_invalidas. No se aplica un filtro de distancia de 12 h "
            "ni un límite inferior de 0,7 MW; se desconoce si corresponden a "
            "paradas reales o problemas de medición."
        ),
        "origen": "load variable.csv, columna Porvenir. Fuente primaria sin documentar.",
    }
}


def marcar_invalidas(serie: pd.Series) -> pd.Series:
    """Marca las horas que hay que imputar: ceros y su recuperación.

    Devuelve una serie booleana con el mismo índice.
    """
    valores = serie.to_numpy(dtype=float)
    cero = valores == 0.0
    invalida = cero.copy()

    # Horas de recuperación: por debajo del umbral y dentro de la ventana
    # posterior al final de una racha de ceros.
    indices = np.flatnonzero(cero)
    if indices.size:
        cortes = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
        for racha in cortes:
            fin = racha[-1]
            for k in range(1, VENTANA_RECUPERACION_H + 1):
                j = fin + k
                if j >= len(valores) or valores[j] >= UMBRAL_RECUPERACION_MW:
                    break
                invalida[j] = True

    return pd.Series(invalida, index=serie.index, name="invalida")


def _imputar(serie: pd.Series, invalida: pd.Series) -> pd.Series:
    """Mediana de la misma hora del día en los días válidos vecinos."""
    limpia = serie.copy()
    validas = serie[~invalida]
    hora = serie.index.hour
    dia = serie.index.normalize()

    ref = pd.DataFrame({"valor": validas.to_numpy()}, index=validas.index)
    ref["hora"] = ref.index.hour
    ref["dia"] = ref.index.normalize()

    for t in serie.index[invalida]:
        vecinos = ref[
            (ref["hora"] == t.hour)
            & ((ref["dia"] - t.normalize()).abs() <= pd.Timedelta(days=DIAS_VECINOS))
        ]["valor"]
        if vecinos.empty:  # ventana sin datos: se amplía al año entero
            vecinos = ref[ref["hora"] == t.hour]["valor"]
        limpia.loc[t] = float(vecinos.median())

    del hora, dia
    return limpia


def limpiar(df_load: pd.DataFrame, columna: str = "Load_MW") -> pd.DataFrame:
    """Devuelve la demanda con los cortes de medidor imputados.

    Añade a `df.attrs` el resumen de lo que se cambió, para que la limpieza sea
    auditable y no silenciosa.
    """
    df = df_load.copy()
    serie = df.set_index("Fecha")[columna].sort_index()

    invalida = marcar_invalidas(serie)
    if not invalida.any():
        df.attrs["limpieza_demanda"] = {"horas_imputadas": 0}
        return df

    limpia = _imputar(serie, invalida)

    energia_antes = float(serie.sum())
    energia_despues = float(limpia.sum())

    df[columna] = df["Fecha"].map(limpia)
    df.attrs["limpieza_demanda"] = {
        "horas_imputadas": int(invalida.sum()),
        "horas_en_cero": int((serie == 0).sum()),
        "horas_de_recuperacion": int(invalida.sum() - (serie == 0).sum()),
        "rachas": int((invalida & ~invalida.shift(fill_value=False)).sum()),
        "energia_antes_MWh": energia_antes,
        "energia_despues_MWh": energia_despues,
        "incremento_%": (energia_despues / energia_antes - 1) * 100,
        "horas_bajas_no_tocadas": int(
            ((serie > 0) & (serie < UMBRAL_RECUPERACION_MW) & ~invalida).sum()
        ),
    }
    return df


def detalle(df_load: pd.DataFrame, columna: str = "Load_MW") -> pd.DataFrame:
    """Tabla hora a hora de lo que cambiaría la limpieza. Para auditar."""
    serie = df_load.set_index("Fecha")[columna].sort_index()
    invalida = marcar_invalidas(serie)
    limpia = _imputar(serie, invalida)
    return pd.DataFrame(
        {
            "Original_MW": serie[invalida],
            "Imputado_MW": limpia[invalida],
            "Motivo": np.where(serie[invalida] == 0, "corte", "recuperación"),
        }
    )


if __name__ == "__main__":
    import logging

    for _r in ("pypsa", "linopy"):
        logging.getLogger(_r).setLevel(logging.ERROR)
    import sensibilidad as S
    import costos

    costos.consola_utf8()
    pd.set_option("display.width", 200)

    g = S.cargar_entorno_notebook()
    crudo = g["df_load"]
    limpio = limpiar(crudo)

    print("RESUMEN DE LA LIMPIEZA")
    for k, v in limpio.attrs["limpieza_demanda"].items():
        print(f"  {k:26s} {v:,.2f}" if isinstance(v, float) else f"  {k:26s} {v:,}")

    print("\nDETALLE (primeras 20 horas imputadas)")
    print(detalle(crudo).head(20).to_string())
