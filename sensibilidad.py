"""
Análisis de sensibilidad del modelo base PV + BESS.

En el caso base (PV 100 MW, BESS 30 MW/120 MWh, tarifa CREG N3 horaria) las tres
heurísticas E1, E2 y E3 quedan separadas por apenas 0,13 puntos porcentuales de
ahorro. Este módulo barre los tres parámetros que pueden romper ese empate y mide
cuál estrategia es más ROBUSTA, es decir, cuál sostiene un buen desempeño en todo
el espacio de diseño y no solo en un punto afortunado.

Barridos:

    A. Capacidad del BESS      duración fija de 4 h, energía variable
    B. Capacidad del PV        el perfil se escala linealmente
    C. Spread tarifario        banda horaria CREG fija, spread variable
    D. Factorial A x B x C     para el ranking de robustez

La tarifa del barrido C se calibra a INGRESO NEUTRO sobre el perfil de importación
del caso sin BESS, con la misma convención que `diagnostico_escenarios.py`: así el
efecto de la FORMA de la señal horaria queda aislado del efecto de su NIVEL.

Métrica principal de robustez: la CAPTURA, definida como la fracción del beneficio
alcanzable que la heurística consigue,

    captura = (costo_sin_BESS - costo_estrategia) / (costo_sin_BESS - costo_óptimo)

donde el óptimo es el despacho PyPSA con previsión perfecta. Captura = 100 % es
operar tan bien como la previsión perfecta. Es preferible al ahorro bruto porque
normaliza escenarios en los que el BESS vale mucho frente a otros en los que vale
poco, y por tanto sí es comparable entre celdas de la grilla.

Uso:
    .venv/bin/python sensibilidad.py            # corre lo que falte y cachea
    .venv/bin/python sensibilidad.py --rehacer  # ignora la caché y recalcula
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# El barrido resuelve cientos de MILP; el log por escenario tapa el progreso.
for _ruidoso in ("pypsa", "linopy"):
    logging.getLogger(_ruidoso).setLevel(logging.ERROR)

NOTEBOOK = "01_Modelo base PV_BESS.ipynb"

# Celdas del notebook que definen funciones y configuración.
# 2: librerías | 4: lectores | 6: configuración | 8: series
# 12: heurísticas | 13: resumen económico | 16: optimizador PyPSA
# El óptimo se resuelve con el mismo gap relativo que usa
# `barrido_economico.py`. Con el gap por defecto (1e-7) el árbol de
# ramificación de las binarias del punto de conexión se dispara en las celdas
# de BESS grande del barrido D: una corrida quedó siete horas sin converger.
# El error que introduce es despreciable para la captura, que es una razón de
# diferencias de costo: 1e-5 sobre 90 mil MM son 0,9 millones de COP frente a
# una banda de 16 mil MM, un 0,006 %.
MIP_REL_GAP = 1e-5

CELDAS = (2, 4, 6, 8, 12, 13, 16)

HEURISTICAS = [
    "E1 — Autoconsumo",
    "E2 — Desplazamiento solar",
    "E3 — Estrategia tarifaria",
]

OPTIMO = "Óptimo"

SALIDA = Path("resultados_sensibilidad")

# Referencia del caso base, para marcar el punto de partida en las gráficas.
PV_BASE_MW = 100.0
CAPACIDAD_BASE_MWh = 120.0
DURACION_H = 4.0  # potencia = energía / duración, se mantiene fija en los barridos


# ============================================================
# Carga del entorno del notebook
# ============================================================


def cargar_entorno_notebook():
    """Ejecuta las celdas de definición del notebook y devuelve su espacio de nombres."""
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    g = {"display": lambda x: None}
    with contextlib.redirect_stdout(io.StringIO()):
        for i in CELDAS:
            exec("".join(nb["cells"][i]["source"]), g)
    return g


# ============================================================
# Construcción de escenarios
# ============================================================


def escalar_pv(df_base, pv_MW, pv_base_MW=PV_BASE_MW):
    """Escala linealmente el perfil solar a una nueva capacidad nominal.

    Se escala el perfil en lugar de releer un archivo porque el recurso solar del
    sitio es el mismo: lo que cambia es cuántos módulos hay instalados. El costo
    del PPA escala solo, porque se liquida sobre PV_MW.
    """
    df = df_base.copy()
    df["PV_MW"] = df["PV_MW"] * (pv_MW / pv_base_MW)
    return df


def tarifa_spread(df_base, imp_sin_bess_MW, spread_rel, horas_caras):
    """Tarifa de compra con un spread horario dado, calibrada a ingreso neutro.

    `spread_rel` es (cara - barata) / media. La calibración iguala el costo total
    de importación del caso SIN BESS al de la tarifa original, de modo que mover
    el spread no cambie de paso el nivel de la factura: cualquier diferencia que
    aparezca entre estrategias es atribuible a la forma de la señal, no al precio.
    """
    alta = df_base["Hora"].isin(horas_caras).to_numpy()
    forma = np.where(alta, 1.0 + spread_rel / 2.0, 1.0 - spread_rel / 2.0)

    imp = imp_sin_bess_MW.to_numpy()
    costo_actual = (df_base["c_imp_COP_MWh"].to_numpy() * imp).sum()
    k = costo_actual / (forma * imp).sum()

    return forma * k


def aplicar_tarifa(df_base, c_imp_COP_MWh):
    df = df_base.copy()
    df["c_imp_COP_MWh"] = c_imp_COP_MWh
    return df


# ============================================================
# Ejecución de un punto de la grilla
# ============================================================


def correr_punto(g, df_esc, capacity_MWh, pv_MW, etiqueta, incluir_optimo=True):
    """Corre Sin BESS + E1/E2/E3 + óptimo sobre un escenario ya construido.

    Devuelve una fila por estrategia. Los parámetros que dependen del tamaño del
    BESS (potencia, SOC, límite de importación) se recalculan aquí y no se toman
    del notebook, que solo aporta el caso base.
    """
    power_MW = capacity_MWh / DURACION_H
    SOC_max = capacity_MWh
    SOC_min = capacity_MWh * (1 - g["DoD"])

    p_import_max_MW = float(df_esc["Load_MW"].max()) + power_MW
    p_export_max_MW = float(pv_MW)

    # El cargo CND/ASIC se declara sobre la capacidad de la planta, así que sigue
    # al PV. Es idéntico entre estrategias, pero mueve el denominador del ahorro.
    meses = df_esc["Fecha"].dt.to_period("M").nunique()
    costo_fijo = g["cnd_asic_COP_kW_mes"] * pv_MW * 1000 * meses

    horas_caras = set(df_esc.loc[df_esc["Franja_alta"], "Hora"].unique())
    if horas_caras:
        precio_alto = float(df_esc.loc[df_esc["Franja_alta"], "c_imp_COP_MWh"].mean())
    else:
        precio_alto = float(df_esc["c_imp_COP_MWh"].mean())

    comunes = dict(
        power_MW=power_MW,
        eff_c=g["eff_c"],
        eff_d=g["eff_d"],
        eff_rt=g["eff_rt"],
        dt=g["dt"],
        hora_inicio_h2=g["hora_inicio_h2"],
        horas_caras=horas_caras,
        precio_alto_COP_MWh=precio_alto,
        p_import_max_MW=p_import_max_MW,
        p_export_max_MW=p_export_max_MW,
    )

    filas = []

    for estrategia in ["Sin BESS"] + HEURISTICAS:
        try:
            df = g["simular_heuristica_ciclica"](
                df_base=df_esc,
                estrategia=estrategia,
                SOC_min=SOC_min,
                SOC_max=SOC_max,
                max_iter=60,
                **comunes,
            )
        except RuntimeError as exc:
            # El SOC periódico no cerró: se registra y se sigue con la grilla.
            filas.append({"Estrategia": estrategia, "Error": str(exc)})
            continue

        fila = g["resumir_resultado"](
            df, g["sitio"], estrategia, costo_fijo, g["c_epsilon_COP_MWh"], g["dt"]
        )
        filas.append(fila)

    if incluir_optimo:
        _, df_opt = g["optimizar_pypsa"](
            df_base=df_esc,
            power_MW=power_MW,
            SOC_min=SOC_min,
            SOC_max=SOC_max,
            eff_c=g["eff_c"],
            eff_d=g["eff_d"],
            pv_nominal_MW=pv_MW,
            p_import_max_MW=p_import_max_MW,
            p_export_max_MW=p_export_max_MW,
            solver_name="highs",
            c_epsilon_COP_MWh=g["c_epsilon_COP_MWh"],
            mip_rel_gap=MIP_REL_GAP,
        )
        filas.append(
            g["resumir_resultado"](
                df_opt, g["sitio"], OPTIMO, costo_fijo, g["c_epsilon_COP_MWh"], g["dt"]
            )
        )

    df_filas = pd.DataFrame(filas)
    for clave, valor in etiqueta.items():
        df_filas[clave] = valor
    df_filas["Potencia_BESS_MW"] = power_MW
    df_filas["Capacidad_BESS_MWh"] = capacity_MWh
    df_filas["PV_MW"] = pv_MW

    return df_filas


def agregar_metricas(df):
    """Añade ahorro y captura del beneficio, por grupo de escenario."""
    claves = [c for c in ("Escenario",) if c in df.columns]

    costo = df.pivot_table(
        index=claves, columns="Estrategia", values="Costo_total_suministro_COP"
    )

    ref_sin = costo["Sin BESS"]
    ref_opt = costo[OPTIMO]
    beneficio_max = ref_sin - ref_opt

    df = df.merge(
        pd.DataFrame(
            {
                "Costo_sin_BESS_COP": ref_sin,
                "Costo_optimo_COP": ref_opt,
                "Beneficio_maximo_COP": beneficio_max,
            }
        ).reset_index(),
        on=claves,
        how="left",
    )

    df["Ahorro_COP"] = df["Costo_sin_BESS_COP"] - df["Costo_total_suministro_COP"]
    df["Ahorro_%"] = df["Ahorro_COP"] / df["Costo_sin_BESS_COP"] * 100

    # Si el BESS no aporta nada en el escenario, la captura no está definida.
    df["Captura_%"] = np.where(
        df["Beneficio_maximo_COP"] > 1.0,
        df["Ahorro_COP"] / df["Beneficio_maximo_COP"] * 100,
        np.nan,
    )
    df["Brecha_vs_optimo_%"] = (
        (df["Costo_total_suministro_COP"] - df["Costo_optimo_COP"])
        / df["Costo_optimo_COP"]
        * 100
    )

    return df


# ============================================================
# Barridos
# ============================================================

REJILLA_BESS = [15.0, 30.0, 60.0, 120.0, 200.0, 300.0, 480.0, 720.0, 1000.0, 1400.0]
REJILLA_PV = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0]
REJILLA_SPREAD = [0.0, 0.0103, 0.025, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60]

# Factorial: menos puntos por eje, pero cubriendo el mismo rango.
FACT_BESS = [30.0, 120.0, 300.0, 720.0, 1400.0]
FACT_PV = [25.0, 50.0, 100.0, 200.0]
FACT_SPREAD = [0.0103, 0.05, 0.15, 0.40]


def _imp_sin_bess(df_base):
    """Perfil de importación del caso sin BESS, base de la calibración tarifaria."""
    return (df_base["Load_MW"] - df_base["PV_MW"]).clip(lower=0)


def barrido_bess(g, df_base):
    print(f"\n--- Barrido A: capacidad del BESS ({len(REJILLA_BESS)} puntos) ---")
    partes = []
    for cap in REJILLA_BESS:
        t = time.time()
        partes.append(
            correr_punto(
                g,
                df_base,
                capacity_MWh=cap,
                pv_MW=PV_BASE_MW,
                etiqueta={"Escenario": f"BESS {cap:.0f} MWh"},
            )
        )
        print(f"  BESS {cap:7.0f} MWh / {cap/DURACION_H:6.1f} MW   {time.time()-t:5.1f} s")
    return agregar_metricas(pd.concat(partes, ignore_index=True))


def barrido_pv(g, df_base):
    print(f"\n--- Barrido B: capacidad del PV ({len(REJILLA_PV)} puntos) ---")
    partes = []
    for pv in REJILLA_PV:
        t = time.time()
        partes.append(
            correr_punto(
                g,
                escalar_pv(df_base, pv),
                capacity_MWh=CAPACIDAD_BASE_MWh,
                pv_MW=pv,
                etiqueta={"Escenario": f"PV {pv:.0f} MW"},
            )
        )
        print(f"  PV {pv:6.0f} MW   {time.time()-t:5.1f} s")
    return agregar_metricas(pd.concat(partes, ignore_index=True))


def barrido_tarifa(g, df_base):
    print(f"\n--- Barrido C: spread tarifario ({len(REJILLA_SPREAD)} puntos) ---")
    imp = _imp_sin_bess(df_base)
    horas_caras = set(df_base.loc[df_base["Franja_alta"], "Hora"].unique())

    partes = []
    for s in REJILLA_SPREAD:
        t = time.time()
        c_imp = tarifa_spread(df_base, imp, s, horas_caras)
        partes.append(
            correr_punto(
                g,
                aplicar_tarifa(df_base, c_imp),
                capacity_MWh=CAPACIDAD_BASE_MWh,
                pv_MW=PV_BASE_MW,
                etiqueta={"Escenario": f"Spread {s*100:.2f}%", "Spread_rel": s},
            )
        )
        print(
            f"  spread {s*100:5.2f} %  "
            f"({c_imp.min()/1000:6.1f} – {c_imp.max()/1000:6.1f} COP/kWh)   "
            f"{time.time()-t:5.1f} s"
        )
    return agregar_metricas(pd.concat(partes, ignore_index=True))


def barrido_factorial(g, df_base):
    combos = list(itertools.product(FACT_BESS, FACT_PV, FACT_SPREAD))
    print(f"\n--- Barrido D: factorial BESS x PV x tarifa ({len(combos)} puntos) ---")

    horas_caras = set(df_base.loc[df_base["Franja_alta"], "Hora"].unique())

    partes = []
    for i, (cap, pv, s) in enumerate(combos, start=1):
        t = time.time()

        df_esc = escalar_pv(df_base, pv)
        # La tarifa se recalibra con el perfil de importación DE ESE escenario:
        # al cambiar el PV cambia lo que el usuario compra, y con ello el nivel
        # que hace neutra a la tarifa.
        c_imp = tarifa_spread(df_esc, _imp_sin_bess(df_esc), s, horas_caras)
        df_esc = aplicar_tarifa(df_esc, c_imp)

        partes.append(
            correr_punto(
                g,
                df_esc,
                capacity_MWh=cap,
                pv_MW=pv,
                etiqueta={
                    "Escenario": f"B{cap:.0f}_P{pv:.0f}_S{s*100:.2f}",
                    "Capacidad_grid_MWh": cap,
                    "PV_grid_MW": pv,
                    "Spread_rel": s,
                },
            )
        )
        print(
            f"  [{i:3d}/{len(combos)}] BESS {cap:6.0f} MWh | PV {pv:5.0f} MW | "
            f"spread {s*100:5.2f} %   {time.time()-t:5.1f} s"
        )

    return agregar_metricas(pd.concat(partes, ignore_index=True))


# ============================================================
# Orquestación
# ============================================================

BARRIDOS = {
    "A_bess": barrido_bess,
    "B_pv": barrido_pv,
    "C_tarifa": barrido_tarifa,
    "D_factorial": barrido_factorial,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rehacer", action="store_true", help="ignora la caché y recalcula todo"
    )
    parser.add_argument(
        "--solo", nargs="*", choices=list(BARRIDOS), help="corre solo estos barridos"
    )
    args = parser.parse_args()

    SALIDA.mkdir(exist_ok=True)

    print("=" * 78)
    print("SENSIBILIDAD DEL MODELO BASE PV + BESS")
    print("=" * 78)

    g = cargar_entorno_notebook()
    df_base = g["df_base"].copy()

    print(f"Caso base: PV {PV_BASE_MW:.0f} MW | BESS {CAPACIDAD_BASE_MWh:.0f} MWh | "
          f"demanda pico {df_base['Load_MW'].max():.1f} MW")
    print(f"Excedente PV diario: {(df_base['PV_MW']-df_base['Load_MW']).clip(lower=0).sum()/365:.1f} MWh")

    objetivo = args.solo or list(BARRIDOS)

    for nombre in objetivo:
        destino = SALIDA / f"{nombre}.csv"
        if destino.exists() and not args.rehacer:
            print(f"\n--- {nombre}: ya existe {destino}, se omite (usa --rehacer) ---")
            continue

        t = time.time()
        df = BARRIDOS[nombre](g, df_base)
        df.to_csv(destino, index=False)
        print(f"--> {destino}  ({len(df)} filas, {time.time()-t:.0f} s)")

    print("\nListo. Las gráficas se generan en 03_Sensibilidad_BESS_PV_tarifa.ipynb")


if __name__ == "__main__":
    main()
