# -*- coding: utf-8 -*-
"""Experimento factorial de operación: sitio x señal x estrategia x configuración.

Diseño congelado en 08D y validado en 08D.1:

    SITIOS         Caribe, Antioquia, Suroccidente, Oriente, Nordeste   (5)
    SEÑALES        T0 cuasi-plana, T1 horaria indexada                  (2)
    ESTRATEGIAS    A, B, C, D, F                                        (5)
    CONFIGURACIONES K1..K4                                              (4)

    200 casos factoriales + 40 de baseline PV sin BESS.

El factor de configuración compara configuraciones representativas de distintos
regímenes de disponibilidad relativa de excedente PV frente a capacidad BESS,
sin volver a ejecutar una optimización de dimensionamiento. K1..K4 cambian PV y
energía a la vez, y K4 cambia además potencia y duración: el factor se
interpreta como «configuración PV+BESS», nunca como «ratio PV/BESS». El valor
`r` que acompaña a cada celda es descriptivo, para etiquetar el régimen.

El baseline sin BESS va aparte porque con capacidad cero las cinco estrategias
producen el mismo despacho: como nivel del factorial serían cincuenta filas
duplicadas.

Motor: `despachar()` del cuaderno 02 y `motor_multianual.simular_horizonte`.
CAPEX M0, trayectoria NREL C3E, 7.200 ciclos, vida calendario 20 años, residual
cero. No interviene ningún módulo de la línea legacy E1-E8.

Uso:
    .venv/Scripts/python.exe factorial_operativo.py            # smoke test + todo
    .venv/Scripts/python.exe factorial_operativo.py --smoke    # solo el smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import costos
import motor_multianual as MM
import solar_tmy as ST
import benchmark_optimo as BO

SALIDA = Path("resultados_nb08")
SITIOS = ["Caribe", "Antioquia", "Suroccidente", "Oriente", "Nordeste"]
CONFIGURACIONES = {                     # PV [MWp], P [MW], E [MWh]
    "K1": (70.0, 20.0, 160.0),
    "K2": (80.0, 20.0, 120.0),
    "K3": (85.0, 20.0, 100.0),
    "K4": (95.0, 30.0, 60.0),
}
PV_BASELINE = sorted({c[0] for c in CONFIGURACIONES.values()})
ANIOS_ANCLA = [1, 5, 10, 15, 20]

# Valores validados para Oriente / K3, contra los que corre el smoke test. Hacen
# falta los dos archivos: C4 cubre A, B, C y D, y C8C añade F.
REFERENCIA_SMOKE = (Path("resultados_nb02") / "C4_comparacion_ABCD.csv",
                    Path("resultados_nb02") / "C8C_estrategia_F.csv")


# ============================================================
# Entorno
# ============================================================


def cargar():
    """Series, reglas y parámetros del cuaderno 02, más los perfiles de los cinco TMY."""
    ns = BO.cargar_entorno_nb02()
    datos = ns["datos"][["Fecha", "Load_MW"]].copy()
    pu = {}
    for sitio in SITIOS:
        d = ST.perfil_pv(sitio, 1.0)[["Fecha", "PV_MW"]].rename(columns={"PV_MW": "pu"})
        m = datos.merge(d, on="Fecha", how="inner")
        if len(m) != 8760:
            raise RuntimeError(
                "{}: {} horas tras alinear con la demanda, se esperaban 8.760".format(
                    sitio, len(m)))
        pu[sitio] = m["pu"].to_numpy()
    ns["PU_SITIO"] = pu
    ns["ESTRATEGIAS_FACTORIAL"] = list(ns["ESTRATEGIAS"]) + [ns["ESTRATEGIA_F"]]
    ns["NOMBRE_TARIFA"] = {k[:2]: k for k in ns["TARIFAS"]}
    return ns


def _regimen(pu_sitio, load, pv_MW, cap_MWh, eta_c, DoD, dia_key):
    """Descriptores del régimen de excedente. No decide nada, solo etiqueta."""
    if cap_MWh <= 0:
        return {"r_medio": np.nan, "pct_dias_r_menor_1": np.nan,
                "cupo_medio_MWh": np.nan, "pct_dias_cupo_positivo": np.nan,
                "Excedente_PV_anual_MWh": np.nan}
    pv = pu_sitio * pv_MW
    exc = np.maximum(pv - np.minimum(pv, load), 0.0)
    exc_dia = pd.Series(exc).groupby(dia_key).sum().to_numpy()
    util = cap_MWh * DoD
    r = exc_dia * eta_c / util
    cupo = np.maximum(0.0, util - np.minimum(util, exc_dia * eta_c))
    return {"r_medio": float(r.mean()),
            "pct_dias_r_menor_1": float((r < 1).mean() * 100),
            "cupo_medio_MWh": float(cupo.mean()),
            "pct_dias_cupo_positivo": float((cupo > 1e-9).mean() * 100),
            "Excedente_PV_anual_MWh": float(exc.sum())}


# ============================================================
# Un caso
# ============================================================


def evaluar(ns, sitio, tarifa, estrategia, pv_MW, p_MW, e_MWh):
    """Un caso completo: despacho del año base, economía a 20 años y validación."""
    despachar, validar = ns["despachar"], ns["validar"]
    load, c_exp = ns["LOAD"], ns["c_exp"]
    c_imp = ns["TARIFAS"][ns["NOMBRE_TARIFA"][tarifa]]
    pu_sitio = ns["PU_SITIO"][sitio]
    DoD, eta_rt = ns["DoD"], ns["ETA_RT"]
    eta_c = eta_d = float(np.sqrt(eta_rt))

    c_deg = costos.costo_degradacion_R1_COP_MWh(p_MW, e_MWh, DoD, eta_rt) if e_MWh else 0.0
    capex_pv = costos.capex_pv(pv_MW)
    capex_bess = costos.capex_bess(p_MW, e_MWh) if e_MWh else 0.0
    opex = costos.opex_pv_anual(pv_MW) + (
        costos.opex_bess_anual(e_MWh, power_MW=p_MW) if e_MWh else 0.0)

    d = despachar(pu_sitio * pv_MW, load, c_imp, c_exp, estrategia,
                  power_MW=p_MW, cap_MWh=e_MWh, c_deg=c_deg)
    r = MM.simular_horizonte(
        despachar, pu_sitio, load, c_imp, c_exp, estrategia, pv_MW, p_MW, e_MWh,
        DoD=DoD, eff_rt=eta_rt, c_deg_COP_MWh=c_deg,
        capex_inicial_COP=capex_pv + capex_bess, opex_anual_COP=opex,
        anios_ancla=ANIOS_ANCLA)

    # --- validaciones físicas y económicas del caso ---
    fallos = list(validar(d, p_MW, e_MWh))
    soc = d.SOC_MWh.to_numpy()
    previo = np.concatenate(([d.attrs["SOC_inicial_MWh"]], soc[:-1]))
    esperado = previo + d.Carga_MW.to_numpy() * eta_c - d.Descarga_MW.to_numpy() / eta_d
    if np.abs(soc - esperado).max() > 1e-5:
        fallos.append("dinámica del SOC")
    glob = (d.PV_MW + d.Import_MW + d.Descarga_MW
            - d.Load_MW - d.Carga_MW - d.Export_MW - d.Vertimiento_MW)
    if glob.abs().max() > 1e-5:
        fallos.append("balance de energía")
    if d.attrs["cierre_ciclico_MWh"] > 1e-3:
        fallos.append("cierre cíclico")
    if not np.isfinite(r["VPN_COP"]):
        fallos.append("VPN no finito")

    reg = _regimen(pu_sitio, load, pv_MW, e_MWh, eta_c, DoD, ns["DIA_KEY"])

    # F no puede comprar de la red más allá del cupo diario que deja el sol
    if estrategia == ns["ESTRATEGIA_F"] and e_MWh > 0:
        exc = np.maximum(pu_sitio * pv_MW - np.minimum(pu_sitio * pv_MW, load), 0.0)
        exc_dia = pd.Series(exc).groupby(ns["DIA_KEY"]).sum().to_numpy()
        cupo = np.maximum(0.0, e_MWh * DoD - np.minimum(e_MWh * DoD, exc_dia * eta_c))
        red = (pd.Series(d.Carga_Red_MW.to_numpy() * eta_c)
               .groupby(ns["DIA_KEY"]).sum().to_numpy())
        if float(np.max(red - cupo)) > 1e-6:
            fallos.append("F excede el cupo de carga desde red")

    eb = float(d.Descarga_Red_MW.sum())
    util = e_MWh * DoD
    fila = {
        "Sitio": sitio, "Senal": tarifa, "Estrategia": estrategia,
        "PV_MW": pv_MW, "P_BESS_MW": p_MW, "E_BESS_MWh": e_MWh,
        "Duracion_h": (e_MWh / p_MW) if p_MW else 0.0,
        **reg,
        "VPN_COP": r["VPN_COP"], "TIR": r["TIR"],
        "Payback_anios": r["Payback_anios"],
        "CAPEX_COP": capex_pv + capex_bess, "CAPEX_PV_COP": capex_pv,
        "CAPEX_BESS_COP": capex_bess, "OPEX_anual_COP": opex,
        "EFC_total": r["EFC_total_descargado"], "EFC_anio1": r["EFC_anio1"],
        "N_renovaciones": r["n_renovaciones"],
        "Anios_renovacion": ";".join(str(a) for a in r["Anios_renovacion"]),
        "Beneficio_anio1_COP": r["Beneficio_anio1_COP"],
        "Generacion_PV_MWh": float(d.PV_MW.sum()),
        "Carga_PV_MWh": float(d.Carga_PV_MW.sum()),
        "Carga_red_MWh": float(d.Carga_Red_MW.sum()),
        "Descarga_demanda_MWh": float(d.Descarga_Demanda_MW.sum()),
        "Exportacion_BESS_MWh": eb,
        "Exportacion_PV_MWh": float(d.Export_MW.sum()) - eb,
        "Exportacion_total_MWh": float(d.Export_MW.sum()),
        "Importacion_MWh": float(d.Import_MW.sum()),
        "Vertimiento_MWh": float(d.Vertimiento_MW.sum()),
        "Fallos": " | ".join(fallos),
    }
    if util > 0:
        fila["EFC_anio1_calc"] = float(d.Descarga_MW.sum()) / util
    return fila


# ============================================================
# Corridas
# ============================================================


def smoke_test(ns, tol=1.0):
    """Oriente / K3 contra los valores validados en 08C. Devuelve (ok, tabla)."""
    ref = pd.concat([pd.read_csv(f)[["Tarifa", "Estrategia", "VPN_COP"]]
                     for f in REFERENCIA_SMOKE], ignore_index=True)
    ref = ref.drop_duplicates(["Tarifa", "Estrategia"], keep="last")
    pv, p, e = CONFIGURACIONES["K3"]
    filas, ok = [], True
    for tarifa in ("T0", "T1"):
        for est in ns["ESTRATEGIAS_FACTORIAL"]:
            f = evaluar(ns, "Oriente", tarifa, est, pv, p, e)
            s = ref[(ref.Tarifa == tarifa) & (ref.Estrategia == est)]
            esperado = float(s.VPN_COP.iloc[0]) if len(s) else np.nan
            dif = abs(f["VPN_COP"] - esperado) if len(s) else np.nan
            bien = bool(len(s)) and dif <= tol and not f["Fallos"]
            ok = ok and bien
            filas.append({"Senal": tarifa, "Estrategia": est,
                          "VPN_mil_MM": f["VPN_COP"] / 1e9,
                          "Esperado_mil_MM": esperado / 1e9 if len(s) else np.nan,
                          "Diferencia_COP": dif, "OK": bien,
                          "Fallos": f["Fallos"]})
    return ok, pd.DataFrame(filas)


def ejecutar_factorial(ns, verbose=True):
    filas, t0, n = [], time.time(), 0
    total = len(SITIOS) * 2 * len(ns["ESTRATEGIAS_FACTORIAL"]) * len(CONFIGURACIONES)
    for sitio in SITIOS:
        for tarifa in ("T0", "T1"):
            for k, (pv, p, e) in CONFIGURACIONES.items():
                for est in ns["ESTRATEGIAS_FACTORIAL"]:
                    filas.append({"Config": k,
                                  **evaluar(ns, sitio, tarifa, est, pv, p, e)})
                    n += 1
            if verbose:
                print("  {:<14} {}  {:>3}/{}  ({:5.1f} min)".format(
                    sitio, tarifa, n, total, (time.time() - t0) / 60), flush=True)
    return pd.DataFrame(filas), time.time() - t0


def ejecutar_baseline(ns, verbose=True):
    """PV sin BESS. Una fila por contexto: las cinco estrategias coinciden."""
    est = ns["ESTRATEGIAS"][0]
    filas, t0, n = [], time.time(), 0
    for sitio in SITIOS:
        for tarifa in ("T0", "T1"):
            for pv in PV_BASELINE:
                f = evaluar(ns, sitio, tarifa, est, pv, 0.0, 0.0)
                for c in ("Estrategia", "r_medio", "pct_dias_r_menor_1",
                          "cupo_medio_MWh", "pct_dias_cupo_positivo",
                          "Carga_PV_MWh", "Carga_red_MWh", "Descarga_demanda_MWh",
                          "Exportacion_BESS_MWh", "EFC_total", "EFC_anio1",
                          "N_renovaciones", "Anios_renovacion", "Duracion_h",
                          "P_BESS_MW", "E_BESS_MWh", "CAPEX_BESS_COP"):
                    f.pop(c, None)
                filas.append(f)
                n += 1
        if verbose:
            print("  baseline {:<14} {:>3}/{}  ({:4.1f} min)".format(
                sitio, n, len(SITIOS) * 2 * len(PV_BASELINE),
                (time.time() - t0) / 60), flush=True)
    return pd.DataFrame(filas), time.time() - t0


def anadir_valor_incremental(fac, base):
    """Delta_VPN_BESS contra el mismo sitio, señal y tamaño de PV, sin batería."""
    ref = base.set_index(["Sitio", "Senal", "PV_MW"]).VPN_COP
    idx = pd.MultiIndex.from_frame(fac[["Sitio", "Senal", "PV_MW"]])
    fac = fac.copy()
    fac["VPN_PV_solo_COP"] = ref.reindex(idx).to_numpy()
    fac["Delta_VPN_BESS_COP"] = fac.VPN_COP - fac.VPN_PV_solo_COP
    fac["Delta_CAPEX_BESS_COP"] = fac.CAPEX_BESS_COP
    return fac


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="ejecuta solo el smoke test y termina")
    args = p.parse_args(argv)

    costos.consola_utf8()
    pd.set_option("display.width", 240)
    SALIDA.mkdir(exist_ok=True)
    L = "=" * 110

    print(L); print("FACTORIAL OPERATIVO — carga del entorno"); print(L)
    ns = cargar()
    print("  {} sitios · 2 señales · {} estrategias · {} configuraciones".format(
        len(SITIOS), len(ns["ESTRATEGIAS_FACTORIAL"]), len(CONFIGURACIONES)))

    print()
    print(L); print("SMOKE TEST — Oriente / K3 contra 08C"); print(L)
    ok, tabla = smoke_test(ns)
    for _, x in tabla.iterrows():
        print("  {:<4}{:<22}{:>14,.6f}{:>16,.6f}   dif {:>10.2e}   {}".format(
            x.Senal, x.Estrategia, x.VPN_mil_MM, x.Esperado_mil_MM,
            x.Diferencia_COP, "OK" if x.OK else "*** DISCREPA ***"))
    if not ok:
        raise SystemExit("\n*** El smoke test no reproduce 08C. No se lanza el factorial.")
    print("\n  El smoke test reproduce 08C. Continúa el factorial.")
    if args.smoke:
        return 0

    print()
    print(L); print("FACTORIAL — 200 casos"); print(L)
    fac, t_fac = ejecutar_factorial(ns)

    print()
    print(L); print("BASELINE — PV sin BESS, 40 casos"); print(L)
    base, t_base = ejecutar_baseline(ns)

    fac = anadir_valor_incremental(fac, base)
    fac.to_csv(SALIDA / "factorial_08.csv", index=False)
    base.to_csv(SALIDA / "baseline_pv_08.csv", index=False)

    resumen = pd.DataFrame([{
        "Casos_factoriales": len(fac), "Casos_baseline": len(base),
        "Tiempo_factorial_min": t_fac / 60, "Tiempo_baseline_min": t_base / 60,
        "Casos_con_fallos": int((fac.Fallos != "").sum() + (base.Fallos != "").sum()),
        "Casos_con_renovacion": int((fac.N_renovaciones > 0).sum()),
        "Renovaciones_totales": int(fac.N_renovaciones.sum()),
        "VPN_min_mil_MM": fac.VPN_COP.min() / 1e9,
        "VPN_max_mil_MM": fac.VPN_COP.max() / 1e9,
        "TIR_min": fac.TIR.min(), "TIR_max": fac.TIR.max(),
        "EFC_min": fac.EFC_total.min(), "EFC_max": fac.EFC_total.max(),
        "Delta_VPN_positivo": int((fac.Delta_VPN_BESS_COP > 0).sum()),
        "Delta_VPN_negativo": int((fac.Delta_VPN_BESS_COP <= 0).sum()),
    }])
    resumen.to_csv(SALIDA / "resumen_ejecucion_08.csv", index=False)
    print()
    print("Guardado en {}/: factorial_08.csv, baseline_pv_08.csv, "
          "resumen_ejecucion_08.csv".format(SALIDA))
    return 0


if __name__ == "__main__":
    sys.exit(main())
