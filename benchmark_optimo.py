# -*- coding: utf-8 -*-
"""Benchmarks de despacho optimo con prevision perfecta para el cuaderno 02.

Dos benchmarks, no uno, porque responden preguntas distintas:

    optimo_full  Prevision perfecta con el conjunto COMPLETO de acciones que
                 permite el modelo: carga desde PV, carga desde red, descarga a
                 demanda y exportacion desde el BESS. Acota superiormente el
                 OBJETIVO OPERATIVO ANUAL dentro de ese espacio ampliado, pero
                 no necesariamente el VPN a 20 anos: ver la nota de abajo.

    optimo_b     Prevision perfecta con carga acotada al excedente PV y sin
                 camino BESS -> exportacion. El reparto del PV entre usuario
                 y exportacion sigue siendo una decision del modelo: no impone
                 la prioridad instantanea de autoconsumo de la regla B.

La brecha B -> optimo_b incluye programacion, asignacion del PV e informacion;
no aisla la calidad de decision de B. Optimo_b -> optimo_full cuantifica el
incremento al ampliar conjuntamente las rutas del BESS. La procedencia solar
de la carga en optimo_b se comprueba en la salida: su tope por excedente no
equivale por si solo a separar fisicamente las fuentes.

Diferencia con `optimizar_pypsa` del cuaderno 01, que NO se modifica y sigue
siendo el productivo de la linea E1-E8: alli el BESS no tiene camino fisico
hacia el bus de exportacion, mientras que las reglas C y D si exportan desde la
bateria. Sin ese camino el optimo no cubre el espacio de acciones de las
heuristicas y no es cota superior del conjunto.

`c_deg_COP_MWh` entra en el objetivo como precio sombra del throughput y NO en
el flujo de caja, igual que en las heuristicas: la bateria ya se paga en el
CAPEX y sus renovaciones las contabiliza `motor_multianual.simular_horizonte`.

CONSECUENCIA, medida en la fase 08G: como `c_deg` es lineal en el throughput y
la envolvente de ciclos es un escalon, el optimizador anual NO internaliza de
forma discreta el costo de una renovacion. Por eso `optimo_full` acota el
objetivo operativo anual dentro de su espacio de acciones, pero NO
necesariamente el VPN a 20 anos: puede ciclar mas, cruzar la envolvente y
renovar. En Caribe/K4/T1 eso ocurre y su VPN queda 1,171 mil MM por debajo del
de `optimo_b`, pese a dominarlo en el objetivo que ambos minimizan. Al comparar
VPN conviene usar `max(VPN_optimo_b, VPN_optimo_full)`, que es el mejor VPN
entre los dos benchmarks operativos evaluados y no una cota superior del VPN.

Las dos funciones devuelven un DataFrame con las mismas columnas que
`despachar()` del cuaderno 02, de modo que se validan con el mismo `validar()`
y se evaluan con el mismo motor multianual.

Procedencia: codigo validado en las fases C5 y C6.2, trasladado aqui sin
cambios. La regresion contra `resultados_nb02/C6_economia_OPT.csv` y
`C62_prevision_vs_flexibilidad.csv` esta en el cuaderno 09.
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

# Celdas del cuaderno 02 que son diagnosticos de una sola vez: no definen nada
# que el modelo necesite y tardan. Se saltan al cargar el entorno.
_DIAGNOSTICOS = (
    "IDENTIDAD DE LA DEMANDA CON 01B",
    "EQUIVALENCIA CON LA MALLA ANTERIOR",
    "IMPACTO DEL CAMBIO DE RECURSO SOLAR",
    "IMPACTO DE LIMPIAR LA DEMANDA",
)

# El corte: hasta `evaluar_finanzas` ya estan definidos los datos, las tarifas,
# `despachar` y `validar`. Lo que sigue son barridos que no hacen falta aqui.
_ANCLA_DE_CORTE = "def evaluar_finanzas"

CUADERNO = Path("02_Optimizacion_portafolio_PV_BESS.ipynb")


def cargar_entorno_nb02(ruta=None):
    """Ejecuta las celdas del cuaderno 02 y devuelve su espacio de nombres.

    Mismo patron que `sensibilidad.cargar_entorno_notebook()` para el cuaderno
    01: el cuaderno es la fuente de verdad y aqui no se reimplementa nada.

    Devuelve un dict con, entre otros: `despachar`, `validar`, `PV_PU`, `LOAD`,
    `c_exp`, `TARIFAS`, `ESTRATEGIAS`, `datos`, `DoD`, `ETA_RT`, `ETA_C`, `ETA_D`.
    """
    ruta = Path(ruta) if ruta is not None else CUADERNO
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    ns = {"display": lambda *a, **k: None, "__name__": "__nb02__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for celda in nb["cells"]:
            if celda["cell_type"] != "code":
                continue
            fuente = "".join(celda["source"])
            if any(m in fuente for m in _DIAGNOSTICOS):
                continue
            exec(compile(fuente, str(ruta), "exec"), ns)
            if _ANCLA_DE_CORTE in fuente:
                break
    faltan = [k for k in ("despachar", "validar", "PV_PU", "LOAD", "c_exp",
                          "TARIFAS", "ESTRATEGIAS") if k not in ns]
    if faltan:
        raise RuntimeError(
            "El cuaderno 02 no definio {}. Cambio el orden de las celdas?".format(faltan))
    return ns


def optimo_full(pv_MW, load_MW, c_imp, c_exp_h, power_MW, cap_MWh, *,
              DoD, eff_c, eff_d, c_deg_COP_MWh, fechas,
              solver_name="highs", mip_rel_gap=1e-7, tol=1e-6):
    """Despacho optimo anual con prevision perfecta, mismo espacio de acciones
    que A/B/C/D. Devuelve un DataFrame con las columnas de `despachar()`."""
    n_h = len(pv_MW)
    soc_max, soc_min = float(cap_MWh), float(cap_MWh) * (1.0 - DoD)
    p_exp_max = float(pv_MW.max() + power_MW) * 1.05
    p_imp_max = float(load_MW.max() + power_MW) * 1.05
    pv_p_nom = float(pv_MW.max())
    pv_pu = np.clip(pv_MW / pv_p_nom, 0.0, 1.0)

    sn = pd.DatetimeIndex(fechas)
    ci = pd.Series(np.asarray(c_imp, dtype=float), index=sn)
    ce = pd.Series(np.asarray(c_exp_h, dtype=float), index=sn)

    n = pypsa.Network()
    n.set_snapshots(sn)
    n.add("Carrier", "electricidad")
    n.add("Carrier", "bateria")
    n.add("Bus", "PV", carrier="electricidad")
    n.add("Bus", "Usuario", carrier="electricidad")
    n.add("Bus", "ExportBESS", carrier="electricidad")
    n.add("Bus", "BESS", carrier="bateria")

    # Recurso solar. Se puede recortar; el PPA no entra en el despacho.
    n.add("Generator", "PV_disponible", bus="PV", p_nom=pv_p_nom,
          p_min_pu=0.0, p_max_pu=pd.Series(pv_pu, index=sn), marginal_cost=0.0)
    n.add("Link", "PV_a_usuario", bus0="PV", bus1="Usuario", p_nom=pv_p_nom,
          p_min_pu=0.0, p_max_pu=1.0, efficiency=1.0)
    n.add("Generator", "Exportacion_PV", bus="PV", sign=-1, p_nom=p_exp_max,
          p_min_pu=0.0, p_max_pu=1.0, marginal_cost=-ce)

    n.add("Load", "Demanda", bus="Usuario",
          p_set=pd.Series(np.asarray(load_MW, dtype=float), index=sn))
    n.add("Generator", "Importacion_red", bus="Usuario", p_nom=p_imp_max,
          p_min_pu=0.0, p_max_pu=1.0, marginal_cost=ci)

    n.add("Store", "BESS_store", bus="BESS", e_nom=soc_max,
          e_min_pu=soc_min / soc_max, e_max_pu=1.0, e_cyclic=True,
          standing_loss=0.0)
    n.add("Link", "BESS_carga", bus0="Usuario", bus1="BESS", p_nom=power_MW,
          p_min_pu=0.0, p_max_pu=1.0, efficiency=eff_c,
          marginal_cost=c_deg_COP_MWh)
    n.add("Link", "BESS_descarga", bus0="BESS", bus1="Usuario",
          p_nom=power_MW / eff_d, p_min_pu=0.0, p_max_pu=1.0, efficiency=eff_d,
          marginal_cost=c_deg_COP_MWh * eff_d)
    # C5: camino BESS -> exportacion. Bus dedicado para que la energia enviada
    # aqui SOLO pueda venderse; evita la degeneracion de rutarla al usuario por
    # dos caminos con el mismo costo, que arruinaria la atribucion.
    n.add("Link", "BESS_a_export", bus0="BESS", bus1="ExportBESS",
          p_nom=power_MW / eff_d, p_min_pu=0.0, p_max_pu=1.0, efficiency=eff_d,
          marginal_cost=c_deg_COP_MWh * eff_d)
    n.add("Generator", "Exportacion_BESS", bus="ExportBESS", sign=-1,
          p_nom=p_exp_max, p_min_pu=0.0, p_max_pu=1.0, marginal_cost=-ce)

    horas_pcc = sn[(ce.to_numpy() > ci.to_numpy())]

    def extra(network, sns):
        m = network.model
        lp = m.variables["Link-p"]
        # La potencia de descarga es UNA sola: atender demanda y exportar
        # comparten el mismo inversor.
        m.add_constraints(
            lp.sel(name="BESS_descarga") + lp.sel(name="BESS_a_export")
            <= power_MW / eff_d,
            name="BESS_potencia_descarga_total")

        # Exclusividad en el punto de conexion. Solo puede atar donde exportar
        # pagaria mas que importar; en el resto es redundante (bajar import y
        # export a la vez cambia el costo en delta*(c_exp - c_imp) <= 0).
        hp = horas_pcc.intersection(pd.DatetimeIndex(sns))
        if len(hp) == 0:
            return
        hp = pd.Index(hp, name="snapshot")
        gp = m.variables["Generator-p"]
        modo = m.add_variables(binary=True, coords=[hp], name="PCC_modo_importacion")
        m.add_constraints(
            gp.sel(name="Importacion_red", snapshot=hp) <= p_imp_max * modo,
            name="PCC_limite_importacion")
        m.add_constraints(
            gp.sel(name="Exportacion_PV", snapshot=hp)
            + gp.sel(name="Exportacion_BESS", snapshot=hp)
            <= p_exp_max * (1 - modo),
            name="PCC_limite_exportacion")

    estado, condicion = n.optimize(
        solver_name=solver_name, extra_functionality=extra,
        solver_options={"mip_rel_gap": mip_rel_gap}, log_to_console=False)
    if estado != "ok" or condicion != "optimal":
        raise RuntimeError("PyPSA no encontro optimo: {} / {}".format(estado, condicion))

    pv_desp = n.generators_t.p["PV_disponible"].to_numpy()
    pv_a_us = n.links_t.p0["PV_a_usuario"].to_numpy()
    imp = n.generators_t.p["Importacion_red"].to_numpy()
    exp_pv = n.generators_t.p["Exportacion_PV"].to_numpy()
    exp_bess = n.generators_t.p["Exportacion_BESS"].to_numpy()
    carga = n.links_t.p0["BESS_carga"].to_numpy()
    dis_dem = -n.links_t.p1["BESS_descarga"].to_numpy()
    dis_exp = -n.links_t.p1["BESS_a_export"].to_numpy()
    soc = n.stores_t.e["BESS_store"].to_numpy()

    # Misma descomposicion contable que usa el cuaderno 01.
    pv_to_load = np.minimum(pv_a_us, load_MW)
    pv_restante = np.maximum(0.0, pv_a_us - pv_to_load)
    ch_pv = np.minimum(carga, pv_restante)
    ch_red = np.maximum(0.0, carga - ch_pv)

    d = pd.DataFrame({
        "PV_MW": pv_MW, "Load_MW": load_MW, "PV_a_Load_MW": pv_to_load,
        "Carga_PV_MW": ch_pv, "Carga_Red_MW": ch_red, "Carga_MW": carga,
        "Descarga_Demanda_MW": dis_dem, "Descarga_Red_MW": dis_exp,
        "Descarga_MW": dis_dem + dis_exp, "SOC_MWh": soc,
        "Red_a_Load_MW": imp - ch_red,
        "Import_MW": imp, "Export_MW": exp_pv + exp_bess,
        "Vertimiento_MW": np.maximum(0.0, pv_MW - pv_desp),
    })
    d.attrs["SOC_inicial_MWh"] = float(soc[-1])
    d.attrs["cierre_ciclico_MWh"] = 0.0
    d.attrs["n_binarias"] = int(len(horas_pcc))
    d.attrs["objetivo_COP"] = float(n.objective)
    return d


def optimo_b(pv_MW, load_MW, c_imp, c_exp_h, power_MW, cap_MWh, *,
             DoD, eff_c, eff_d, c_deg_COP_MWh, fechas,
             solver_name="highs", mip_rel_gap=1e-7, tol=1e-6):
    """Prevision perfecta con el espacio de acciones de B Merito diario."""
    soc_max, soc_min = float(cap_MWh), float(cap_MWh) * (1.0 - DoD)
    p_exp_max = float(pv_MW.max() + power_MW) * 1.05
    p_imp_max = float(load_MW.max() + power_MW) * 1.05
    pv_p_nom = float(pv_MW.max())
    pv_pu = np.clip(pv_MW / pv_p_nom, 0.0, 1.0)

    sn = pd.DatetimeIndex(fechas)
    ci = pd.Series(np.asarray(c_imp, dtype=float), index=sn)
    ce = pd.Series(np.asarray(c_exp_h, dtype=float), index=sn)

    # El tope de carga de B: solo el excedente solar de la propia hora.
    excedente = np.maximum(0.0, np.asarray(pv_MW) - np.asarray(load_MW))
    tope_carga = pd.Series(np.clip(excedente / power_MW, 0.0, 1.0), index=sn)

    n = pypsa.Network()
    n.set_snapshots(sn)
    n.add("Carrier", "electricidad")
    n.add("Carrier", "bateria")
    n.add("Bus", "PV", carrier="electricidad")
    n.add("Bus", "Usuario", carrier="electricidad")
    n.add("Bus", "BESS", carrier="bateria")

    n.add("Generator", "PV_disponible", bus="PV", p_nom=pv_p_nom,
          p_min_pu=0.0, p_max_pu=pd.Series(pv_pu, index=sn), marginal_cost=0.0)
    n.add("Link", "PV_a_usuario", bus0="PV", bus1="Usuario", p_nom=pv_p_nom,
          p_min_pu=0.0, p_max_pu=1.0, efficiency=1.0)
    n.add("Generator", "Exportacion_PV", bus="PV", sign=-1, p_nom=p_exp_max,
          p_min_pu=0.0, p_max_pu=1.0, marginal_cost=-ce)

    n.add("Load", "Demanda", bus="Usuario",
          p_set=pd.Series(np.asarray(load_MW, dtype=float), index=sn))
    n.add("Generator", "Importacion_red", bus="Usuario", p_nom=p_imp_max,
          p_min_pu=0.0, p_max_pu=1.0, marginal_cost=ci)

    n.add("Store", "BESS_store", bus="BESS", e_nom=soc_max,
          e_min_pu=soc_min / soc_max, e_max_pu=1.0, e_cyclic=True,
          standing_loss=0.0)
    # Restriccion 1: la carga no puede superar el excedente PV de la hora.
    # Es el mismo tope que aplica `despachar()` en la rama de excedente de B,
    # y cierra por completo la carga desde red.
    n.add("Link", "BESS_carga", bus0="Usuario", bus1="BESS", p_nom=power_MW,
          p_min_pu=0.0, p_max_pu=tope_carga, efficiency=eff_c,
          marginal_cost=c_deg_COP_MWh)
    # Restriccion 2: no existe camino BESS -> exportacion.
    n.add("Link", "BESS_descarga", bus0="BESS", bus1="Usuario",
          p_nom=power_MW / eff_d, p_min_pu=0.0, p_max_pu=1.0, efficiency=eff_d,
          marginal_cost=c_deg_COP_MWh * eff_d)

    horas_pcc = sn[(ce.to_numpy() > ci.to_numpy())]

    def extra(network, sns):
        m = network.model
        hp = horas_pcc.intersection(pd.DatetimeIndex(sns))
        if len(hp) == 0:
            return
        hp = pd.Index(hp, name="snapshot")
        gp = m.variables["Generator-p"]
        modo = m.add_variables(binary=True, coords=[hp], name="PCC_modo_importacion")
        m.add_constraints(gp.sel(name="Importacion_red", snapshot=hp)
                          <= p_imp_max * modo, name="PCC_limite_importacion")
        m.add_constraints(gp.sel(name="Exportacion_PV", snapshot=hp)
                          <= p_exp_max * (1 - modo), name="PCC_limite_exportacion")

    estado, condicion = n.optimize(
        solver_name=solver_name, extra_functionality=extra,
        solver_options={"mip_rel_gap": mip_rel_gap}, log_to_console=False)
    if estado != "ok" or condicion != "optimal":
        raise RuntimeError("OPT_B no encontro optimo: {} / {}".format(estado, condicion))

    pv_desp = n.generators_t.p["PV_disponible"].to_numpy()
    pv_a_us = n.links_t.p0["PV_a_usuario"].to_numpy()
    imp = n.generators_t.p["Importacion_red"].to_numpy()
    exp_pv = n.generators_t.p["Exportacion_PV"].to_numpy()
    carga = n.links_t.p0["BESS_carga"].to_numpy()
    dis_dem = -n.links_t.p1["BESS_descarga"].to_numpy()
    soc = n.stores_t.e["BESS_store"].to_numpy()

    pv_to_load = np.minimum(pv_a_us, load_MW)
    pv_restante = np.maximum(0.0, pv_a_us - pv_to_load)
    ch_pv = np.minimum(carga, pv_restante)
    ch_red = np.maximum(0.0, carga - ch_pv)

    d = pd.DataFrame({
        "PV_MW": pv_MW, "Load_MW": load_MW, "PV_a_Load_MW": pv_to_load,
        "Carga_PV_MW": ch_pv, "Carga_Red_MW": ch_red, "Carga_MW": carga,
        "Descarga_Demanda_MW": dis_dem, "Descarga_Red_MW": np.zeros(len(soc)),
        "Descarga_MW": dis_dem, "SOC_MWh": soc,
        "Red_a_Load_MW": imp - ch_red, "Import_MW": imp, "Export_MW": exp_pv,
        "Vertimiento_MW": np.maximum(0.0, pv_MW - pv_desp),
    })
    d.attrs["SOC_inicial_MWh"] = float(soc[-1])
    d.attrs["cierre_ciclico_MWh"] = 0.0
    d.attrs["n_binarias"] = int(len(horas_pcc))
    d.attrs["objetivo_COP"] = float(n.objective)
    return d
