"""
Supuestos de CAPEX y OPEX del sistema PV + BESS, con la fuente de cada número.

Hasta ahora el modelo base (`01_Modelo base PV_BESS.ipynb`) comparaba las
estrategias solo por su COSTO OPERATIVO anual: PPA solar, cargo CND/ASIC, compra
de energía a la red y venta de excedentes. Ese costo responde "¿cómo conviene
operar la batería?" pero no responde "¿conviene tener la batería?", porque no
incluye lo que cuesta comprarla ni mantenerla.

Este módulo agrega las dos piezas que faltaban:

    CAPEX  inversión inicial, se paga una sola vez en el año 0
    OPEX   operación y mantenimiento, se paga todos los años

y las convierte en indicadores comparables entre estrategias: costo anual
equivalente, VPN, TIR, payback descontado y costo nivelado de suministro.

--------------------------------------------------------------------------------
QUÉ ENTRA EN LA INVERSIÓN Y QUÉ NO
--------------------------------------------------------------------------------

El caso base remunera la energía solar mediante un PPA de 300 COP/kWh liquidado
sobre toda la generación disponible (`ppa_solar_COP_kWh` en el notebook 01). Bajo
esa figura la planta fotovoltaica NO es un activo del usuario: es un contrato de
compra, y su costo ya está adentro de `Costo_total_suministro_COP`. Sumarle
además el CAPEX del PV sería contarla dos veces.

Por eso, en la configuración por defecto:

    - El PV entra como COSTO OPERATIVO (el PPA), no como inversión.
    - El BESS entra como INVERSIÓN (CAPEX + OPEX).

El BESS es además lo único que cambia entre los cinco escenarios: los cinco
tienen los mismos 100 MW de PV y la misma demanda. `Sin BESS` tiene CAPEX cero, y
las otras cuatro pagan exactamente la misma batería. La comparación económica se
reduce entonces a una sola pregunta: ¿el ahorro operativo que consigue cada regla
alcanza para pagar la batería?

Para el caso en que el PV sí sea propio (autogeneración, sin PPA) las funciones
aceptan `pv_propio=True` y agregan CAPEX y OPEX de la planta solar. Ese escenario
se reporta aparte en el notebook 06, porque cambia el nivel de todos los
indicadores pero no el orden entre estrategias: es un costo común.

--------------------------------------------------------------------------------
FUENTES DE LOS SUPUESTOS
--------------------------------------------------------------------------------

Todas las referencias están en el diccionario `REFERENCIAS` de abajo, con URL y
fecha de consulta, y se imprimen con `costos.ficha()`. Resumen:

  BESS — CAPEX
    Se usa la convención de NREL: el costo se parte en un componente de ENERGÍA
    (USD/kWh, celdas y contenedor) y uno de POTENCIA (USD/kW, PCS, transformador
    y conexión), de modo que el modelo sirva para cualquier duración:

        costo total [USD/kWh] = componente_energía + componente_potencia / horas

    Valor base: 220 USD/kWh + 120 USD/kW → 250 USD/kWh para un sistema de 4 h,
    que es la duración del caso base (30 MW / 120 MWh).

    El rango publicado para 2025 es muy ancho y conviene tenerlo presente:

        125 USD/kWh   Ember (oct-2025), proyecto llave en mano fuera de China y
                      EE. UU.: ~75 equipo + ~50 instalación y conexión.
        192 USD/kWh   IRENA (2024), promedio mundial ponderado de costo total
                      instalado de almacenamiento en baterías.
        250 USD/kWh   ESTE MODELO. Punto medio: equipo asiático con sobrecosto
                      logístico, arancelario y de EPC en Colombia.
        321 USD/kWh   NREL 2025, proyección "mid" para 2025, mercado EE. UU.
        350 USD/kWh   NREL 2025, proyección "high" para 2025.

    Se eligió 250 USD/kWh y no el dato de EE. UU. porque el benchmark de NREL es
    un costeo bottom-up del mercado estadounidense, con costos de mano de obra,
    permisos e interconexión que no aplican a Colombia; y no el dato de Ember
    porque 125 USD/kWh corresponde a proyectos de escala mucho mayor con acceso
    directo a fabricantes. La sensibilidad del VPN a este número se calcula
    explícitamente en el notebook 06.

  BESS — OPEX
    3,0 USD/kWh-año sobre la capacidad instalada, que para 4 h equivale a
    12 USD/kW-año. Es el extremo BAJO del rango de Lazard (3,0–8,0 USD/kWh-año
    para almacenamiento independiente de escala de servicio público, 100 MW/400
    MWh). Se toma el extremo bajo a propósito, porque el rango alto de Lazard y
    el FOM de NREL (4 % del costo en USD/kW, ≈ 40 USD/kW-año) incluyen
    AUMENTACIÓN —añadir celdas para compensar la degradación— y en este modelo la
    degradación ya se paga por separado, como reposición del BESS al agotar sus
    ciclos. Cargarla también al OPEX sería contarla dos veces.

  PV — CAPEX y OPEX (solo si `pv_propio=True`)
    850 USD/kW y 14 USD/kW-año. IRENA reporta 691 USD/kW como promedio mundial
    ponderado de 2024 y Lazard 1.150–1.600 USD/kW para EE. UU.; 850 USD/kW es un
    valor intermedio coherente con proyectos de escala de servicio público en
    Colombia. El OPEX de 14 USD/kW-año es el extremo alto del rango de Lazard
    (11–14 USD/kW-año).

  TRM
    3.126,08 COP/USD, TRM vigente al 5 de septiembre de 2026, utilizada como
    supuesto de conversión para la evaluación económica. Se adopta la TRM vigente
    en lugar del promedio de 2025 para representar las condiciones de mercado
    observables al momento de la evaluación.

    Es un supuesto congelado, no una consulta dinámica: el valor está fijado en
    el módulo y no se recalcula en cada ejecución.

    Convención de unidades. La economía se modela internamente en COP, porque las
    tarifas, el precio de bolsa y los costos horarios son señales colombianas. La
    TRM se usa en dos direcciones: para llevar a COP los componentes de inversión
    que están originalmente en USD, y para reportar en USD los resultados
    económicos —VPN, CAPEX y las diferencias derivadas de ellos—.

    Supuesto anterior, conservado como referencia histórica: 4.052,86 COP/USD,
    promedio aritmético de la TRM diaria de 2025. Los resultados calculados bajo
    ese supuesto están archivados con el sufijo `_TRM_2025_LEGACY`.

  Vida útil y ciclos
    7.200 ciclos equivalentes a DoD nominal y 20 años de vida calendario. Son los
    valores que usa el modelo: `CICLOS_VIDA_BESS` y `VIDA_CALENDARIO_BESS`.

    Los respalda una fuente comercial confidencial de fabricante de sistemas
    BESS, 2026, que documenta un término contractual de 20 años y un programa de
    garantía y O&M construido sobre 1 ciclo por día, del que resultan del orden
    de 7.300 ciclos acumulados al año 20. **Los 7.200 del modelo son una
    aproximación conservadora y redondeada de ese orden de magnitud, no una
    cifra textual del documento.**

    Qué NO respalda esa fuente, y por tanto sigue sin cita propia:

      - el criterio exacto de fin de vida (EOL);
      - el porcentaje de capacidad retenida al final del término;
      - la degradación anual (`DEGRADACION_BESS_ANUAL`, todavía PENDIENTE);
      - la política de reposición;
      - el valor residual.

    Esos cinco puntos requieren evidencia adicional y no deben inferirse de esta
    fuente.

    Cómo debe leerse entonces el límite de 7.200 ciclos: como una **envolvente de
    utilización y de vida contractual** que el modelo adopta, no como un fin de
    vida electroquímico asociado a un estado de salud conocido. El modelo **no
    asigna ningún SOH** a ese límite. Cualquier lectura que le atribuya 60 %,
    70 % u 80 % de capacidad retenida es una inferencia que la fuente no soporta.

    Convención anterior, ya superada: 6.000 ciclos y 15 años, apoyados en Lazard
    (350 ciclos/año durante 20 años) y en los 15 años de vida calendario que
    adopta NREL para sistemas de 4 h. Queda solo como registro histórico. Si
    vuelve a usarse debe ser como sensibilidad declarada y con fuente
    independiente que la justifique, nunca como supuesto base.

    Con los ~330 ciclos/año que hacen las heurísticas del caso base, el límite
    que manda sigue siendo el CALENDARIO, no los ciclos: 7.200/330 = 21,8 > 20.

  Financieros
    Horizonte 20 años (vida de proyecto que usa Lazard para almacenamiento),
    WACC real 10 % en pesos y degradación del recurso solar de 0,5 %/año.

--------------------------------------------------------------------------------
ADVERTENCIA SOBRE OTROS MÓDULOS DEL REPOSITORIO
--------------------------------------------------------------------------------

Antes de este archivo, tres módulos definían costos por su cuenta y NO coinciden
entre sí. Se dejaron como estaban para no alterar resultados ya publicados:

    02_Optimizacion_portafolio_PV_BESS.ipynb   220 USD/kWh + 120 USD/kW, TRM 4.200
    estrategias_avanzadas.py                   250 USD/kWh, TRM 4.200
    05_distribucion_probabilidad_PV_BESS.py    300 USD/kWh, TRM 4.000

`costos.py` es la referencia vigente. Unificar los otros tres está en la lista de
pendientes de INFORME.md.

Uso:
    import costos
    costos.ficha()                       # imprime supuestos y fuentes
    costos.capex_bess(30, 120)           # COP
    costos.evaluar(resumen, 30, 120)     # DataFrame con la economía completa
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================
# Tasa de cambio
# ============================================================

# TRM vigente al 5 de septiembre de 2026, utilizada como supuesto de conversión
# para la evaluación económica. Supuesto congelado: no se consulta dinámicamente.
# Fuente: Superintendencia Financiera de Colombia, TRM oficial vigente.
TRM_REFERENCIA_COP_USD = 3_126.08
TRM_FECHA_REFERENCIA = "2026-09-05"

# Nombre histórico, conservado para no romper los módulos que ya lo importan.
# Es el mismo valor: hay una sola TRM productiva en el proyecto.
TRM_COP_USD = TRM_REFERENCIA_COP_USD

# CVOM por unidad de la línea térmica integrada, instrucción del usuario (18/09/2026).
# Orden explícito G1, G2, G3, G4; USD por MWh eléctrico generado.
# Reemplaza el CVOM uniforme de 11. No altera los resultados archivados.
# El cuaderno 11 conserva sus supuestos históricos para poder reproducirlo.
CVOM_TERMICO_USD_MWH = (9.28, 9.28, 15.32, 15.32)

# Escenario de gas elevado elegido por el usuario para la comparación integrada.
# Valor constante de estudio; no cotización observada ni pronóstico de precios.
GAS_TERMICO_USD_MMBTU = 20.0

# Factor de emisión de CO2 del gas natural genérico colombiano, FECOC 2016 (UPME),
# sobre poder calorífico inferior (PCI). Misma base que los consumos específicos de
# fabricante de las turbinas G1–G4 (explícita para G4; convención ISO para G1–G3).
FACTOR_EMISION_GAS_KG_TJ = 55_539.11
GJ_POR_MMBTU = 1.05505585262   # Btu internacional (IT)
FACTOR_EMISION_GAS_T_MMBTU = FACTOR_EMISION_GAS_KG_TJ * GJ_POR_MMBTU / 1e6   # 0,058597 t/MMBtu

# Precio interno de carbono (versión 2 de parámetros). Parámetro de escenario: cero
# reproduce los resultados vigentes. Se suma al costo variable de cada unidad como
# HR·FE_gas·precio y, en la modalidad `gas_y_red`, a la señal de compra como FE_red·precio.
# En `solo_gas` no grava las compras de red, aunque sus emisiones se reportan igual.
# No es un precio observado ni una tarifa tributaria; ver `carbono.py`.
PRECIO_CO2_USD_T = 0.0
MODALIDADES_CARBONO = ('gas_y_red', 'solo_gas')
PRECIOS_CO2_EXPLORATORIOS_USD_T = (0., 10., 20., 26., 50., 100.)

# Factor de emisión del Sistema Interconectado Nacional para las compras de red (emisiones asociadas a la
# electricidad comprada, basadas en ubicación). Decisión de la usuaria del 26/09/2026:
# - principal: XM, factor del SIN de 2025, el mismo año de la demanda y los precios del modelo;
# - sensibilidad: UPME, factor oficial para inventarios de GEI del año 2024 (año de El Niño).
# El factor MDL de margen combinado estima reducciones de proyectos y no se usa para las compras.
# Mantenerlo constante durante el horizonte financiero es un supuesto de escenario, no un pronóstico.
FACTOR_EMISION_RED_T_MWH = 0.097018445
FACTOR_EMISION_RED_UPME_2024_T_MWH = 0.220

# Supuesto anterior, solo como referencia documental. NO lo usa ningún cálculo.
TRM_PROMEDIO_2025_COP_USD = 4_052.86

# ============================================================
# BESS
# ============================================================

# HISTÓRICO. El CAPEX ya no sale de estas constantes sino de las curvas del
# Excel (ver `capex_bess`). Se conservan porque `REFERENCIAS` las cita y
# porque documentan el orden de magnitud del costo plano equivalente.
CAPEX_BESS_ENERGIA_USD_KWH = 220.0  # celdas, módulos, contenedor, BMS
CAPEX_BESS_POTENCIA_USD_KW = 120.0  # PCS, transformador, obra civil, conexión

OPEX_BESS_USD_KWH_ANIO = 3.0  # HISTÓRICO: hoy el OPEX es 1,5 % del CAPEX
OPEX_BESS_FRACCION_CAPEX = 0.015  # 1,5 % del CAPEX del BESS al año
OPEX_PV_FRACCION_CAPEX = 0.020  # 2 % del CAPEX solar al año

# Baseline técnico de trabajo, respaldado por una fuente comercial confidencial
# de fabricante de sistemas BESS, 2026: término contractual de 20 años y
# garantía a 1 ciclo/día, del orden de 7.300 ciclos acumulados al año 20. Los
# 7.200 son una aproximación conservadora y redondeada de ese orden de
# magnitud, no una cifra textual del documento. La fuente NO
# respalda el criterio de fin de vida, la capacidad retenida, la degradación
# anual, la política de reposición ni el valor residual: ver el docstring.
CICLOS_VIDA_BESS = 7_200  # ciclos equivalentes a DoD nominal
VIDA_CALENDARIO_BESS = 20  # años
# A ~326 ciclos/año el límite por ciclos daría 22,1 años, así que manda el
# calendario: el BESS dura el horizonte completo y no hay reposición. El
# límite por ciclos sigue mordiendo por encima de 360 ciclos/año.

# ============================================================
# PV  (solo aplica si el PV es propio; bajo PPA no se usa)
# ============================================================

CAPEX_PV_USD_KW = 850.0
OPEX_PV_USD_KW_ANIO = 14.0

# ============================================================
# Financieros
# ============================================================

HORIZONTE_ANIOS = 20
WACC = 0.10  # tasa de descuento real en COP
DEGRADACION_PV_ANUAL = 0.005  # 0,5 %/año
# Pérdida de capacidad útil del BESS. No confundir con `costo_degradacion`
# de `estrategias_avanzadas`, que es un precio sombra para el despacho
# DENTRO del año: esto es el desvanecimiento a lo largo del horizonte.
# PENDIENTE: falta la cita de respaldo, exigida por CLAUDE.md.
DEGRADACION_BESS_ANUAL = 0.02  # 2 %/año de capacidad útil


# ============================================================
# Referencias
# ============================================================

REFERENCIAS = {
    "NREL 2025": {
        "cita": (
            "Cole, W., Ramasamy, V. y Turan, M. (2025). Cost Projections for "
            "Utility-Scale Battery Storage: 2025 Update. NREL/TP-6A40-93281. "
            "Golden, CO: National Renewable Energy Laboratory."
        ),
        "url": "https://www.nrel.gov/docs/fy25osti/93281.pdf",
        "consultada": "2026-08-25",
        "aporta": (
            "Costo de capital de un BESS de 4 h: 334 USD/kWh en 2024 (USD de "
            "2024, costeo bottom-up del mercado de EE. UU.) y 295 / 321 / 350 "
            "USD/kWh en 2025 para las trayectorias low / mid / high. Convención "
            "de separar el costo en componente de energía y de potencia. FOM "
            "del 4 % del costo en USD/kW, con aumentación incluida. Vida "
            "calendario 15 años y RTE 85 %."
        ),
    },
    "Lazard 2025": {
        "cita": (
            "Lazard (2025). Lazard's Levelized Cost of Energy+ (LCOE+), "
            "versión 18.0, junio de 2025."
        ),
        "url": "https://www.lazard.com/media/5tlbhyla/lazards-lcoeplus-june-2025-_vf.pdf",
        "consultada": "2026-08-25",
        "aporta": (
            "Almacenamiento independiente de escala de servicio público de "
            "100 MW / 400 MWh (4 h): capital DC 107–232 USD/kWh, capital AC "
            "25–67 USD/kW, EPC 28–116 USD/kWh, O&M 3,0–8,0 USD/kWh-año, "
            "garantía extendida 0,66–1,85 % del capital desde el año 3, vida de "
            "proyecto 20 años, 350 ciclos/año a 90 % de DoD, RTE 86–92 %. "
            "Solar de escala de servicio público: capital 1.150–1.600 USD/kW y "
            "O&M fijo 11–14 USD/kW-año."
        ),
    },
    "IRENA 2024": {
        "cita": (
            "IRENA (2025). Renewable Power Generation Costs in 2024. "
            "Abu Dhabi: International Renewable Energy Agency."
        ),
        "url": (
            "https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/"
            "Jul/IRENA_TEC_RPGC_in_2024_2025.pdf"
        ),
        "consultada": "2026-08-25",
        "aporta": (
            "Promedio mundial ponderado de costo total instalado en 2024: "
            "691 USD/kW para solar fotovoltaica de escala de servicio público "
            "(caída del 87 % desde 2010) y 192 USD/kWh para almacenamiento en "
            "baterías (caída del 93 % desde 2 571 USD/kWh). LCOE mundial de la "
            "solar fotovoltaica: 0,043 USD/kWh."
        ),
    },
    "BNEF 2025": {
        "cita": (
            "BloombergNEF (2025). Lithium-Ion Battery Price Survey 2025, "
            "publicada el 9 de diciembre de 2025."
        ),
        "url": (
            "https://about.bnef.com/insights/clean-transport/"
            "lithium-ion-battery-pack-prices-fall-to-108-per-kilowatt-hour-"
            "despite-rising-metal-prices-bloombergnef/"
        ),
        "consultada": "2026-08-25",
        "aporta": (
            "Precio promedio mundial de paquetes de baterías de ion de litio: "
            "108 USD/kWh en 2025, 8 % menos que en 2024. El segmento de "
            "almacenamiento estacionario cayó a 70 USD/kWh, 45 % menos que en "
            "2024, y pasó a ser el más barato de todos. Es precio de PAQUETE: "
            "no incluye PCS, contenedor, EPC ni conexión."
        ),
    },
    "Ember 2025": {
        "cita": (
            "Ember (2025). How cheap is battery storage? Octubre de 2025."
        ),
        "url": "https://ember-energy.org/latest-insights/how-cheap-is-battery-storage/",
        "consultada": "2026-08-25",
        "aporta": (
            "CAPEX llave en mano de un proyecto BESS de larga duración (4 h o "
            "más) fuera de China y EE. UU.: ~125 USD/kWh, de los cuales "
            "~75 USD/kWh es equipo importado de China y ~50 USD/kWh es "
            "instalación y conexión. Corroborado con subastas recientes."
        ),
    },
    "Banco de la República": {
        "cita": (
            "Banco de la República de Colombia. Tasa Representativa del "
            "Mercado (TRM), serie diaria."
        ),
        "url": "https://www.banrep.gov.co/es/glosario/tasa-cambio-trm",
        "consultada": "2026-09-05",
        "aporta": (
            "TRM vigente al 5 de septiembre de 2026: 3.126,08 COP/USD, que es "
            "el supuesto de conversión del proyecto. Referencia histórica: la "
            "TRM promedio de 2025 fue 4.052,86 COP/USD (promedio aritmético "
            "simple de la serie diaria del año)."
        ),
    },
    "CREG 119 de 2007": {
        "cita": (
            "Comisión de Regulación de Energía y Gas (2007). Resolución CREG "
            "119 de 2007: fórmula tarifaria general del servicio de energía "
            "eléctrica para usuarios regulados."
        ),
        "url": "https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0119_2007.htm",
        "consultada": "2026-08-25",
        "aporta": (
            "Estructura del costo unitario de prestación del servicio (CU) que "
            "usa el modelo para valorar la energía comprada a la red. Las "
            "bandas horarias vienen de la Resolución CREG 079 de 1997. No es "
            "una fuente de CAPEX ni de OPEX: define el precio de la energía, "
            "que es la otra mitad del cálculo económico."
        ),
    },
    "Fuente comercial confidencial 2026": {
        "cita": "Fuente comercial confidencial de fabricante de sistemas BESS, 2026.",
        # Documento confidencial: no se publica URL ni ningun dato que permita
        # identificarlo. Es una excepcion deliberada a la convencion de
        # CLAUDE.md, que exige cita, URL y fecha de consulta.
        "url": "no divulgable — documento confidencial",
        "consultada": "2026-09-01",
        "aporta": (
            "Término contractual de 20 años y programa de garantía y O&M "
            "construido sobre 1 ciclo por día, del que resultan del orden de "
            "7.300 ciclos acumulados al año 20. Respalda la VIDA CALENDARIO de "
            "20 años y el ORDEN DE MAGNITUD de los ciclos de vida; los 7.200 "
            "que usa el modelo son una aproximación conservadora y redondeada, "
            "no una cifra textual del documento. NO respalda el criterio de fin "
            "de vida, el porcentaje de capacidad retenida, la degradación "
            "anual, la política de reposición ni el valor residual: esos cinco "
            "puntos siguen requiriendo evidencia adicional."
        ),
    },
    "UPME GeoLCOE": {
        "cita": (
            "UPME. GeoLCOE — Costos nivelados de generación de electricidad en "
            "Colombia (aplicativo web)."
        ),
        "url": "https://lcoev2.upme.gov.co/",
        "consultada": "2026-08-25",
        "aporta": (
            "Referencia local para contrastar el LCOE que produce este modelo "
            "con los costos nivelados que publica la UPME para Colombia. No se "
            "usó como fuente numérica directa de CAPEX en esta versión: queda "
            "como validación pendiente."
        ),
    },
}


# ============================================================
# CAPEX y OPEX
# ============================================================


def _curvas():
    """Importación diferida: `costos_curvas` importa este módulo para la TRM."""
    import costos_curvas

    return costos_curvas


def capex_bess(power_MW, capacity_MWh, trm=TRM_COP_USD):
    """Inversión inicial del BESS en COP, desde la curva del Excel.

    El costo unitario depende del tamaño y de la duración: a 4 h va de 168 a
    210 USD/kWh, con un cambio de segmento entre 100 y 110 MWh. Esa
    discontinuidad es lo que fija el dimensionamiento óptimo (INFORME §7.3), y
    se perdería con un costo plano.
    """
    if capacity_MWh <= 0:
        return 0.0
    return _curvas().capex_bess_COP(power_MW, capacity_MWh, trm=trm)


def opex_bess_anual(capacity_MWh, trm=TRM_COP_USD, power_MW=None):
    """O&M anual del BESS: 1,5 % del CAPEX. No incluye reposición.

    `power_MW` hace falta porque el CAPEX depende de la potencia además de la
    energía. Si no se pasa, se asume la duración de 4 h del caso base.
    """
    if capacity_MWh <= 0:
        return 0.0
    if power_MW is None:
        power_MW = capacity_MWh / 4.0
    return OPEX_BESS_FRACCION_CAPEX * capex_bess(power_MW, capacity_MWh, trm)


def capex_pv(pv_MW, trm=TRM_COP_USD):
    """Inversión inicial de la planta solar en COP, desde la curva del Excel."""
    if pv_MW <= 0:
        return 0.0
    return _curvas().capex_pv_COP(pv_MW, trm=trm)


def opex_pv_anual(pv_MW, trm=TRM_COP_USD):
    """O&M anual de la planta solar: 2 % del CAPEX."""
    if pv_MW <= 0:
        return 0.0
    return OPEX_PV_FRACCION_CAPEX * capex_pv(pv_MW, trm)


def opex_pv_anual_plano(pv_MW, trm=TRM_COP_USD):
    """O&M anual de la planta solar en COP. Solo si el PV es propio."""
    return OPEX_PV_USD_KW_ANIO * pv_MW * 1_000 * trm


def capex_bess_por_kwh(power_MW, capacity_MWh):
    """Costo unitario en USD/kWh de un BESS concreto, según la curva."""
    if capacity_MWh <= 0:
        return 0.0
    return capex_bess(power_MW, capacity_MWh) / (capacity_MWh * 1_000) / TRM_COP_USD


def capex_bess_por_kwh_4h():
    """Costo equivalente en USD/kWh para la duración del caso base (4 h)."""
    return CAPEX_BESS_ENERGIA_USD_KWH + CAPEX_BESS_POTENCIA_USD_KW / 4.0


def costo_degradacion_COP_MWh(capacity_MWh, DoD, eff_rt, trm=TRM_COP_USD):
    """Precio sombra del throughput: COP por MWh ciclado (carga + descarga).

    Reparte el CAPEX de energía entre todo el throughput de la vida útil. Guía el
    despacho pero NO entra al flujo de caja: la batería ya se paga en el CAPEX, y
    cobrarla otra vez por ciclo sería contarla dos veces. Misma convención que
    `estrategias_avanzadas.costo_degradacion`.
    """
    if capacity_MWh <= 0:
        return 0.0
    util = capacity_MWh * DoD
    capex_energia = CAPEX_BESS_ENERGIA_USD_KWH * capacity_MWh * 1_000 * trm
    throughput_vida = CICLOS_VIDA_BESS * (util + util / eff_rt)
    return capex_energia / throughput_vida


# ============================================================
# Vida útil
# ============================================================


def vida_bess_anios(descarga_anual_MWh, capacity_MWh, DoD):
    """Años de vida del BESS: el menor entre el límite por ciclos y el calendario.

    Devuelve también los ciclos equivalentes por año, porque es el número que
    explica cuál de los dos límites manda.
    """
    if capacity_MWh <= 0:
        return np.inf, 0.0
    util = capacity_MWh * DoD
    ciclos_anio = descarga_anual_MWh / util if util > 0 else 0.0
    if ciclos_anio <= 0:
        return float(VIDA_CALENDARIO_BESS), 0.0
    vida_por_ciclos = CICLOS_VIDA_BESS / ciclos_anio
    return float(min(vida_por_ciclos, VIDA_CALENDARIO_BESS)), float(ciclos_anio)


# ============================================================
# Indicadores financieros
# ============================================================


def factor_recuperacion_capital(r=WACC, n=None):
    """CRF: fracción del CAPEX que hay que recuperar cada año.

    Convierte una inversión de hoy en la anualidad equivalente durante n años, y
    es lo que permite sumar CAPEX y costo operativo en la misma unidad.
    """
    n = HORIZONTE_ANIOS if n is None else n
    if n <= 0:
        return 0.0
    if r == 0:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def tir(flujos, lo=-0.95, hi=3.0, tol=1e-9, max_iter=400):
    """Tasa interna de retorno por bisección. Devuelve NaN si no hay cambio de signo."""
    def vpn_a(r):
        return sum(f / (1 + r) ** t for t, f in enumerate(flujos))

    f_lo, f_hi = vpn_a(lo), vpn_a(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return np.nan

    for _ in range(max_iter):
        medio = (lo + hi) / 2
        f_medio = vpn_a(medio)
        if abs(f_medio) < tol:
            return medio
        if f_lo * f_medio < 0:
            hi = medio
        else:
            lo, f_lo = medio, f_medio
    return (lo + hi) / 2


def consola_utf8():
    """Fuerza UTF-8 en la salida estándar y de error.

    En Windows la consola usa cp1252 y cualquier símbolo fuera de esa página
    aborta el script con `UnicodeEncodeError`, aunque el trabajo ya esté hecho
    y escrito en disco. Llamarla al entrar en `__main__` evita perder una
    corrida de media hora por un carácter de una tabla.
    """
    import sys

    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def anio_de_reposicion(vida_activo_anios, horizonte=HORIZONTE_ANIOS):
    """Año en que hay que reponer el activo, o None si aguanta el horizonte."""
    if (
        vida_activo_anios is None
        or not np.isfinite(vida_activo_anios)
        or vida_activo_anios >= horizonte
    ):
        return None
    return int(np.ceil(vida_activo_anios))


def valor_residual(capex_COP, vida_activo_anios, horizonte=HORIZONTE_ANIOS):
    """Valor del activo que sobra al terminar el horizonte, sin descontar.

    Convención financiera, no un hecho físico. El baseline del proyecto es
    residual = 0, que es la opción conservadora; pero conservador no quiere decir
    neutro: **residual cero puede penalizar a las configuraciones que renuevan
    cerca del final del horizonte**, porque se les cobra un activo completo del
    que solo se aprovecha una fracción. La variante lineal de abajo existe para
    medir ese efecto como sensibilidad.

    Si el BESS dura 15 años y el horizonte es de 20, al año 15 se compra una
    batería nueva que al año 20 todavía tiene 10 de sus 15 años por delante.
    Cargarle al proyecto la batería completa sin acreditarle esa vida sobrante
    exagera el costo, así que se reconoce el remanente de forma lineal. Es la
    convención habitual de valor residual contable.
    """
    if capex_COP <= 0 or vida_activo_anios is None or not np.isfinite(vida_activo_anios):
        return 0.0
    if vida_activo_anios <= 0:
        return 0.0
    anio = anio_de_reposicion(vida_activo_anios, horizonte)
    inicio_ultimo = anio if anio is not None else 0
    usados = horizonte - inicio_ultimo
    restante = vida_activo_anios - usados
    if restante <= 0:
        return 0.0
    return capex_COP * restante / vida_activo_anios


def factor_capacidad_bess(t, anio_reposicion, degradacion=DEGRADACION_BESS_ANUAL):
    """Fracción de capacidad útil que conserva el BESS en el año `t`.

    La reposición reinicia el contador: la batería nueva arranca al 100 %.
    """
    if anio_reposicion is None or t < anio_reposicion:
        antiguedad = t - 1
    else:
        antiguedad = t - anio_reposicion
    return (1.0 - degradacion) ** antiguedad


def flujo_de_caja(
    beneficio_anual_COP,
    capex_COP,
    opex_anual_COP,
    vida_activo_anios=None,
    costo_reposicion_COP=0.0,
    horizonte=HORIZONTE_ANIOS,
    degradacion=DEGRADACION_PV_ANUAL,
    incluir_residual=True,
    beneficio_bess_COP=0.0,
    degradacion_bess=DEGRADACION_BESS_ANUAL,
):
    """Flujos anuales del proyecto, con el año 0 en la posición 0.

    El beneficio del año 1 decae `degradacion` cada año por la pérdida de recurso
    solar. Si el activo se agota antes del horizonte se resta la reposición en el
    año en que ocurre, y al final del horizonte se acredita el valor residual de
    la última unidad instalada.
    """
    flujos = [-capex_COP]
    anio_rep = anio_de_reposicion(vida_activo_anios, horizonte)

    # El beneficio se separa en dos partes con desvanecimientos distintos: la
    # solar decae con el recurso y la del BESS con su capacidad útil.
    beneficio_pv = beneficio_anual_COP - beneficio_bess_COP

    for t in range(1, horizonte + 1):
        flujo = beneficio_pv * (1 - degradacion) ** (t - 1)
        flujo += (
            beneficio_bess_COP
            * (1 - degradacion) ** (t - 1)
            * factor_capacidad_bess(t, anio_rep, degradacion_bess)
        )
        flujo -= opex_anual_COP
        if costo_reposicion_COP > 0 and anio_rep is not None and t == anio_rep:
            flujo -= costo_reposicion_COP
        if incluir_residual and t == horizonte:
            base = costo_reposicion_COP if anio_rep is not None else capex_COP
            flujo += valor_residual(base, vida_activo_anios, horizonte)
        flujos.append(flujo)
    return flujos


def vpn(flujos, r=WACC):
    return float(sum(f / (1 + r) ** t for t, f in enumerate(flujos)))


def payback_descontado(flujos, r=WACC):
    """Primer año en que el acumulado descontado se vuelve no negativo."""
    acumulado = np.cumsum([f / (1 + r) ** t for t, f in enumerate(flujos)])
    cruces = np.where(acumulado >= 0)[0]
    return int(cruces[0]) if len(cruces) else np.nan


# ============================================================
# Evaluación de un conjunto de estrategias
# ============================================================


def evaluar(
    resumen,
    power_MW,
    capacity_MWh,
    pv_MW,
    DoD,
    demanda_MWh=None,
    columna_costo="Costo_total_suministro_COP",
    columna_descarga="Descarga_BESS_MWh",
    referencia="Sin BESS",
    pv_propio=False,
    costo_referencia_COP=None,
    r=WACC,
    horizonte=HORIZONTE_ANIOS,
    trm=TRM_COP_USD,
):
    """Agrega CAPEX, OPEX e indicadores financieros a una tabla de estrategias.

    `resumen` es la tabla que produce `resumir_resultado` del notebook 01: una
    fila por estrategia, con el costo operativo anual y la energía descargada.

    Convenciones:

      - El contrafactual lo define `costo_referencia_COP`. Si no se pasa, es el
        costo operativo de la estrategia `referencia` ("Sin BESS"), y entonces la
        pregunta que responde el VPN es "¿se paga la batería, dado que el PV ya
        está?". Con `pv_propio=True` hay que pasar el costo de NO hacer nada
        —comprarle toda la demanda a la red, sin PV y sin BESS—, porque de lo
        contrario se le carga al proyecto el CAPEX solar sin acreditarle la
        energía que la planta sustituye.
      - El beneficio operativo de cada estrategia es el ahorro anual frente a esa
        referencia, ANTES de pagar la batería.
      - El costo anual equivalente suma el costo operativo, el OPEX y el CAPEX
        anualizado con el factor de recuperación de capital, calculado sobre la
        vida del propio activo y no sobre el horizonte. Es la cifra que hay que
        mirar para decidir si la batería se paga sola, y su signo coincide con
        el del VPN por construcción.
      - El costo nivelado de suministro reparte todos los costos descontados
        entre toda la demanda descontada, y hace comparable a `Sin BESS` con las
        demás.
    """
    df = resumen.copy()

    if referencia not in set(df["Estrategia"]):
        raise ValueError(
            f"No está la estrategia de referencia '{referencia}' en el resumen."
        )

    # Costo operativo de la MISMA planta sin batería. Es la base contra la que
    # se mide lo que aporta el BESS, y por tanto lo que se desvanece.
    costo_sin_bess = float(
        df.loc[df["Estrategia"] == referencia, columna_costo].iloc[0]
    )

    if costo_referencia_COP is None:
        costo_ref = costo_sin_bess
    else:
        costo_ref = float(costo_referencia_COP)

    if demanda_MWh is None:
        if "Demanda_MWh" not in df.columns:
            raise ValueError(
                "Hace falta la demanda anual: pásala en `demanda_MWh` o incluye "
                "la columna 'Demanda_MWh' en el resumen."
            )
        demanda_MWh = float(df["Demanda_MWh"].iloc[0])

    capex_pv_COP = capex_pv(pv_MW, trm) if pv_propio else 0.0
    opex_pv_COP = opex_pv_anual(pv_MW, trm) if pv_propio else 0.0

    filas = []
    for _, fila in df.iterrows():
        tiene_bess = fila["Estrategia"] != referencia

        capex_b = capex_bess(power_MW, capacity_MWh, trm) if tiene_bess else 0.0
        opex_b = opex_bess_anual(capacity_MWh, trm, power_MW) if tiene_bess else 0.0

        descarga = float(fila.get(columna_descarga, 0.0) or 0.0)
        vida, ciclos = vida_bess_anios(descarga, capacity_MWh if tiene_bess else 0.0, DoD)

        capex_total = capex_b + capex_pv_COP
        opex_total = opex_b + opex_pv_COP

        costo_op = float(fila[columna_costo])
        beneficio = costo_ref - costo_op
        # Parte del beneficio atribuible a la batería: lo que ahorra frente a
        # la MISMA planta sin ella. Es la única que pierde capacidad con los
        # años; el resto lo aporta el PV.
        beneficio_bess = (costo_sin_bess - costo_op) if tiene_bess else 0.0

        # La reposición aplica al BESS. El PV dura el horizonte completo.
        reposicion = capex_b if tiene_bess else 0.0

        flujos = flujo_de_caja(
            beneficio_anual_COP=beneficio,
            capex_COP=capex_total,
            opex_anual_COP=opex_total,
            vida_activo_anios=vida if tiene_bess else None,
            costo_reposicion_COP=reposicion,
            horizonte=horizonte,
            beneficio_bess_COP=beneficio_bess,
        )

        # Cada activo se anualiza sobre SU PROPIA vida, no sobre el horizonte.
        # Anualizar el BESS a 20 años cuando dura 15 lo abarataría, y el costo
        # anual equivalente dejaría de coincidir en signo con el VPN, que sí
        # paga la reposición.
        vida_bess = min(vida, horizonte) if tiene_bess else horizonte
        capex_anualizado = capex_b * factor_recuperacion_capital(r, vida_bess)
        capex_anualizado += capex_pv_COP * factor_recuperacion_capital(r, horizonte)

        # Costo nivelado: todo lo que cuesta suministrar la demanda, descontado,
        # sobre toda la demanda, descontada. En COP/kWh.
        fd = np.array([(1 + r) ** -t for t in range(1, horizonte + 1)])
        decaimiento = (1 - DEGRADACION_PV_ANUAL) ** np.arange(horizonte)
        costos_desc = capex_total + ((opex_total + costo_op) * decaimiento * fd).sum()
        anio_rep = anio_de_reposicion(vida if tiene_bess else None, horizonte)
        if reposicion and anio_rep is not None:
            costos_desc += reposicion * (1 + r) ** -anio_rep
        residual = valor_residual(
            reposicion if anio_rep is not None else capex_b,
            vida if tiene_bess else None,
            horizonte,
        )
        costos_desc -= residual * (1 + r) ** -horizonte
        demanda_desc = (demanda_MWh * fd).sum()
        costo_nivelado = costos_desc / demanda_desc / 1_000

        filas.append(
            {
                "CAPEX_BESS_COP": capex_b,
                "CAPEX_PV_COP": capex_pv_COP,
                "CAPEX_total_COP": capex_total,
                "OPEX_BESS_anual_COP": opex_b,
                "OPEX_PV_anual_COP": opex_pv_COP,
                "OPEX_total_anual_COP": opex_total,
                "CAPEX_anualizado_COP": capex_anualizado,
                "Costo_referencia_COP": costo_ref,
                "Costo_operativo_anual_COP": costo_op,
                "Costo_anual_equivalente_COP": costo_op + opex_total + capex_anualizado,
                "Beneficio_operativo_anual_COP": beneficio,
                "Beneficio_neto_anual_COP": beneficio - opex_total,
                "Ciclos_equivalentes_anio": ciclos,
                "Vida_BESS_anios": (
                    min(vida, horizonte) if tiene_bess else np.nan
                ),
                "Anio_reposicion_BESS": (anio_rep if tiene_bess else np.nan),
                "Valor_residual_COP": residual,
                "VPN_COP": vpn(flujos, r),
                "TIR": tir(flujos) if tiene_bess or pv_propio else np.nan,
                "Payback_descontado_anios": (
                    payback_descontado(flujos, r) if tiene_bess or pv_propio else np.nan
                ),
                "Costo_nivelado_COP_kWh": costo_nivelado,
            }
        )

    economia = pd.DataFrame(filas, index=df.index)
    salida = pd.concat([df, economia], axis=1)

    # El costo anual equivalente de la referencia es el punto de comparación de
    # todo lo demás: si una estrategia queda por encima, la batería no se paga.
    cae_ref = float(
        salida.loc[salida["Estrategia"] == referencia, "Costo_anual_equivalente_COP"].iloc[0]
    )
    salida["Ahorro_anual_equivalente_COP"] = cae_ref - salida["Costo_anual_equivalente_COP"]
    salida["Ahorro_anual_equivalente_%"] = (
        salida["Ahorro_anual_equivalente_COP"] / cae_ref * 100
    )

    return salida


# El CAPEX máximo con VPN ≥ 0 —el "precio de entrada" del almacenamiento— NO se
# calcula aquí en forma cerrada. Una fórmula cerrada tendría que ignorar el
# decaimiento del recurso solar, la reposición y el valor residual, y daría un
# número distinto del que produce el flujo de caja completo. En vez de eso,
# `comparacion_economica.barrido_capex` recorre el rango de CAPEX evaluando el
# VPN real, y `precio_de_entrada` reporta dónde cruza el cero.


# ============================================================
# Ficha de supuestos
# ============================================================


def ficha(power_MW=30.0, capacity_MWh=120.0, pv_MW=100.0):
    """Imprime los supuestos, sus fuentes y el CAPEX del caso que se le pase."""
    linea = "=" * 78
    print(linea)
    print("SUPUESTOS DE CAPEX Y OPEX")
    print(linea)
    print(f"TRM                              : {TRM_COP_USD:,.2f} COP/USD  (vigente al {TRM_FECHA_REFERENCIA})")
    print()
    print("BESS")
    print(f"  CAPEX componente de energía    : {CAPEX_BESS_ENERGIA_USD_KWH:,.0f} USD/kWh")
    print(f"  CAPEX componente de potencia   : {CAPEX_BESS_POTENCIA_USD_KW:,.0f} USD/kW")
    print(f"  CAPEX equivalente a 4 h        : {capex_bess_por_kwh_4h():,.0f} USD/kWh")
    print(f"  OPEX                           : {OPEX_BESS_USD_KWH_ANIO:,.1f} USD/kWh-año "
          f"({OPEX_BESS_USD_KWH_ANIO * 4:,.0f} USD/kW-año a 4 h)")
    print(f"  Ciclos de vida                 : {CICLOS_VIDA_BESS:,.0f}")
    print(f"  Vida calendario                : {VIDA_CALENDARIO_BESS} años")
    print()
    print("PV  (solo si el PV es propio; bajo PPA no aplica)")
    print(f"  CAPEX                          : {CAPEX_PV_USD_KW:,.0f} USD/kW")
    print(f"  OPEX                           : {OPEX_PV_USD_KW_ANIO:,.0f} USD/kW-año")
    print()
    print("Financieros")
    print(f"  Horizonte                      : {HORIZONTE_ANIOS} años")
    print(f"  WACC real                      : {WACC:.0%}")
    print(f"  Degradación del recurso solar  : {DEGRADACION_PV_ANUAL:.1%}/año")
    print(f"  Factor de recuperación de capital: {factor_recuperacion_capital():.4f}")
    print()
    print(linea)
    print(f"CASO: PV {pv_MW:,.0f} MW | BESS {power_MW:,.0f} MW / {capacity_MWh:,.0f} MWh")
    print(linea)
    print(f"  CAPEX BESS                     : {capex_bess(power_MW, capacity_MWh)/1e9:,.2f} mil MM COP")
    print(f"  OPEX BESS anual                : {opex_bess_anual(capacity_MWh)/1e9:,.3f} mil MM COP")
    print(f"  CAPEX BESS anualizado          : "
          f"{capex_bess(power_MW, capacity_MWh) * factor_recuperacion_capital()/1e9:,.2f} mil MM COP/año")
    print(f"  CAPEX PV si fuera propio       : {capex_pv(pv_MW)/1e9:,.2f} mil MM COP")
    print()
    print(linea)
    print("FUENTES")
    print(linea)
    for nombre, ref in REFERENCIAS.items():
        print(f"\n[{nombre}]")
        print(f"  {ref['cita']}")
        print(f"  {ref['url']}")
        print(f"  Consultada: {ref['consultada']}")
        print(f"  Aporta: {ref['aporta']}")


# ============================================================
# Renovación del bloque de almacenamiento (C3A)
# ============================================================
#
# El BESS no se vuelve a comprar entero cuando se agota su envolvente de
# utilización: se renueva el bloque que escala con energía. Ver
# `costos_curvas.capex_bess_bloque_almacenamiento`.


def costo_renovacion_COP(power_MW, capacity_MWh, anio_proyecto,
                         escenario="medio", trm=TRM_COP_USD):
    """Costo de renovar el bloque de almacenamiento en el año `anio_proyecto`.

        R1_t = R1_suministro,0 · I_batería,t + R1_local,0 · I_local,t

    Correspondencia temporal, explícita: el año 0 del proyecto es el año base de
    los costos, así que

        anio_indice = costos_curvas.ANIO_COSTOS + anio_proyecto

    Con `anio_proyecto = 0` los dos índices valen 1,00 y el resultado es
    exactamente R1_0: la función no aplica ninguna reducción en el año base.

    Los índices son reales, en USD constantes del año base, coherentes con un
    WACC real y una TRM fija.
    """
    if capacity_MWh <= 0 or power_MW <= 0:
        return 0.0
    cc = _curvas()
    r1 = cc.capex_bess_bloque_almacenamiento(power_MW, capacity_MWh)
    anio_indice = cc.ANIO_COSTOS + int(anio_proyecto)
    i_bat = cc.indice_trayectoria(anio_indice, "bateria", escenario)
    i_loc = cc.indice_trayectoria(anio_indice, "local", escenario)
    usd = r1["suministro_USD"] * i_bat + r1["local_energia_USD"] * i_loc
    return usd * trm


def costo_degradacion_R1_COP_MWh(power_MW, capacity_MWh, DoD, eff_rt,
                                 trm=TRM_COP_USD):
    """Precio sombra del throughput, amortizando R1 y no el CAPEX completo.

        c_deg = R1_0 / (CICLOS_VIDA_BESS · (E_util + E_util / RTE))

    Es el costo marginal de consumir la envolvente inicial de utilización del
    activo. Si al agotarla solo se renueva el bloque de almacenamiento, cobrar
    el CAPEX completo por ciclar sobreestima el desgaste.

    Deliberadamente NO aplica la trayectoria tecnológica: el precio sombra mira
    el activo ya instalado, no el que se comprará en el futuro.

    Mantiene la convención de throughput del resto del proyecto: se penaliza
    carga más descarga, con `E_util = capacidad · DoD`.
    """
    if capacity_MWh <= 0 or power_MW <= 0:
        return 0.0
    r1_cop = _curvas().capex_bess_bloque_almacenamiento(
        power_MW, capacity_MWh)["total_R1_USD"] * trm
    util = capacity_MWh * DoD
    return r1_cop / (CICLOS_VIDA_BESS * (util + util / eff_rt))


def tabla_referencias():
    """Las referencias como DataFrame, para mostrarlas en un notebook."""
    return pd.DataFrame(
        [
            {
                "Fuente": nombre,
                "Referencia": ref["cita"],
                "URL": ref["url"],
                "Consultada": ref["consultada"],
                "Qué aporta": ref["aporta"],
            }
            for nombre, ref in REFERENCIAS.items()
        ]
    )


if __name__ == "__main__":
    ficha()


# Cargo térmico: exploración económica, no calibración del mantenimiento de planta.
# El coeficiente se aplica por arranque a cada unidad. No impone un número anual.
CARGOS_ARRANQUE_EXPLORATORIOS_USD = (0., 100., 500., 2000.)
REFERENCIAS['cargo_arranque_exploratorio'] = {
    'naturaleza': 'Malla de sensibilidad elegida para explorar respuesta; no rango empírico ni probabilístico.',
    'valores_USD_arranque': CARGOS_ARRANQUE_EXPLORATORIOS_USD,
    'cita': 'NREL/Intertek, Power Plant Cycling Costs, 2012, tablas 1-1 y 1-3; no fuente de esta malla.',
    'url': 'https://docs.nlr.gov/docs/fy12osti/55433.pdf',
    'consultada': '2026-09-14',
    'aporta': 'Metodología para separar componentes; no calibra la malla. Valores publicados en USD 2011 y para otras clases de turbinas; no trasladados directamente. CVOM sin desagregación.',
    'detalle': 'COSTO_ARRANQUE.md',
}


REFERENCIAS['cvom_termico_usuario_20260917'] = {
    'cita': 'Instrucción explícita del usuario del 17/09/2026: CVOM de 11 USD/MWh y TRM del estudio.',
    'url': '',
    'consultada': '2026-09-17',
    'aporta': 'Antecedente: CVOM uniforme usado el 17/09/2026; sustituido el 18/09/2026 por valores por unidad.',
    'valor_USD_MWh': 11.0,
    'estado': 'sustituido',
    'detalle': 'CAMBIO_CVOM.md',
}


REFERENCIAS['gas_termico_usuario_20260917'] = {
    'cita': 'Instrucción explícita del usuario del 17/09/2026: adoptar gas a 20 USD/MMBtu como escenario del estudio.',
    'url': '',
    'consultada': '2026-09-17',
    'aporta': 'Precio constante para nuevas corridas integradas; escenario de gas elevado, no precio contratado ni pronóstico.',
    'valor_USD_MMBtu': GAS_TERMICO_USD_MMBTU,
    'detalle': 'CAMBIO_CVOM.md',
}
REFERENCIAS['contexto_gas_ideam_20260917'] = {
    'cita': 'IDEAM. El Niño en Colombia. Estado actualizado el 10 de septiembre de 2026.',
    'url': 'https://www.ideam.gov.co/fenomeno-del-nino',
    'consultada': '2026-09-17',
    'aporta': 'Contexto climático y presiones posibles sobre el recurso hídrico. No establece un precio del gas de 20 USD/MMBtu.',
}
REFERENCIAS['contexto_gas_upme_20260917'] = {
    'cita': 'UPME. Estudio Técnico para el Plan de Abastecimiento de Gas Natural.',
    'url': 'https://www.upme.gov.co/simec/planeacion-energetica/pang/',
    'consultada': '2026-09-17',
    'aporta': 'Escenarios de abastecimiento e importación para complementar la producción nacional. No cuantifica el supuesto de precio adoptado.',
}


REFERENCIAS['cvom_termico_usuario_20260918'] = {
    'cita': 'Tabla e instrucción explícita del usuario del 18/09/2026: G1/G2 = 9,28 y G3/G4 = 15,32 USD/MWh.',
    'url': '',
    'consultada': '2026-09-18',
    'aporta': 'CVOM por unidad para nuevas corridas integradas. Mantiene gas 20 USD/MMBtu y TRM del estudio.',
    'unidades': ('G1', 'G2', 'G3', 'G4'),
    'valores_USD_MWh': CVOM_TERMICO_USD_MWH,
    'detalle': 'CAMBIO_CVOM.md',
}


REFERENCIAS['factor_emision_gas_fecoc_2016'] = {
    'cita': ('UPME, Fundación Natura y Red de Investigación en Combustión Avanzada (2016). Factores de '
             'Emisión de los Combustibles Colombianos (FECOC). Informe final, tabla de gas natural genérico.'),
    'url': 'https://app.upme.gov.co/Calculadora_Emisiones1/new/Informe_Final_FECOC_Correcciones_UPME_FunNatura.pdf',
    'consultada': '2026-09-20',
    'valor_kg_CO2_TJ': FACTOR_EMISION_GAS_KG_TJ,
    'valor_t_CO2_MMBtu': FACTOR_EMISION_GAS_T_MMBTU,
    'base': 'PCI (poder calorífico inferior), convención IPCC; declarada en el informe y confirmada con la '
            'razón factor por m³ / factor por TJ de la base de datos de la calculadora.',
    'aporta': 'Emisiones directas de combustión en sitio; no incluye red, ciclo de vida ni CH4/N2O.',
    'detalle': 'Adoptado el 24/09/2026 para la versión 2 de parámetros térmicos, en lugar del factor EPA '
               'sobre PCS usado en el artículo. Los consumos específicos se interpretan sobre PCI como supuesto de '
               'trabajo (explícito solo para G4); la base del gas facturado es desconocida y no se convierte. '
               'Referencia secundaria propuesta (anexo del inventario nacional en unfccc.int) no verificable: HTTP 404 '
               'el 26/09/2026.',
}
REFERENCIAS['precio_co2_escenario'] = {
    'cita': 'Supuesto de escenario de la versión 2 de parámetros; propuesta del director de tesis del 24/09/2026.',
    'naturaleza': 'Parámetro de escenario de la versión 2; no es una tarifa vigente ni un precio observado.',
    'valor_USD_t': PRECIO_CO2_USD_T,
    'url': '',
    'consultada': '2026-09-24',
    'aporta': 'Precio a las emisiones directas; los valores del barrido y su anclaje regulatorio quedan pendientes.',
}
REFERENCIAS['factor_emision_red_xm_2025'] = {
    'cita': 'XM. Informe de operación y mercado 2025, sección 2.5, tabla 15 "Factor de emisión del SIN 2025" (p. 50 del PDF).',
    'url': 'https://www.xm.com.co/sites/default/files/documents/Informe%20operaci%C3%B3n%20y%20mercado%20XM%202025.pdf#page=50',
    'consultada': '2026-09-26',
    'anio_representado': 2025,
    'valor_t_CO2e_MWh': FACTOR_EMISION_RED_T_MWH,
    'emisiones_SIN_2025_t_CO2e': 8_208_280.53,
    'aporta': ('Factor principal de la electricidad comprada: mismo año de la serie del modelo. El informe atribuye su '
               'nivel bajo al fin de El Niño en mayo de 2024. Publicación de XM, no resolución de UPME.'),
}
REFERENCIAS['factor_emision_red_upme_2024'] = {
    'cita': ('UPME. Resolución 000085 de 23-02-2026, "Por la cual se actualiza el factor de emisión del Sistema '
             'Interconectado Nacional del año 2024 para inventarios de emisiones de GEI y proyectos de mitigación de '
             'GEI", art. 1, literal B (p. 3).'),
    'url': 'https://docs.upme.gov.co/Normatividad/085_2026.pdf',
    'consultada': '2026-09-26',
    'anio_representado': 2024,
    'valor_inventarios_t_CO2eq_MWh': FACTOR_EMISION_RED_UPME_2024_T_MWH,
    'aporta': ('Sensibilidad oficial para inventarios de GEI (año 2024). Los factores MDL de la misma resolución (literal A) '
               'estiman reducciones de proyectos y no se usan para contabilizar compras.'),
}
