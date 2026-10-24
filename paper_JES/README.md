# Artículo para el Journal of Energy Storage: experimentos adicionales y figuras

Material del artículo *Economic Comparison of Heuristic Dispatch Rules and
Optimization Benchmarks for an Industrial Behind-the-Meter Energy System*
(N. Barros, E. Ángel, S. Zapata; Universidad EIA). El estudio base (reglas,
benchmarks MILP anuales, factorial integrado, VPN a 20 años y extensión de
carbono) es el resto de este repositorio; esta carpeta agrega los experimentos
corridos para el artículo y los scripts que dibujan sus figuras y las tablas del
material suplementario.

*English: R1–R3 experiments (rolling horizon, gas price, start-up charge), figure
and supplementary-table scripts of the article. Run from the repository root.*

## Contenido

| Ruta | Contenido |
|---|---|
| `scripts/comun.py` | Utilidades comunes; reutiliza los módulos del repositorio sin modificarlos |
| `scripts/r1_horizonte_rodante.py` | R1: MILP de horizonte rodante (RH-24, RH-24S, RH-48) sobre C280 |
| `scripts/r23_sensibilidades.py` | R2: precio del gas (12, 16, 24 USD/MMBtu); R3: cargo por arranque (0, 500, 1.000 USD) |
| `scripts/correr_refuerzos.py` | Lanza cada caso de R1–R3 en un proceso aparte; no repite los ya guardados |
| `scripts/consolidar_refuerzos.py` | Consolida R1–R3 en CSV y en `RESUMEN.md` |
| `scripts/figuras_jes.py` | Figuras y *graphical abstract* del artículo (PDF y PNG) |
| `scripts/suplementario_jes.py` | Tablas LaTeX y figura S1 del material suplementario |
| `fuentes/` | Tablas agregadas del estudio base que usan las figuras (distancias del factorial, resumen del VPN) |
| `resultados_refuerzos/` | Resultados de R1–R3: trayectorias horarias (`.npz`), metadatos de cada caso (`.json`) y `consolidado/*.csv` |

Los niveles base (gas a 20 USD/MMBtu, cargo por arranque de 100 USD) se leen de
la corrida cerrada `resultados_correccion_numerica/20260920/` y no se recalculan.

## Nombres en el código

| Artículo | Código |
|---|---|
| C280 | `K3` |
| FLAT / WMI | `T0` / `T1` |
| SC / DM / DMX | `A_INT` / `B_INT` / `D_INT` |
| OPT-B / OPT-X | `OPT_AB` / `OPT_D` |
| Grid+Gas / PV-only | `RED_TERMICA` / `PV_SIN_BESS` |
| RH-24 / RH-24S / RH-48 | `RH24` / `RH24A6` / `RH48` |
| Conjuntos de acciones de OPT-B / OPT-X | `B` / `X` |

## Reproducción

Crear el entorno del repositorio (`requirements.txt`) y agregar `psutil`, que el
coordinador usa para limitar la memoria. Desde la raíz del repositorio:

```sh
# Figuras y tablas del suplemento a partir de los resultados guardados
.venv/bin/python paper_JES/scripts/figuras_jes.py
.venv/bin/python paper_JES/scripts/suplementario_jes.py

# Tablas consolidadas de R1–R3
.venv/bin/python paper_JES/scripts/consolidar_refuerzos.py

# Un caso: RH-48 con el conjunto B bajo FLAT, u OPT-B a 12 USD/MMBtu bajo FLAT
.venv/bin/python paper_JES/scripts/r1_horizonte_rodante.py --caso 48 B T0
.venv/bin/python paper_JES/scripts/r23_sensibilidades.py --caso R2 g12 T0 OPT_AB
```

`correr_refuerzos.py` repite el conjunto completo (90 trayectorias anuales; los
benchmarks anuales usan HiGHS con un límite de 600 s, así que toma varias horas).
Como no repite los casos presentes en `resultados_refuerzos/`, para recalcular
todo hay que mover antes esa carpeta. Las salidas van a `paper_JES/manuscript/`
y `paper_JES/resultados_refuerzos/`.

No se publican los registros de ejecución ni una primera versión descartada de
RH-24S (sin el cierre seguro al final de la ventana; ver la sección S3 del
material suplementario). No se divulgan la identidad ni la ubicación de la
instalación industrial.
