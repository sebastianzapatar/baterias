# Journal of Energy Storage article: additional experiments and figures

Material for a journal article on the dispatch of an industrial behind-the-meter
PV–battery system with on-site gas units. The base study (rules, annual MILP
benchmarks, integrated factorial, 20-year NPV and carbon extension) is the rest
of this repository; this folder adds the
experiments run for the article and the scripts that draw its figures and the
tables of its supplementary material. The manuscript itself is not included.

## Contents

| Path | Contents |
|---|---|
| `scripts/comun.py` | Shared utilities; reuses the repository modules without modifying them |
| `scripts/r1_horizonte_rodante.py` | R1: rolling-horizon MILP (RH-24, RH-24S, RH-48) on C280 |
| `scripts/r23_sensibilidades.py` | R2: gas price (12, 16, 24 USD/MMBtu); R3: start-up charge (0, 500, 1,000 USD) |
| `scripts/r4_mpc_pronosticos.py` | R4: 48-h model predictive control (MPC-48) with forecasts ranging from perfect foresight to persistence (α = 0–1); hourly or daily re-optimization |
| `scripts/correr_refuerzos.py` | Runs each R1–R3 case in a separate process; skips cases already saved |
| `scripts/consolidar_refuerzos.py` | Consolidates R1–R4 into CSV files and `RESUMEN.md` |
| `scripts/figuras_jes.py` | Article figures and graphical abstract (PDF and PNG) |
| `scripts/suplementario_jes.py` | LaTeX tables and Fig. S1 of the supplementary material |
| `fuentes/` | Aggregated base-study tables used by the figures (factorial distances, NPV summary) |
| `resultados_refuerzos/` | R1–R4 results: hourly trajectories (`.npz`), per-case metadata (`.json`) and `consolidado/*.csv` |

Base levels (gas at 20 USD/MMBtu, start-up charge of 100 USD) are read from the
closed run `resultados_correccion_numerica/20260920/` and are not recomputed.

## Names in the code

The code and its comments are in Spanish. Article names map to code names as follows:

| Article | Code |
|---|---|
| C280 | `K3` |
| FLAT / WMI | `T0` / `T1` |
| SC / DM / DMX | `A_INT` / `B_INT` / `D_INT` |
| OPT-B / OPT-X | `OPT_AB` / `OPT_D` |
| Grid+Gas / PV-only | `RED_TERMICA` / `PV_SIN_BESS` |
| RH-24 / RH-24S / RH-48 | `RH24` / `RH24A6` / `RH48` |
| MPC-48 hourly / daily with slack / daily with plan | `MPC48H` / `MPC48S` / `MPC48` |
| Action sets of OPT-B / OPT-X | `B` / `X` |

## Reproduction

Create the repository environment (`requirements.txt`) and add `psutil`, which
the runner uses to limit memory. From the repository root:

```sh
# Figures and supplementary tables from the saved results
.venv/bin/python paper_JES/scripts/figuras_jes.py
.venv/bin/python paper_JES/scripts/suplementario_jes.py

# Consolidated R1–R4 tables
.venv/bin/python paper_JES/scripts/consolidar_refuerzos.py

# One case: RH-48 with action set B under FLAT, or OPT-B at 12 USD/MMBtu under FLAT
.venv/bin/python paper_JES/scripts/r1_horizonte_rodante.py --caso 48 B T0
.venv/bin/python paper_JES/scripts/r23_sensibilidades.py --caso R2 g12 T0 OPT_AB

# R4: MPC-48 re-optimized every hour with pure persistence (α = 1) under FLAT (about 2 h)
.venv/bin/python paper_JES/scripts/r4_mpc_pronosticos.py --caso 1 T0 --ejecucion horario
```

`correr_refuerzos.py` reruns the full R1–R3 set (90 annual trajectories; the
annual benchmarks use HiGHS with a 600 s limit, so it takes several hours).
Because it skips cases already present in `resultados_refuerzos/`, move that
folder first to recompute everything. Outputs go to `paper_JES/manuscript/` and
`paper_JES/resultados_refuerzos/`.

Run logs and a discarded first version of RH-24S (without the conservative
terminal condition; see Section S3 of the supplementary material) are not
published. The identity and location of the industrial facility are not
disclosed.
