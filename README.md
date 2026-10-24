# Cuadernos PV + BESS + térmica · Universidad EIA

Estudio de *operación horaria y dimensionamiento de portafolios fotovoltaicos con
almacenamiento en baterías (BESS) e integración térmica a gas*, sobre series
horarias reales del año 2025 (8.760 horas) de un usuario industrial conectado a la
red colombiana.

| | |
|---|---|
| **Plataforma de consulta** | <https://sebastianzapatar.github.io/baterias/>; también se abre `index.html` con doble clic, sin instalar nada |
| **Manual completo** | [`Manual_Ejecucion_Cuadernos_EIA.pdf`](Manual_Ejecucion_Cuadernos_EIA.pdf) |
| **Cuadernos** | `01_…ipynb` a `15_…ipynb`, ya ejecutados: se pueden leer en GitHub sin correr nada |

---

## 1. Descargar

Con Git:

```bash
git clone https://github.com/sebastianzapatar/baterias.git
cd baterias
```

Sin Git: botón verde **Code → Download ZIP** y descomprimir. La carpeta se llama
`baterias-main`.

> Descargue el repositorio **completo**. Un cuaderno copiado aislado no se puede
> ejecutar: necesita los módulos `.py`, la carpeta `datos/` y los resultados.

## 2. Preparar el ambiente (una sola vez)

Requisito: **Python 3.12 o superior** desde [python.org](https://www.python.org/downloads/)
(en Windows, marque *Add python.exe to PATH* al instalar).

Abra una terminal **dentro de la carpeta del repositorio** y ejecute:

| Paso | Windows (PowerShell) | macOS / Linux |
|---|---|---|
| Crear el ambiente | `py -3.12 -m venv .venv` | `python3 -m venv .venv` |
| Activarlo | `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| Instalar librerías | `python -m pip install -r requirements.txt` | `python -m pip install -r requirements.txt` |

La terminal debe mostrar `(.venv)` al inicio de la línea. Si PowerShell bloquea la
activación, ejecute una vez
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

La carpeta `.venv/` es local de cada equipo y está excluida en `.gitignore`: no se sube
al repositorio.

## 3. Ejecutar los cuadernos

```bash
jupyter lab
```

Abra JupyterLab **desde la carpeta raíz** del repositorio: todas las rutas se
resuelven desde ahí. En cada cuaderno use *Kernel → Restart Kernel and Run All Cells*.

Para ejecutarlos desde la terminal, sin abrir el navegador:

```bash
jupyter nbconvert --to notebook --execute --inplace "01_Modelo base PV_BESS.ipynb"
```

### Orden recomendado

| Cuaderno | Qué hace | Tiempo aprox. |
|---|---|---|
| `01_Modelo base PV_BESS` | Caso de referencia: sin batería, tres reglas heurísticas y despacho óptimo con previsión perfecta | ≈ 15 s |
| `01B_Caracterizacion_dimensionamiento` | Mira las series horarias para acotar tamaños de PV y batería. No optimiza | ≈ 5 s |
| `02_Optimizacion_portafolio_PV_BESS` | Define la demanda y las señales tarifarias T0/T1 y dimensiona el portafolio. **Es el único cuaderno largo** | 15–35 min |
| `08_Sensibilidad_5_sitios_CAPEX_curvas` | Etapa I · de qué depende el valor de la batería sin generación térmica (200 casos) | segundos |
| `09_Comparacion_reglas_operativas` | Etapa I · distancia de las reglas heurísticas a los benchmarks óptimos | segundos |
| `11_Integracion_generacion_termica` | Etapa II · red + cuatro unidades a gas, mínimos técnicos y arranques | segundos |
| `12_Integracion_PV_termica` | Etapa II · red + térmica + PV de tamaño fijo, sin batería | segundos |
| `13_Integracion_PV_BESS_termica` | Etapa II · primera comparación con batería que carga solo desde PV | segundos |
| `14_Comparacion_anual_reglas_integradas` | Etapa II · las cinco reglas integradas frente a los benchmarks en un año completo | segundos |
| `15_Sintesis_integral_tesis_V2` | Reúne las tres etapas, corre simulaciones de contraste y genera las ocho figuras de síntesis | 2–5 min |

**01, 01B y 02 recalculan** a partir de los datos de entrada. **08 a 15 leen
resultados ya calculados** en las carpetas `resultados_*` y los presentan; por eso
son rápidos. Los tiempos dependen del equipo.

## 4. Estructura del repositorio

```
baterias/
├── README.md                          este archivo
├── Manual_Ejecucion_Cuadernos_EIA.pdf manual completo
├── index.html                         plataforma web de consulta (GitHub Pages)
├── requirements.txt                   librerías y versiones fijadas
├── requirements-vpn.txt               aceleración opcional del cuaderno 15
├── 01_…ipynb … 15_…ipynb              los diez cuadernos
├── datos/                             datos de entrada
│   ├── load variable.csv              demanda horaria [MW], separada por «;»
│   ├── Precio_Bolsa_Nacional_($kwh)_2025.xlsx   precio de bolsa horario (XM)
│   ├── CurvasCapexPVBESS.xlsx         curvas de CAPEX solar y del BESS
│   ├── *_tmy-2025.csv                 año meteorológico típico NSRDB de las 5 regiones
│   ├── NREL_TP-6A40-93281_tabla2.csv  trayectoria publicada del costo de la batería
│   └── Oriente.xlsx                   serie solar histórica (solo contraste en el 02)
├── *.py                               biblioteca interna que importan los cuadernos
├── .build/                            biblioteca interna de los cuadernos 11, 14 y 15
├── resultados_*/                      resultados calculados que leen los cuadernos 08–15
├── paper_JES/                         artículo para el Journal of Energy Storage: experimentos R1–R4,
│                                      figuras y tablas del suplemento (ver paper_JES/README.md)
└── figuras_*/                         figuras que generan los cuadernos
```

No mueva ni renombre los cuadernos, los `.py`, `.build/` ni las carpetas `datos/`,
`resultados_*` y `figuras_*`: los cuadernos los buscan por nombre.

## 5. Cambiar los datos de entrada

Para estudiar otro caso se reemplazan los archivos de `datos/` **conservando su
nombre, sus hojas y el orden de filas y columnas**, y se ejecutan de nuevo 01, 01B
y 02. Antes de cambiar nada, haga una copia de la carpeta.

| Para cambiar… | Archivo | Lo usan |
|---|---|---|
| La demanda | `datos/load variable.csv` (columna `Porvenir`) | 01, 01B, 02 |
| El precio de bolsa | `datos/Precio_Bolsa_Nacional_($kwh)_2025.xlsx` | 01, 02 |
| El sitio o recurso solar | `datos/<Región>_tmy-2025.csv` | 01, 01B, 02 |
| Los costos de inversión | `datos/CurvasCapexPVBESS.xlsx` | 01 (secc. 9), 02 |

El formato exacto de cada archivo y las celdas que se leen están en la sección 6 del
manual y en la pestaña *Cambiar datos* de la plataforma.

## 6. Aceleración opcional del cuaderno 15

El cuaderno 15 funciona sin esto; con Numba tarda menos (≈ 2 min en vez de ≈ 5 min).
Con el ambiente activado:

```bash
# Windows (PowerShell)
python -m pip install --no-deps --target .venv\numba -r requirements-vpn.txt
# macOS / Linux
python -m pip install --no-deps --target .venv/numba -r requirements-vpn.txt
```

Se instala en una subcarpeta aparte para no cambiar la versión de NumPy del ambiente
principal; el cuaderno la detecta solo.

## 7. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `FileNotFoundError` al leer un CSV o Excel | Jupyter se abrió desde otra carpeta, o se movió o renombró un archivo de `datos/` | Cierre Jupyter y ábralo desde la raíz del repositorio; restaure los nombres originales |
| `ModuleNotFoundError` | El ambiente no está activado o el kernel no es el de `.venv` | Active `.venv` y elija ese kernel en Jupyter |
| «Kernel died» o falta de memoria | Poca RAM libre | Cierre otros programas y ejecute un cuaderno a la vez |
| Excel no deja guardar o el cuaderno no ve los cambios | El libro sigue abierto en Excel | Guarde, cierre Excel y vuelva a ejecutar |

## 8. Plataforma en GitHub Pages

La plataforma pública se despliega solo desde el repositorio
[`sebastianzapatar/baterias`](https://github.com/sebastianzapatar/baterias), con el
flujo `.github/workflows/pages.yml`: publica `index.html` y el manual cada vez que
cambian en `main`. Requisito de una sola vez en ese repositorio: en
**Settings → Pages**, poner **Source = GitHub Actions**.
