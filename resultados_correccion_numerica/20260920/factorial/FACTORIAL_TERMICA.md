# Factorial operativo anual con térmica

**Completo: 40 políticas, 24 benchmarks y 8 referencias auditados (72 archivos).**

Año 2025, 8.760 horas, recurso solar Oriente; gas 20 USD/MMBtu, TRM 3.126,08 COP/USD. 
CVOM G1/G2: 9,28; G3/G4: 15,32 USD/MWh. Cargo de arranque supuesto: 100 USD/unidad/evento.

## Alcance

Se compara operación a tamaños dados. K es un factor categórico que modifica conjuntamente la penetración PV, potencia y energía del BESS. No permite atribuir el cambio exclusivamente al tamaño de la batería.
A/B/D cargan solo desde excedentes PV. C/F son un bloque adicional que admite carga desde red. La térmica no carga el BESS ni exporta. SOC cíclico anual libre por trayectoria, térmica inicialmente libre y máximo un arranque y una parada diarios por unidad.
El costo neto incluye gas, CVOM, compras, cargos de arranque e ingresos de exportación; excluye CAPEX y OPEX fijo. Esta comparación no es un VPN ni selecciona el dimensionamiento económicamente óptimo.

| Configuración | PV [MWp DC] | Penetración anual PV | BESS [MW/MWh] |
|---|---:|---:|---:|
| K1 | 114,42 | 100,0 % | 50 / 400 |
| K2 | 105,03 | 91,8 % | 45 / 180 |
| K3 | 114,42 | 100,0 % | 45 / 280 |
| K4 | 94,96 | 83,0 % | 50 / 100 |

## Costo operativo neto [mil millones COP/año]

| Tarifa | K | A | B | D | C | F | Mejor regla |
|---|---|---:|---:|---:|---:|---:|---|
| T0 | K1 | 10,91 | 13,28 | 13,84 | 35,53 | 35,53 | A |
| T0 | K2 | 26,40 | 26,23 | 26,20 | 38,08 | 38,09 | D |
| T0 | K3 | 14,82 | 15,70 | 16,12 | 35,53 | 35,53 | A |
| T0 | K4 | 38,26 | 38,08 | 38,03 | 41,76 | 41,77 | D |
| T1 | K1 | 9,27 | 10,38 | 10,83 | 24,82 | 24,82 | A |
| T1 | K2 | 22,97 | 22,88 | 22,88 | 29,22 | 29,23 | D |
| T1 | K3 | 12,64 | 13,02 | 13,21 | 25,18 | 25,18 | A |
| T1 | K4 | 34,17 | 33,78 | 33,76 | 35,67 | 35,71 | D |

## Validación y lectura de los benchmarks

Los 72 archivos superan la reconstrucción del costo, balances horarios, rutas de energía, exclusiones, eventos diarios y cierre del SOC. Las 40 políticas y ocho referencias se recalcularon con el evaluador corregido. Los 24 benchmarks conservan exactamente sus archivos y hashes originales. K1 comparte la referencia PV sin BESS de K3; el techo de exportación no la limita.
Brecha máxima del solver: **0,4667 %**. Casos por encima de la meta del 0,1 %: **2/24**. Se conservan condición y cotas de cada solución, incluida K3; un límite de tiempo no se presenta como óptimo exacto.
La distancia de cada regla se acota entre su costo menos la mejor trayectoria factible compatible y su costo menos la cota dual. OPT_AB permite A/B; OPT_D añade exportación del BESS; OPT_CF añade carga desde red. Una trayectoria compatible mejor que el incumbente del solver se registra con su origen.
La descomposición factorial es descriptiva, exacta y sin réplicas. No hay p-valores ni pruebas de significancia. Se informa A/B/D por separado y las cinco reglas como bloque ampliado.

## Archivos

- `resumen.csv`: 72 casos y trazabilidad de origen.
- `comparacion.csv`: 40 distancias compatibles y ahorros frente al PV sin BESS.
- `cotas.csv`: 24 intervalos de costo óptimo.
- `ranking.csv`: orden por configuración y tarifa.
- `descomposicion_ABD.csv` y `descomposicion_todas.csv`: contribuciones descriptivas.
- `figuras/`: costos, distancias y uso de térmica/BESS en PNG y SVG.

La corrección elimina infeasibilidades por redondeo en los mínimos técnicos. Se usaron como máximo dos procesos de despacho anual. Los resultados anteriores se conservan; los benchmarks reutilizados mantienen su condición y cotas originales. Carga_BESS es la carga total: Carga_PV + Carga_Red. Los benchmarks están condicionados a la prioridad del PV a demanda y a la prohibición de importar y exportar simultáneamente.
