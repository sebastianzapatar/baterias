"""Precio interno de carbono y contabilidad de emisiones de la línea térmica (versión 2).

Alcance de las emisiones
------------------------
* **CO2 térmico directo:** generación efectiva × consumo específico × FE_gas (FECOC 2016, base PCI).
  Los consumos específicos se interpretan sobre PCI como supuesto de trabajo (explícito solo para G4).
  El modelo no representa combustible de arranque: el cargo por arranque es monetario y no se convierte
  en gas consumido, así que esas emisiones quedan fuera.
* **CO2e de la electricidad comprada:** importación bruta × FE_red (principal XM 2025; sensibilidad UPME
  2024). Incluye cualquier compra que cargue el BESS. No se restan exportaciones ni se atribuyen créditos a
  la energía exportada, y la descarga del BESS no vuelve a contabilizarse.
* Las columnas de CO2 térmico y CO2e de red se reportan por separado; su suma son **emisiones operativas
  contabilizadas con este alcance parcial**, no un inventario completo de GEI ni un análisis de ciclo de vida.

Precio interno de carbono
-------------------------
Parámetro de escenario, no precio observado ni tarifa tributaria. Entra en las decisiones:

    CV_g ajustado = HR_g·PrecioGas + CVOM_g + p_C·HR_g·FE_gas
    c_imp,t ajustado = c_imp,t financiero + I_red·p_C·FE_red,   I_red = 1 en `gas_y_red`, 0 en `solo_gas`

La remuneración de exportación no cambia. Los motores reciben parámetros y señal ajustados; la contabilidad
reconstruye desde las series el costo financiero y el cargo de carbono por separado y exige que su suma sea
el objetivo del motor, de modo que el carbono no se cobra dos veces.

Todo es explícito: `EscenarioCarbono` es inmutable y las funciones devuelven objetos nuevos, sin tocar
variables de módulo durante los barridos.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

import costos

VERSION = 'carbono_v2_20260926'
MODALIDADES = costos.MODALIDADES_CARBONO
TOL_OBJETIVO_USD = 1e-3      # misma tolerancia que `termica_vpn.contabilidad`
TOL_SENAL_USD_MWH = 1e-9


@dataclass(frozen=True)
class EscenarioCarbono:
    """Escenario de precio interno de carbono. Con precio cero no altera ninguna decisión."""
    precio_USD_t: float = 0.
    modalidad: str = 'gas_y_red'
    fe_gas_t_MMBtu: float = costos.FACTOR_EMISION_GAS_T_MMBTU
    fe_red_t_MWh: float = costos.FACTOR_EMISION_RED_T_MWH
    factor_red: str = 'XM_2025'

    def __post_init__(self):
        if self.modalidad not in MODALIDADES:
            raise ValueError(f'modalidad debe ser una de {MODALIDADES}')
        for nombre in ('precio_USD_t', 'fe_gas_t_MMBtu', 'fe_red_t_MWh'):
            v = getattr(self, nombre)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ValueError(f'{nombre} debe ser finito y no negativo')
        if self.fe_gas_t_MMBtu <= 0 or self.fe_red_t_MWh <= 0:
            raise ValueError('Los factores de emisión deben ser positivos')

    @property
    def grava_red(self) -> bool:
        return self.modalidad == 'gas_y_red'

    def cargo_termico_USD_MWh(self, hr) -> np.ndarray:
        return self.precio_USD_t * np.asarray(hr, float) * self.fe_gas_t_MMBtu

    def cargo_red_USD_MWh(self) -> float:
        return self.precio_USD_t * self.fe_red_t_MWh if self.grava_red else 0.

    def clave(self) -> dict:
        """Identificación completa para cachés y metadatos."""
        return dict(asdict(self), version=VERSION)

    def nombre(self) -> str:
        return f"p{self.precio_USD_t:g}_{self.modalidad}_{self.factor_red}"


def escenario_upme_2024(precio_USD_t, modalidad='gas_y_red') -> EscenarioCarbono:
    """Sensibilidad del factor de red: UPME, inventarios de GEI, año 2024."""
    return EscenarioCarbono(precio_USD_t, modalidad, fe_red_t_MWh=costos.FACTOR_EMISION_RED_UPME_2024_T_MWH,
                            factor_red='UPME_2024')


def intensidad_termica_t_MWh(P, esc: EscenarioCarbono) -> np.ndarray:
    """t CO2 por MWh eléctrico de cada unidad: HR × FE_gas (derivado, no medición de chimenea)."""
    return np.asarray(P['hr'], float) * esc.fe_gas_t_MMBtu


def parametros_ajustados(P, esc: EscenarioCarbono) -> dict:
    """Copia de los parámetros térmicos con el cargo de carbono dentro del costo variable.

    Comprueba antes que `P['cv']` sea el costo financiero HR·gas + CVOM: impide aplicar el cargo dos veces."""
    hr = np.asarray(P['hr'], float)
    cv_fin = np.asarray(P['cv'], float)
    esperado = hr * float(P['gas']) + np.asarray(P['cvom'], float)
    if not np.allclose(cv_fin, esperado, rtol=0, atol=1e-9):
        raise ValueError('P["cv"] no es el costo financiero HR·gas + CVOM: ¿ya incluye carbono?')
    Q = dict(P)
    Q['cv'] = cv_fin + esc.cargo_termico_USD_MWh(hr)
    Q['cv_financiero'] = cv_fin.copy()
    Q['cargo_carbono_USD_MWh'] = esc.cargo_termico_USD_MWh(hr)
    Q['escenario_carbono'] = esc.clave()
    return Q


def parametros_financieros(Q) -> dict:
    """Recupera los parámetros financieros a partir de unos ajustados (o los devuelve si ya lo son)."""
    if 'cv_financiero' not in Q:
        return dict(Q)
    P = {k: v for k, v in Q.items() if k not in ('cv_financiero', 'cargo_carbono_USD_MWh', 'escenario_carbono')}
    P['cv'] = np.asarray(Q['cv_financiero'], float).copy()
    return P


def senal_ajustada(c_imp_usd, esc: EscenarioCarbono) -> np.ndarray:
    """Señal de compra con el cargo de carbono de la red (solo en `gas_y_red`)."""
    return np.asarray(c_imp_usd, float) + esc.cargo_red_USD_MWh()


def contabilidad(r, P, esc: EscenarioCarbono, c_imp_financiero_usd, tol=TOL_OBJETIVO_USD) -> dict:
    """Emisiones y costos anuales reconstruidos desde las series de una trayectoria.

    `P` son los parámetros financieros (o ajustados: se recupera el financiero). `c_imp_financiero_usd` es la
    señal sin carbono. Comprueba que la señal usada por el motor sea la financiera más el cargo de red y que
    costo financiero + cargo de carbono = objetivo del motor. Δt = 1 h."""
    Pf = parametros_financieros(P)
    hr, cvom = np.asarray(Pf['hr'], float), np.asarray(Pf['cvom'], float)
    pot = np.asarray(r['potencia'], float)
    n = len(pot)
    cr = np.asarray(r.get('Carga_Red', np.zeros(n)), float)
    ci_fin = np.broadcast_to(np.asarray(c_imp_financiero_usd, float), (n,))
    ci_motor = np.asarray(r['C_imp_USD'], float)
    desvio = float(np.abs(ci_motor - (ci_fin + esc.cargo_red_USD_MWh())).max())
    if desvio > TOL_SENAL_USD_MWH:
        raise AssertionError(('señal de compra del motor distinta de la financiera + cargo de red', desvio))
    energia_u = pot[:, :4].sum(axis=0)
    gas_u = energia_u * hr
    co2_termico_u = gas_u * esc.fe_gas_t_MMBtu
    importacion = pot[:, 4] + cr
    co2_red = float(importacion.sum()) * esc.fe_red_t_MWh
    ventas = float((np.asarray(r['Export_PV'], float) + np.asarray(r['Descarga_export'], float))
                   @ np.asarray(r['C_exp_USD'], float))
    combustible = float(gas_u @ np.full(4, float(Pf['gas'])))
    cvom_usd = float(energia_u @ cvom)
    compras = float(importacion @ ci_fin)
    arranques = float(np.sum(np.asarray(r['arranques']) @ np.broadcast_to(np.asarray(r['Cargo'], float), (4,))))
    financiero = math.fsum([combustible, cvom_usd, compras, arranques, -ventas])
    cargo_termico = esc.precio_USD_t * float(co2_termico_u.sum())
    cargo_red = esc.precio_USD_t * co2_red if esc.grava_red else 0.
    ajustado = financiero + cargo_termico + cargo_red
    error = abs(ajustado - float(r['Objetivo']))
    if error > tol:
        raise AssertionError((r.get('Caso'), 'costo ajustado distinto del objetivo del motor', ajustado, r['Objetivo']))
    out = dict(Escenario=esc.nombre(), Precio_CO2_USD_t=esc.precio_USD_t, Modalidad=esc.modalidad,
               Factor_red=esc.factor_red, FE_red_t_MWh=esc.fe_red_t_MWh, FE_gas_t_MMBtu=esc.fe_gas_t_MMBtu,
               Costo_financiero_USD=financiero, Cargo_carbono_termico_USD=cargo_termico,
               Cargo_carbono_red_USD=cargo_red, Cargo_carbono_USD=cargo_termico + cargo_red,
               Costo_ajustado_USD=ajustado, Error_objetivo_USD=error,
               Combustible_USD=combustible, CVOM_USD=cvom_usd, Compras_red_financiero_USD=compras,
               Arranques_USD=arranques, Ingreso_exportacion_USD=ventas,
               Gas_MMBtu=float(gas_u.sum()), Generacion_termica_MWh=float(energia_u.sum()),
               Importacion_MWh=float(importacion.sum()),
               Exportacion_MWh=float((np.asarray(r['Export_PV'], float) + np.asarray(r['Descarga_export'], float)).sum()),
               Arranques=int(np.asarray(r['arranques']).sum()),
               CO2_termico_t=float(co2_termico_u.sum()), CO2e_red_t=co2_red,
               Emisiones_operativas_t=float(co2_termico_u.sum()) + co2_red,
               Emisiones_cobradas_t=float(co2_termico_u.sum()) + (co2_red if esc.grava_red else 0.))
    for j, u in enumerate(('G1', 'G2', 'G3', 'G4')):
        out[f'Generacion_{u}_MWh'] = float(energia_u[j])
        out[f'CO2_{u}_t'] = float(co2_termico_u[j])
    return out


def vpn_ajustado(vpn_financiero, precios_USD_t, emisiones_cobradas_ref_t, emisiones_cobradas_proy_t,
                 trm, tasa=costos.WACC):
    """VPN ajustado = VPN financiero + Σ_y p_y·(E_ref,y − E_proy,y)/(1+r)^y, años 1..H, en COP.

    El componente de carbono es una valoración interna, no un ingreso de caja ni una obligación tributaria.
    Ambos VPN deben referirse a las mismas trayectorias de operación."""
    p = np.asarray(precios_USD_t, float)
    er, ep = np.asarray(emisiones_cobradas_ref_t, float), np.asarray(emisiones_cobradas_proy_t, float)
    if not (p.shape == er.shape == ep.shape and p.ndim == 1):
        raise ValueError('Precios y emisiones deben ser series anuales de igual longitud')
    anios = np.arange(1, len(p) + 1)
    valor = float(np.sum(p * (er - ep) * trm / (1 + tasa) ** anios))
    return float(vpn_financiero) + valor, valor


def vpn_financiero_y_ajustado(anual, *, pv_MWp, P_MW, E_MWh, DoD, RTE, capex_COP, opex_anual_COP, trm,
                              horizonte=None, wacc=None):
    """VPN financiero con el motor multianual existente y VPN ajustado por carbono sobre las MISMAS trayectorias.

    `anual(anio, escala_pv)` resuelve (o lee) el año explícito del proyecto y de su referencia red + térmica
    bajo el mismo escenario y devuelve: `costo_proy_fin_USD`, `costo_ref_fin_USD`, `emis_cobradas_proy_t`,
    `emis_cobradas_ref_t`, `precio_USD_t` y `efc_h` (perfil horario de ciclos del BESS). El beneficio que ve el
    motor es solo financiero; el carbono se agrega después con `vpn_ajustado`. Exige los `horizonte` años
    explícitos: no extrapola el primer año ni interpola."""
    import motor_multianual as MM
    horizonte = costos.HORIZONTE_ANIOS if horizonte is None else horizonte
    wacc = costos.WACC if wacc is None else wacc
    registro = {}

    def evaluar(anio, escala):
        a = anual(anio, escala)
        registro[anio] = a
        return dict(beneficio_COP=(float(a['costo_ref_fin_USD']) - float(a['costo_proy_fin_USD'])) * trm,
                    efc_h=np.asarray(a['efc_h'], float))

    out = MM.simular_horizonte(None, None, np.zeros(8760), None, None, 'carbono_v2', pv_MWp, P_MW, E_MWh,
                               DoD=DoD, eff_rt=RTE, c_deg_COP_MWh=0., capex_inicial_COP=capex_COP,
                               opex_anual_COP=opex_anual_COP, horizonte=horizonte, wacc=wacc, trm=trm,
                               evaluar_anio=evaluar)
    if out['n_despachos'] != horizonte or sorted(registro) != list(range(1, horizonte + 1)):
        raise AssertionError('El VPN exige los años explícitos del horizonte, sin interpolación')
    anios = list(range(1, horizonte + 1))
    vpn_aj, valor = vpn_ajustado(out['VPN_COP'], [registro[a]['precio_USD_t'] for a in anios],
                                 [registro[a]['emis_cobradas_ref_t'] for a in anios],
                                 [registro[a]['emis_cobradas_proy_t'] for a in anios], trm, wacc)
    return dict(VPN_financiero_COP=float(out['VPN_COP']), VPN_ajustado_COP=vpn_aj,
                Valor_carbono_evitado_VP_COP=valor, Renovaciones=int(out['n_renovaciones']), salida=out,
                anual=registro)
