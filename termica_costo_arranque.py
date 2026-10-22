"""Estimación paramétrica de arranques; no calibra parámetros de planta.

Las referencias NREL se mantienen en USD de 2011, separadas de los precios
actuales del gas. La malla del despacho es una exploración, no una distribución
de costos probables. No se infiere desgaste a partir del CVOM existente.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from costos import CARGOS_ARRANQUE_EXPLORATORIOS_USD

SAL = Path('resultados_nb13')


def componentes(q_extra, precio_gas, factor_eoh, provision_eoh, auxiliares_extra=0.):
    """USD/arranque. q: MMBtu adicionales; f: EOH/arranque; provisión: USD/EOH.

    Todos los consumos son adicionales al despacho. No asigna por defecto
    q, f ni provisión a ninguna unidad. Admite vectores por unidad.
    """
    q, gas, f, k, aux = np.broadcast_arrays(*[np.asarray(v, float) for v in
                                           (q_extra, precio_gas, factor_eoh, provision_eoh, auxiliares_extra)])
    if not all(np.isfinite(v).all() and (v >= 0).all() for v in (q, gas, f, k, aux)):
        raise ValueError('Parámetros finitos y no negativos requeridos')
    return dict(Combustible_USD=q*gas, Desgaste_USD=f*k, Auxiliares_USD=aux,
                Total_USD=q*gas+f*k+aux)


def generar():
    SAL.mkdir(exist_ok=True)
    combustible = [dict(Q_extra_MMBtu=q, Gas_USD_MMBtu=g, Costo_combustible_USD=q*g,
                        Naturaleza='malla_parametrica_no_calibrada')
                   for q in np.arange(0., 40.1, .5) for g in (8., 16., 20.)]
    desgaste = [dict(Factor_EOH_arranque=f, Provision_USD_EOH=k, Costo_desgaste_USD=f*k,
                    Naturaleza='malla_parametrica_no_calibrada')
                for f in range(51) for k in (10., 25., 50., 100.)]
    pd.DataFrame(combustible).to_csv(SAL/'costo_arranque_combustible.csv', index=False)
    pd.DataFrame(desgaste).to_csv(SAL/'costo_arranque_desgaste.csv', index=False)
    iso=isocostos()
    iso.to_csv(SAL/'costo_arranque_isocostos.csv',index=False)
    # Transcripción de las tablas 1-1 y 1-3, arranque frío, de NREL/Intertek 55433.
    # No son estimaciones de G1–G4 ni se suman a precios actuales sin conversión.
    ref = []
    for tecnologia, valores, q, aux in [('CT_industrial_grande', (31., 103., 118.), .22, .95),
                                      ('CT_aeroderivada', (12., 32., 61.), 1.53, 1.90)]:
        for pct, valor in zip((25, 50, 75), valores):
            ref.append(dict(Tecnologia_referencia=tecnologia, Tipo='frio', Percentil=pct,
                            CM_USD2011_MW_arranque=valor, Gas_MMBtu_MW_arranque=q,
                            Otros_USD2011_MW_arranque=aux, Paginas_impresas='12;30',
                            URL='https://docs.nlr.gov/docs/fy12osti/55433.pdf',
                            Aplicabilidad='referencia_externa_no_calibracion_de_planta'))
    pd.DataFrame(ref).to_csv(SAL/'costo_arranque_referencias.csv', index=False)
    pd.DataFrame([dict(Unidad=g, Q_extra_MMBtu='PENDIENTE', Factor_EOH_arranque='PENDIENTE',
                       Provision_USD_EOH='PENDIENTE', Auxiliares_extra_USD='PENDIENTE',
                       Cobertura_CVOM='NO_VERIFICADA') for g in ('G1','G2','G3','G4')]).to_csv(
                           SAL/'costo_arranque_parametros_pendientes.csv', index=False)
    graficar_componentes(pd.DataFrame(combustible), pd.DataFrame(desgaste))


def isocostos():
    """Combinaciones no negativas que explicarían cada cargo; no estimaciones de planta."""
    return pd.DataFrame([dict(Cargo_USD_arranque=c,Gas_USD_MMBtu=16.,Q_extra_MMBtu=q,
                              Desgaste_mas_auxiliares_USD=max(0.,c-16*q))
                         for c in CARGOS_ARRANQUE_EXPLORATORIOS_USD if c>0
                         for q in sorted(set(np.linspace(0.,min(40.,c/16),81)))])


def graficar_componentes(combustible, desgaste):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), layout='none')
    fig.subplots_adjust(left=.06, right=.985, bottom=.16, top=.77, wspace=.35)
    for g, estilo in zip((8.,16.,20.), ('--','-',':')):
        t=combustible[combustible.Gas_USD_MMBtu==g]
        ax[0].plot(t.Q_extra_MMBtu,t.Costo_combustible_USD,color='#7048c0',ls=estilo,label=f'Gas {g:g} USD/MMBtu')
    for k, estilo in zip((10.,25.,50.,100.), (':','--','-.','-')):
        t=desgaste[desgaste.Provision_USD_EOH==k]
        ax[1].plot(t.Factor_EOH_arranque,t.Costo_desgaste_USD,color='#c07a00',ls=estilo,label=f'{k:g} USD/EOH supuesto')
    ax[0].set_xlabel('Gas adicional por arranque [MMBtu]'); ax[0].set_ylabel('Costo de combustible [USD/arranque]')
    ax[1].set_xlabel('Factor de desgaste [EOH/arranque]'); ax[1].set_ylabel('Provisión por desgaste [USD/arranque]')
    iso=isocostos()
    for c,estilo in zip([v for v in CARGOS_ARRANQUE_EXPLORATORIOS_USD if v>0],(':','--','-')):
        t=iso[iso.Cargo_USD_arranque==c]
        ax[2].plot(t.Q_extra_MMBtu,t.Desgaste_mas_auxiliares_USD,color='#7048c0',ls=estilo,label=f'Total {c:g} USD')
    ax[2].set_xlabel('Gas adicional [MMBtu], a 16 USD/MMBtu')
    ax[2].set_ylabel('Desgaste + auxiliares necesarios [USD]')
    for a in ax:
        a.grid(alpha=.2); a.legend(fontsize=8)
        a.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'.replace('.',',')))
    fig.suptitle('Estimación paramétrica del costo de arranque\n'
                 'Ejes y provisiones exploratorios: no son rangos certificados ni costos asignados a G1–G4.\n'
                 'Combustible adicional + desgaste + auxiliares adicionales; evitar duplicación con CVOM.',fontsize=10)
    fig.canvas.draw(); Path('figuras_nb13').mkdir(exist_ok=True)
    fig.savefig('figuras_nb13/13_4_componentes_costo_arranque.png',dpi=160,bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    generar()
    print('Curvas paramétricas y referencias guardadas. No se asignó un costo real a ninguna unidad.')
