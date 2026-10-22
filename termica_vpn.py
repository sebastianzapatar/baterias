"""VPN de reglas integradas: costos completos y despachos anuales explícitos.

Reutiliza año 1; cada año posterior reduce únicamente el perfil PV según costos.
No interpola costos ni throughput, no cambia tamaños y no maximiza el VPN.
El motor financiero común conserva el tratamiento horario de las renovaciones.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

import costos
import motor_multianual as MM
import termica_cf as CF
import disponibilidad_termica as DT
import termica_politicas as POL
import termica_reglas as REG
from termica_politicas_anual import decodificar, leer_caso, codificar

FUENTE_AB = Path('resultados_politicas/anual_gas20_cvom_unidades_20260918')
FUENTE_CF = Path('resultados_politicas/cf_anual_gas20_cvom_unidades_20260918')
FUENTE_SIN = Path('resultados_reglas_termica/sin_bess_gas20_cvom_unidades_20260918')
SALIDA = Path('resultados_vpn_termica/gas20_cvom_unidades_20260918')
POLITICAS = ('A_INT', 'B_INT', 'D_INT', 'C_INT', 'F_INT')
BENCHMARKS = ('OPT_AB', 'OPT_D', 'OPT_CF')
BASES = ('RED_TERMICA', 'PV_SIN_BESS')
SERIES = ('PV', 'Carga', 'PV_directo', 'Carga_BESS', 'Descarga_demanda',
          'Descarga_export', 'Export_PV', 'Vertimiento', 'SOC', 'potencia',
          'compromiso', 'arranques', 'paradas', 'C_imp_USD', 'C_exp_USD')
CODIGO = ('termica_vpn.py', 'motor_multianual.py', 'termica_dp_acelerada.py',
          'termica_politicas.py', 'termica_cf.py', 'termica_reglas.py',
          'termica_pypsa.py', 'termica_pv.py', 'termica_bess.py',
          'termica.py', 'costos.py', 'costos_curvas.py',
          '02_Optimizacion_portafolio_PV_BESS.ipynb', 'datos/CurvasCapexPVBESS.xlsx')
_ENTORNO_A = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_guardar(path, valor):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(valor, ensure_ascii=False, indent=2, default=codificar), encoding='utf-8')
    tmp.replace(path)


def meta_fuente():
    m = json.loads((FUENTE_AB/'ejecucion.json').read_text(encoding='utf-8'), object_hook=decodificar)
    np.testing.assert_array_equal(m['parametros_termicos']['cvom'], costos.CVOM_TERMICO_USD_MWH)
    assert m['parametros_termicos']['gas'] == costos.GAS_TERMICO_USD_MMBTU
    assert m['TRM'] == costos.TRM_COP_USD and m['minutos_eventos'] == 0
    return m


def cargar_entrada(senal):
    with np.load(FUENTE_AB/f'{senal}_A_INT.npz', allow_pickle=False) as z:
        return (pd.DatetimeIndex(z['Fecha']), *[z[k].copy() for k in
                ('PV', 'Carga', 'C_imp_USD', 'C_exp_USD')])


def auditar(r, p):
    if r['Caso'] in ('C_INT', 'F_INT', 'OPT_CF'):
        return CF.validar_cf(r, p)
    if r['Caso'] == 'A_INT':
        return POL.validar_politica(r, p)
    if r['Caso'] == 'B_INT':
        return POL.validar_b_int(r, p)
    if r['Caso'] == 'D_INT':
        return POL.validar_d_int(r, p)
    return REG.validar_integrado(r, p, r['Caso'] == 'OPT_D')


def contabilidad(r, p, escenario=None, c_imp_financiero=None):
    """Costo anual completo USD; no incluye CAPEX, OPEX fijo ni precio sombra.

    Versión 2: con `escenario` (carbono.EscenarioCarbono) la trayectoria se resolvió con parámetros y señal
    ajustados por carbono. `Costo_operativo_USD` es entonces el costo **financiero** y el cargo de carbono,
    las emisiones y el costo ajustado se devuelven aparte (`carbono.contabilidad`). Sin escenario el
    resultado es el de la versión 1."""
    if escenario is not None:
        import carbono
        ci_fin = r['C_imp_USD'] if c_imp_financiero is None else c_imp_financiero
        k = carbono.contabilidad(r, p, escenario, ci_fin)
        return dict(Costo_operativo_USD=k['Costo_financiero_USD'], Gas_USD=k['Combustible_USD'], CVOM_USD=k['CVOM_USD'],
                    Compras_red_USD=k['Compras_red_financiero_USD'], Arranques_USD=k['Arranques_USD'],
                    Ingreso_exportacion_USD=k['Ingreso_exportacion_USD'], Arranques=k['Arranques'],
                    Descarga_MWh=float(np.sum(r['Descarga_demanda']+r['Descarga_export'])),
                    PV_MWh=float(r['PV'].sum()), Cierre_SOC_MWh=float(r['SOC'][-1]-r['SOC_inicial']),
                    **{c: v for c, v in k.items() if c.startswith(('Cargo_', 'Costo_ajustado', 'CO2', 'Emisiones',
                                                                   'Escenario', 'Precio_CO2', 'Modalidad', 'Factor_red',
                                                                   'Importacion', 'Exportacion'))})
    pot = np.asarray(r['potencia'], float)
    cr = np.asarray(r.get('Carga_Red', np.zeros(len(pot))), float)
    energias = pot[:, :4].sum(axis=0)
    gas = float(energias @ (p['hr']*p['gas']))
    cvom = float(energias @ p['cvom'])
    compras = float((pot[:, 4]+cr) @ r['C_imp_USD'])
    arranques = float(np.sum(r['arranques'] @ np.broadcast_to(r['Cargo'], (4,))))
    ventas = float((r['Export_PV']+r['Descarga_export']) @ r['C_exp_USD'])
    costo = gas + cvom + compras + arranques - ventas
    assert abs(costo-r['Objetivo']) <= 1e-3, (r['Caso'], costo, r['Objetivo'])
    return dict(Costo_operativo_USD=costo, Gas_USD=gas, CVOM_USD=cvom,
                Compras_red_USD=compras, Arranques_USD=arranques,
                Ingreso_exportacion_USD=ventas, Arranques=int(r['arranques'].sum()),
                Descarga_MWh=float(np.sum(r['Descarga_demanda']+r['Descarga_export'])),
                PV_MWh=float(r['PV'].sum()),
                Cierre_SOC_MWh=float(r['SOC'][-1]-r['SOC_inicial']))


def resolver_base(caso, datos, m, disponible=None):
    """DP anual exacta sin BESS, idénticas restricciones térmicas y exportación PV.

    `disponible` (versión 2): disponibilidad horaria de la flota; se registra en el resultado."""
    f, pv, load, ci, ce = datos
    if caso == 'RED_TERMICA':
        pv = np.zeros_like(pv)
    x = np.minimum(pv, load)
    p = m['parametros_termicos']
    d = DT.alinear(disponible, f)
    ev = POL.evaluador_dia(load-x, ci, p['cv'], p['pmin'], p['pmax'], f,
                          m['cargo_arranque_USD'], 'libre', disponible=d)
    POL.validar_evaluador(ev, load-x, ci, p['cv'], p['pmin'], p['pmax'], f,
                         m['cargo_arranque_USD'], 'libre', disponible=d)
    xp = np.where(ce > 0, np.minimum(pv-x, m['P_EXP_MAX']), 0.)
    cero = np.zeros(len(f))
    b = dict(m['bess'], P_MW=0., E_MWh=0.)
    r = dict(Caso=caso, Fechas=f, PV=pv, Carga=load, PV_directo=x,
             Carga_BESS=cero.copy(), Descarga_demanda=cero.copy(), Descarga_export=cero.copy(),
             Export_PV=xp, Vertimiento=pv-x-xp, SOC=cero.copy(), SOC_inicial=0.,
             C_imp_USD=ci, C_exp_USD=ce, BESS=b, Cargo=m['cargo_arranque_USD'],
             Objetivo=ev['objetivo']-float(xp@ce), Condicion='DP/optimo', Disponible=d,
             **{k: ev[k] for k in ('potencia', 'compromiso', 'arranques', 'paradas')})
    r['Cota'] = r['Objetivo']
    r['Brecha_solver'] = 0.
    REG.validar_integrado(r, p, False)
    return r


def resolver(caso, datos, m, segundos, brecha, disponible=None):
    """Una anualidad. Versión 2: `disponible` se propaga a reglas, benchmarks y bases; el carbono entra por
    `m['parametros_termicos']` y la señal de `datos` ya ajustados (`carbono.parametros_ajustados`)."""
    global _ENTORNO_A
    f, pv, load, ci, ce = datos
    p, b, cargo, pe, trm = (m[k] for k in
                           ('parametros_termicos', 'bess', 'cargo_arranque_USD', 'P_EXP_MAX', 'TRM'))
    if caso in BASES:
        return resolver_base(caso, datos, m, disponible=disponible)
    if caso == 'A_INT':
        if _ENTORNO_A is None:
            _ENTORNO_A = REG.BO.cargar_entorno_nb02()
        r = POL.simular_a_int(caso, REG.entorno(_ENTORNO_A, f), f, pv, load,
                             ci*trm, ce*trm, trm, b, pe, p, cargo, 'libre', disponible=disponible)
    elif caso in ('B_INT', 'D_INT', 'C_INT', 'F_INT'):
        if caso in ('B_INT', 'D_INT'):
            fun = POL.simular_b_int if caso == 'B_INT' else POL.simular_d_int
            simular = lambda e: fun(f, pv, load, ci, ce, b, pe, p, cargo, 'libre', e, disponible=disponible)
        else:
            simular = lambda e: CF.simular_cf(f, pv, load, ci, ce, b, pe, p, cargo, 'libre', e,
                                             reserva=(caso == 'F_INT'), disponible=disponible)
        ciclo = POL.ciclo_soc_politica(simular, b, max_iter=25, tol=1e-4)
        if ciclo['estado'] != 'CONVERGE':
            raise RuntimeError(f'{caso}: SOC {ciclo["estado"]}; no se calcula VPN de una trayectoria abierta')
        r = ciclo['resultado']
    elif caso == 'OPT_CF':
        r = CF.optimo_cf(f, pv, load, ci, ce, b, pe, p, cargo,
                         segundos=segundos, brecha=brecha, disponible=disponible)
    else:
        r = REG.optimo(caso, f, pv, load, ci, ce, p, b, pe, cargo=cargo,
                       frontera_soc='ciclica', segundos=segundos, brecha=brecha,
                       exclusion='binaria', aceptar_limite_tiempo=True, disponible=disponible)
    auditar(r, p)
    return r


def rutas(salida, senal, caso, anio):
    prefijo = Path(salida)/'anuales'/f'{senal}_{caso}_a{anio:02d}'
    return prefijo.with_suffix('.npz'), prefijo.with_suffix('.json')


def guardar(r, m, salida, senal, caso, anio, origen, segundos):
    npz, js = rutas(salida, senal, caso, anio)
    assert not js.exists(), 'No se sobrescribe un caso terminado'
    series = {k: np.asarray(r[k]) for k in SERIES}
    series['Carga_Red'] = np.asarray(r.get('Carga_Red', np.zeros(len(r['Fechas']))))
    series['Fecha'] = r['Fechas'].to_numpy()
    series['SOC_inicial'] = np.asarray(r['SOC_inicial'])
    temporal = npz.with_suffix('.tmp.npz')
    np.savez_compressed(temporal, **series)
    temporal.replace(npz)
    fila = dict(Senal=senal, Caso=caso, Anio=anio,
                Escala_PV=0. if caso == 'RED_TERMICA' else (1-costos.DEGRADACION_PV_ANUAL)**(anio-1),
                Origen=origen, Segundos=segundos, Estado='OK',
                Condicion=r.get('Condicion', 'politica/CONVERGE'),
                Cota_USD=float(r.get('Cota', np.nan)),
                Brecha_solver=float(r.get('Brecha_solver', np.nan)),
                SHA256_npz=sha(npz), **contabilidad(r, m['parametros_termicos']))
    json_guardar(js, fila)
    return fila


def comprobar_archivo(salida, senal, caso, anio):
    npz, js = rutas(salida, senal, caso, anio)
    row = json.loads(js.read_text(encoding='utf-8'))
    assert sha(npz) == row['SHA256_npz'], str(npz)
    assert row['Estado'] == 'OK' and abs(row['Cierre_SOC_MWh']) <= REG.TOL_CIERRE_MWh
    return row


def inicializar_worker(acelerar):
    if acelerar:
        from termica_dp_acelerada import activar
        activar()
    for nombre in ('pypsa', 'linopy'):
        logging.getLogger(nombre).setLevel(logging.ERROR)


def tarea(salida, senal, caso, anio, segundos, brecha):
    m = meta_fuente()
    npz, js = rutas(salida, senal, caso, anio)
    if js.exists():
        return comprobar_archivo(salida, senal, caso, anio)
    tic = time.perf_counter()
    datos = list(cargar_entrada(senal))
    datos[1] *= (1-costos.DEGRADACION_PV_ANUAL)**(anio-1)
    if anio == 1 and caso in POLITICAS+BENCHMARKS:
        fuente = FUENTE_CF if caso in ('C_INT', 'F_INT', 'OPT_CF') else FUENTE_AB
        r = leer_caso(fuente, f'{senal}_{caso}')
        for k, v in zip(('PV', 'Carga', 'C_imp_USD', 'C_exp_USD'), datos[1:]):
            np.testing.assert_array_equal(r[k], v)
        auditar(r, m['parametros_termicos'])
        origen = str(fuente/f'{senal}_{caso}.json.gz')
    else:
        r = resolver(caso, datos, m, segundos, brecha)
        origen = 'despacho_explicito'
    if caso == 'PV_SIN_BESS' and anio == 1:
        ref = pd.read_csv(FUENTE_SIN/'resumen.csv')
        ref = ref[(ref.Senal == senal) & (ref.Caso == 'SIN_BESS')].iloc[0]
        assert abs(r['Objetivo']-ref['Costo_operativo_neto_USD']) < 1e-3
    return guardar(r, m, salida, senal, caso, anio, origen, time.perf_counter()-tic)


def preparar(salida, segundos, brecha):
    salida = Path(salida)
    (salida/'anuales').mkdir(parents=True, exist_ok=True)
    m = meta_fuente()
    # Tamaño DC de la fuente canónica; se comprueba su perfil contra el archivo congelado.
    e = REG.TP.cargar_entradas_pv()
    np.testing.assert_array_equal(e['PV_MW'], cargar_entrada('T0')[1])
    fuentes = [FUENTE_AB/'ejecucion.json', FUENTE_CF/'ejecucion.json', FUENTE_SIN/'resumen.csv']
    fuentes += list(FUENTE_AB.glob('*.json.gz'))+list(FUENTE_CF.glob('*.json.gz'))
    config = dict(horizonte=costos.HORIZONTE_ANIOS, wacc=costos.WACC,
                  degradacion_pv=costos.DEGRADACION_PV_ANUAL, ciclos=costos.CICLOS_VIDA_BESS,
                  calendario=costos.VIDA_CALENDARIO_BESS, pv_MWp=e['PV_MWp'],
                  segundos=segundos, brecha=brecha, TRM=m['TRM'],
                  codigo={f: sha(f) for f in CODIGO}, fuentes={str(f): sha(f) for f in fuentes},
                  convencion='20 años representativos explícitos con SOC cíclico anual; térmica libre al inicio de cada año',
                  parametros_termicos=m['parametros_termicos'], bess=m['bess'])
    path = salida/'configuracion.json'
    if path.exists():
        anterior = json.loads(path.read_text(encoding='utf-8'), object_hook=decodificar)
        for k in ('horizonte','wacc','degradacion_pv','ciclos','calendario','pv_MWp','segundos','brecha','TRM','codigo','fuentes'):
            assert anterior[k] == config[k], f'Configuración distinta: {k}; use otra salida'
    else:
        json_guardar(path, config)
    return config


def financiero(salida, senal, caso, referencia, config):
    """Evaluación sin interpolación; costos de las dos trayectorias para cada año."""
    b = config['bess']
    completo = referencia == 'RED_TERMICA'
    tiene_bess = caso not in BASES
    trm = config['TRM']
    cap_b = costos.capex_bess(b['P_MW'], b['E_MWh'], trm) if tiene_bess else 0.
    op_b = costos.opex_bess_anual(b['E_MWh'], trm, b['P_MW']) if tiene_bess else 0.
    cap_pv = costos.capex_pv(config['pv_MWp'], trm) if completo else 0.
    op_pv = costos.opex_pv_anual(config['pv_MWp'], trm) if completo else 0.
    filas = []

    def evaluar(anio, escala):
        a = comprobar_archivo(salida, senal, caso, anio)
        nr = 1 if referencia == 'RED_TERMICA' else anio
        ref = comprobar_archivo(salida, senal, referencia, nr)
        assert abs(a['Escala_PV']-escala) < 1e-12
        with np.load(rutas(salida, senal, caso, anio)[0], allow_pickle=False) as z:
            descarga = z['Descarga_demanda']+z['Descarga_export']
        assert descarga.min() >= -POL.TOL_FLUJO_MW
        # Solo elimina negativos del orden de redondeo ya aceptados por el validador físico.
        efc = np.maximum(descarga, 0)/(b['E_MWh']*b['DoD']) if tiene_bess else np.zeros(len(descarga))
        ahorro = (ref['Costo_operativo_USD']-a['Costo_operativo_USD'])*trm
        filas.append(dict(Senal=senal, Caso=caso, Referencia=referencia, Anio=anio,
                          Costo_operativo_COP=a['Costo_operativo_USD']*trm,
                          Costo_referencia_COP=ref['Costo_operativo_USD']*trm,
                          Condicion=a['Condicion'], Brecha_solver=a['Brecha_solver']))
        return dict(beneficio_COP=ahorro, efc_h=efc)

    out = MM.simular_horizonte(None, None, np.zeros(8760), None, None, caso,
            config['pv_MWp'], b['P_MW'] if tiene_bess else 0., b['E_MWh'] if tiene_bess else 0.,
            DoD=b['DoD'], eff_rt=b['RTE'], c_deg_COP_MWh=0.,
            capex_inicial_COP=cap_b+cap_pv, opex_anual_COP=op_b+op_pv,
            horizonte=config['horizonte'], wacc=config['wacc'], degradacion_pv=config['degradacion_pv'],
            ciclos_vida=config['ciclos'], vida_calendario=config['calendario'],
            trm=trm, evaluar_anio=evaluar)
    tab = out['tabla'].merge(pd.DataFrame(filas), on='Anio', validate='one_to_one')
    tab['OPEX_fijo_COP'] = op_b+op_pv
    tab['Flujo_COP'] = out['flujos'][1:]
    tab['Flujo_descontado_COP'] = tab.Flujo_COP/(1+config['wacc'])**tab.Anio
    assert np.isclose(out['EFC_total_descargado'],out['EFC_en_bloques_retirados']+out['EFC_bloque_activo_final'],atol=1e-7,rtol=0)
    assert np.isclose(out['VPN_COP'],-(cap_b+cap_pv)+tab.Flujo_descontado_COP.sum(),atol=.001,rtol=0)
    assert out['n_despachos'] == config['horizonte'] and tab.Explicito.all()
    fila = dict(Senal=senal, Caso=caso, Referencia=referencia,
                CAPEX_COP=cap_b+cap_pv, OPEX_anual_COP=op_b+op_pv,
                VPN_COP=out['VPN_COP'], TIR=out['TIR'], Payback_anios=out['Payback_anios'],
                EFC_total=out['EFC_total_descargado'], Renovaciones=out['n_renovaciones'],
                Costo_renovaciones_VP_COP=out['Costo_renovaciones_VP_COP'],
                Tipo='Despacho operativo PyPSA factible; VPN no optimizado' if caso in BENCHMARKS else 'Regla' if tiene_bess else 'PV sin BESS')
    return fila, tab, out['renovaciones']


def consolidar(salida):
    salida = Path(salida)
    cfg = json.loads((salida/'configuracion.json').read_text(encoding='utf-8'), object_hook=decodificar)
    filas, tablas, renov = [], [], []
    for s in ('T0','T1'):
        for c in ('PV_SIN_BESS',)+POLITICAS+BENCHMARKS:
            for ref in ('RED_TERMICA',) if c == 'PV_SIN_BESS' else BASES:
                fila, tab, rr = financiero(salida,s,c,ref,cfg)
                filas.append(fila); tablas.append(tab)
                renov.extend(dict(Senal=s,Caso=c,Referencia=ref,**r) for r in rr)
    df = pd.DataFrame(filas)
    # Identidad de aditividad: VPN total = VPN PV + VPN incremental del BESS.
    for s in ('T0','T1'):
        q = df[df.Senal==s]
        vp_pv = q[q.Caso=='PV_SIN_BESS'].VPN_COP.iloc[0]
        for c in POLITICAS+BENCHMARKS:
            d = q[q.Caso==c].set_index('Referencia')
            assert abs(d.loc['RED_TERMICA','VPN_COP']-d.loc['PV_SIN_BESS','VPN_COP']-vp_pv)<.01
    df.to_csv(salida/'vpn_resumen.csv',index=False)
    pd.concat(tablas,ignore_index=True).to_csv(salida/'flujos_anuales.csv',index=False)
    json_guardar(salida/'renovaciones.json',renov)
    anuales=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((salida/'anuales').glob('*.json'))]
    pd.DataFrame(anuales).to_csv(salida/'despachos_anuales.csv',index=False)
    return df


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--salida',type=Path,default=SALIDA)
    ap.add_argument('--etapa',choices=('preparar','correr','consolidar'),required=True)
    ap.add_argument('--casos',nargs='+',choices=BASES+POLITICAS+BENCHMARKS,default=BASES+POLITICAS+BENCHMARKS)
    ap.add_argument('--anios',nargs='+',type=int,default=list(range(1,21)))
    ap.add_argument('--senales',nargs='+',choices=('T0','T1'),default=['T0','T1'])
    ap.add_argument('--procesos',type=int,default=1)
    ap.add_argument('--segundos',type=int,default=60)
    ap.add_argument('--brecha',type=float,default=.001)
    ap.add_argument('--acelerar',action='store_true')
    args=ap.parse_args()
    assert args.procesos>0 and args.segundos>0 and 0<args.brecha<=.001
    assert all(1<=a<=costos.HORIZONTE_ANIOS for a in args.anios)
    if args.etapa=='preparar':
        preparar(args.salida,args.segundos,args.brecha)
        return
    if args.etapa=='consolidar':
        print(consolidar(args.salida).to_string(index=False))
        return
    cfg=json.loads((args.salida/'configuracion.json').read_text(encoding='utf-8'))
    assert cfg['segundos']==args.segundos and cfg['brecha']==args.brecha
    assert cfg['codigo']=={f:sha(f) for f in CODIGO},'Cambió el código durante la corrida'
    tareas=[(args.salida,s,c,a,args.segundos,args.brecha)
            for a in args.anios for s in args.senales for c in args.casos
            if not(c=='RED_TERMICA' and a!=1)]
    pendientes=[]
    for t in tareas:
        if rutas(*t[:4])[1].exists():
            comprobar_archivo(*t[:4])
        else:
            pendientes.append(t)
    print(json.dumps(dict(etapa='inicio',pendientes=len(pendientes),procesos=args.procesos)),flush=True)
    with ProcessPoolExecutor(max_workers=args.procesos,initializer=inicializar_worker,initargs=(args.acelerar,)) as pool:
        futuros={pool.submit(tarea,*t):t for t in pendientes}
        for i,fu in enumerate(as_completed(futuros),1):
            row=fu.result()
            print(json.dumps(dict(terminados=i,total=len(pendientes),**row),ensure_ascii=False),flush=True)
    print('ETAPA TERMINADA',flush=True)


if __name__=='__main__':
    main()
