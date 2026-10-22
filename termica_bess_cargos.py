"""Respuesta anual al cargo de arranque, sin límite anual de eventos.

Conserva el límite diario y todos los demás supuestos del caso BESS fijo.
Revalúa candidatos factibles a cada cargo. Las cotas de corridas con límite
anual NO se trasladan al problema sin ese límite. No calibra costos de planta.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from costos import CARGOS_ARRANQUE_EXPLORATORIOS_USD

SAL=Path('resultados_nb13')
CARPETA=SAL/'cargos_bess_candidatos'
CARGOS=tuple(CARGOS_ARRANQUE_EXPLORATORIOS_USD)
TOL=1e-4


def leer(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def firma(E,P,b):
    from tarifas_termica_v2 import huella_entradas
    return huella_entradas(E,P,dict(bess=b,minutos=0,inicial='libre',limite_diario=1,
                                   estudio='cargo_variable_sin_limite_anual_v1'))


def contar(r):
    u=np.asarray(r['compromiso'])
    assert np.isin(u,[0,1]).all()
    arr=(np.diff(u,axis=0,prepend=u[:1])==1).astype(int)
    np.testing.assert_array_equal(arr,r['arranques'])
    return int(arr.sum())


def energia(r,E,P,s):
    return float((r['potencia'][:,:4]@P['cv']).sum()+r['potencia'][:,4]@E['C_IMP_USD'][s])


def validar(r,E,P,b,s,cargo):
    import termica_bess as TB
    assert pd.DatetimeIndex(r['Fecha']).equals(E['FECHA'])
    np.testing.assert_allclose(r['pv_disponible'],E['PV_MW'],rtol=0,atol=1e-9)
    costo=energia(r,E,P,s)+cargo*contar(r)
    if 'objetivo' in r:
        assert abs(float(r['objetivo'])-costo)<=TOL
    ver=TB.validar_bess(dict(r,objetivo=costo),E['LOAD'],E['C_IMP_USD'][s],P['cv'],P['pmin'],P['pmax'],
                       E['FECHA'],E['PV_MW'],cargo,b)
    return costo,ver


def guardar(r,E,P,b,s,cargo,segundos):
    CARPETA.mkdir(exist_ok=True)
    nombre=f'{s}_{cargo:g}_{uuid4().hex}'
    temp=CARPETA/(nombre+'.pendiente.npz'); destino=CARPETA/(nombre+'.npz')
    m=dict(Senal=s,Cargo_origen_USD=float(cargo),Limite_anual_origen=None,
           Huella_contexto=firma(E,P,b),Costo_origen_USD=float(r['objetivo']),
           Costo_energia_USD=energia(r,E,P,s),Arranques=contar(r),Cota_origen_USD=float(r['cota_dual']),
           Segundos=float(r['segundos']),Presupuesto_s=segundos,Condicion_solver=r['condicion_solver'],
           Fecha_UTC=datetime.now(timezone.utc).isoformat())
    claves=('potencia','compromiso','arranques','paradas','pv_usado','pv_directo','bess_carga','bess_descarga','soc')
    np.savez_compressed(temp,**{k:r[k] for k in claves},Fecha=E['FECHA'].to_numpy(),pv_disponible=E['PV_MW'],
                        objetivo=r['objetivo'],metadatos=json.dumps(m))
    costo,_=validar(leer(temp),E,P,b,s,cargo)
    assert np.isfinite(m['Cota_origen_USD']) and m['Cota_origen_USD']<=costo+TOL
    os.replace(temp,destino)


def leer_candidato(path,E,P,b):
    r=leer(path); m=json.loads(str(r.pop('metadatos')))
    assert m['Huella_contexto']==firma(E,P,b),f'caché incompatible: {path}'
    assert m['Limite_anual_origen'] is None
    costo,_=validar(r,E,P,b,m['Senal'],m['Cargo_origen_USD'])
    assert abs(costo-m['Costo_origen_USD'])<=TOL and contar(r)==m['Arranques']
    assert abs(energia(r,E,P,m['Senal'])-m['Costo_energia_USD'])<=TOL
    assert np.isfinite(m['Cota_origen_USD']) and m['Cota_origen_USD']<=costo+TOL
    return r,dict(m,Archivo=str(path))


def cargar_pool(E,P,b):
    import termica_bess_limites as BL
    originales,_=BL.referencias(E,P,b)
    anteriores=BL.cargar_pool(E,P,b,originales)
    pool=[]
    for r,m in anteriores:
        e=energia(r,E,P,m['Senal']); n=contar(r)
        mm=dict(Senal=m['Senal'],Cargo_origen_USD=100.,Limite_anual_origen=m['Limite_origen'],
                Huella_contexto=firma(E,P,b),Costo_energia_USD=e,Arranques=n,
                Costo_origen_USD=e+100*n,Archivo=m['Archivo'],Condicion_solver=m['Condicion_solver'])
        # 365 no agrega restricción al límite diario del año 2025; los demás sí.
        if m['Limite_origen']==365 and len(E['FECHA'].normalize().unique())<=365:
            mm['Limite_anual_origen']=None
            mm['Cota_origen_USD']=m['Cota_USD']
        else:
            mm['Cota_origen_USD']=None
        pool.append((r,mm))
    for path in sorted(CARPETA.glob('*.npz')):
        if '.pendiente.' in path.name: continue
        pool.append(leer_candidato(path,E,P,b))
    return pool


def escoger(pool,s,cargo,nmax):
    candidatos=[(r,m) for r,m in pool if m['Senal']==s]
    r,m=min(candidatos,key=lambda v:(v[1]['Costo_energia_USD']+cargo*v[1]['Arranques'],v[1]['Arranques'],v[1].get('Archivo','')))
    upper=m['Costo_energia_USD']+cargo*m['Arranques']
    validas=[mm for _,mm in candidatos if mm.get('Limite_anual_origen') is None
             and mm.get('Cota_origen_USD') is not None]
    cotas=[(0.,'costos_no_negativos')]
    # f(c) >= L(c0)+(c-c0)*Nmin si c>=c0; para c<c0 usamos Nmax válido.
    for a in validas:
        delta=cargo-a['Cargo_origen_USD']
        cotas.append((a['Cota_origen_USD']+min(0.,delta)*nmax,a.get('Archivo','')))
    # f es cóncava: la cuerda entre cotas de dos cargos también es cota inferior.
    for a in validas:
        for z in validas:
            ca,cz=a['Cargo_origen_USD'],z['Cargo_origen_USD']
            if ca<cargo<cz:
                w=(cargo-ca)/(cz-ca)
                cotas.append(((1-w)*a['Cota_origen_USD']+w*z['Cota_origen_USD'],
                              'cuerda:'+a.get('Archivo','')+'|'+z.get('Archivo','')))
    lower,origen=max(cotas,key=lambda v:v[0])
    assert lower<=upper+TOL,(s,cargo,lower,upper)
    return r,m,upper,lower,origen


def puntos_envolvente(pool,s):
    puntos=set(np.linspace(min(CARGOS),max(CARGOS),201))|set(CARGOS)
    politicas=[m for _,m in pool if m['Senal']==s]
    # Incluir todos los cruces evita interpolar por debajo de la envolvente factible.
    for a in politicas:
        for z in politicas:
            dn=a['Arranques']-z['Arranques']
            if dn:
                cruce=(z['Costo_energia_USD']-a['Costo_energia_USD'])/dn
                if min(CARGOS)<cruce<max(CARGOS): puntos.add(float(cruce))
    return sorted(puntos)


def resumir(pool,E,P,b,completo=False):
    import termica_bess as TB
    # La cota cero requiere costos energéticos no negativos.
    assert np.isfinite(P['cv']).all() and (np.asarray(P['cv'])>=0).all()
    for senal in ('T0','T1'):
        precios=np.asarray(E['C_IMP_USD'][senal])
        assert np.isfinite(precios).all() and (precios>=0).all()
    nmax=4*len(E['FECHA'].normalize().unique())
    filas=[]; controles=[]; unidades=[]
    resueltos={(m['Senal'],m['Cargo_origen_USD']) for _,m in pool if m.get('Limite_anual_origen') is None}
    for s in ('T0','T1'):
        for c in CARGOS:
            if (s,c) not in resueltos: continue
            r,m,u,l,origen=escoger(pool,s,c,nmax)
            rr=dict(r,objetivo=u)
            _,v=validar(rr,E,P,b,s,c)
            i=TB.indicadores_bess(rr,E['LOAD'],E['C_IMP_USD'][s],P,E['PV_MW'],c,b)
            filas.append(dict(Senal=s,Cargo_USD_arranque=c,Sin_limite_anual=True,
                Costo_energia_USD=m['Costo_energia_USD'],Costo_comun_100_USD=m['Costo_energia_USD']+100*m['Arranques'],
                Cota_dual_USD=l,Brecha=max(0.,(u-l)/u),Archivo=m['Archivo'],Archivo_cota=origen,
                Cargo_origen_trayectoria=m['Cargo_origen_USD'],Limite_anual_origen_trayectoria=m['Limite_anual_origen'],
                Huella_contexto=firma(E,P,b),**i))
            controles.append(dict(Senal=s,Cargo_USD_arranque=c,Estado='OK',**v))
            unidades+=TB.tabla_unidades(rr,dict(Senal=s,Cargo_USD_arranque=c))
    t=pd.DataFrame(filas)
    for s in ('T0','T1'):
        a=t[t.Senal==s].sort_values('Cargo_USD_arranque')
        assert (np.diff(a.Arranques)<=0).all(), 'envolvente no monótona en arranques'
        assert (np.diff(a.Costo_energia_USD)>=-TOL).all(), 'energía no monótona'
        assert (np.diff(a.Costo_operativo_total_USD)>=-TOL).all()
    if completo: assert len(t)==8
    t.to_csv(SAL/'cargos_bess_resumen.csv',index=False)
    pd.DataFrame(controles).to_csv(SAL/'cargos_bess_validaciones.csv',index=False)
    pd.DataFrame(unidades).to_csv(SAL/'cargos_bess_unidades.csv',index=False)
    pd.DataFrame([m for _,m in pool]).to_csv(SAL/'cargos_bess_candidatos.csv',index=False)
    densas=[]
    for s in ('T0','T1'):
        for c in puntos_envolvente(pool,s):
            r,m,u,l,origen=escoger(pool,s,c,nmax)
            densas.append(dict(Senal=s,Cargo_USD_arranque=c,Costo_factible_USD=u,Cota_inferior_USD=l,
                              Arranques=m['Arranques'],Archivo=m['Archivo'],
                              Metodo='envolvente_de_candidatos_no_nueva_optimizacion'))
    pd.DataFrame(densas).to_csv(SAL/'cargos_bess_envolvente.csv',index=False)
    return t


def graficar(t,densa):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='none',sharex=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.09,top=.86,wspace=.28,hspace=.33)
    for s,color,marker in [('T0','#7048c0','o'),('T1','#c07a00','s')]:
        a=t[t.Senal==s].sort_values('Cargo_USD_arranque'); d=densa[densa.Senal==s]
        x=a.Cargo_USD_arranque
        axs[0,0].scatter(x,a.Arranques,color=color,marker=marker,label=s)
        axs[0,0].step(d.Cargo_USD_arranque,d.Arranques,where='post',color=color,alpha=.45,lw=1)
        axs[0,1].fill_between(d.Cargo_USD_arranque,d.Cota_inferior_USD/1e6,d.Costo_factible_USD/1e6,color=color,alpha=.12)
        axs[0,1].plot(d.Cargo_USD_arranque,d.Costo_factible_USD/1e6,color=color,lw=1)
        axs[0,1].scatter(x,a.Costo_operativo_total_USD/1e6,color=color,marker=marker,label=s)
        axs[1,0].scatter(x,a.Energia_termica_MWh/1000,color=color,marker=marker,label=s)
        axs[1,1].scatter(x,a.Compras_red_MWh/1000,color=color,marker=marker,label=s)
    for ax,label in zip(axs.flat,['Arranques totales','Energía + cargo [MUSD/año]','Generación térmica [GWh/año]','Compras de red [GWh/año]']):
        ax.set_xlabel('Cargo común [USD/arranque]'); ax.set_ylabel(label)
        ax.set_xticks(CARGOS); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'.replace('.',',')))
    fig.suptitle('Respuesta al cargo de arranque · PV y BESS fijos · sin límite anual\n'
                 'Puntos: cargos resueltos. Líneas: mejores trayectorias archivadas. Bandas: cotas del costo.\n'
                 'Malla exploratoria, no calibración de mantenimiento. Límite diario conservado.',fontsize=10)
    fig.canvas.draw(); Path('figuras_nb13').mkdir(exist_ok=True)
    fig.savefig('figuras_nb13/13_5_respuesta_cargo_arranque.png',dpi=160,bbox_inches='tight'); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segundos',type=int,default=600)
    p.add_argument('--caso',nargs=2)
    p.add_argument('--solo-resumen',action='store_true')
    p.add_argument('--solo-figura',action='store_true')
    args=p.parse_args()
    if args.solo_figura:
        graficar(pd.read_csv(SAL/'cargos_bess_resumen.csv'),pd.read_csv(SAL/'cargos_bess_envolvente.csv')); return
    if args.segundos<=0: p.error('segundos debe ser positivo')
    import termica_bess as TB
    import termica_pv as TP
    from termica_pypsa import resolver_pypsa
    for nombre in ('pypsa','linopy'): logging.getLogger(nombre).setLevel(logging.ERROR)
    E,P,b=TP.cargar_entradas_pv(),TP.parametros_termicos(),TB.parametros_bess()
    pool=cargar_pool(E,P,b)
    casos=[(s,c) for s in ('T0','T1') for c in CARGOS if c!=100]
    if args.caso:
        caso=(args.caso[0],float(args.caso[1]))
        if caso not in casos: p.error('caso fuera de malla')
        casos=[caso]
    for s,c in ([] if args.solo_resumen else casos):
        if not args.caso and any(m['Senal']==s and m['Cargo_origen_USD']==c and m['Limite_anual_origen'] is None for _,m in pool):
            print('REUTILIZADO',s,c,flush=True); continue
        print(f'INICIO {s}/cargo {c:g}; sin límite anual; {args.segundos} s',flush=True)
        r=resolver_pypsa(E['LOAD'],E['C_IMP_USD'][s],P['cv'],P['pmin'],P['pmax'],E['FECHA'],c,
                        limite_segundos=args.segundos,brecha=1e-6,pv_disponible=E['PV_MW'],bess=b,
                        aceptar_limite_tiempo=True,bess_exclusion='relajada',max_arranques_anuales=None)
        r=TB.reparar_simultaneidad(r,b,E['PV_MW'])
        guardar(r,E,P,b,s,c,args.segundos)
        pool=cargar_pool(E,P,b); resumir(pool,E,P,b)
        print(f'COMPLETADO {s}/{c:g}: costo {r["objetivo"]:.2f}; cota {r["cota_dual"]:.2f}; '
              f'brecha {r["brecha"]:.4%}; arranques {contar(r)}',flush=True)
    t=resumir(pool,E,P,b,completo=not args.caso)
    graficar(t,pd.read_csv(SAL/'cargos_bess_envolvente.csv'))
    print(f'{len(t)}/8 escenarios validados; referencias conservadas.',flush=True)


if __name__=='__main__': main()
