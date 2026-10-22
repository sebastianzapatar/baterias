"""Lectura editorial del factorial y VPN terminados; biblioteca estándar, sin modelos."""
from pathlib import Path, PureWindowsPath
import csv
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING
import hashlib
import json

FACTORIAL = Path('resultados_correccion_numerica/20260920/factorial')
VPN = Path('resultados_correccion_numerica/20260920/vpn')
POLITICAS = ('A_INT','B_INT','D_INT','C_INT','F_INT')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt(x, dec=2, modo=ROUND_HALF_UP):
    z=Decimal(str(x)).quantize(Decimal(1).scaleb(-dec),rounding=modo)
    return f'{z:,.{dec}f}'.translate(str.maketrans({',':'.','.':','}))


def tabla(cab, filas):
    return '\n'.join(['| '+' | '.join(cab)+' |','|'+'---|'*len(cab)]
                     +['| '+' | '.join(map(str,f))+' |' for f in filas])


def generar(raiz, *, factorial=None, vpn=None):
    FACTORIAL=factorial or globals()['FACTORIAL']
    VPN=vpn or globals()['VPN']
    raiz=Path(raiz)
    fuentes={}
    def js(p):
        fuentes[p.as_posix()]=sha(raiz/p)
        return json.loads((raiz/p).read_text(encoding='utf-8'))
    def csv_leer(p):
        fuentes[p.as_posix()]=sha(raiz/p)
        with (raiz/p).open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    def verificar(rutas):
        for p,h in rutas.items():
            ruta = str(p).replace('\\', '/')
            if Path(ruta).is_absolute() or PureWindowsPath(ruta).is_absolute():
                prefijos = ('resultados_correccion_numerica/20260920/',
                            'resultados_vpn_termica/gas20_cvom_unidades_20260918/',
                            'resultados_factorial_termica/gas20_cvom_unidades_20260919/')
                prefijo = next((q for q in prefijos if q in ruta), None)
                assert prefijo is not None, f'Ruta absoluta no reconocida: {p}'
                ruta = prefijo + ruta.split(prefijo, 1)[1]
            local = (raiz / ruta).resolve()
            assert local.is_relative_to(raiz.resolve()), f'Fuente fuera del checkout: {p}'
            assert sha(local)==h, f'Fuente alterada: {p}'
    vf=js(FACTORIAL/'validacion_final.json')
    vv=js(VPN/'trazabilidad_financiera.json')
    assert vf['estado']==vv['estado']=='COMPLETO'
    assert (vf['casos'],vf['politicas'],vf['benchmarks'])==(72,40,24)
    assert (vv['despachos'],vv['flujos'],vv['anios'])==(362,680,20)
    verificar(vf['fuentes']); verificar(vf['artefactos'])
    verificar(vv['archivos']); verificar(vv['configuraciones'])
    cfg=js(FACTORIAL/'configuracion.json')
    cv=js(VPN/'configuracion.json')
    fac=csv_leer(FACTORIAL/'resumen.csv')
    comp=csv_leer(FACTORIAL/'comparacion.csv')
    dec=csv_leer(FACTORIAL/'descomposicion_ABD.csv')
    fin=csv_leer(VPN/'vpn_resumen.csv')
    limites=csv_leer(VPN/'vpn_intervalos.csv')
    flujos=csv_leer(VPN/'flujos_anuales.csv')
    assert (len(fac),len(comp),len(fin),len(limites),len(flujos))==(72,40,34,12,680)
    assert all(r['Estado']=='OK' for r in fac)
    assert cfg['TRM']==cv['TRM'] and cv['horizonte']==20
    t={(r['K'],r['Senal'],r['Caso']):r for r in fac}
    f={(r['Senal'],r['Caso'],r['Referencia']):r for r in fin}
    vi={(r['Senal'],r['Benchmark'],r['Referencia']):r for r in limites}
    n=lambda r,k:Decimal(r[k])
    valores={}; exactos={}
    def guardar(k,x,decimales=2):
        exactos[k]=str(x);valores[k]=fmt(x,decimales)
    filas=[]; mejor={};ventajas=[]
    for s in ('T0','T1'):
        for k in cfg['diseno']:
            costos={p:n(t[k,s,p],'Costo_operativo_USD') for p in POLITICAS}
            mejor[k,s]=min(costos,key=costos.get)
            filas.append([f'{k} / {s}',*[fmt(costos[p]*Decimal(str(cfg['TRM']))/Decimal('1e9')) for p in POLITICAS],mejor[k,s].replace('_INT','')])
            if k in ('K2','K4'):
                ventajas.append(100*(costos['A_INT']-costos['D_INT'])/costos['A_INT'])
                assert mejor[k,s]=='D_INT'
            else:
                assert mejor[k,s]=='A_INT'
    valores['FACTORIAL_COSTOS']=tabla(['K / tarifa','A','B','D','C','F','Menor costo'],filas)
    valores['FACTORIAL_CONFIG']=tabla(['K','PV [MWp]','PV anual / demanda','BESS [MW/MWh]'],[
        [k,fmt(cv['pv_MWp']*v['f_PV']),fmt(100*v['f_PV'],1)+' %',fmt(v['P_MW'],0)+' / '+fmt(v['E_MWh'],0)] for k,v in cfg['diseno'].items()])
    guardar('VENTAJA_D_MIN',min(ventajas));guardar('VENTAJA_D_MAX',max(ventajas))
    guardar('FACTORIAL_BRECHA_MAX',100*vf['brecha_solver_max'],4)
    guardar('FACTORIAL_SS_K',next(Decimal(r['Fraccion_pct']) for r in dec if r['Efecto']=='Configuracion'))
    # La identidad económica explica el contraste, sin atribuir causalidad no ensayada.
    for k in ('K1','K2'):
        a,d=(t[k,'T0',p] for p in ('A_INT','D_INT'))
        term=n(d,'Gas_USD')+n(d,'CVOM_USD')-n(a,'Gas_USD')-n(a,'CVOM_USD')
        red=n(d,'Compras_red_USD')-n(a,'Compras_red_USD')
        arr=n(d,'Arranques_USD')-n(a,'Arranques_USD')
        ingreso=n(d,'Ingreso_exportacion_USD')-n(a,'Ingreso_exportacion_USD')
        dif=n(d,'Costo_operativo_USD')-n(a,'Costo_operativo_USD')
        assert abs(term+red+arr-ingreso-dif)<Decimal('.001')
        # La prosa publica direcciones, no solo magnitudes absolutas.
        assert term>0 and red<0 and ingreso>0
        assert (arr>0 and dif>0) if k=='K1' else (arr<0 and dif<0)
        for nombre,x in [('TERM',term),('RED',red),('ARR',arr),('EXP',ingreso),('NETO',dif)]:
            guardar(f'{k}_{nombre}',abs(x)/1000,1)
    for s in ('T0','T1'):
        for p in POLITICAS:
            total=n(f[s,p,'RED_TERMICA'],'VPN_COP')
            inc=n(f[s,p,'PV_SIN_BESS'],'VPN_COP')
            pv=n(f[s,'PV_SIN_BESS','RED_TERMICA'],'VPN_COP')
            assert abs(total-inc-pv)<Decimal('.01')
            guardar(f'VPN_{s}_{p}',inc/Decimal('1e9'))
        assert max(POLITICAS,key=lambda p:n(f[s,p,'PV_SIN_BESS'],'VPN_COP'))=='A_INT'
        assert all(n(f[s,p,'PV_SIN_BESS'],'VPN_COP')<0 for p in ('C_INT','F_INT'))
    valores['VPN_TABLA']=tabla(['Regla','T0 total','T0 BESS','T1 total','T1 BESS'],[
        [p.replace('_INT',''),*[fmt(n(f[s,p,ref],'VPN_COP')/Decimal('1e9')) for s in ('T0','T1') for ref in ('RED_TERMICA','PV_SIN_BESS')]] for p in POLITICAS])
    bandas=[]
    for b in ('OPT_AB','OPT_D','OPT_CF'):
        fila=[b]
        for s in ('T0','T1'):
            r=vi[s,b,'PV_SIN_BESS']
            lo=n(r,'VPN_factible_COP')/Decimal('1e9');hi=n(r,'VPN_cota_superior_COP')/Decimal('1e9')
            assert lo<=hi
            fila.append('['+fmt(lo,2,ROUND_FLOOR)+'; '+fmt(hi,2,ROUND_CEILING)+']')
        bandas.append(fila)
    valores['VPN_BANDAS']=tabla(['Espacio compatible','T0 VPN BESS','T1 VPN BESS'],bandas)
    guardar('CAPEX_PV',n(f['T0','PV_SIN_BESS','RED_TERMICA'],'CAPEX_COP')/Decimal('1e9'))
    guardar('CAPEX_BESS',n(f['T0','A_INT','PV_SIN_BESS'],'CAPEX_COP')/Decimal('1e9'))
    guardar('CAPEX_TOTAL',n(f['T0','A_INT','RED_TERMICA'],'CAPEX_COP')/Decimal('1e9'))
    assert all(Decimal(r['Renovaciones'])==0 for r in fin)
    guardar('EFC_MAX_20',max(Decimal(r['EFC_total']) for r in fin),0)
    for p in (FACTORIAL/'figuras/01_costos_reglas.png',FACTORIAL/'figuras/02_distancia_benchmarks.png',
              FACTORIAL/'figuras/03_arranques_ciclos.png',VPN/'figuras/01_vpn_total_incremental.png',
              VPN/'figuras/02_recuperacion_bess.png',VPN/'figuras/03_costos_anuales.png'):
        fuentes[p.as_posix()]=sha(raiz/p)
    return valores,exactos,fuentes


def agregar_celdas(md,code,corregido=True):
    emitir=code
    def code(texto):
        if not corregido:
            texto=texto.replace('AMP, EXACTOS_AMP, FUENTES_AMP = generar(Path.cwd())',
                "FACTORIAL=Path('resultados_factorial_termica/gas20_cvom_unidades_20260919')\n"
                "VPN=Path('resultados_vpn_termica/gas20_cvom_unidades_20260918')\n"
                "AMP, EXACTOS_AMP, FUENTES_AMP = generar(Path.cwd(),factorial=FACTORIAL,vpn=VPN)")
        emitir(texto)
    md('''
## 11 · Comparación factorial anual con térmica

Se contrasta si el orden de las reglas se conserva en K1–K4 y T0/T1, con el mismo
recurso solar, demanda, gas y CVOM. Son 40 políticas, 24 benchmarks y ocho referencias
únicas. K combina PV, potencia y energía del BESS: se trata como factor categórico,
sin atribuir su efecto exclusivamente a la capacidad de batería.

El lector verifica los hashes de los resultados completos. No importa modelos,
no ejecuta nuevas optimizaciones ni altera los archivos de las corridas.
''')
    code('''
sys.path.insert(0,str(Path.cwd()/'.build'))
from ampliacion_tesis import generar, FACTORIAL, VPN
AMP, EXACTOS_AMP, FUENTES_AMP = generar(Path.cwd())
from IPython.display import Image
display(Markdown('**Configuraciones congeladas**\\n\\n'+AMP['FACTORIAL_CONFIG']))
display(Markdown('**Costo operativo neto [mil millones COP/año]**\\n\\n'+AMP['FACTORIAL_COSTOS']))
display(Image(filename=str(FACTORIAL/'figuras/01_costos_reglas.png')))
''')
    code('''
display(Markdown('A tiene el menor costo en K1/K3; D en K2/K4, bajo ambas tarifas. '
    'La ventaja de D frente a A en K2/K4 es de '+AMP['VENTAJA_D_MIN']+'–'+AMP['VENTAJA_D_MAX']+
    ' %. Son diferencias entre trayectorias heurísticas calculadas; la brecha del solver afecta '
    'su distancia al benchmark, no ese orden entre reglas.'))
fac_comp=pd.read_csv(FACTORIAL/'comparacion.csv')
fac_comp['Distancia mínima [millones COP/año]']=fac_comp.Distancia_min_USD*META['TRM']/1e6
fac_comp['Distancia máxima [millones COP/año]']=fac_comp.Distancia_max_USD*META['TRM']/1e6
display(fac_comp[['K','Senal','Caso','Benchmark','Distancia mínima [millones COP/año]',
                  'Distancia máxima [millones COP/año]']].style.format(precision=2,decimal=',',thousands='.'))
display(Image(filename=str(FACTORIAL/'figuras/02_distancia_benchmarks.png')))
display(Image(filename=str(FACTORIAL/'figuras/03_arranques_ciclos.png')))
display(pd.read_csv(FACTORIAL/'descomposicion_ABD.csv').style.format({'Fraccion_pct':'{:.3f}'}))
''')
    md('''
**Interpretación.** Una menor importación no garantiza menor costo: puede acompañarse
de más combustible y CVOM. En K1, la descarga inmediata de A reduce más el suministro
residual; los mayores ingresos de D no compensan sus costos adicionales. En K2/K4,
la combinación de red, térmica y exportación de D produce una ventaja pequeña.
Esta lectura contable explica las diferencias observadas; no prueba el efecto aislado
de la exportación o de la capacidad. Las reglas conservan sus horizontes y limitaciones.

K domina la variación del costo absoluto de A/B/D, pero incluye recursos e inversiones
distintos. La descomposición es descriptiva, sin réplicas ni p-valores. No mide la
irrelevancia económica de elegir una regla ni demuestra un tamaño óptimo.

Dos benchmarks K2/T0 terminaron por tiempo: OPT_D (0,135 %) y OPT_CF (0,467 %).
Se conservan sus cotas. Una solución factible no se presenta como óptimo exacto.
''')
    md('''
## 12 · Comparación financiera de K3 durante veinte años

Esta sección añade inversión, OPEX fijo y renovación del almacenamiento a **K3**.
Cada una de las veinte anualidades tiene despacho horario explícito. La producción
PV disminuye 0,5 % anual; demanda, tarifas, gas y CVOM se mantienen en términos reales.
Tasa real de descuento del 10 %, sin impuestos, financiación, incentivos ni valor residual.

El VPN total compara PV+BESS con red+térmica; el incremental del BESS compara el
mismo PV con y sin batería. La térmica existente no añade inversión. El CAPEX es
común a las reglas; el OPEX fijo es 2 % del CAPEX PV y 1,5 % del CAPEX BESS por año.
La renovación se evalúa con 7.200 EFC o veinte años, sin reposición al cierre exacto.
No se modela pérdida progresiva de capacidad del BESS.
''')
    code('''
display(Markdown('**Inversión [mil millones COP]:** PV '+AMP['CAPEX_PV']+', BESS '+AMP['CAPEX_BESS']+
    ', total '+AMP['CAPEX_TOTAL']+'.\\n\\n**VPN [mil millones COP reales]**\\n\\n'+AMP['VPN_TABLA']))
display(Image(filename=str(VPN/'figuras/01_vpn_total_incremental.png')))
display(Markdown('**Intervalos del VPN incremental compatible [mil millones COP]**\\n\\n'+AMP['VPN_BANDAS']))
display(Image(filename=str(VPN/'figuras/02_recuperacion_bess.png')))
display(Image(filename=str(VPN/'figuras/03_costos_anuales.png')))
''')
    md('''
**Resultado.** A obtiene el mayor VPN entre las cinco reglas en K3. A/B/D crean valor
incremental del BESS bajo ambas tarifas. C/F tienen VPN total positivo, pero VPN
incremental negativo: la rentabilidad del PV no justifica añadir la batería bajo
cualquier regla. Ninguna trayectoria alcanza una renovación antes de cerrar el horizonte.

**Benchmarks.** PyPSA minimiza costo operativo anual; su VPN no se declara máximo
financiero. Los intervalos usan el mejor VPN factible compatible como cota inferior
y las cotas operativas, sin costo de renovación no negativo, como cota superior.
Cada año tiene SOC cíclico propio y estado térmico inicial libre. No es una trayectoria
continua de compromiso durante veinte años ni un pronóstico de precios.

**Conclusión conjunta.** No hay una regla ganadora para todas las configuraciones
operativas evaluadas. En K3, A también lidera el criterio financiero. El factorial
no permite extrapolar ese VPN a K1/K2/K4 ni seleccionar una inversión óptima.
''')
    code('''
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in FUENTES_AMP.items())
(OUT/'14_ampliacion_trazabilidad.json').write_text(json.dumps(dict(
    factorial_casos=72,vpn_despachos=362,vpn_evaluaciones=34,flujos=680,
    fuentes_sha256=FUENTES_AMP,valores_prosa=EXACTOS_AMP,optimizaciones_ejecutadas=0),
    ensure_ascii=False,indent=2),encoding='utf-8')
print('Ampliación completa: factorial y VPN validados desde archivos; cero optimizaciones.')
''')
