"""Ejecutor multianual V2: VPN financiero y ajustado por carbono de K3 con 20 anualidades explícitas.

Reutiliza sin duplicar lógica: el despacho y la contabilidad del ejecutor anual V2 (`termica_carbono_anual.resolver`,
que llama a los motores de A_INT, B_INT, D_INT y a `termica_vpn.resolver_base`), el calendario de disponibilidad
vigente, `carbono.vpn_financiero_y_ajustado` (motor multianual existente) y las funciones de CAPEX y OPEX de
`costos`. K3 no se redimensiona: 114,415019 MWp y 45 MW / 280 MWh.

Anualidades. Cada año `a` aplica la degradación solar vigente, escala (1 − 0,5 %)^(a−1), y se resuelve con el
escenario de carbono y la disponibilidad V2; se valida (balances, térmica, disponibilidad, cierre del SOC) y
se contabiliza (costo financiero, cargo de carbono, emisiones, ciclos). La referencia red + térmica no tiene PV:
sus entradas son idénticas todos los años, así que el año 1 se resuelve y los años 2–20 se registran como
reutilizados solo después de comprobar que la huella de sus entradas coincide. Nada se extrapola ni interpola.

Comparaciones, con las inversiones que corresponden a cada una:

    Proyecto completo   regla frente a red + térmica      CAPEX y OPEX de PV y BESS
    Incremental BESS    regla frente a PV sin BESS        CAPEX y OPEX del BESS
    Solo PV             PV sin BESS frente a red + térmica CAPEX y OPEX del PV (control de aditividad)

    VPN ajustado = VPN financiero + Σ_y p_C·(E_ref,y − E_proy,y)·TRM/(1+r)^y   (emisiones cobradas)

El VPN financiero corresponde a una operación elegida con el carbono del escenario; no es un máximo financiero.

Etapas (desde la raíz; Numba aislado opcional con PYTHONPATH=.venv/numba):

    python termica_carbono_vpn.py --salida resultados_carbono_v2/vpn_K3_20260926 --etapa anio1 --procesos 8
    python termica_carbono_vpn.py --salida ... --etapa anual --procesos N
    python termica_carbono_vpn.py --salida ... --etapa financiero
    python termica_carbono_vpn.py --salida ... --etapa verificar

Cada etapa devuelve código 1 si algún caso requerido no quedó completo; el VPN de una comparación no se calcula
si falta cualquiera de sus 40 anualidades.

Malla de dimensionamiento (26/09/2026): `tarea_anual`, `clave_anual`, `huella_entradas` y `evaluar_vpn` aceptan
`config` = {f_PV, P_MW, E_MWh} (None: K3, comportamiento anterior). `evaluar_vpn` admite además carpetas
distintas para el proyecto y la referencia y lee el `.npz` de una anualidad reutilizada desde `Ruta_npz`.
El coordinador de la malla es `termica_carbono_malla.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import carbono as C
import costos
import termica_carbono_anual as TA
import termica_politicas as POL
import termica_politicas_anual as AN

ESCENARIOS_VPN = ('V2_p0', 'V2_p50')
CASOS_RESUELTOS = TA.POLITICAS + ('PV_SIN_BESS',)
H = costos.HORIZONTE_ANIOS
DEG = costos.DEGRADACION_PV_ANUAL
PILOTO = Path('resultados_carbono_v2/piloto_K3_20260926')
CODIGO = TA.CODIGO + ('termica_carbono_vpn.py',)
COMPARACIONES = {'proyecto_completo': 'RED_TERMICA', 'incremental_BESS': 'PV_SIN_BESS'}


def escala(anio):
    return (1 - DEG) ** (anio - 1)


def etiqueta(senal, caso, anio):
    return f'{senal}_{caso}_a{anio:02d}'


def huella_entradas(nombre, senal, caso, anio, config=None):
    """Huella de todo lo que entra al despacho de una anualidad (el PV de la referencia es cero)."""
    ctx = TA.contexto()
    esc, con_disp = TA.escenario(nombre)
    pv = np.zeros(8760) if caso == 'RED_TERMICA' else TA.pv_anualidad(config, escala(anio))
    partes = [TA.sha_arreglo(pv), TA.sha_arreglo(ctx['E']['LOAD']), TA.sha_arreglo(ctx['C_IMP_COP'][senal]),
              TA.sha_arreglo(ctx['C_EXP_COP']), json.dumps(esc.clave(), sort_keys=True),
              TA.sha_arreglo(ctx['disp_v2'].to_numpy()) if con_disp else 'sin_disponibilidad', caso]
    return TA.firma(partes)


def clave_anual(nombre, senal, caso, anio, config=None):
    k = TA.clave(nombre, senal, caso, 600, 1e-6, config=config)
    k.update(anio=anio, escala_pv=escala(anio) if caso != 'RED_TERMICA' else 0.,
             entradas_anualidad=huella_entradas(nombre, senal, caso, anio, config),
             codigo={f: TA.sha_archivo(f) for f in CODIGO})
    return k


def ram_pico_MB():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return round(getattr(info, 'peak_wset', info.rss) / 1e6, 1)
    except Exception:
        return None


def tarea_anual(salida, nombre, senal, caso, anio, config=None):
    salida = Path(salida)
    carpeta = salida/nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    et = etiqueta(senal, caso, anio)
    meta = carpeta/f'{et}_meta.json'
    k = clave_anual(nombre, senal, caso, anio, config)
    fk = TA.firma(k)
    if meta.exists():
        guardado = json.loads(meta.read_text(encoding='utf-8'))
        if guardado['firma'] != fk:
            raise RuntimeError(f'{nombre}/{et}: clave distinta de la guardada; usar otra carpeta')
        return guardado
    t0 = time.perf_counter()
    if caso == 'RED_TERMICA' and anio > 1:
        m1 = carpeta/f'{etiqueta(senal, caso, 1)}_meta.json'
        if not m1.exists():
            raise RuntimeError(f'{nombre}/{et}: falta el año 1 que se reutilizaría')
        d1 = json.loads(m1.read_text(encoding='utf-8'))
        if d1.get('Estado') != 'OK' or d1['clave']['entradas_anualidad'] != k['entradas_anualidad']:
            raise RuntimeError(f'{nombre}/{et}: entradas distintas del año 1; no se puede reutilizar')
        fila = {c: v for c, v in d1.items() if c not in ('clave', 'firma', 'Segundos', 'RAM_pico_MB')}
        fila.update(Anio=anio, Etiqueta=et, Origen='reutilizada', Fuente=etiqueta(senal, caso, 1),
                    Segundos=time.perf_counter() - t0, firma=fk)
    else:
        out = TA.resolver(nombre, senal, caso, 600, 1e-6, escala_pv=escala(anio), config=config)
        ciclo = out.get('ciclo')
        fila = dict(Escenario_piloto=nombre, Senal=senal, Caso=caso, Anio=anio, Escala_PV=k['escala_pv'],
                    Etiqueta=et, Origen='resuelta', firma=fk)
        convergio = out['estado'] == 'OK' and (ciclo is None or ciclo['estado'] == 'CONVERGE')
        if ciclo is not None:
            (carpeta/f'{et}_ciclo.json').write_text(json.dumps(ciclo, ensure_ascii=False, indent=1,
                                                               default=AN.codificar), encoding='utf-8')
            fila['Ciclo'] = ciclo['estado']
        if not convergio:
            fila.update(Estado='NO_CONVERGE', Escenario_carbono=TA.escenario(nombre)[0].nombre())
        else:
            r = out['resultado']
            AN.guardar_caso(carpeta, et, r)
            cont = dict(out['contabilidad'])
            fila['Escenario_carbono'] = cont.pop('Escenario')
            b = TA.dimensionar(config)[1]
            descarga = np.maximum(np.asarray(r['Descarga_demanda']) + np.asarray(r['Descarga_export']), 0.)
            fila.update(cont, Estado='OK', Cierre_SOC_MWh=float(r['SOC'][-1] - r['SOC_inicial']),
                        SOC_inicial_MWh=float(r['SOC_inicial']), PV_MWh=float(np.sum(r['PV'])),
                        Descarga_MWh=float(descarga.sum()),
                        EFC_anio=float(descarga.sum()/(b['E_MWh']*b['DoD'])) if caso in TA.POLITICAS else 0.,
                        SHA256_npz=TA.sha_archivo(carpeta/f'{et}.npz'))
        fila.update(Segundos=time.perf_counter() - t0, RAM_pico_MB=ram_pico_MB())
    meta.write_text(json.dumps(dict(fila, clave=k), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    error = carpeta/f'{et}_error.json'
    if error.exists():
        error.replace(carpeta/f'{et}_error_resuelto.json')
    return dict(fila, clave=k)


# ------------------------------------------------------------------------------------ inventario y completitud
def inventario():
    filas = [dict(Escenario_piloto=n, Senal=s, Caso=c, Anio=a, Etiqueta=etiqueta(s, c, a))
             for n in ESCENARIOS_VPN for s in TA.SENALES for c in CASOS_RESUELTOS + ('RED_TERMICA',)
             for a in range(1, H + 1)]
    inv = pd.DataFrame(filas)
    assert not inv.duplicated(['Escenario_piloto', 'Etiqueta']).any()
    return inv


def evaluar_completitud(salida, inv):
    salida = Path(salida)
    filas = []
    for _, e in inv.iterrows():
        carpeta = salida/e.Escenario_piloto
        meta, error = carpeta/f'{e.Etiqueta}_meta.json', carpeta/f'{e.Etiqueta}_error.json'
        fila = dict(e)
        if not meta.exists():
            fila['Estado_caso'] = 'FALLIDO' if error.exists() else 'FALTANTE'
        else:
            d = json.loads(meta.read_text(encoding='utf-8'))
            k = d.get('clave', {})
            if (k.get('escenario'), k.get('senal'), k.get('caso'), k.get('anio'), d.get('Etiqueta')) != \
                    (e.Escenario_piloto, e.Senal, e.Caso, int(e.Anio), e.Etiqueta):
                fila['Estado_caso'] = 'IDENTIDAD_INCOHERENTE'
            elif d.get('Estado') != 'OK':
                fila['Estado_caso'] = 'NO_CONVERGE'
            elif e.Caso in TA.POLITICAS and abs(float(d.get('Cierre_SOC_MWh', np.inf))) > POL.TOL_CIERRE_MWh:
                fila['Estado_caso'] = 'NO_CONVERGE'
            else:
                fila['Estado_caso'] = 'COMPLETADO'
                fila['Origen'] = d.get('Origen')
        filas.append(fila)
    return pd.DataFrame(filas)


def trabajos(anios):
    t = [(n, s, c, a) for a in anios for n in ESCENARIOS_VPN for s in TA.SENALES
         for c in CASOS_RESUELTOS + (('RED_TERMICA',) if a == 1 else ())]
    return sorted(t, key=lambda x: ({'D_INT': 0, 'B_INT': 1, 'A_INT': 2}.get(x[2], 3), x[3]))


def ejecutar(lista, salida, procesos, acelerar=True):
    progreso = Path(salida)/'progreso.log'
    fallos = 0
    with ProcessPoolExecutor(max_workers=procesos, initializer=TA.inicializar, initargs=(acelerar,)) as pool:
        futuros = {pool.submit(tarea_anual, salida, *t): t for t in lista}
        for fut in as_completed(futuros):
            t = futuros[fut]
            try:
                f = fut.result()
                linea = (f"{pd.Timestamp.now():%H:%M:%S} {t[0]} {t[1]} {t[2]} a{t[3]:02d} {f['Estado']} "
                         f"{f.get('Origen', '')} {f.get('Segundos', 0):.1f}s {f.get('RAM_pico_MB')}MB")
            except Exception as e:
                fallos += 1
                carpeta = Path(salida)/t[0]
                carpeta.mkdir(parents=True, exist_ok=True)
                (carpeta/f'{etiqueta(t[1], t[2], t[3])}_error.json').write_text(json.dumps(dict(
                    Tipo=type(e).__name__, Mensaje=str(e)[:2000], Fecha=pd.Timestamp.now().isoformat()),
                    ensure_ascii=False, indent=1), encoding='utf-8')
                linea = f"{pd.Timestamp.now():%H:%M:%S} {t[0]} {t[1]} {t[2]} a{t[3]:02d} ERROR {type(e).__name__}: {e}"
            print(linea, flush=True)
            with open(progreso, 'a', encoding='utf-8') as fh:
                fh.write(linea + '\n')
    return fallos


def registrar_reutilizaciones(salida):
    """Años 2–20 de la referencia red + térmica, después de que exista su año 1 (sin procesos)."""
    for n in ESCENARIOS_VPN:
        for s in TA.SENALES:
            for a in range(2, H + 1):
                tarea_anual(salida, n, s, 'RED_TERMICA', a)


# ------------------------------------------------------------------------------------ financiero
def leer_meta(salida, n, s, c, a):
    return json.loads((Path(salida)/n/f'{etiqueta(s, c, a)}_meta.json').read_text(encoding='utf-8'))


def ruta_npz(salida, n, s, c, a):
    """`.npz` de la anualidad; si fue reutilizada de otra corrida, el metadato apunta a su archivo original."""
    m = leer_meta(salida, n, s, c, a)
    return Path(m['Ruta_npz']) if m.get('Ruta_npz') else Path(salida)/n/f'{etiqueta(s, c, a)}.npz'


def efc_horario(salida, n, s, c, a, b):
    with np.load(ruta_npz(salida, n, s, c, a), allow_pickle=False) as z:
        descarga = z['Descarga_demanda'] + z['Descarga_export']
    assert descarga.min() >= -POL.TOL_FLUJO_MW
    return np.maximum(descarga, 0.)/(b['E_MWh']*b['DoD'])


def evaluar_vpn(salida, n, s, proyecto, referencia, tabla_comp, config=None, salida_ref=None):
    """VPN financiero y ajustado de una comparación; exige las 40 anualidades completas.

    `config`: tamaños del proyecto (None: K3). `salida_ref`: carpeta de la referencia (por defecto, la misma)."""
    ctx = TA.contexto()
    trm = ctx['TRM']
    _, b, _, pv_MWp = TA.dimensionar(config)
    salida_ref = salida if salida_ref is None else salida_ref
    faltan = tabla_comp[(tabla_comp.Escenario_piloto == n) & (tabla_comp.Senal == s)
                        & tabla_comp.Caso.isin([proyecto, referencia]) & (tabla_comp.Estado_caso != 'COMPLETADO')]
    if len(faltan):
        return None, f'{len(faltan)} anualidades no completadas: {sorted(faltan.Etiqueta)[:5]}'
    tiene_bess = proyecto in TA.POLITICAS
    completo = referencia == 'RED_TERMICA'
    con_pv = completo and proyecto != 'RED_TERMICA'
    capex = (costos.capex_bess(b['P_MW'], b['E_MWh'], trm) if tiene_bess else 0.) + (costos.capex_pv(pv_MWp, trm) if con_pv else 0.)
    opex = ((costos.opex_bess_anual(b['E_MWh'], trm, b['P_MW']) if tiene_bess else 0.)
            + (costos.opex_pv_anual(pv_MWp, trm) if con_pv else 0.))
    anual = {}

    def leer(a, esc_pv):
        mp, mr = leer_meta(salida, n, s, proyecto, a), leer_meta(salida_ref, n, s, referencia, a)
        if proyecto != 'RED_TERMICA':
            assert abs(mp['Escala_PV'] - esc_pv) < 1e-12, (proyecto, a, mp['Escala_PV'], esc_pv)
        if referencia == 'PV_SIN_BESS':
            assert abs(mr['Escala_PV'] - esc_pv) < 1e-12
        assert mp['Precio_CO2_USD_t'] == mr['Precio_CO2_USD_t'] and mp['Modalidad'] == mr['Modalidad']
        anual[a] = (mp, mr)
        return dict(costo_proy_fin_USD=mp['Costo_financiero_USD'], costo_ref_fin_USD=mr['Costo_financiero_USD'],
                    emis_cobradas_proy_t=mp['Emisiones_cobradas_t'], emis_cobradas_ref_t=mr['Emisiones_cobradas_t'],
                    precio_USD_t=mp['Precio_CO2_USD_t'],
                    efc_h=efc_horario(salida, n, s, proyecto, a, b) if tiene_bess else np.zeros(8760))

    v = C.vpn_financiero_y_ajustado(leer, pv_MWp=pv_MWp, P_MW=b['P_MW'] if tiene_bess else 0.,
                                    E_MWh=b['E_MWh'] if tiene_bess else 0., DoD=b['DoD'], RTE=b['RTE'],
                                    capex_COP=capex, opex_anual_COP=opex, trm=trm)
    o = v['salida']
    tab = o['tabla'].copy()
    # Reconstrucción independiente de flujos y VPN
    tasa = costos.WACC
    filas = []
    for _, fila in tab.iterrows():
        a = int(fila.Anio)
        mp, mr = anual[a]
        ben_fin = (mr['Costo_financiero_USD'] - mp['Costo_financiero_USD'])*trm
        ben_carb = mp['Precio_CO2_USD_t']*(mr['Emisiones_cobradas_t'] - mp['Emisiones_cobradas_t'])*trm
        assert abs(ben_fin - fila.Beneficio_COP) <= 1e-6*max(1., abs(ben_fin)), (a, ben_fin, fila.Beneficio_COP)
        flujo = ben_fin - opex - fila.Costo_renovacion_COP
        assert abs(flujo - o['flujos'][a]) <= 1e-6*max(1., abs(flujo)), (a, flujo, o['flujos'][a])
        filas.append(dict(Escenario_piloto=n, Senal=s, Proyecto=proyecto, Referencia=referencia, Anio=a,
                          Escala_PV=fila.Escala_PV, Beneficio_financiero_COP=ben_fin, OPEX_fijo_COP=opex,
                          Renovacion_COP=fila.Costo_renovacion_COP, Flujo_financiero_COP=flujo,
                          Flujo_financiero_desc_COP=flujo/(1 + tasa)**a, Valor_carbono_COP=ben_carb,
                          Valor_carbono_desc_COP=ben_carb/(1 + tasa)**a, EFC_anio=fila.EFC_anio,
                          EFC_acumulado=fila.EFC_acum_despues, Renovaciones_en_anio=fila.Renovaciones_en_anio,
                          Motivo_renovacion=fila.Motivo_renovacion))
    flujos = pd.DataFrame(filas)
    vpn_fin = -capex + flujos.Flujo_financiero_desc_COP.sum()
    valor = flujos.Valor_carbono_desc_COP.sum()
    assert abs(vpn_fin - v['VPN_financiero_COP']) <= 1e-6*abs(vpn_fin) + 1e-3
    assert abs((vpn_fin + valor) - v['VPN_ajustado_COP']) <= 1e-6*abs(vpn_fin + valor) + 1e-3
    if anual[1][0]['Precio_CO2_USD_t'] == 0:
        assert v['VPN_ajustado_COP'] == v['VPN_financiero_COP'] and valor == 0.
    resumen = dict(Escenario_piloto=n, Escenario_carbono=anual[1][0]['Escenario_carbono'], Senal=s,
                   Proyecto=proyecto, Referencia=referencia, CAPEX_COP=capex, OPEX_anual_COP=opex,
                   VPN_financiero_COP=v['VPN_financiero_COP'], VPN_ajustado_COP=v['VPN_ajustado_COP'],
                   Valor_carbono_evitado_VP_COP=v['Valor_carbono_evitado_VP_COP'],
                   VPN_financiero_MUSD=v['VPN_financiero_COP']/trm/1e6, VPN_ajustado_MUSD=v['VPN_ajustado_COP']/trm/1e6,
                   TIR=o['TIR'], Payback_anios=o['Payback_anios'], Renovaciones=o['n_renovaciones'],
                   Anios_renovacion=json.dumps(o['Anios_renovacion']), EFC_total=o['EFC_total_descargado'],
                   Costo_renovaciones_VP_COP=o['Costo_renovaciones_VP_COP'])
    return (resumen, flujos), None


def financiero(salida):
    salida = Path(salida)
    comp = evaluar_completitud(salida, inventario())
    resumenes, flujos, omitidos = [], [], []
    for n in ESCENARIOS_VPN:
        for s in TA.SENALES:
            pares = [(p, ref) for p in TA.POLITICAS for ref in ('RED_TERMICA', 'PV_SIN_BESS')] + [('PV_SIN_BESS', 'RED_TERMICA')]
            for proyecto, referencia in pares:
                res, motivo = evaluar_vpn(salida, n, s, proyecto, referencia, comp)
                if res is None:
                    omitidos.append(dict(Escenario_piloto=n, Senal=s, Proyecto=proyecto, Referencia=referencia, Motivo=motivo))
                    continue
                resumenes.append(res[0])
                flujos.append(res[1])
    vpn = pd.DataFrame(resumenes)
    if len(vpn):
        # Aditividad: proyecto completo = solo PV + incremental del BESS (financiero y ajustado)
        for (n, s), g in vpn.groupby(['Escenario_piloto', 'Senal']):
            g = g.set_index(['Proyecto', 'Referencia'])
            if ('PV_SIN_BESS', 'RED_TERMICA') not in g.index:
                continue
            pv = g.loc[('PV_SIN_BESS', 'RED_TERMICA')]
            for p in TA.POLITICAS:
                if (p, 'RED_TERMICA') in g.index and (p, 'PV_SIN_BESS') in g.index:
                    for col in ('VPN_financiero_COP', 'VPN_ajustado_COP'):
                        dif = g.loc[(p, 'RED_TERMICA'), col] - pv[col] - g.loc[(p, 'PV_SIN_BESS'), col]
                        assert abs(dif) <= 1e-6*abs(g.loc[(p, 'RED_TERMICA'), col]) + 1., (n, s, p, col, dif)
    return vpn, (pd.concat(flujos, ignore_index=True) if flujos else pd.DataFrame()), pd.DataFrame(omitidos), comp


def informar(salida, comp, nombre):
    comp.to_csv(Path(salida)/f'completitud_{nombre}.csv', index=False)
    conteo = comp.Estado_caso.value_counts().to_dict()
    completo = conteo.get('COMPLETADO', 0) == len(comp)
    (Path(salida)/f'completitud_{nombre}.json').write_text(json.dumps(dict(
        Estado_entrega='COMPLETO' if completo else 'INCOMPLETO', Casos_esperados=len(comp), **conteo,
        Pendientes=comp[comp.Estado_caso != 'COMPLETADO'][['Escenario_piloto', 'Etiqueta', 'Estado_caso']].to_dict('records')),
        ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"completitud {nombre}: {'COMPLETO' if completo else 'INCOMPLETO'} {conteo}", flush=True)
    return completo


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--salida', type=Path, required=True)
    p.add_argument('--etapa', required=True, choices=('anio1', 'anual', 'financiero', 'verificar'))
    p.add_argument('--procesos', type=int, default=4)
    p.add_argument('--sin-numba', action='store_true')
    a = p.parse_args(argv)
    a.salida.mkdir(parents=True, exist_ok=True)
    if a.etapa in ('anio1', 'anual'):
        anios = [1] if a.etapa == 'anio1' else list(range(1, H + 1))
        ejecutar(trabajos(anios), a.salida, a.procesos, not a.sin_numba)
        if a.etapa == 'anual':
            registrar_reutilizaciones(a.salida)
        inv = inventario()
        inv = inv[inv.Anio == 1] if a.etapa == 'anio1' else inv
        return 0 if informar(a.salida, evaluar_completitud(a.salida, inv), a.etapa) else 1
    if a.etapa == 'verificar':
        return 0 if informar(a.salida, evaluar_completitud(a.salida, inventario()), 'total') else 1
    import termica_carbono_vpn_informe as INF
    return INF.informe(a.salida)


if __name__ == '__main__':
    sys.exit(main())
