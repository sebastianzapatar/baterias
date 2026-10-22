"""Compromiso térmico PyPSA equivalente al caso diario, con duración de eventos.

Cuatro unidades, paso horario, frontera inicial libre y sin parada terminal.
No modifica los resultados del motor de programación dinámica.

`pv_disponible` (opcional, MW AC por hora) agrega un generador PV sin costo
variable y con vertimiento permitido: la potencia utilizada queda entre cero y
la disponible, y sin componente de exportación toda la oferta debe igualar la
demanda. Con `pv_disponible=None` el modelo es exactamente el térmico anterior.

`bess` (opcional, exige PV) agrega un almacenamiento que **solo carga desde el
PV**. Con `bess=None` el modelo con PV es exactamente el anterior. Topología:

    PV ──PV_directo──► Demanda ◄── G1–G4, Red
     └──BESS_carga──► [BESS] ──BESS_descarga──► Demanda

Los enlaces son unidireccionales y el bus del PV no recibe ninguna otra fuente,
así que la red y la térmica no pueden cargar el BESS. No hay exportación.
Claves de `bess`: P_MW (límite AC de carga y de descarga), E_MWh (energía
nominal), eta_c, eta_d, soc_min_pu, soc_max_pu y soc0_pu (SOC inicial, que se
exige también al final del año). Carga y descarga simultáneas se excluyen con
una binaria por hora; solo hace falta en las horas con PV disponible, porque en
las demás la carga es cero por construcción.
"""
import time
import numpy as np
import pandas as pd
import pypsa
import xarray as xr

CLAVES_BESS = ('P_MW', 'E_MWh', 'eta_c', 'eta_d', 'soc_min_pu', 'soc_max_pu', 'soc0_pu')
TOL_DEFICIT_MW = 1e-9   # umbral de déficit de `despachar` en el cuaderno 02 (`def_i > 1e-9`)


def resolver_pypsa(carga, red, cv, pmin, pmax, fechas, costo_arranque=0.,
                   limite_segundos=120, brecha=0., minutos=0, pv_disponible=None, max_arranques_anuales=None,
                   aceptar_limite_tiempo=False, bess=None, bess_exclusion='binaria', solar=None, fijar=None,
                   disponible=None):
    """`solar` (opcional, excluye `pv_disponible` y `bess`): topología del bloque A/B/D de
    `REGLAS_TERMICA.md`. `carga` es la demanda residual tras el PV directo, que se descuenta
    fuera (prioridad solar). Claves: `excedente` (MW tras autoconsumo), `c_exp` (USD/MWh),
    `p_exp_max` (MW, límite común de exportación), `bess` (P_MW, E_MWh, eta_c, eta_d,
    soc_min_pu, soc_max_pu y soc0_pu si `soc='fijo'`), `exportar_bess` (bool), `soc`
    ('ciclico' o 'fijo') y `exclusion` ('binaria' o 'relajada', solo con exportación del BESS):

        Excedente PV ──► Exportacion_PV
             └──BESS_carga──► [BESS] ──BESS_descarga──► Demanda ◄── G1–G4, Red
                                  └──BESS_a_export──► Exportacion_BESS   (solo si exportar_bess)

    Por defecto, red y térmica solo llegan al bus de demanda. `solar['cargar_red']=True`
    habilita exclusivamente para C/F una ruta RedCarga→BESS, con costo de compra y límites
    compartidos. La térmica no accede a esa ruta. Esta opción exige exclusión binaria y
    añade exclusión importación/exportación. `fijar['carga_red']` se mide en MW AC.
    `potencia[:,4]` sigue siendo red a demanda, no compra total; `solar_carga_red` va aparte.
    `BESS_a_export` tiene potencia máxima cero en las horas con déficit tras el PV directo
    (`carga > TOL_DEFICIT_MW`): máscara fija derivada de las entradas, sin binarias. Es la
    posibilidad temporal de la D original, que solo exporta desde el BESS en la rama sin
    déficit solar, e impide exportar mientras red o térmica atienden demanda.
    Sin `solar` el modelo es exactamente el anterior.

    Frontera del SOC con `solar` (PyPSA 1.3.0, `define_store_constraints`):
    - `soc='fijo'`: e[0] = e_initial + p[0] y, además, e[T−1] = e_initial (restricción `BESS-solar-SOC-final`).
    - `soc='ciclico'`: e[0] = e[T−1] + p[0]; `e_initial` se ignora y no se añade ninguna restricción que fije
      el nivel. El SOC antes de la primera hora es e[T−1] (libre dentro de límites) y e[0] es el SOC al final
      de la primera hora.
    Se registran `solar_soc_modo`, `soc_inicial` (antes de la primera hora), `solar_soc_fin_primera_hora`,
    `solar_soc_final`, `solar_residuo_cierre` y `solar_residuo_primera_hora`.

    `fijar` (opcional, solo con `solar`): fija decisiones de una trayectoria externa para comprobar que pertenece
    al conjunto factible de este mismo modelo. Claves opcionales: `carga` (MW tomados del excedente),
    `descarga_demanda` (MW entregados), `descarga_export` (MW entregados al punto de exportación desde el
    BESS, solo con `exportar_bess`), `export_pv` (MW), `soc` (MWh al final de cada hora), `potencia`
    (n×5: G1..G4, Red), `compromiso` (n×4) y `tol` (banda absoluta de las igualdades, MW o MWh; 1e−7 por
    defecto). Las claves ausentes quedan libres. Sin `fijar` el modelo no cambia."""
    carga, red, cv, pmin, pmax = [np.asarray(x, float) for x in (carga, red, cv, pmin, pmax)]
    fechas = pd.DatetimeIndex(fechas, name='snapshot')
    cargos = np.broadcast_to(np.asarray(costo_arranque, float), (4,)).copy()
    if minutos not in (0, 10, 20):
        raise ValueError('Escenarios auditados: 0, 10 o 20 minutos por evento')
    if not len(carga) or carga.shape != red.shape or len(fechas) != len(carga):
        raise ValueError('Series incompatibles')
    if any(x.shape != (4,) for x in (cv, pmin, pmax)):
        raise ValueError('Se requieren cuatro unidades')
    if not all(np.isfinite(x).all() for x in (carga, red, cv, pmin, pmax, cargos)):
        raise ValueError('Entradas no finitas')
    if (carga < 0).any() or (pmin < 0).any() or (pmax <= 0).any() or (pmin > pmax).any() or (cargos < 0).any():
        raise ValueError('Parámetros físicos o cargos inválidos')
    if not fechas.is_unique or not ((fechas[1:]-fechas[:-1]) == pd.Timedelta(hours=1)).all():
        raise ValueError('Se requiere calendario horario continuo')
    pv = None if pv_disponible is None else np.asarray(pv_disponible, float)
    if pv is not None and (pv.shape != carga.shape or not np.isfinite(pv).all() or (pv < 0).any()):
        raise ValueError('PV disponible incompatible: se requieren potencias finitas, no negativas y horarias')
    limites = None
    if max_arranques_anuales is not None:
        limites = np.broadcast_to(np.asarray(max_arranques_anuales, float), (4,)).copy()
        if not np.isfinite(limites).all() or (limites < 0).any() or (limites != np.floor(limites)).any():
            raise ValueError('Límite anual: entero no negativo, escalar o cuatro valores')
    sc = None
    if solar is not None:
        if pv is not None or bess is not None:
            raise ValueError('`solar` excluye `pv_disponible` y `bess`: el PV directo ya está descontado de la carga')
        exc = np.asarray(solar['excedente'], float)
        ce = np.asarray(solar['c_exp'], float)
        if exc.shape != carga.shape or ce.shape != carga.shape or not np.isfinite(exc).all() \
                or not np.isfinite(ce).all() or (exc < 0).any():
            raise ValueError('Excedente PV o precio de exportación incompatibles')
        p_exp = float(solar.get('p_exp_max', 0.))
        if not np.isfinite(p_exp) or p_exp < 0:
            raise ValueError('p_exp_max debe ser finito y no negativo')
        bs = solar['bess']
        faltan = [k for k in CLAVES_BESS if k != 'soc0_pu' and k not in bs]
        if faltan:
            raise ValueError(f'Faltan parámetros del BESS solar: {faltan}')
        modo_soc = solar.get('soc', 'ciclico')
        if modo_soc not in ('ciclico', 'fijo'):
            raise ValueError("soc: 'ciclico' o 'fijo'")
        sc = dict(exc=exc, ce=ce, p_exp=p_exp, Pb=float(bs['P_MW']), Eb=float(bs['E_MWh']),
                  eta_c=float(bs['eta_c']), eta_d=float(bs['eta_d']), s_min=float(bs['soc_min_pu']),
                  s_max=float(bs['soc_max_pu']), soc=modo_soc,
                  s0=float(bs['soc0_pu']) if modo_soc == 'fijo' else np.nan,
                  exclusion=solar.get('exclusion', 'binaria'))
        sc['cargar_red'] = bool(solar.get('cargar_red', False))
        sc['ruta_export'] = bool(solar.get('exportar_bess', False)) and p_exp > 0
        sc['deficit'] = carga > TOL_DEFICIT_MW
        if sc['Pb'] <= 0 or sc['Eb'] <= 0 or not (0 < sc['eta_c'] <= 1 and 0 < sc['eta_d'] <= 1) \
                or not (0 <= sc['s_min'] <= sc['s_max'] <= 1):
            raise ValueError('Parámetros del BESS solar inválidos')
        if modo_soc == 'fijo' and not (sc['s_min'] <= sc['s0'] <= sc['s_max']):
            raise ValueError('SOC inicial fuera de límites')
        if sc['exclusion'] not in ('binaria', 'relajada'):
            raise ValueError("exclusion: 'binaria' o 'relajada'")
    if sc is not None and sc['cargar_red'] and sc['exclusion'] != 'binaria':
        raise ValueError('C/F requiere exclusión binaria de cargas, descargas e intercambio')
    if bess is not None:
        if pv is None:
            raise ValueError('El BESS de este modelo solo carga desde PV: se requiere pv_disponible')
        faltan = [k for k in CLAVES_BESS if k not in bess]
        if faltan:
            raise ValueError(f'Faltan parámetros del BESS: {faltan}')
        Pb, Eb, eta_c, eta_d = (float(bess[k]) for k in ('P_MW', 'E_MWh', 'eta_c', 'eta_d'))
        s_min, s_max, s0 = (float(bess[k]) for k in ('soc_min_pu', 'soc_max_pu', 'soc0_pu'))
        if Pb < 0 or Eb < 0 or not (0 < eta_c <= 1 and 0 < eta_d <= 1) or not (0 <= s_min <= s0 <= s_max <= 1):
            raise ValueError('Parámetros del BESS inválidos')
        # 'binaria': exclusión exacta con una binaria por hora con PV.
        # 'relajada': sin binarias; es una relajación del modelo binario. Si su solución no tiene
        # carga y descarga simultáneas, es factible para el binario; la optimalidad depende de la brecha.
        # Si aparece simultaneidad, se requiere una reparación validada que conserve costo y SOC, o resolver con 'binaria'.
        if bess_exclusion not in ('binaria', 'relajada'):
            raise ValueError("bess_exclusion: 'binaria' o 'relajada'")
    inicio = time.perf_counter()
    n = pypsa.Network()
    n.set_snapshots(fechas)
    n.add('Bus', 'Demanda')
    n.add('Load', 'Consumo', bus='Demanda', p_set=carga)
    unidades = ['G1', 'G2', 'G3', 'G4']
    for j, g in enumerate(unidades):
        n.add('Generator', g, bus='Demanda', p_nom=pmax[j], committable=True,
              p_min_pu=pmin[j]/pmax[j], marginal_cost=cv[j], start_up_cost=cargos[j],
              min_up_time=0, min_down_time=0, up_time_before=0, down_time_before=0)
    n.add('Generator', 'Red', bus='Demanda', p_nom=max(float(carga.max()), 1.), marginal_cost=red)
    if pv is not None and bess is None:
        # PV propio: sin costo variable; p_max_pu acota la potencia usada por la
        # disponible y la diferencia es vertimiento. No hay exportación.
        p_nom_pv = max(float(pv.max()), 1.)
        n.add('Generator', 'PV', bus='Demanda', p_nom=p_nom_pv, p_max_pu=pv/p_nom_pv, marginal_cost=0.)
    elif bess is not None:
        # PV en bus propio. La generación PV no usada es vertimiento; el bus solo
        # alimenta la demanda (directo) o el BESS. Ningún enlace entra al bus PV.
        p_nom_pv = max(float(pv.max()), 1.)
        n.add('Bus', 'PV')
        n.add('Bus', 'BESS')
        n.add('Generator', 'PV', bus='PV', p_nom=p_nom_pv, p_max_pu=pv/p_nom_pv, marginal_cost=0.)
        n.add('Link', 'PV_directo', bus0='PV', bus1='Demanda', p_nom=p_nom_pv, efficiency=1.,
              p_min_pu=0., p_max_pu=1.)
        # Potencia medida en AC: la carga toma hasta P del PV; la descarga entrega hasta P a la demanda.
        n.add('Link', 'BESS_carga', bus0='PV', bus1='BESS', p_nom=Pb, efficiency=eta_c, p_min_pu=0., p_max_pu=1.)
        n.add('Link', 'BESS_descarga', bus0='BESS', bus1='Demanda', p_nom=Pb/eta_d, efficiency=eta_d,
              p_min_pu=0., p_max_pu=1.)
        n.add('Store', 'BESS', bus='BESS', e_nom=Eb, e_min_pu=s_min, e_max_pu=s_max,
              e_initial=Eb*s0, e_cyclic=False, standing_loss=0.)
    if sc is not None:
        # El bus de excedente solo recibe el PV que sobra tras el autoconsumo; ningún enlace
        # entra a él ni al BESS desde la demanda, así que red y térmica no cargan ni exportan.
        p_nom_exc = max(float(sc['exc'].max()), 1.)
        n.add('Bus', 'Excedente')
        n.add('Bus', 'BESS')
        n.add('Generator', 'PV_excedente', bus='Excedente', p_nom=p_nom_exc,
              p_max_pu=sc['exc']/p_nom_exc, marginal_cost=0.)
        if sc['p_exp'] > 0:
            n.add('Generator', 'Exportacion_PV', bus='Excedente', sign=-1, p_nom=sc['p_exp'],
                  marginal_cost=pd.Series(-sc['ce'], index=fechas))
        n.add('Link', 'BESS_carga', bus0='Excedente', bus1='BESS', p_nom=sc['Pb'], efficiency=sc['eta_c'],
              p_min_pu=0., p_max_pu=1.)
        n.add('Link', 'BESS_descarga', bus0='BESS', bus1='Demanda', p_nom=sc['Pb']/sc['eta_d'],
              efficiency=sc['eta_d'], p_min_pu=0., p_max_pu=1.)
        if sc['ruta_export']:
            n.add('Bus', 'ExportBESS')
            # Sin exportación del BESS en horas con déficit tras el PV directo (rama de D sin déficit).
            n.add('Link', 'BESS_a_export', bus0='BESS', bus1='ExportBESS', p_nom=sc['Pb']/sc['eta_d'],
                  efficiency=sc['eta_d'], p_min_pu=0.,
                  p_max_pu=pd.Series(np.where(sc['deficit'], 0., 1.), index=fechas))
            n.add('Generator', 'Exportacion_BESS', bus='ExportBESS', sign=-1, p_nom=sc['p_exp'],
                  marginal_cost=pd.Series(-sc['ce'], index=fechas))
        if sc['cargar_red']:
            n.add('Bus', 'RedCarga')
            n.add('Generator', 'Red_BESS', bus='RedCarga', p_nom=sc['Pb'], marginal_cost=red)
            n.add('Link', 'BESS_carga_red', bus0='RedCarga', bus1='BESS', p_nom=sc['Pb'],
                  efficiency=sc['eta_c'], p_min_pu=0., p_max_pu=1.)
        n.add('Store', 'BESS', bus='BESS', e_nom=sc['Eb'], e_min_pu=sc['s_min'], e_max_pu=sc['s_max'],
              e_cyclic=sc['soc'] == 'ciclico', e_initial=sc['Eb']*sc['s0'] if sc['soc'] == 'fijo' else 0.,
              standing_loss=0.)
    m = n.optimize.create_model()
    u, y, z = [m.variables['Generator-'+a] for a in ('status', 'start_up', 'shut_down')]
    # Las transiciones nativas suponen una frontera previa fija. Se sustituyen
    # para que u[0] sea libre, sin eventos ficticios. Igualdad + exclusión
    # hace exactos los indicadores incluso con cargo cero.
    m.remove_constraints(['Generator-com-transition-start-up', 'Generator-com-transition-shut-down'])
    m.add_constraints(y.isel(snapshot=0) == 0, name='Inicial-sin-arranque')
    m.add_constraints(z.isel(snapshot=0) == 0, name='Inicial-sin-parada')
    if len(fechas) > 1:
        anterior = u.shift(snapshot=1)
        m.add_constraints((u-anterior-y+z).isel(snapshot=slice(1, None)) == 0, name='Transicion-exacta')
    m.add_constraints(y+z <= 1, name='Eventos-excluyentes')
    dias = xr.DataArray(fechas.normalize().to_numpy(), coords={'snapshot': fechas},
                        dims='snapshot', name='dia')
    m.add_constraints(y.groupby(dias).sum() <= 1, name='Un-arranque-diario')
    m.add_constraints(z.groupby(dias).sum() <= 1, name='Una-parada-diaria')
    # Versión 2: disponibilidad horaria. Una unidad no disponible no puede estar comprometida (status = 0);
    # bajar p_min_pu no bastaría, porque permitiría mantenerla comprometida sin generar para evitar el
    # cargo de rearranque. La parada forzada es una parada ordinaria y el rearranque paga su cargo.
    disp = None
    if disponible is not None:
        disp = np.asarray(disponible)
        if disp.dtype != bool:
            raise ValueError('La disponibilidad debe ser booleana')
        if disp.shape == (len(fechas),):
            disp = np.repeat(disp[:, None], 4, axis=1)
        if disp.shape != (len(fechas), 4):
            raise ValueError('La disponibilidad debe tener forma (n,) o (n, 4)')
        for j, g in enumerate(unidades):
            horas = fechas[~disp[:, j]]
            if len(horas):
                m.add_constraints(u.sel(name=g, snapshot=horas) == 0, name=f'Disponibilidad-{g}')
    if limites is not None:
        anios = xr.DataArray(fechas.year.to_numpy(), coords={'snapshot': fechas}, dims='snapshot', name='anio')
        techo = xr.DataArray(limites, coords={'name': unidades}, dims='name')
        m.add_constraints(y.groupby(anios).sum() <= techo, name='Presupuesto-arranques-anual')
    if minutos:
        # La parada al inicio de t+1 descuenta capacidad de la última hora ON t.
        # Ambos eventos se suman si la unidad permanece encendida una sola hora.
        # La última hora no recibe una parada ficticia fuera del horizonte.
        nominal = xr.DataArray(pmax, coords={'name': unidades}, dims='name')
        p = m.variables['Generator-p'].sel(name=unidades)
        tau = minutos/60.
        if len(fechas) > 1:
            lhs = p-nominal*u+tau*nominal*y+tau*nominal*z.shift(snapshot=-1)
            m.add_constraints(lhs.isel(snapshot=slice(None, -1)) <= 0,
                              name='Capacidad-eventos')
        m.add_constraints((p-nominal*u+tau*nominal*y).isel(snapshot=-1) <= 0,
                          name='Capacidad-eventos-terminal')
    n_binarias_bess = 0
    if bess is not None:
        # SOC final igual al inicial explícito: sin energía gratuita ni vaciado al cierre.
        m.add_constraints(m.variables['Store-e'].sel(name='BESS').isel(snapshot=-1) == Eb*s0,
                          name='BESS-SOC-final')
        horas = fechas[pv > 0]
        if bess_exclusion == 'binaria' and Pb > 0 and Eb > 0 and len(horas):
            # Exclusión de carga y descarga simultáneas. Fuera de `horas` el PV es cero
            # y la carga ya está acotada a cero, así que la reducción es exacta.
            horas = pd.Index(horas, name='snapshot')
            modo = m.add_variables(binary=True, coords=[horas], name='BESS-modo-carga')
            lp = m.variables['Link-p']
            m.add_constraints(lp.sel(name='BESS_carga', snapshot=horas) - Pb*modo <= 0, name='BESS-carga-si-modo')
            m.add_constraints(lp.sel(name='BESS_descarga', snapshot=horas) + (Pb/eta_d)*modo <= Pb/eta_d,
                              name='BESS-descarga-si-no-modo')
            n_binarias_bess = len(horas)
    n_binarias_solar = 0
    if sc is not None:
        lps = m.variables['Link-p']
        if sc['soc'] == 'fijo':
            m.add_constraints(m.variables['Store-e'].sel(name='BESS').isel(snapshot=-1) == sc['Eb']*sc['s0'],
                              name='BESS-solar-SOC-final')
        if sc['ruta_export']:
            gps = m.variables['Generator-p']
            # Un solo inversor: descarga a demanda y a exportación comparten la potencia.
            m.add_constraints(lps.sel(name='BESS_descarga') + lps.sel(name='BESS_a_export') <= sc['Pb']/sc['eta_d'],
                              name='BESS-solar-descarga-total')
            m.add_constraints(gps.sel(name='Exportacion_PV') + gps.sel(name='Exportacion_BESS') <= sc['p_exp'],
                              name='PCC-exportacion-total')
            # Descarga a demanda solo ocurre con residual positivo, donde el excedente es cero: no
            # coincide con la carga. La exportación del BESS sí puede coincidir con carga en horas de
            # excedente; se excluye con binaria o se repara después de la relajación.
            horas_s = fechas[sc['exc'] > 0]
            if sc['exclusion'] == 'binaria' and len(horas_s) and not sc['cargar_red']:
                horas_s = pd.Index(horas_s, name='snapshot')
                modo_s = m.add_variables(binary=True, coords=[horas_s], name='BESS-solar-modo-carga')
                m.add_constraints(lps.sel(name='BESS_carga', snapshot=horas_s) - sc['Pb']*modo_s <= 0,
                                  name='BESS-solar-carga-si-modo')
                m.add_constraints(lps.sel(name='BESS_a_export', snapshot=horas_s) + (sc['Pb']/sc['eta_d'])*modo_s
                                  <= sc['Pb']/sc['eta_d'], name='BESS-solar-exporta-si-no-modo')
                n_binarias_solar = len(horas_s)
        if sc['cargar_red']:
            # La red puede cargar en cualquier hora: ya no sirve la máscara de horas solares.
            horas_cf = pd.Index(fechas, name='snapshot')
            modo_cf = m.add_variables(binary=True, coords=[horas_cf], name='BESS-CF-modo-carga')
            ch_cf = lps.sel(name='BESS_carga') + lps.sel(name='BESS_carga_red')
            dis_cf = lps.sel(name='BESS_descarga')
            if sc['ruta_export']:
                dis_cf = dis_cf + lps.sel(name='BESS_a_export')
            m.add_constraints(ch_cf <= sc['Pb']*modo_cf, name='BESS-CF-carga-compartida')
            m.add_constraints(dis_cf <= (sc['Pb']/sc['eta_d'])*(1-modo_cf), name='BESS-CF-descarga-compartida')
            n_binarias_solar = len(fechas)
            if sc['p_exp'] > 0:
                gps = m.variables['Generator-p']
                imp_cf = gps.sel(name='Red') + gps.sel(name='Red_BESS')
                exp_cf = gps.sel(name='Exportacion_PV')
                if sc['ruta_export']:
                    exp_cf = exp_cf + gps.sel(name='Exportacion_BESS')
                modo_pcc = m.add_variables(binary=True, coords=[horas_cf], name='PCC-CF-modo-importacion')
                # Cota derivada del balance de demanda y del inversor, no un límite externo de red.
                techo_imp = xr.DataArray(carga+sc['Pb'], coords={'snapshot': fechas}, dims='snapshot')
                m.add_constraints(imp_cf <= techo_imp*modo_pcc, name='PCC-CF-importacion')
                m.add_constraints(exp_cf <= sc['p_exp']*(1-modo_pcc), name='PCC-CF-exportacion')
    if fijar is not None:
        if sc is None:
            raise ValueError('`fijar` solo se admite con `solar`')
        tf = float(fijar.get('tol', 1e-7))
        if not np.isfinite(tf) or tf < 0:
            raise ValueError('tol de `fijar` debe ser finita y no negativa')
        lpf, gpf = m.variables['Link-p'], m.variables['Generator-p']

        def banda(expr, valor, nombre):
            valor = np.asarray(valor, float)
            if valor.shape != carga.shape or not np.isfinite(valor).all():
                raise ValueError(f'fijar: {nombre} con forma o valores inválidos')
            da = xr.DataArray(valor, coords={'snapshot': fechas}, dims='snapshot')
            m.add_constraints(expr <= da + tf, name=f'Fijar-{nombre}-sup')
            m.add_constraints(expr >= da - tf, name=f'Fijar-{nombre}-inf')
        if 'carga' in fijar:
            banda(lpf.sel(name='BESS_carga'), fijar['carga'], 'carga')
        if 'carga_red' in fijar:
            if sc['cargar_red']:
                banda(lpf.sel(name='BESS_carga_red'), fijar['carga_red'], 'carga_red')
            elif np.abs(np.asarray(fijar['carga_red'], float)).max() > tf:
                raise ValueError('fijar: carga de red sin ruta habilitada')
        if 'descarga_demanda' in fijar:   # Link-p es p0 (lado BESS): entregado = η_d·p0
            banda(lpf.sel(name='BESS_descarga'), np.asarray(fijar['descarga_demanda'], float)/sc['eta_d'], 'descarga')
        if 'descarga_export' in fijar:   # entregado al punto de exportación = eta_d · p0 del enlace
            if sc['ruta_export']:
                banda(lpf.sel(name='BESS_a_export'), np.asarray(fijar['descarga_export'], float)/sc['eta_d'],
                      'descarga_export')
            elif np.abs(np.asarray(fijar['descarga_export'], float)).max() > tf:
                raise ValueError('fijar: exportación del BESS positiva sin ruta de exportación')
        if 'export_pv' in fijar:
            if sc['p_exp'] > 0:
                banda(gpf.sel(name='Exportacion_PV'), fijar['export_pv'], 'export_pv')
            elif np.abs(np.asarray(fijar['export_pv'], float)).max() > tf:
                raise ValueError('fijar: exportación PV positiva sin ruta de exportación')
        if 'soc' in fijar:
            banda(m.variables['Store-e'].sel(name='BESS'), fijar['soc'], 'soc')
        if 'potencia' in fijar:
            pf = np.asarray(fijar['potencia'], float)
            if pf.shape != (len(carga), 5):
                raise ValueError('fijar: potencia debe ser n×5')
            for j, g in enumerate(unidades + ['Red']):
                banda(gpf.sel(name=g), pf[:, j], f'p-{g}')
        if 'compromiso' in fijar:
            cf = np.asarray(fijar['compromiso'], float)
            if cf.shape != (len(carga), 4) or not np.isin(cf, (0., 1.)).all():
                raise ValueError('fijar: compromiso debe ser n×4 y exactamente binario')
            for j, g in enumerate(unidades):
                m.add_constraints(u.sel(name=g) == xr.DataArray(cf[:, j], coords={'snapshot': fechas}, dims='snapshot'),
                                  name=f'Fijar-u-{g}')
    status, condition = n.optimize.solve_model(solver_name='highs',
        solver_options={'time_limit': limite_segundos, 'mip_rel_gap': brecha,
                        'mip_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9,
                        'log_to_console': False})
    info = m.solver_model.getInfo()
    factible_por_tiempo = (aceptar_limite_tiempo and status == 'ok' and condition == 'time_limit'
                            and np.isfinite(info.objective_function_value) and info.primal_solution_status == 2)
    if condition != 'optimal' and not factible_por_tiempo:
        raise RuntimeError(f'Solución no certificada: {status}/{condition}; gap={info.mip_gap}')
    potencia = n.generators_t.p[unidades+['Red']].to_numpy()
    pv_usado = n.generators_t.p['PV'].to_numpy() if pv is not None else np.zeros(len(carga))
    if pv is not None:
        assert (pv_usado >= -1e-7).all() and (pv_usado <= pv+1e-7).all()
    extra = {}
    suministro_pv = pv_usado
    if bess is not None:
        directo = n.links_t.p0['PV_directo'].to_numpy()
        carga_b = n.links_t.p0['BESS_carga'].to_numpy()
        descarga_b = -n.links_t.p1['BESS_descarga'].to_numpy()
        soc = n.stores_t.e['BESS'].to_numpy()
        assert (directo >= -1e-7).all() and (carga_b >= -1e-7).all() and (descarga_b >= -1e-7).all()
        np.testing.assert_allclose(directo+carga_b, pv_usado, rtol=0, atol=1e-7)
        assert (carga_b <= Pb+1e-7).all() and (descarga_b <= Pb+1e-7).all()
        assert (soc >= Eb*s_min-1e-6).all() and (soc <= Eb*s_max+1e-6).all()
        previo = np.r_[Eb*s0, soc[:-1]]
        np.testing.assert_allclose(soc, previo+eta_c*carga_b-descarga_b/eta_d, rtol=0, atol=1e-6)
        assert abs(soc[-1]-Eb*s0) <= 1e-6
        suministro_pv = directo+descarga_b
        extra = dict(pv_directo=directo, bess_carga=carga_b, bess_descarga=descarga_b, soc=soc,
                     bess_binarias=n_binarias_bess, bess=dict(bess), bess_exclusion=bess_exclusion,
                     bess_horas_simultaneas=int(((carga_b > 1e-6) & (descarga_b > 1e-6)).sum()))
    ingreso_total = 0.
    costo_carga_red = np.zeros(len(carga))
    if sc is not None:
        cero = np.zeros(len(carga))
        carga_s = n.links_t.p0['BESS_carga'].to_numpy()
        carga_red_s = n.links_t.p0['BESS_carga_red'].to_numpy() if sc['cargar_red'] else cero
        carga_total_s = carga_s + carga_red_s
        costo_carga_red = carga_red_s*red
        desc_dem = -n.links_t.p1['BESS_descarga'].to_numpy()
        desc_exp = -n.links_t.p1['BESS_a_export'].to_numpy() if sc['ruta_export'] else cero
        exp_pv = n.generators_t.p['Exportacion_PV'].to_numpy() if sc['p_exp'] > 0 else cero
        exp_bess = n.generators_t.p['Exportacion_BESS'].to_numpy() if sc['ruta_export'] else cero
        exc_usado = n.generators_t.p['PV_excedente'].to_numpy()
        soc_s = n.stores_t.e['BESS'].to_numpy()
        for x in (carga_s, desc_dem, desc_exp, exp_pv, exp_bess, exc_usado):
            assert (x >= -1e-7).all()
        assert (exc_usado <= sc['exc'] + 1e-7).all()
        np.testing.assert_allclose(exc_usado, carga_s + exp_pv, rtol=0, atol=1e-7)
        np.testing.assert_allclose(exp_bess, desc_exp, rtol=0, atol=1e-7)
        assert (exp_bess[sc['deficit']] <= 1e-7).all(), 'exportación del BESS en hora con déficit'
        assert (soc_s >= sc['Eb']*sc['s_min'] - 1e-6).all() and (soc_s <= sc['Eb']*sc['s_max'] + 1e-6).all()
        soc_ini_s = float(soc_s[-1]) if sc['soc'] == 'ciclico' else sc['Eb']*sc['s0']
        previo_s = np.r_[soc_ini_s, soc_s[:-1]]
        np.testing.assert_allclose(soc_s, previo_s + sc['eta_c']*carga_total_s - (desc_dem + desc_exp)/sc['eta_d'],
                                   rtol=0, atol=1e-6)
        suministro_pv = desc_dem
        ingreso = (exp_pv + exp_bess)*sc['ce']
        ingreso_total = float(ingreso.sum())
        extra = dict(solar_carga=carga_s, solar_descarga_demanda=desc_dem, solar_descarga_export=desc_exp,
                     solar_export_pv=exp_pv, solar_vertido=sc['exc'] - exc_usado, soc=soc_s, soc_inicial=soc_ini_s,
                     ingreso_exportacion=ingreso, solar_binarias=n_binarias_solar, solar_exclusion=sc['exclusion'],
                     solar_horas_export_bloqueada=int(sc['deficit'].sum()) if sc['ruta_export'] else 0,
                     solar_horas_simultaneas=int(((carga_total_s > 1e-6) & ((desc_dem + desc_exp) > 1e-6)).sum()),
                     solar_soc_modo=sc['soc'], solar_soc_fin_primera_hora=float(soc_s[0]), solar_soc_final=float(soc_s[-1]),
                     solar_residuo_cierre=float(soc_s[-1] - soc_ini_s),
                     solar_residuo_primera_hora=float(soc_s[0] - (soc_ini_s + sc['eta_c']*carga_total_s[0]
                                                                  - (desc_dem[0] + desc_exp[0])/sc['eta_d'])),
                     solar_fijado=sorted(k for k in (fijar or {}) if k != 'tol'))
        if sc['cargar_red']:
            np.testing.assert_allclose(carga_red_s, n.generators_t.p['Red_BESS'], rtol=0, atol=1e-7)
            assert (carga_red_s >= -1e-7).all() and (carga_total_s <= sc['Pb']+1e-7).all()
            assert (desc_dem+desc_exp <= sc['Pb']+1e-7).all()
            assert not ((carga_total_s > 1e-6) & (desc_dem+desc_exp > 1e-6)).any()
            assert not ((potencia[:,4]+carga_red_s > 1e-6) & (exp_pv+desc_exp > 1e-6)).any()
            extra.update(solar_carga_red=carga_red_s, solar_importacion_total=potencia[:,4]+carga_red_s,
                         solar_costo_carga_red=costo_carga_red, solar_binarias_pcc=len(fechas) if sc['p_exp'] > 0 else 0)
    bruto = n.generators_t.status[unidades].to_numpy()
    compromiso = np.rint(bruto).astype(int)
    np.testing.assert_allclose(bruto, compromiso, rtol=0, atol=1e-7)
    delta = np.diff(np.vstack([compromiso[0], compromiso]), axis=0)
    arr, par = (delta == 1).astype(int), (delta == -1).astype(int)
    np.testing.assert_allclose(n.generators_t.start_up[unidades], arr, rtol=0, atol=1e-7)
    np.testing.assert_allclose(n.generators_t.shut_down[unidades], par, rtol=0, atol=1e-7)
    assert (potencia >= -1e-7).all()
    assert (potencia[:, :4] >= compromiso*pmin-1e-7).all()
    assert (potencia[:, :4] <= compromiso*pmax+1e-7).all()
    capacidad = pmax*(compromiso-minutos/60.*(arr+np.vstack([par[1:], np.zeros(4,int)])))
    assert (potencia[:, :4] <= capacidad+1e-7).all()
    np.testing.assert_allclose(potencia.sum(axis=1)+suministro_pv, carga, rtol=0, atol=1e-7)
    assert (pd.DataFrame(np.c_[arr, par], index=fechas.normalize()).groupby(level=0).sum().to_numpy() <= 1).all()
    if limites is not None:
        assert (pd.DataFrame(arr, index=fechas.year).groupby(level=0).sum().to_numpy() <= limites).all()
    energia = (potencia[:, :4]*cv).sum(axis=1)+potencia[:, 4]*red
    energia = energia + costo_carga_red
    cargos_h = arr@cargos
    # Con `solar`, el objetivo descuenta el ingreso de exportación; sin él, ingreso_total = 0.
    np.testing.assert_allclose(energia.sum()+cargos_h.sum()-ingreso_total, n.objective, rtol=0, atol=1e-5)
    if disp is not None:
        assert not (np.asarray(compromiso).astype(bool) & ~disp).any(), 'unidad comprometida sin disponibilidad'
        extra = dict(extra, disponible=disp)
    return dict(potencia=potencia, compromiso=compromiso, arranques=arr, paradas=par,
                costo=energia, costo_arranques=cargos_h, costo_total=energia+cargos_h,
                objetivo=float(n.objective), brecha=float(info.mip_gap),
                cota_dual=float(info.mip_dual_bound), segundos=time.perf_counter()-inicio,
                pypsa_version=pypsa.__version__, minutos=minutos, capacidad_horaria=capacidad,
                estado_solver=status, condicion_solver=condition, pv_usado=pv_usado,
                pv_vertido=(pv-pv_usado) if pv is not None else np.zeros(len(carga)), **extra)
