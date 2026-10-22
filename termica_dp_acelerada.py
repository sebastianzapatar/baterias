"""Aceleración opcional del paso DP, con los mismos desempates y sin fastmath.

Solo sustituye el núcleo numérico durante una corrida explícitamente activada.
El evaluador, las políticas y los validadores originales se conservan.
Requiere numba 0.67.0; sin él se puede ejecutar toda la cadena sin activar esto.
"""
import numpy as np
from numba import njit
from termica import TOL_REDONDEO_REL


@njit(cache=True, fastmath=False)
def _despacho(carga, red, cv, pmin, capacidad, u):
    n = len(carga)
    pot = np.zeros((n, 5))
    costos = np.empty(n)
    for h in range(n):
        base = np.empty(4)
        suma = 0.
        factible = True
        for j in range(4):
            base[j] = pmin[j]*u[h,j]
            suma += base[j]
            if capacidad[h,j] < base[j]-1e-12:
                factible = False
        residual = carga[h]-suma
        tol = TOL_REDONDEO_REL * max(1., abs(carga[h]), abs(suma))
        factible = factible and residual >= -tol
        c = np.empty(5)
        cap = np.empty(5)
        for j in range(4):
            c[j] = cv[j]
            cap[j] = max(capacidad[h,j]-base[j],0.)
        c[4],cap[4] = red[h],max(residual,0.)
        orden = np.arange(5)
        for j in range(1,5):
            k=j
            while k>0 and c[orden[k]] < c[orden[k-1]]:
                orden[k],orden[k-1] = orden[k-1],orden[k]
                k-=1
        acum = 0.
        for j in range(5):
            k=orden[j]
            acum += cap[k]
            previo = acum-cap[k]
            pot[h,k] = min(cap[k],max(residual-previo,0.))
        for j in range(4):
            pot[h,j] += base[j]
        costo = 0.
        for j in range(5):
            costo += pot[h,j]*c[j]
        costos[h] = costo if factible else np.inf
    return pot,costos


@njit(cache=True, fastmath=False)
def _nucleo(valor, narr, hora, destinos, adicionales, conteos):
    n = len(valor)
    nuevo = np.full(n, np.inf)
    na = np.full(n, 10**9, np.int64)
    pred = np.full(n, -1, np.int64)
    for k in range(16):
        if not np.isfinite(hora[k]):
            continue
        for s in range(n):
            d = destinos[k, s]
            if d < 0 or not np.isfinite(valor[s]):
                continue
            c = (valor[s] + adicionales[k, s]) + hora[k]
            a = narr[s] + conteos[k, s]
            if (c < nuevo[d] or (c == nuevo[d] and
                    (a < na[d] or (a == na[d] and s < pred[d])))):
                nuevo[d], na[d], pred[d] = c, a, s
    return nuevo, na, pred


def activar():
    import termica_politicas as pol
    original = pol._paso
    cache = {}

    def paso(valor, narr, hora, transicion, cargos):
        destinos, arranques = transicion
        key = (id(destinos), np.asarray(cargos).tobytes())
        if key not in cache:
            cache[key] = (arranques @ cargos, arranques.sum(axis=2))
        adicional, conteos = cache[key]
        return _nucleo(valor, narr, hora, destinos, adicional, conteos)

    pol._paso = paso
    pol.despacho_capacidad_eventos = _despacho
    return original
