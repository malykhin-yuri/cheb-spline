"""Remez algorithm for uniform polynomial approximation."""

from bisect import bisect_right
import logging

import numpy as np

from common import fmax


def linear_combination(coef, basis):
    # P = sum c_k * f_k(x)
    def func(xs):
        s = xs * 0
        for c, f in zip(coef, basis):
            s += c * f(xs)
        return s
    return func


def get_polynomial_basis(deg):
    def power(k):
        def func(xs):
            return xs**k
        return func
    return [power(k) for k in range(deg + 1)]


def solve(f, a, b, basis, tol=0.01, max_iter=1000, fmax_kwargs=None):
    """Solve the approximation problem:
        max_{a<=x<=b}|f(x)-P(x)| -> min
        P is a combination of basis functions (e.g. monomials)
    Args:
        f: function to approximate (np.array->np.array)
        a, b: segment of approximation
        basis: list of basis functions (np.array->np.array)
        fmax_kwargs: kwargs for inner fmax calls (e.g. increase grid size for best accuracy)
    """
    n = len(basis) - 1  # number of alternance points is n+2
    xgrid = np.linspace(a, b, n+2)
    coef = None
    if fmax_kwargs is None:
        fmax_kwargs = {}

    # Cycle invariant:
    # D >= E_n(f) >= |L|
    L = 0
    it = 0
    D = fmax(lambda xs: np.abs(f(xs)), a, b, **fmax_kwargs)[1]
    while D - np.abs(L) > tol and it < max_iter:
        it += 1
        L, coef = _alternans(f, xgrid, basis)
        P = linear_combination(coef, basis)

        # находим D=max|f-P|
        (x, D) = fmax(lambda xs: np.abs(f(xs)-P(xs)), a, b, **fmax_kwargs)
        logging.info("iter: %d", it);
        logging.info("L: %f, D: %f", L, D);
        logging.info("xgrid: %s", xgrid)

        x0 = np.array([x])
        xsign = 1 if (f(x0)-P(x0))[0] >= 0.0 else -1
        Lsign = 1 if L >= 0.0 else -1

        # we require list methods to interchange points
        xlist = list(xgrid)

        # меняем сетку
        # xlist[i] <= x < xlist[i+1]
        i = bisect_right(xlist, x) - 1
        if i == -1:
            # x левее точек сетки
            if (xsign == Lsign):  # f(x_0)-P(x_0) = L
                # знаки разности f-P в x и в x_0 совпадают, заменяем x_0->x
                xlist[0] = x
            else:
                # выкидываем x[-1], добавляем в начало x
                xlist.pop()
                xlist.insert(0, x)
        elif i == len(xlist)-1:
            # x правее точек сетки
            rs = Lsign * (1 if (n+1) % 2 == 0 else -1)   # f(x_{n+1})-P(x_{n+1}) = L*(-1)^(n+1)
            if xsign == rs:
                # заменяем x[-1]->x
                xlist[-1] = x
            else:
                xlist.pop(0)
                xlist.append(x)
        else:
            # xlist[i] <= x < xlist[i+1]
            ls = Lsign * (1 if i % 2 == 0 else -1)  # f(x_i)-P(x_i) = L*(-1)^i
            if xsign == ls:
                # заменяем x_i->x
                xlist[i] = x
            else:
                xlist[i+1] = x

        xgrid = np.array(xlist)

    return {
        "coef": coef,
        "error": D,
        "grid": xgrid,
        "iter": it,
    };


def _alternans(f, xs, basis):
    if len(xs) != len(basis) + 1:
        raise ValueError("Wrong number of alternance points!")

    # solve in P, L: f(x_k)-P(x_k)=(-1)^k*L, k = 0, ..., n-1
    # P = c_0 * f_0(x) + ... + c_{d-1} * f_{d-1}(x)
    # so, k-th equation in (L,c_0,c_1,...,c_{d-1}) becomes:
    # (-1)^k * L + f_0(x_k) * c_0 + ... + c_{d-1} * f_{d-1}(x_k) = f(x_k)
    k = np.arange(len(xs))
    signs = 1.0 - 2 * (k % 2)
    fval = []
    for fj in basis:
        fval.append(fj(xs))
    M = np.matrix([signs] + fval).transpose()
    v = np.linalg.solve(M, f(xs))
    return v[0], v[1:]
