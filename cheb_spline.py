import numpy as np
import scipy as sp

from common import fmax
from remez import solve, get_polynomial_basis, linear_combination


def get_phi_n_r(n, r, tau_list=()):
    n_ = n // 2
    if len(tau_list) != n_:
        raise ValueError("Wrong number of knots, expected {}".format(n_))

    def phi_positive(xs):
        s = xs * 0
        s += np.maximum(xs, 0)**r
        for k, tau in enumerate(tau_list):
            s += (-1)**(k+1) * 2 * np.maximum(xs - tau, 0)**r
        return s

    def phi(xs):
        return phi_positive(xs) + (-1)**(n+r) * phi_positive(-xs)

    return phi


def get_xnr(n, r, tau_list=(), solve_kwargs=None):
    """Returns r!x_{n,r}."""
    n_ = n // 2
    if len(tau_list) != n_:
        raise ValueError("Wrong number of knots, expected {}".format(n_))
    if solve_kwargs is None:
        solve_kwargs = {}

    monomials = get_polynomial_basis(r-1)
    phi = get_phi_n_r(n, r, tau_list)
    result = solve(phi, -1, 1, monomials, tol=1e-10, max_iter=1000, **solve_kwargs)

    poly = linear_combination(result['coef'], monomials)
    def spline(xs):
        return phi(xs) - poly(xs)
    return (result['error'], spline)


def get_x23_r_tau(n, r, solve_kwargs=None):
    """Search for optimal knot tau for x_{2,r} or x_{3,r}."""
    if solve_kwargs is None:
        solve_kwargs = {}
    if n not in [2, 3]:
        raise ValueError("Only n=2 or n=3 implemented.")
    monomials = get_polynomial_basis(r - 1)
    def err(tau):
        func = get_phi_n_r(n, r, [tau])
        result = solve(func, -1, 1, monomials, tol=1e-10, max_iter=1000, **solve_kwargs)
        return -result['error']

    (tau, minus_err) = fmax(err, 0, 1, grid_size=50, coef=0.1, n_iter=10, vectorized=False)
    return tau


def get_kappas(n):
    if n == 1:
        return np.array([])
    elif n == 2:
        return np.array([sp.special.erfinv(0.5)])
    elif n == 3:
        return np.array([np.log(2)**0.5])
    else:
        raise NotImplementedError


def get_A(n):
    if n == 1:
        return np.pi**(-0.5)
    kappas = get_kappas(n)
    n_ = n // 2
    k = np.arange(n_)
    sgns = 1 - 2*(k % 2)
    return (-1)**(n_-1) * 2 * np.pi**(-0.5) * np.sum(sgns * kappas**(n-1) * np.exp(-kappas**2))
