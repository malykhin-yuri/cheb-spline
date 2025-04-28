import numpy as np

from common import fmax
from remez import solve, get_polynomial_basis, linear_combination


def get_phi_n_r(n, r, tau_list=()):
    n_ = n // 2
    if len(tau_list) != n_:
        raise ValueError("Wrong number of knots!")

    def phi_positive(xs):
        s = xs * 0
        s += np.maximum(xs, 0)**r
        for k, tau in enumerate(tau_list):
            s += (-1)**(k+1) * 2 * np.maximum(xs - tau, 0)**r
        return s

    def phi(xs):
        return phi_positive(xs) + (-1)**(n+r) * phi_positive(-xs)

    return phi


def get_xnr(n, r, tau_list=()):
    """Returns r!x_{n,r}."""
    n_ = n // 2
    if len(tau_list) != n_:
        raise ValueError("Wrong number of knots!")

    monomials = get_polynomial_basis(r-1)
    result = solve(get_phi_n_r(n, r), -1, 1, monomials, tol=1e-10, max_iter=200)

    spline = linear_combination(result['coef'], monomials)
    return (result['error'], spline)


def get_x2r_tau(r):
    """Search for optimal knot tau for x_{2,r}."""
    n = 2
    monomials = get_polynomial_basis(r - 1)
    def err(tau):
        func = get_phi_n_r(n, r, [tau])
        result = solve(func, -1, 1, monomials, tol=1e-10, max_iter=1000)
        return -result['error']

    (tau, minus_err) = fmax(err, 0, 1, grid_size=20, coef=0.1, n_iter=10, vectorized=False)
    return tau
