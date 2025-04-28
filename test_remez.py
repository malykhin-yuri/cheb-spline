import logging

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from common import assert_equal, assert_range, fmax
from remez import solve, get_polynomial_basis, linear_combination
from cheb_spline import get_phi_n_r, get_x2r_tau


def test_cheb():
    N = 10
    monomials = get_polynomial_basis(N)
    for n in range(1, N):
        func = lambda xs: xs**n
        result = solve(func, -1, 1, monomials[:n], tol=1e-10, max_iter=1000)
        expect_error = 2**(1-n)
        assert_equal(expect_error, result['error'], expect_error * 1e-3)
    print("test_cheb ok!")


def test_bernstein():
    bernstein_constant_lo = 0.26
    bernstein_constant_hi = 0.29
    for n in range(5, 10):
        monomials = get_polynomial_basis(2 * n)
        result = solve(np.abs, -1, 1, monomials, tol=1e-10, max_iter=1000)
        assert_range(result['error'] * 2 * n, bernstein_constant_lo, bernstein_constant_hi)
    print("test_bernstein ok!")


def test_cheb_spline_x1r():
    # see Tikhomirov's book
    expected_norm = [0.5, 0.171, 0.0741, 0.0326, 0.0148, 0.00681, 0.00318, 0.00149, 0.000703, 0.000337]
    for r in range(1, 11):
        func = get_phi_n_r(n=1, r=r)
        expected = expected_norm[r-1]

        result = solve(func, -1, 1, get_polynomial_basis(r-1), tol=1e-10, max_iter=200)
        assert_equal(expected, result['error'], expected*1e-2)

        # E_{r-1} = E_{n+r-1}
        result2 = solve(func, -1, 1, get_polynomial_basis(r), tol=1e-10, max_iter=200)
        assert_equal(expected, result2['error'], expected*1e-2)

    print("test_cheb_spline_x1r ok!")


def test_cheb_spline_x2r():
    n = 2
    for r in [5, 6]:
        tau = get_x2r_tau(r) 
        print('n: 2, r:', r, ', tau:', tau)

        func = get_phi_n_r(n, r, [tau])
        result = solve(func, -1, 1, get_polynomial_basis(r - 1), tol=1e-10, max_iter=1000)

        # check that E_{r-1} = E_{n+r-1} (due to the alternance property)
        result2 = solve(func, -1, 1, get_polynomial_basis(n + r - 1), tol=1e-10, max_iter=1000)
        assert_equal(result['error'], result2['error'], result['error']*1e-2)

    print("test_cheb_spline_x2r ok!")


if __name__ == "__main__":
    test_cheb()
    test_bernstein()
    test_cheb_spline_x1r()
    test_cheb_spline_x2r()
