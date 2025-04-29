import logging

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from common import assert_equal, assert_range, fmax
from remez import solve, get_polynomial_basis, linear_combination
from cheb_spline import get_phi_n_r, get_x23_r_tau


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


def test_cheb_spline_one_knot():
    for n in [2, 3]:
        for r in [5, 6]:
            tau = get_x23_r_tau(n=n, r=r) 
            print('n:', n, 'r:', r, 'tau:', tau)

            func = get_phi_n_r(n, r, [tau])
            result = solve(func, -1, 1, get_polynomial_basis(r - 1), tol=1e-10, max_iter=1000)

            # check that E_{r-1} = E_{n+r-1} (due to the alternance property)
            result2 = solve(func, -1, 1, get_polynomial_basis(n + r - 1), tol=1e-10, max_iter=1000)
            assert_equal(result['error'], result2['error'], result['error']*1e-2)

    print("test_cheb_spline_one_knot ok!")


if __name__ == "__main__":
    test_cheb_spline_x1r()
    test_cheb_spline_one_knot()
