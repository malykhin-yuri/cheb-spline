import logging

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from common import assert_equal, assert_range, fmax
from remez import solve, get_polynomial_basis, linear_combination


def test_cheb():
    """
    Well-known Chebyshev polynomials.
    Check that the error of the best approximation to x^n equals 2^{1-n}.
    """
    N = 10
    monomials = get_polynomial_basis(N)
    for n in range(1, N):
        func = lambda xs: xs**n
        result = solve(func, -1, 1, monomials[:n], tol=1e-10, max_iter=1000)
        expect_error = 2**(1-n)
        assert_equal(expect_error, result['error'], expect_error * 1e-3)
    print("test_cheb ok!")


def test_bernstein():
    """
    It is known that lim E_{2n}(|x|)*2n exists; it is known as "Bernstein constaint"
    and equals approx 0.28
    """
    bernstein_constant_lo = 0.26
    bernstein_constant_hi = 0.29
    for n in range(5, 10):
        monomials = get_polynomial_basis(2 * n)
        result = solve(np.abs, -1, 1, monomials, tol=1e-10, max_iter=1000)
        assert_range(result['error'] * 2 * n, bernstein_constant_lo, bernstein_constant_hi)
    print("test_bernstein ok!")


if __name__ == "__main__":
    test_cheb()
    test_bernstein()
