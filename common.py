import numpy as np


def assert_equal(a, b, tolerance=1e-10):
    if np.abs(a-b) > tolerance:
        raise ValueError("Values not equal: {} vs {}".format(a, b))


def assert_range(x, lo, hi):
    if x < lo or x > hi:
        raise ValueError("Value {} not in range [{}, {}]".format(x, lo, hi))


def fmax(f, a0, b0, grid_size=2000, coef=0.1, n_iter=10, vectorized=True):
    a = a0
    b = b0
    for it in range(n_iter):
        l = b - a
        grid = np.linspace(a, b, grid_size)
        (x, fx) = _fmax_grid(f, grid, vectorized)
        a = max(a0, x - coef * l/2)
        b = min(b0, x + coef * l/2)
    return (x, fx)


def _fmax_grid(f, xs, vectorized):
    if vectorized:
        xv = ((x, y) for x,y in zip(xs, f(xs)))
    else:
        xv = ((x, f(x)) for x in xs)
    return max(xv, key=lambda x: x[1])
