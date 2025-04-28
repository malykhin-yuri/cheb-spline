import numpy as np
import scipy as sp


from common import assert_equal


def delta(m, h, f, x):
    k = np.arange(m + 1)
    return np.sum(sp.special.binom(m, k) * f(x + (m-k)*h) * (1.0 - 2*(k % 2)))


def ffact(n):
    prod = 1
    while n > 0:
        prod *= n
        n -= 2
    return prod


def lambda_m(m, f):
    if m <= 0:
        raise ValueError("Lambda_m not defined!")
    k = np.arange(0, m+1)
    cs = np.cos(np.pi * k / m)
    w = 1.0 - 2*(k % 2)
    w[0] *= 0.5
    w[-1] *= 0.5
    return np.sum(f(cs) * w) / m


def test_lem_3_4():
    for m in range(1, 15):
        for j in range(m+1):
            monom = lambda x: x**j
            expect = 0.0 if j < m else 2**(1-m)
            assert_equal(expect, lambda_m(m, monom))


def test_lem_3_5():
    for m in range(6, 15):
        if m % 2 == 0:
            cmpfunc = lambda x: np.tan(np.pi * x/2)
        else:
            cmpfunc = lambda x: 1/np.cos(np.pi * x/2)
        for nu in range(m):
            func = lambda x: np.maximum(x, 0)**(m-nu)
            if nu == 0:
                expect = 2**(-m)
            elif nu % 2 == 0:
                expect = 0
            else:
                nu_ = (nu - 1)/2
                expect = (-1)**nu_ * (1/m) * 2**(-m+nu-1) * delta(m-nu, 2/m, cmpfunc, -1+nu/m)

            assert_equal(expect, lambda_m(m, func))


def test_lem_3_6():
    for nu_ in range(0, 1):
        nu = 2*nu_ + 1
        for m in range(2*nu, 18*nu + 10):
            expect = (-1)**nu_ * 2 * np.pi**(-0.5) * 2**(-m+nu_) * m**(-nu_-1/2) * ffact(nu - 2)
            func = lambda x: np.maximum(x, 0)**(m-nu)
            got = lambda_m(m, func)
            print(expect / got)


def test_proof():
    test_lem_3_4()
    test_lem_3_5()
    test_lem_3_6()


def main():
    test_proof()


if __name__ == "__main__":
    main()
