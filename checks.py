import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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


def Lambda_m(m, f):
    if m <= 0:
        raise ValueError("Lambda_m not defined!")
    k = np.arange(0, m+1)
    cs = np.cos(np.pi * k / m)
    w = 1.0 - 2*(k % 2)
    w[0] *= 0.5
    w[-1] *= 0.5
    return np.sum(f(cs) * w) / m


def lambda_m_nu(m, nu):
    def func(tau):
        s = tau * 0
        s += 0.5 * (1-tau)**(m-nu)
        for k in range(1, m):
            sgn = 1 - 2*(k % 2)
            s += sgn * np.maximum(0, np.cos(np.pi*k/m) - tau)**(m-nu)
        return s/m
    return func

def lambda_m_nu_approx(m, nu):
    coef = 2**(-m) * m**(-nu/2)
    if nu == 0:
        def func(tau):
            return coef * (1 - sp.special.erf(tau * m**0.5))
        return func
    else:
        herm = sp.special.hermite(nu - 1)
        def func(tau):
            return coef * 2 * np.pi**(-0.5) * herm(tau * m**0.5) * np.exp(-m*tau**2)
        return func
        

def test_lem_3_4():
    for m in range(1, 15):
        for j in range(m+1):
            monom = lambda x: x**j
            expect = 0.0 if j < m else 2**(1-m)
            assert_equal(expect, Lambda_m(m, monom))


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

            assert_equal(expect, Lambda_m(m, func))


def test_lem_3_6():
    for nu_ in range(0, 1):
        nu = 2*nu_ + 1
        for m in range(2*nu, 18*nu + 10):
            expect = (-1)**nu_ * 2 * np.pi**(-0.5) * 2**(-m+nu_) * m**(-nu_-1/2) * ffact(nu - 2)
            func = lambda x: np.maximum(x, 0)**(m-nu)
            got = Lambda_m(m, func)
            print(expect / got)


def test_lem_3_8():
    m = 20
    nu = 2

    f1 = lambda_m_nu(m, nu)
    f2 = lambda_m_nu_approx(m, nu)

    taus = np.linspace(-1, 1, 200)
    plt.plot(taus, f1(taus))
    plt.plot(taus, f2(taus))
    plt.show()


def test_lem_3_11_p2():
    for m in range(5, 10):
        theta = np.linspace(0, 2 * np.pi, 1000)
        Tm = np.cos(theta * m)
        Tm4 = np.cos(theta * (m-4))
        wm = np.cos(theta)
        for k in range(m-1):
            wm *= (np.cos(theta) - np.cos(np.pi*k/(m-2)))
        print('m:', m)
        print('diff:', np.max(np.abs(wm - 2**(1-m)*(Tm-Tm4))))


def test_proof():
    test_lem_3_4()
    test_lem_3_5()
    test_lem_3_6()
    test_lem_3_8()
    test_lem_3_11_p2()

def main():
    test_proof()


if __name__ == "__main__":
    main()
