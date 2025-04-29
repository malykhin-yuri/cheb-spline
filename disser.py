
import logging

import numpy as np
import matplotlib.pyplot as plt

from remez import solve, get_polynomial_basis, linear_combination
from cheb_spline import get_x23_r_tau, get_xnr, get_kappas, get_A


def main():
    r = 10
    solve_kwargs = {"fmax_kwargs": {"grid_size": 5000, "coef": 0.05, "n_iter": 8}}
    for n in [1, 2, 3]:
        kappas = get_kappas(n)
        A = get_A(n)
        tau_list = []
        if n > 1:
            tau = get_x23_r_tau(n, r, solve_kwargs=solve_kwargs)
            tau_list.append(tau)

        (norm, spline) = get_xnr(n, r, tau_list=tau_list, solve_kwargs=solve_kwargs)
        print('n:', n, 'r:', r, 'taus:', tau_list)
        print('||xnr|| / ||x0r|| =', norm * 2**(r-1))
        print('A:', A)
        print('A r^{-n/2}:', A * r**(-n/2))
        if len(tau_list):
            print('tau:', tau_list[0])
            print('tau * r^{1/2}', tau_list[0] * r**0.5)
            print('kappas:', kappas)
        print('')


if __name__ == "__main__":
    main()
