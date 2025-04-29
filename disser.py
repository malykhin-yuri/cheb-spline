import logging
import pickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from remez import solve, get_polynomial_basis, linear_combination
from cheb_spline import get_phi_n_r, get_x23_r_tau, get_xnr, get_kappas, get_A


def save_spline_coef(output_file):
    n = 2
    r = 5
    solve_kwargs = {"fmax_kwargs": {"grid_size": 5000, "coef": 0.05, "n_iter": 8}}
    tau = get_x23_r_tau(n, r, solve_kwargs=solve_kwargs)
    result = solve(get_phi_n_r(n, r, [tau]), -1, 1, get_polynomial_basis(r-1), tol=1e-10, max_iter=1000, **solve_kwargs)
    
    output = {
        "n": n,
        "r": r,
        "tau": tau,
        "coef": result["coef"],
        "error": result["error"],
    }
    with open(output_file, 'wb') as fh:
        pickle.dump(output, fh)


def draw_plot(input_file, output_file):
    with open(input_file, 'rb') as fh:
        data = pickle.load(fh)

    n, r, tau = data["n"], data["r"], data["tau"]
    phi = get_phi_n_r(n, r, [tau])
    monomials = get_polynomial_basis(r-1)
    poly = linear_combination(data["coef"], monomials)

    def xnr(xs):
        return (phi(xs) - poly(xs)) / sp.special.factorial(r)

    norm = data["error"] / sp.special.factorial(r)

    x = np.linspace(-1, 1, 1000)
    y = xnr(x)
    yr = 1 - 2 * np.sign(np.maximum(x - tau, 0)) - 2 * np.sign(np.maximum((-x) - tau, 0))
    
    fig, axs = plt.subplots(2, 1, figsize=(6, 9), dpi=600)
    ax0 = axs[0]
    ax0.set_title('$x_{n,r}$')
    ax0.set_yticks([-norm, 0.0, norm])
    ax0.set_xticks([-1.0, 0.0, 1.0])
    ax0.grid()
    ax0.plot(x, y)

    ax1 = axs[1]
    ax1.set_title('$x_{n,r}^{(r)}$')
    ax1.set_yticks([-1.0, 0.0, 1.0])
    ax1.set_xticks(np.round([-tau, tau], decimals=3))
    ax1.grid()
    ax1.plot(x, yr)

    fig.savefig(output_file, bbox_inches='tight', pad_inches=0.1)


def print_values():
    solve_kwargs = {"fmax_kwargs": {"grid_size": 5000, "coef": 0.05, "n_iter": 8}}
    for n in [1, 2, 3]:
        for r in [5, 10]:
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
    print_values()
    save_spline_coef("x_2_5.dump")
    draw_plot("x_2_5.dump", "x25.png")
