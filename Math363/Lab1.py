import numpy as np
from math import ceil
from scipy.integrate import odeint
from random import uniform
from plotting import *


def solve_ode(f, args, t0, tf, h, w0):
    tspace = np.linspace(t0, tf, ceil((tf - t0)/h))
    solns = odeint(f, w0, tspace, args)
    return tspace, solns


def lorenz(w, t, σ, β, ρ):
    x, y, z = w
    dxdt = σ * (y - x)
    dydt = x * (ρ - z) - y
    dzdt = x * y - β * z
    return np.array([dxdt, dydt, dzdt])


def rossler(w, t, a, b, c):
    x, y, z = w
    dxdt = -y - z
    dydt = x + a*y
    dzdt = b + z*(x - c)
    return np.array([dxdt, dydt, dzdt])


def lotka_volterra(x, t, r1, r2, α1, α2):
    x1, x2 = x
    dx1dt = r1*x1*(1 - x1 - α1*x2)
    dx2dt = r2*x2*(1 - x2 - α2*x1)
    return np.array([dx1dt, dx2dt])


def question_1():
    y0 = [1, 1, 1]
    t0, tf = 0, 100
    h = 0.01
    args = (10, 8/3, 28)
    t, w = solve_ode(lorenz, args, t0, tf, h, y0)
    phase_plot_3d(w, y0, "tab:green")
    time_series_plot(w, t, "-")
    

def question_2():
    w0 = [2, 1, 1]
    t0, tf = 0, 100
    h = 0.01
    args = (0.1, 0.1, 14)
    t, w = solve_ode(rossler, args, t0, tf, h, w0)
    phase_plot_3d(w, w0, "tab:purple")
    time_series_plot(w, t, "-")


def question_3():
    x0 = [50, 50]
    t0, tf = 0, 2
    h = 0.01
    r1, r2, α1, α2 = 1, 0.72, 1, 1.5
    t, x = solve_ode(lotka_volterra, (r1, r2, α1, α2), t0, tf, h, x0)
    phase_plot_2d(x, x0)
    time_series_plot(x, t, "-")
    

def question_4():
    x0 = [0.1, 0.1]
    t0, tf = -5, 5
    h = 0.01
    A = np.array([[uniform(-2, 2) for _ in range(2)] for _ in range(2)])
    b = np.array([uniform(-2, 2), uniform(-2, 2)])
    linear_system = lambda x, t: np.matmul(A, x) + b 
    t, x = solve_ode(linear_system, (), t0, tf, h, x0)
    v = np.linalg.eig(A)[1]
    axes = phase_plot_2d(x, x0)
    m1 = v[0, 1] - v[0, 0]
    m2 = v[1, 1] - v[1, 0]
    axes.plot(t, m1*t, color="tab:red")
    axes.plot(t, m2*t, color="tab:green")
    plt.show()
    time_series_plot(x, t, "-")


def question_5():
    f = lambda x, t, r: r*x - x**3
    x0 = 1
    t0, tf = 0, 10
    h = 0.01
    r_range = np.linspace(-5, 5, 5)
    plots = []
    for r in r_range:
        t, x, = solve_ode(f, (r,), t0, tf, h, x0)
        tup = t, x, f"r={r:.2f}", "-"
        plots.append(tup)
    plot_values(plots, xlabel="t", ylabel="x")
    plt.show()

question_5()