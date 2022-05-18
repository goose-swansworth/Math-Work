import numpy as np
from numpy import array, exp, cos, sin, zeros, pi, arange
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from time import perf_counter
from math353 import plot_values

#//-----------------------Plotting------------------------//#
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

def plot_values(plot_tups, xlabel="", ylabel="", title="",
                color=None, xlim=None, ylim=None):
    """General plotting function"""
    axes = plt.axes()
    for tup in plot_tups:
        x, y, label = tup
        if color is not None:
            axes.plot(x, y, label=label, color=color)
        else:
            axes.plot(x, y, label=label)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    plt.legend(loc="best")
    return axes
#//-------------------------------------------------------//#


def forward_euler(f, h, t0, t_final, y0, n):
    """Apply forward euler method to with the function f: Rn->Rn
    Returns the time mesh and approximate solution as arrays as well
    as the time taken to execute"""
    start = perf_counter()
    tspace = arange(t0, t_final + h, h)
    w = zeros((len(tspace), n))
    w[0] = y0
    for i in range(len(tspace) - 1):
        w[i + 1] = w[i] + h * f(w[i], tspace[i])
    end = perf_counter()
    return tspace, w, end - start


def heun_method(f, h, t0, t_final, y0, n):
    """Apply heuns method to with the function f: Rn->Rn
    Returns the time mesh and approximate solution as arrays as well
    as the time taken to execute"""
    start = perf_counter()
    t = arange(t0, t_final + h, h)
    w = zeros((len(t), n))
    w[0] = y0
    for i in range(len(t) - 1):
        w_tilde = w[i] + h * f(w[i], t[i])
        w[i + 1] = w[i] + h/2 * (f(w[i], t[i]) + f(w_tilde, t[i + 1]))
    end = perf_counter()
    return t, w, end - start


def f_oscillator(w, t):
    """(w_1, w_2)'"""
    return array([w[1], -sin(w[0])])


def question5(method):
    h = 0.05
    t0, tf = (0, 6*pi)
    plots = []
    errors = 0
    times = 0
    for k in [-1, 0, 1]:
        y0 = [pi/10, k]
        if method == "Euler":
            t, w, time = forward_euler(f_oscillator, h, t0, tf, y0, 2)
            true_soln = odeint(f_oscillator, y0, t)
        else:
            t, w, time = heun_method(f_oscillator, h, t0, tf, y0, 2)
            true_soln = odeint(f_oscillator, y0, t)
        plots.append( (t, w[:,0], r"$y_0=(\frac{\pi}{10},$" + f"{k}" + r"$)$") )
        errors += np.max(np.abs(true_soln - w))
        times += time
    
    print(f"Average max error = {errors/3:.5f}")
    print(f"{method}: Average time taken: {(time/3)*1000:.2f}ms")
    plot_values(plots, xlabel="t", ylabel="y(t)", title=method)
    plt.grid()
    plt.show()


question5("Euler")
question5("Heun")