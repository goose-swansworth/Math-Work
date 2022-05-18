from numpy import array, exp, cos, sin, zeros, pi, linspace, arange
from scipy.integrate import odeint
from math import ceil
import matplotlib.pyplot as plt
from time import perf_counter
from math353 import *

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

def find_num_nodes(f, a, b, true_value, dec_places, verbose=False):
    """Return dx required for the smallest number of nodes required for a value correct to  dp decimal places"""
    n_nodes = 2
    approx = None
    error = float("inf")
    while error > 10**(-dec_places):
        xs = np.linspace(a, b, n_nodes)
        fs = zeros(len(xs))
        for i, x in enumerate(xs):
            fs[i] = f(x)
        approx = np.trapz(fs, xs)
        error = abs(true_value - approx)
        n_nodes += 1
        if verbose:
            print(f" I = {approx}, error = {error}, n = {n_nodes}")
    return n_nodes


def print_method_info(method_name, h, tspace, w, true_soln, time):
    print(f"~{method_name}~")
    print(f"h = {h}, ", end="")
    print(f"wf = {w[-1]:.4f}, ", end="")
    print(f"yf = {true_soln(tspace[-1]):4f}, ", end="")
    print(f"max error = {max(np.abs(array([true_soln(t) for t in tspace]) - w)):.4f}, ", end="")
    print(f"time taken: {(time)*1000:.2f}ms")


def forward_euler(f, h, t0, t_final, y0, n, true_soln=None):
    start = perf_counter()
    tspace = np.arange(t0, t_final + h, h)
    w = zeros((len(tspace), n))
    w[0] = y0
    for i in range(len(tspace) - 1):
        w[i + 1] = w[i] + h * f(w[i], tspace[i])
    end = perf_counter()
    if true_soln:
        print_method_info("Forward Euler", h, tspace, w, true_soln, end - start)
    return tspace, w, end - start


def implicit_method(phi, h, t0, t_final, y0, true_soln=None, method_name=""):
    start = perf_counter()
    tspace = np.arange(t0, t_final + h, h)
    w = zeros(len(tspace))
    w[0] = y0
    for i in range(len(tspace) - 1):
        w[i + 1] = phi(tspace[i], w[i], h)
    end = perf_counter()
    return tspace, w, end - start


def heun_method(f, h, t0, t_final, y0, n, true_soln=None):
    start = perf_counter()
    t = np.arange(t0, t_final + h, h)
    w = zeros((len(t), n))
    w[0] = y0
    for i in range(len(t) - 1):
        w_tilde = w[i] + h * f(w[i], t[i])
        w[i + 1] = w[i] + h/2 * (f(w[i], t[i]) + f(w_tilde, t[i + 1]))
    end = perf_counter()

    if true_soln:
        print_method_info("Heun Method", h, t, w, true_soln, end - start)
    return t, w, end - start


def question1():
    f = lambda x: exp(3*x) * cos(2*x)
    print(find_num_nodes(f, 0.2, 0.4, 0.40376, 5))
    g = lambda x: (sin(x))**2
    print(find_num_nodes(g, 0, pi/3, 0.30709, 5))
    h = lambda x: x**(1/3)
    print(find_num_nodes(h, 0, 1, 0.75, 5))

def question2():
    f = lambda t, w: w * cos(t)
    y = lambda t: exp(sin(t))
    forward_euler(f, 0.5, 0, 1, 1, y)
    forward_euler(f, 0.1, 0, 1, 1, y)
    forward_euler(f, 0.05, 0, 1, 1, y)

def question3():
    f = lambda y, t: (2/t)*y + (t**2)*exp(t)
    y = lambda t: (t**2)*(exp(t) - exp(1))
    phi1 = lambda t, w, h: ((t + h)/(t - h))*(w + (h*(t + h)**2)*exp(t+h))
    phi2 = lambda t, w, h: ((t+h)/t)*(w + (h/t)*w + (h/2)*(t**2)*exp(t) + exp(t+h)*(h/2)*(t+h)**2)
    h = 0.05
    t0, tf = 1, 2
    y0 = 0
    tspace = arange(t0, tf + h, h)
    w = heun_method(f, h, t0, tf, y0, 1)[1]
    # w1 = implicit_method(phi1, h, t0, tf, y0)[1]
    # w3 = heun_method(f, h, t0, tf, y0, 1)[1]
    # w3 = implicit_method(phi2, h, t0, tf, y0)[1]

    # plot_values([(tspace, y_true, "y(t)"), (tspace, w, "w(t)"), (tspace, np.abs(y_true - w[:,0]), "Error")], xlabel="t")
    # plt.show()

    hs = arange(0.001, 3, 0.001)
    errors = zeros(len(hs))
    for i in range(len(hs)):
        t, w = forward_euler(f, hs[i], t0, tf, y0, 1)[:2]
        y_true = array([y(t) for ti in t])
        errors[i] = np.max(np.abs(y_true - w[:,0]))
    plot_values([(np.log(hs), np.log(errors), "")], xlabel="h", ylabel="Max Error")
    plt.show()

def f_oscillator(w, t):
    """f Function for problem 5, t is the current time value and w is a 2d column vector [w1, w2]^T"""
    return array([w[1], -sin(w[0])])

def question5(method):
    h = 0.1
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

def lorenz(w, t):
    sigma = 10
    beta = 8/3
    pho = 28
    x, y, z = w
    dxdt = sigma * (y - x)
    dydt = x * (pho - z) - y
    dzdt = x * y - beta * z
    return array([dxdt, dydt, dzdt])

def question6():
    y0 = [1, 1, 1]
    t0, tf = 0, 100
    h = 0.01
    w = heun_method(lorenz, h, t0, tf, y0, 3)[1]
    x, y, z = w[:, 0], w[:, 1], w[:, 2]
    axes = plt.figure().add_subplot(projection='3d')
    axes.plot(x, y, z, label=f"y0={y0}")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    plt.legend(loc="best")
    plt.show()
    # print(f"y0={y0} [x, y, z](100) = {w[-1]}")
    # y0 = [1 + 10**-5, 1, 1]
    # w = heun_method(lorenz, h, t0, tf, y0, 3)[1]
    # print(f"y0={y0} [x, y, z](100) = {w[-1]}")



question6()
