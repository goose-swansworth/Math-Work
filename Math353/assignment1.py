import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, array, linspace, trapz
from numpy.linalg import norm, solve, LinAlgError
from random import randint

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.style.use('ggplot')

#-----------------------------------------------------------------------------#
#--------------Helper functions for ploting and root finding------------------#
#-----------------------------------------------------------------------------#

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
    plt.legend(loc="best", prop={'size': 15})
    return axes

def find_sign_changes(f, x_min, x_max, min_step):
    """Return intervals (a, b) where f changes sign"""
    found = False
    step = 1
    result_intervals = []
    while step > min_step:
        x = x_min
        while x <= x_max and not found:
            if f(x) * f(x + step) < 0:
                found = True
                result_intervals.append((x, x + step))
                x_min = x + step
            else:
                x += step
        if x > x_max and not found:
            step = step / 2
        else:
            found = False
    return result_intervals

def numerical_jacobian(x, F):
    """Numerical Approximation of Jf(x) - From Lab 2"""
    n = len(x)
    J = np.zeros((n, n))
    x_nh = x.copy()
    h = 1e-7
    for i in range(n):
        x_nh[i] = x[i] + h
        J[:,i] = (F(x_nh) - F(x)) / h
        x_nh = x.copy()
    return J

def newton_method(f, f_prime, xk, tol, max_iters):
    x_ks = []
    x_ks.append(xk)
    i = 0
    while abs(f(xk)) > tol and i < max_iters:
        xk = xk - (f(xk) / f_prime(xk))
        i += 1
        x_ks.append(xk)
    return x_ks, i

def newton_method_num_jac(F, x0, tol, max_iterations):
    i = 0
    xk = x0
    xk_s = []
    while norm(F(xk)) > tol and i < max_iterations:
        xk_s.append(xk)
        J = numerical_jacobian(xk, F)
        delta = solve(J, -F(xk))
        xk = xk + delta
        i += 1
    return xk_s, i

def falsi_helper(x_ks, f):
    """Return maximim x_k_prime such that f(x_k)f(x_k_prime) < 0"""
    i = len(x_ks) - 1
    x_k = x_ks[i]
    x_k_prime = x_ks[i - 1]
    found = False
    while not found:
        sign = f(x_k)*f(x_k_prime)
        if sign <= 0:
            found = True
        else:
            i -= 1
            x_k_prime = x_ks[i - 1]
    return x_k_prime

def regular_falsi(interval, f, tol, max_iters):
    a, b = interval
    x_ks = []
    x_ks.append(b)
    x_ks.append(a)
    x_k = a
    i = 0
    while abs(f(x_k)) > tol and i < max_iters:
        x_k_prime = falsi_helper(x_ks, f)
        qk = (f(x_k) - f(x_k_prime)) / (x_k - x_k_prime)
        x_k = x_k - (1/qk)*f(x_k)
        i += 1
        x_ks.append(x_k)
    return x_k

#-----------------------------------------------------------------------------#
#-----------------------------Assignment Code---------------------------------#
#-----------------------------------------------------------------------------#

def pressure_system(x):
    """P(x) Function for approximating x1, x2, x3. 
       p and r are 3-tuples with experimental constants"""
    x1, x2, x3 = x
    p1, p2, p3 = [170.3085, 178.4465, 190.6695]
    r1, r2, r3 = [5/100, 7/100, 10/100]
    f1 = x1*exp(x2*r1) + x3*r1 - p1
    f2 = x1*exp(x2*r2) + x3*r2 - p2
    f3 = x1*exp(x2*r3) + x3*r3 - p3
    return array([f1, f2, f3])

def assignment_1():
    tolerance = 10**-4
    xspace = linspace(-10, 10, 100)
    x0 = [randint(0, 100), randint(0, 100), randint(0, 100)]

    converged = False
    while not converged:
        try:
            xks, i = newton_method_num_jac(pressure_system, x0, tolerance, 100)
        except LinAlgError:
            x0 = [randint(0, 100), randint(0, 100), randint(0, 100)]
        else:
            converged = True
            print(x0)

    x1, x2, x3 = xks[-1]
    # f(r) One variable function for finding radius
    f_radius = lambda r: x1*exp(x2*r) + x3*r - 1015 
    # f'(r)
    f_r_prime = lambda r: x1*x2*exp(x2*r) + x3 
    r_spa = newton_method(f_radius, f_r_prime, 0, tolerance, 100)[0][-1]
    frs = [f_radius(r) for r in xspace]
    plot_values([(xspace, frs, r"$f(r)=x_1e^{x_2r} + x_3r - 1015$")], color="tab:red")
    plt.axvline(r_spa, color="tab:blue", linestyle="dashed", label=r"$r_{spa}$")
    plt.axvline(0, color="tab:gray")
    plt.axhline(0, color="tab:gray")
    plt.show()


def recursive_func_sequence(n, s0, i0, beta, N, gamma):
    """Helper function for question 2, computes the first n 
       items in the Sn and In sequences"""
    Sn = lambda Sn, In: Sn - (beta/N)*Sn*In
    In = lambda Sn, In: In + (beta/N)*Sn*In - gamma*In
    Sns = [s0]
    Ins = [i0]
    for i in range(1, n+1):
        Sns.append(Sn(Sns[i - 1], Ins[i - 1]))
        Ins.append(In(Sns[i - 1], Ins[i - 1]))
    return Sns, Ins

def assignment_2():
    #constants
    N = 157759
    beta = 3.9928
    gamma = 3.517
    i0 = 3
    s0 = N - i0
    T_max = 48
    Sns, Ins = recursive_func_sequence(T_max, s0, i0, beta, N, gamma)
    print(f"Max number of I_n: {max(Ins)} at week {Ins.index(max(Ins))}")
    print(f"Total num Inf: {sum(Ins)}")

    #part a
    plot_values([(range(len(Sns)), Sns, "")],
               xlabel=r"$n$", ylabel=r"$S_n$", color="tab:green")
    plt.show()
    plot_values([(range(len(Ins)), Ins, "")], 
               xlabel=r"$n$", ylabel=r"$I_n$", color="tab:purple")
    plt.show()

    #part b i
    plots = []
    for v_prop in range(2, 14, 2):
        prop = (1-v_prop/100)
        Ins_v = recursive_func_sequence(T_max, prop*N-i0, i0, beta, N, gamma)[1]
        plots.append((range(len(Ins_v)), Ins_v, f"{v_prop}% vaxination"))
    plot_values(plots, xlabel=r"$n$", ylabel=r"$I_n$")
    plt.show()

    #part b ii
    plots = []
    for bq in range(2, 14, 2):
        prop = (1-bq/100)
        Ins = recursive_func_sequence(T_max, s0, i0, prop*beta, N, gamma)[1]
        plots.append((range(len(Ins)), Ins, f"|β - γ| = {abs(prop*beta - gamma):.3f}"))
    plot_values(plots, xlabel=r"$n$", ylabel=r"$I_n$")
    plt.show()


def mu(gamma_dot, mu_inf_mu_0, lambd, n):
    """Non-linear function of gamma_dot, the measure of the deformation 
       of the material"""
    return mu_inf_mu_0 + (1 - mu_inf_mu_0)*(1 + (lambd*gamma_dot)**2)**((n - 1) / 2)

def assignment_3():
    #constants
    mu_inf_mu_0 = 0.01
    lambd = 0
    u = lambda y: 1/2 - 1/2 * y**2 #true u(y)

    #part b
    y_f = linspace(0, 1, 201)
    #gamma_dot and n are arbitrary when lambda = 0
    gamma_f = y_f / mu(1, mu_inf_mu_0, lambd, 1) 
    plot_values([(y_f, gamma_f, "")], r"$y_f$", r"$\dot\gamma_f$")
    plt.show()

    #part c
    u_N = [trapz(gamma_f[i:], y_f[i:]) for i in range(len(y_f))]
    max_uN = max(u_N)
    plot_values([(y_f, u_N / max_uN, r"$u(y_f)$")], 
                xlabel=r"$y_f$", ylabel=r"$u(y_f) / \max\{u\}$", color="tab:blue")
    plt.show()

    #part d
    u_exact = [u(y) for y in y_f]
    error = [abs(u_N[i] - u_exact[i]) for i in range(len(u_exact))]
    plot_values([(y_f, error, "")], 
                r"$y_f$", r"$|u_N - u|$", color="tab:green")
    plt.show()

    #part e
    lambd = 10
    n = 1/4

    # plot g
    g_first = lambda gamma_dot: mu(gamma_dot, mu_inf_mu_0, lambd, n) * gamma_dot
    gs = [g_first(gamma) for gamma in linspace(-10, 10, 100)]
    plot_values([(gs, linspace(-10, 10, 100), r"$g(\dot\gamma)$")])
    plt.show()

    # find a bound on the roots of g
    g_last = lambda gamma_dot: mu(gamma_dot, mu_inf_mu_0, lambd, n) * gamma_dot - 1
    bound = max(find_sign_changes(g_last, 0, 500, 10**-3))[1] 
    
    # find the gamma dots
    tolerance = 10**-8
    max_iterations = 100
    gamma_f = []
    for yf_i in y_f: 
        g = lambda gamma_dot: mu(gamma_dot, mu_inf_mu_0, lambd, n) * gamma_dot - yf_i
        gamma_f.append(regular_falsi((0, bound), g, tolerance, max_iterations))

    plot_values([(y_f, gamma_f, "")], 
                xlabel=r"$y_f$", ylabel=r"$\dot\gamma$", color="tab:purple")
    plt.show()

    # intergate the gamma dots over the yfs
    u_NN = array([trapz(gamma_f[i:], y_f[i:]) for i in range(len(y_f))])
    max_uNN = max(u_NN)
    plot_values([(y_f, u_NN / max_uNN, "")], 
                xlabel=r"$y_f$", ylabel=r"$u(y_f) / \max\{u\}$", color="tab:orange")
    plt.show()

    plot_values([(y_f, u_N/ max_uN, "Newtonian"), 
                (y_f, u_NN / max_uNN, "Non Newtonian")], 
                xlabel=r"$y_f$", ylabel=r"$u(y_f) / \max\{u\}$")
    plt.show()

assignment_3()