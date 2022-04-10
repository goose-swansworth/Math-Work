import matplotlib.pyplot as plt
from numpy import exp, array, matrix, linspace, trapz
import numpy as np
from numpy.linalg import norm, solve, det

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.style.use('ggplot')


def plot_values(x, y, xlabel, ylabel, plot_label, title, color=None, xlim=None, ylim=None):
    """General plotting function"""
    axes = plt.axes()
    if color is not None:
        axes.plot(x, y, label=plot_label, color=color)
    else:
        axes.plot(x, y, label=plot_label)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    return axes



def numerical_jacobian(x, F):
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


def newton_method_analytical_jac(F, Jf, x_0, tol, max_iterations):
    i = 0
    x_k = x_0
    x_ks = []
    while norm(F(x_k)) > tol and i < max_iterations:
        x_ks.append(x_k)
        delta = solve(Jf(x_k), -F(x_k))
        x_k = x_k + delta
        i += 1
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

def pressure_system(x):
    """P(x) Function for approximating x1, x2, x3. p and r are 3-tuples with experimental constants"""
    x1, x2, x3 = x
    p1, p2, p3 = [170.3085, 178.4465, 190.6695]
    r1, r2, r3 = [5/100, 7/100, 10/100]
    f1 = x1*exp(x2*r1) + x3*r1 - p1
    f2 = x1*exp(x2*r2) + x3*r2 - p2
    f3 = x1*exp(x2*r3) + x3*r3 - p3
    return array([f1, f2, f3])


def assignment_1():
    x0 = [100, 10, 100]
    tolerance = 10**-4
    xspace = np.linspace(-10, 10, 100)

    xks, i = newton_method_num_jac(pressure_system, x0, tolerance, 100)
    x1, x2, x3 = xks[-1]
    print(xks[-1])
    f_radius = lambda r: x1*exp(x2*r) + x3*r - 1015 # f(r) One variable function for finding radius
    f_r_prime = lambda r: x1*x2*exp(x2*r) + x3 # f'(r)
    r_spa = newton_method(f_radius, f_r_prime, 0, tolerance, 100)[0][-1]
    plt.plot(xspace, [f_radius(r) for r in xspace], label=r"$f(r)=x_1e^{x_2r} + x_3r - 1015$", color="tab:red")
    plt.axvline(r_spa, color="tab:blue", linestyle="dashed", label=r"$r_{spa}$")
    plt.axvline(0, color="tab:gray")
    plt.axhline(0, color="tab:gray")
    plt.legend(loc="best")
    print(r_spa)
    plt.show()


def recursive_func_sequnce(n, s0, i0, beta, N, gamma):
    """Helper function for question 2, computes the first n items in the Sn and In sequences"""
    Sn = lambda Sn, In: Sn - (beta/N)*Sn*In
    In = lambda Sn, In: In + (beta/N)*Sn*In - gamma*In
    Sns = [s0]
    Ins = [i0]
    for i in range(1, n+1):
        Sns.append(Sn(Sns[i - 1], Ins[i - 1]))
        Ins.append(In(Sns[i - 1], Ins[i - 1]))
    return Sns, Ins


def assignment_2_a():
    #constants
    N = 157759
    beta = 3.9928
    gamma = 3.517
    s0 = N*0.88 - 3
    i0 = 3
    T_max = 48
    Sns, Ins = recursive_func_sequnce(T_max, s0, i0, beta, N, gamma)
    print(f"Max number of I_n: {max(Ins)} at week {Ins.index(max(Ins))}")
    print(f"Total num Inf: {sum(Ins)}")
    #plot Sn and In
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Iterative SIR Model")
    ax1.plot(range(len(Sns)), Sns, label=r"$S_n$", color="tab:green")
    ax2.plot(range(len(Ins)), Ins, label=r"$I_n$", color="tab:purple")
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    plt.show()

def assignment_2_bi():
    #constants
    N = 157759
    beta = 3.9928
    gamma = 3.517
    i0 = 3
    T_max = 48
    #plot In for various values of v_prop
    plt.title("Infected numbers with different vaxination proportions")
    temp = None
    for v_prop in range(2, 14, 2):
        Ins_v = recursive_func_sequnce(T_max, (1-v_prop/100)*N-i0, i0, beta, N, gamma)[1]
        plt.plot(range(len(Ins_v)), Ins_v, label=f"{v_prop}%")
        temp = max(Ins_v)
    plt.legend(loc="best")
    plt.ylabel(r"$I_n$")
    plt.xlabel(r"$n$")
    plt.show()

def assignment_2_bii():
    #constants
    N = 157759
    beta = 3.9928
    gamma = 3.517
    i0 = 3
    s0 = N - i0
    T_max = 48
    #plot In for various values of B_q
    Ins = recursive_func_sequnce(T_max, s0, i0, beta, N, gamma)[1]
    plt.title("Infected numbers with different contact rates")
    for bq in range(2, 14, 2):
        Ins = recursive_func_sequnce(T_max, s0, i0, (1-bq/100)*beta, N, gamma)[1]
        plt.plot(range(len(Ins)), Ins, label=f"|β - γ| = {abs((1-bq/100)*beta - gamma):.3f}")
    plt.legend(loc="best")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$I_n$")
    plt.show()


def mu(gamma_dot, mu_inf_mu_0, lambd, n):
    """Non-linear function of gamma_dot, the measure of the deformation of the material"""
    return mu_inf_mu_0 + (1 - mu_inf_mu_0)*(1 + (lambd*gamma_dot)**2)**((n - 1) / 2)

def assignment_3_b():
    #constants
    mu_inf_mu_0 = 0.01
    lambd = 0
    u = lambda y: 1/2 - 1/2 * y**2 #true u(y)

    #part b
    y_f = linspace(0, 1, 201)
    gamma_f = y_f / mu(1, mu_inf_mu_0, lambd, 1) #gamma_dot and n are arbitrary when lambda = 0
    plot_values(y_f, gamma_f, r"$y_f$", r"$\dot\gamma_f$", "", "")
    plt.show()

    #part c
    u_N = [trapz(gamma_f[i:], y_f[i:]) for i in range(len(y_f))]
    max_u = max(u_N)
    plot_values(y_f, [u_N[i] / max_u for i in range(len(u_N))], r"$y_f$", r"$u(y_f) / \max\{u\}$", "", "")
    plt.show()

    #part d
    u_exact = [u(y) for y in y_f]
    error = [abs(u_N[i] - u_exact[i]) for i in range(len(u_exact))]
    plot_values(y_f, error, r"$y_f$", r"$|u_N - u|$", "", "")
    plt.show()

assignment_3_b()