import matplotlib.pyplot as plt
import numpy as np
from  numpy.linalg import solve
import math as m
from math353 import plot_values

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.style.use('ggplot')

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

def heun_method(f, h, t0, t_final, y0, args, n):
    """Use heuns method to solve y' = f(w, t), where f: Rn -> Rn, 
    h is the step size, (t0, t_final) is the time interval to solve over
    and y0 is the initial condition"""
    t = np.arange(t0, t_final + h, h)
    w = np.zeros((len(t), n))
    w[0] = y0
    for i in range(len(t) - 1):
        print(t[i])
        w_tilde = w[i] + h * f(w[i], t[i], *args)
        w[i + 1] = w[i] + h/2 * (f(w[i], t[i], *args) + f(w_tilde, t[i + 1], *args))
    return t, w

def SEIR_function(w, t, lambd, beta0, beta1, gamma, sigma):
    """Derivative function used in SEIR model"""
    S, E, I = w
    beta = lambda t: beta0 * (1 + beta1 * np.cos(2*np.pi*t))

    dSdt = lambd*(1 - S) - beta(t)*S*I
    dEdt = beta(t)*S*I - (lambd + sigma)*E
    dIdt = sigma*E - (gamma + lambd)*I

    return np.array([dSdt, dEdt, dIdt])

def write_soln_file(filename, t, w):
    """Write the solution to a text file"""
    S, E, I = w[:, 0], w[:, 1], w[:, 2]
    with open(filename, "x") as out_file:
        out_file.write("S,E,I\n")
        for i in range(len(t)):
            out_file.write(f"{S[i]},{E[i]},{I[i]}\n")

def read_soln_file(filename):
    """Read in soln data from file"""
    in_file = np.loadtxt(filename, delimiter=",", skiprows=1)
    return in_file[:, 0], in_file[:, 1], in_file[:, 2]



def part_1():
    """Create required plots for part 1 of assignment"""
    lambd, beta0, beta1, gamma, sigma = 0.02, 1250, 0, 73, 45.625
    h = 10**-4
    w0 = [0.9, 0.05, 0.05]
    t0, tf = 0, 100
    args = (lambd, beta0, beta1, gamma, sigma)
    #t, w = heun_method(SEIR_function, h, t0, tf, w0, args, 3)

    #write_soln_file("part1_a.txt", t, w)

    S, E, I = read_soln_file("part1_c.txt")
    t = np.arange(t0, tf + h, h)
    index = m.floor(len(t) * 0.6)

    plot_values([(t, S, "S(t)"), (t, E, "E(t)"), (t, I, "I(t)")], xlabel="t")
    plt.grid()
    plt.show()
    axes = plt.figure().add_subplot(projection='3d')
    
    axes.plot(S[index:], E[index:], I[index:])
    axes.set_xlabel("S")
    axes.set_ylabel("E")
    axes.set_zlabel("I")
    plt.show()


def part_2():
    u = lambda x: np.sin(2*x)
    u_prime = lambda x: 2*np.cos(2*x)
    diff_func = lambda u, x, h: (u(x - 2*h) -4*u(x - h) + 3*u(x)) / (2*h)

    h_values = np.logspace(-1, -15, 10)
    error = np.zeros(len(h_values))

    for i in range(len(h_values)):
        error[i] = abs(u_prime(1) - diff_func(u, 1, h_values[i]))

    plot_values([(np.log(h_values), np.log(error), "")], xlabel="h", ylabel=r"$|u'(1) - \tilde{u}'(1)}|$")
    plt.show()

def build_matrix(n, h, b):
    A = np.zeros((n-1, n-1))
    for i in range(n - 1):
        xi = (i + 1)*h
        if i == 0:
            A[i, i] = -4
            A[i, i+1] = 2 - h*b(xi)
        elif i == n - 2:
            A[i, n-3] = 2 + h*b(xi)
            A[i, n-2] = -4
        else:
            A[i, i-1] = 2 + h*b(xi)
            A[i, i] = -4
            A[i, i+1] = 2 - h*b(xi)
    return A


def build_rhs_vector(n, h, a, b):
    B = np.zeros((n - 1, 1))
    for i in range(n - 1):
        xi = (i + 1)*h
        if i == 0:
            B[i] = (2*h**2)*a(xi) - (2+h*b(xi))
        elif i == n - 2:
            B[i] = (2*h**2)*a(xi) - (2-h*b(xi))
        else:
            B[i] = (2*h**2)*a(xi)
    return B



def part_3():
    n =  6#round(1 / 0.01)
    s = lambda x: 1
    s_prime = lambda x: 0
    a = lambda x: (-6*s_prime(x)) / (s(x))**3
    b = lambda x: (-3*s_prime(x)) / (s(x))
    A = build_matrix(n, 0.01, b)
    B = build_rhs_vector(n, 0.01, a, b)
    print(solve(A, B))
    
part_3()