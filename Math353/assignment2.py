import matplotlib.pyplot as plt
import matplotx
import numpy as np
import math as m
from  numpy.linalg import solve

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.style.use(matplotx.styles.ayu["light"])

def plot_values(plot_tups, xlabel="", ylabel="", title="",
                color=None, xlim=None, ylim=None, xticks=None, yticks=None):
    """General plotting function"""
    axes = plt.axes()
    for tup in plot_tups:
        x, y, label = tup
        axes.plot(x, y, label=label, color=color)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    if xticks is not None:
        ticks, labels = xticks
        axes.set_xticks(ticks)
        axes.set_xticklabels(labels)
    if yticks is not None:
        ticks, labels = yticks
        axes.set_yticks(ticks)
        axes.set_yticklabels(labels)
    plt.legend(loc="best", fontsize="large", framealpha=0.5)
    axes.grid(True)
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
    """Solve the system in (1) and create the required plots"""
    λ, β0, β1, γ, σ = 0.02, 1250, 0, 73, 45.625
    h = 10**-4
    w0 = [0.9, 0.05, 0.05]
    t0, tf = 0, 100
    args = (λ, β0, β1, γ, σ)
    S, E, I = read_soln_file("part1_c.txt")
    t = np.arange(t0, tf + h, h)
    index = m.floor(len(t) * 0.6)
    xticks = range(0, 110, 10)
    yticks = [round(x, 2) for x in np.linspace(0, 0.2, 10)]

    plot_values([(t, S, "S(t)"), (t, E, "E(t)"), (t, I, "I(t)")], xlabel="t", xticks=(xticks, xticks), yticks=(yticks, yticks), ylim=(0, 0.2))
    plt.show()
    axes = plt.figure().add_subplot(projection='3d')
    
    axes.plot(S[index:], E[index:], I[index:], color="tab:purple")
    axes.set_xlabel("S")
    axes.set_ylabel("E")
    axes.set_zlabel("I")
    plt.savefig("part_c_trajectory", dpi=400)


def part_2():
    """Create the required plots for part 2 of assignment"""
    u = lambda x: np.sin(2*x)
    u_prime = lambda x: 2*np.cos(2*x)
    diff_func = lambda u, x, h: (u(x - 2*h) -4*u(x - h) + 3*u(x)) / (2*h)

    h_values = np.logspace(-1, -15, 10)
    error = np.zeros(len(h_values))

    for i in range(len(h_values)):
        error[i] = abs(u_prime(1) - diff_func(u, 1, h_values[i]))

    plot_values([(np.log(h_values), np.log(error), "")], xlabel="h", ylabel=r"$u'(1) - \tilde{u}'(1)}$")
    plt.show()

def build_matrix(n, h, b):
    """Create the (n-1) by (n-1) matrix of coefficients for solving the BVP in (3)"""
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
    """Create the column used in the linear system in (3)"""
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


def solve_bvp(h, n, s, s_prime):
    """Solve the BVP in (3) with the given s(x)"""
    a = lambda x: (-6*s_prime(x)) / (s(x))**3
    b = lambda x: (-3*s_prime(x)) / (s(x))
    A = build_matrix(n, h, b)
    B = build_rhs_vector(n, h, a, b)
    return solve(A, B)



def part_3():
    """Solve the BVP in (3) and create the required plots"""
    h = 0.01
    n =  round(1 / h)
    xspace = np.linspace(0, 1, n-1)

    s = lambda x: 1
    s_prime = lambda x: 0
    p1 = solve_bvp(h, n, s, s_prime)

    s = lambda x: x/2 + 1
    s_prime = lambda x: 1/2
    p2 = solve_bvp(h, n, s, s_prime)

    s = lambda x: x**2/2 + 1
    s_prime = lambda x: x
    p3 = solve_bvp(h, n, s, s_prime)

    xticks = [round(x, 1) for x in np.arange(0, 1.1, 0.1)]
    yticks = [round(y, 2) for y in np.arange(1, 1.5, 0.05)]
    plot_values([(xspace, p1, r"$s(x)=1$"),
                (xspace, p2, r"$s(x)=x/2+1$"),
                (xspace, p3, r"$s(x)=x^2/2+1$")],
                xticks=(xticks, xticks), yticks=(yticks, yticks))
    plt.show()
    
part_1()