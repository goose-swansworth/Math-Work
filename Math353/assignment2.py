import matplotlib.pyplot as plt
import matplotx
import numpy as np
import math as m
from  numpy.linalg import solve

#/--------------------- Plotting ---------------------#/

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.style.use(matplotx.styles.ayu["light"])

def plot_values(plot_tups, xlabel="", ylabel="", title="",
                color=None, xlim=None, ylim=None,
                xticks=None, yticks=None):
    """General plotting function"""
    axes = plt.axes()
    for tup in plot_tups:
        x, y, label = tup
        axes.plot(x, y, label=label, color=color)
    axes.set_xlabel(xlabel, fontsize=10)
    axes.set_ylabel(ylabel, fontsize=10)
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

#/--------------------- Part 1 ---------------------#/

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

    # Uncomment to generate solution
    # β1 = 0
    # t, w = heun_method(SEIR_function, h, t0, tf, w0, (λ, β0, β1, γ, σ), 3)
    # write_soln_file("part_a_soln.txt", t, w)

    # β1 = 0.1
    # t, w = heun_method(SEIR_function, h, t0, tf, w0, (λ, β0, β1, γ, σ), 3)
    # write_soln_file("part_b_soln.txt", t, w)

    # β1 = 0.2
    # t, w = heun_method(SEIR_function, h, t0, tf, w0, (λ, β0, β1, γ, σ), 3)
    # write_soln_file("part_c_soln.txt", t, w)

    t = np.arange(t0, tf + h, h)
    index = m.floor(len(t) * 0.6)
    xticks = range(0, 110, 10)
    yticks = [round(x, 2) for x in np.linspace(0, 0.2, 10)]
    for part in ["a", "b", "c"]:
        S, E, I = read_soln_file(f"part_{part}_soln.txt")
        plot_values([(t, S, "S(t)"), (t, E, "E(t)"), (t, I, "I(t)")],
                    xlabel="t", xticks=(xticks, xticks), yticks=(yticks, yticks),
                    ylim=(0, 0.2))

        plt.savefig(f"part_{part}_tplot", dpi=400)

        axes = plt.figure().add_subplot(projection='3d')
        axes.plot(S[index:], E[index:], I[index:], color="tab:purple")
        axes.tick_params(axis="both", labelsize=5)
        axes.set_xlabel("S")
        axes.set_ylabel("E")
        axes.set_zlabel("I")
        plt.savefig(f"part_{part}_trajectory", dpi=400)

#/--------------------- Part 2 ---------------------#/

def part_2():
    """Create the required plots for part 2 of assignment"""
    u = lambda x: np.sin(2*x)
    u_prime = lambda x: 2*np.cos(2*x)
    diff_func = lambda u, x, h: (u(x - 2*h) -4*u(x - h) + 3*u(x)) / (2*h)

    h_values = np.logspace(-1, -5, 15)
    error = np.zeros(len(h_values))

    for i in range(len(h_values)):
        error[i] = abs(u_prime(1) - diff_func(u, 1, h_values[i]))
    
    axes = plt.axes()
    axes.plot(np.log(h_values), np.log(error), color="red")
    axes.set_xlabel("h")
    axes.set_ylabel(r"$|u'(1) - \tilde{u}'(1)|$")
    axes.grid(True)
    plt.show()

    m, b = np.polyfit(np.log(h_values), np.log(error), 1)
    print(m, b)
    #plt.savefig("part2_plot", dpi=400)

#/--------------------- Part 3 ---------------------#/

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
    print(A)
    B = build_rhs_vector(n, h, a, b)
    return solve(A, B)

def add_endpoints(array, left, right):
    """Returns a copy of an array with the value left and right added 
    to the start and end"""
    n = len(array) + 2
    out = np.zeros(n)
    for i in range(n):
        if i == 0:
            out[i] = left
        elif i == n - 1:
            out[i] = right
        else:
            out[i] = array[i - 1]
    return out

def part_3():
    """Solve the BVP in (3) and create the required plots"""
    h = 0.01
    n =  round(1 / h)
    p0, pn = 1, 1

    xspace = np.linspace(0, 1, n)
    xticks = [round(x, 1) for x in np.arange(0, 1.1, 0.1)]
    yticks = [round(y, 2) for y in np.arange(1, 1.5, 0.05)]

    s1 = lambda x: 1
    s_prime = lambda x: 0
    p1 = solve_bvp(h, n-1, s1, s_prime)
    p1 = add_endpoints(p1, p0, pn)

    s2 = lambda x: x/2 + 1
    s_prime = lambda x: 1/2
    p2 = solve_bvp(h, n-1, s2, s_prime)
    p2 = add_endpoints(p2, p0, pn)

    s3 = lambda x: x**2/2 + 1
    s_prime = lambda x: x
    p3 = solve_bvp(h, n-1, s3, s_prime)
    p3 = add_endpoints(p3, p0, pn)
   
    plot_values([(xspace, p1, r"$s(x)=1$"),
                (xspace, p2, r"$s(x)=x/2+1$"),
                (xspace, p3, r"$s(x)=x^2/2+1$")],
                xticks=(xticks, xticks), yticks=(yticks, yticks),
                xlabel=r"$x$",
                ylabel=r"$p(x)$")

    plt.savefig("partc_plot", dpi=400)
    

prod_list = [("a", 1/8, 2), ("b", 3, 1/3)]
for product in prod_list:
    name, s1, s2 = product
    print(f"Product Name: {name}, s1:{s1:.2f}, s2:{s2:.2f}")

