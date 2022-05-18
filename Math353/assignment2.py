import matplotlib.pyplot as plt
import numpy as np
import math as m

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
    S, E, I = read_soln_file("part1_a.txt")
    t = np.arange(t0, tf + h, h)
    

    plot_values([(t, S, "S(t)"), (t, E, "E(t)"), (t, I, "I(t)")], xlabel="t")
    plt.show()
    axes = plt.figure().add_subplot(projection='3d')
    index = m.floor(len(t) * 0.6)
    axes.plot(S[index:], E[index:], I[index:])
    axes.set_xlabel("S")
    axes.set_ylabel("E")
    axes.set_zlabel("I")
    plt.show()


read_soln_file("part1_a.txt")


