import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from math import ceil

#/________________________Plotting________________________/#

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.style.use("ggplot")

def plot_values(plot_tups, xlabel="", ylabel="", title="",
                color=None, xlim=None, ylim=None,
                xticks=None, yticks=None):
    """General plotting function"""
    axes = plt.axes()
    for tup in plot_tups:
        x, y, label, f_str = tup
        axes.plot(x, y, f_str, label=label, color=color)
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
    plt.legend(loc="best", framealpha=0.5)
    axes.grid(True)
    return axes

#/________________________Question 1________________________/#

def tri_diagonal(n, a, b):
    A = np.zeros((n - 1, n - 1))
    for i in range(len(A)):
        if i == 0:
            A[i][i] = a
            A[i][i + 1] = b
        elif i == len(A) - 1:
            A[i][i - 1] = b
            A[i][i] = a
        else:
            A[i][i - 1] = b
            A[i][i] = a
            A[i][i + 1] = b
    return A
    

def question1():
    u_true = lambda x: -np.exp(x) + np.exp(1)*x + 1
    plots = []
    error_plots = []

    for h in [0.1, 0.01]:
        n = ceil(1 / h)
        xspace = np.linspace(0, 1, n)

        #The true solution
        u = np.array([u_true(x) for x in xspace])

        #Solve the BVP
        f = lambda x: np.exp(x)
        A = tri_diagonal(n-1, 2, -1)
        b = np.array([(h**2)*f(x) for x in xspace[1:-1]])
        b[-1] += 1
        u_tilde = solve(A, b)
        u_tilde = np.insert(u_tilde, 0, [0])
        u_tilde = np.append(u_tilde, [1])

        plots.append((xspace, u_tilde, f"h={h}", "-"))
        error_plots.append((xspace, np.abs(u - u_tilde), f"Error h={h}", "-"))
        print(f"max error = {np.max(np.abs(u - u_tilde))}")
        if h == 0.01:
            plots.append((xspace, u, "True", "-"))
        
    xticks = np.round(np.arange(0, 1.1, 0.1), 2)
    plot_values(plots, xlabel=r"$x$", ylabel=r"$u(x)$",
                xticks=(xticks, xticks), yticks=(xticks, xticks))
    plt.show()
    plot_values(error_plots, xlabel=r"$x$", ylabel=r"$|u(x) - u_d(x)|$",
                xticks=(xticks, xticks))
    plt.show()

def question2():
    u = lambda x: np.sin(x) / np.sin(1)
    plots = []
    for h in [0.1, 0.01]:
        n = ceil(1 / h)
        xspace = np.linspace(0, 1, n)
        b = [0 for _ in range(n - 2)]
        b[-1] = 1
        A = tri_diagonal(n-1, 2+h**2, -1)
        u_tilde = solve(A, b)
        plots.append((xspace, u_tilde, f"h={h}", "-"))

        if h == 0.01:
            u_exact = [u(x) for x in xspace]
            plots.append((xspace, u_exact, "Exact", "-"))
    plot_values(plots, xlabel=r"$x$", ylabel=r"$u(x)$")
    plt.show()