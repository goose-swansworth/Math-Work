import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from math import ceil
from math353 import plot_values

plt.style.use("ggplot")

def upper_trianglar(n, a, b):
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
        A = upper_trianglar(n-1, 2, -1)
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
    plot_values(plots, xlabel=r"$x$", ylabel=r"$u(x)$", xticks=(xticks, xticks), yticks=(xticks, xticks))
    plt.show()
    plot_values(error_plots, xlabel=r"$x$", ylabel=r"$|u(x) - u_d(x)|$", xticks=(xticks, xticks))
    plt.show()

question1()