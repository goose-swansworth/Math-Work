from cProfile import label
from unittest import runner
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from math import ceil
from math353 import plot_values

e = lambda x: np.exp(x)
 
plt.style.use("ggplot")


def tri_diagonal(n, left, mid, right):
    A = np.zeros((n - 1, n - 1))
    for i in range(len(A)):
        if i == 0:
            A[i][i] = mid
            A[i][i + 1] = right
        elif i == len(A) - 1:
            A[i][i - 1] = left
            A[i][i] = mid
        else:
            A[i][i - 1] = left
            A[i][i] = mid
            A[i][i + 1] = right
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
    plot_values(plots, xlabel=r"$x$", ylabel=r"$u(x)$", xticks=(xticks, xticks), yticks=(xticks, xticks))
    plt.savefig("Question1_plot", dpi=400)
    plt.show()
    plot_values(error_plots, xlabel=r"$x$", ylabel=r"$|u(x) - u_d(x)|$", xticks=(xticks, xticks))
    plt.savefig("Question1_error_plot", dpi=400)

def question2():
    u = lambda x: np.sin(x) / np.sin(1)
    plots = []
    for h in [0.1, 0.01]:
        n = ceil(1 / h)
        xspace = np.linspace(0, 1, n)
        b = [0 for _ in range(n - 2)]
        b[-1] = 1
        print(b)
        A = tri_diagonal(n-1, 2-h**2, -1)
        print(A)
        u_tilde = solve(A, b)
        u_tilde = np.append(u_tilde, 1)
        u_tilde = np.insert(u_tilde, 0, 0)
        plots.append((xspace, u_tilde, f"h={h}", "-"))

        if h == 0.01:
            u_exact = [u(x) for x in xspace]
            plots.append((xspace, u_exact, "Exact", "-"))
    plot_values(plots, xlabel=r"$x$", ylabel=r"$u(x)$")
    plt.show()

def question3():
    plots = []
    ϵ = 1/250
    β = 1 
    u = lambda x: (1/(e(β/ϵ) - 1 ))*e((β/ϵ)*x) - 1/((e(β/ϵ)) - 1)
    for h in [0.01]:
        n = ceil(1 / h)
        xspace = np.linspace(0, 1, n)

        #Centered difference
        b = [0 for _ in range(n - 2)]
        b[-1] = -2*ϵ + h*β
        A = tri_diagonal(n-1, -2*ϵ-h*β, 4*ϵ, -2*ϵ+h*β)
        u_tilde = solve(A, b)
        u_tilde = np.append(u_tilde, 1)
        u_tilde = np.insert(u_tilde, 0, 0)
        plots.append((xspace, u_tilde, f"Centered", ""))

        #Upwind Difference
        b = [0 for _ in range(n - 2)]
        b[-1] = ϵ
        A = tri_diagonal(n-1, -ϵ-h*β, 2*ϵ+h*β, -ϵ)
        u_tilde = solve(A, b)
        u_tilde = np.append(u_tilde, 1)
        u_tilde = np.insert(u_tilde, 0, 0)
        plots.append((xspace, u_tilde, f"Upwind", ""))

        if h == 0.01:
            u_exact = [u(x) for x in xspace]
            plots.append((xspace, u_exact, "Exact", "-"))
    plot_values(plots, xlabel=r"$x$", ylabel=r"$u(x)$")
    plt.show()
    

def question4():
    plots = []
    ϵ = 1/250
    β = 1 
    u = lambda x: (1/(e(β/ϵ) - 1 ))*e((β/ϵ)*x) - 1/((e(β/ϵ)) - 1)
    for h in [0.01]:
        n = ceil(1 / h)
        xspace = np.linspace(0, 1, n)

        #Centered difference

        ϵ = ϵ*(1 + (β*h)/(2*ϵ)) #ϵₕ

        b = [0 for _ in range(n - 2)]
        b[-1] = -2*ϵ + h*β
        A = tri_diagonal(n-1, -2*ϵ-h*β, 4*ϵ, -2*ϵ+h*β)
        u_tilde = solve(A, b)
        u_tilde = np.append(u_tilde, 1)
        u_tilde = np.insert(u_tilde, 0, 0)
        plots.append((xspace, u_tilde, f"Centered", ""))

        #Upwind Difference
        b = [0 for _ in range(n - 2)]
        b[-1] = ϵ
        A = tri_diagonal(n-1, -ϵ-h*β, 2*ϵ+h*β, -ϵ)
        u_tilde = solve(A, b)
        u_tilde = np.append(u_tilde, 1)
        u_tilde = np.insert(u_tilde, 0, 0)
        plots.append((xspace, u_tilde, f"Upwind", ""))

        if h == 0.01:
            u_exact = [u(x) for x in xspace]
            plots.append((xspace, u_exact, "Exact", "-"))
    plot_values(plots, xlabel=r"$x$", ylabel=r"$u(x)$")
    plt.show()


def gn(x, n):
    if 0 <= x and x < 1/(2*n):
        return 2*x*n**2
    elif 1/(2*n) <= x and x < 1/n:
        return -2*x*n**2 + 2*n
    else:
        return 0

xspace = np.linspace(0, 1, 1000)
for n in range(1, 11):
    g = [gn(x, n) for x in xspace]
    plt.plot(xspace, g, label=f"g{n}")
    plt.legend(loc="best")
plt.show()
