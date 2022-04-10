import matplotlib.pyplot as plt
import numpy as np
from random import uniform

from scipy.fftpack import diff


def even_spaced_nodes(f, a, b, n, perturbate=False):
    """Return list of n pairs (x_i, f(x_i)) of even spaced points in (a, b)"""
    h = (b - a) / (n - 1)
    if perturbate:
        epsilon = [(10**-3) * uniform(0, 1) for _ in range(n)]
        return [(a + i*h, f(a + i*h) + epsilon[i]) for i in range(n)]
    else:
        return [(a + i*h, f(a + i*h)) for i in range(n)]


def newton_div_diff_wrapper(nodes, i, j):
    """Wrapper for recursive divided difference function so we can memoize"""
    cache = dict()
    def newton_div_diff(nodes, i, j):
        """Returns f[x_i,...,x_j]"""
        if (i, j) in cache:
            return cache[(i, j)]
        else:
            if i == j:
                return nodes[i][1]
            else:
                f_ij = (newton_div_diff(nodes, i + 1, j) - 
                newton_div_diff(nodes, i, j - 1)) / (nodes[j][0] - nodes[i][0])
                cache[(i, j)] = f_ij
                return f_ij
    return newton_div_diff(nodes, i, j)


def newton_poly(nodes, x):
    """Evaluate the newton polynomial (x-x0)(x-x1)...(x-xn) at x"""
    out = 1
    for i in range(len(nodes)):
        xi = nodes[i][0]
        out *= (x - xi)
    return out
    

def intrp_poly_with_diffs(nodes, diffs, x):
    """Evalute the newton interpolating polynomial 
    pn(x) = f0 + f[x0,x1]w1(x) + ... + f[x0,...,xn]wn(x) at x"""
    out = 0
    for k in range(len(nodes)):
        out += newton_poly(nodes[:k], x) * diffs[k]
    return out


def newton_interpolating_poly(nodes, xspace):
    diffs = []
    for m in range(len(nodes)):
        diffs.append(newton_div_diff_wrapper(nodes, 0, m))
    interp_poly = [intrp_poly_with_diffs(nodes, diffs, x) for x in xspace]
    return interp_poly


def max_poly_error(fxs, pxs):
    error = [abs(fxs[i] - pxs[i]) for i in range(len(fxs))]
    return max(error)



def question_1(perturbate_data=False):
    a, b = (0, 1)
    xspace = np.linspace(a, b, 200)
    cos_2pix = lambda x: np.cos(10*np.pi*x)
    max_nodes = 40
    max_error = float("inf")
    n_nodes = 2

    #compute interpolating polynomail and check if the error bound is reached
    while max_error >= 4*10**-4 and n_nodes <= max_nodes:
        nodes = even_spaced_nodes(cos_2pix, a, b, n_nodes, perturbate_data)
        diffs = []
        for m in range(len(nodes)):
            f_m = newton_div_diff_wrapper(nodes, 0, m)
            diffs.append(f_m)
        interp_poly = [intrp_poly_with_diffs(nodes, diffs, x) for x in xspace]
        max_error = max_poly_error([cos_2pix(x) for x in xspace], interp_poly)
        n_nodes += 1

    interp_poly = [intrp_poly_with_diffs(nodes, diffs, x) for x in xspace]
    xs = [nodes[i][0] for i in range(len(nodes))]
    ys = [nodes[i][1] for i in range(len(nodes))]
    plt.scatter(xs, ys, label=r"$f(x)$")
    plt.title(f"Nodes: {n_nodes - 1}, Max Error" + r"$<4\times10^{-4}$" + 
    f": {n_nodes - 1 < max_nodes}")
    plt.plot(xspace, interp_poly, label=r"$p(x)$")
    plt.grid()
    plt.legend(loc="best")
    plt.show()

def question_2():

    f = lambda x: 1 / (1 + 25*x**2)

    a, b = -1, 1

    xspace = np.linspace(-1, 1, 100)

    fs = [f(x) for x in xspace]

    degrees = [1, 3, 6, 10, 15, 21, 28, 37]

    for degree in degrees:
        nodes = even_spaced_nodes(f, a, b, degree+1)
        diffs = []
        for m in range(len(nodes)):
            diffs.append(newton_div_diff_wrapper(nodes, 0, m))
        interp_poly = [intrp_poly_with_diffs(nodes, diffs, x) for x in xspace]
        plt.plot(xspace, interp_poly, label=f"p{degree}(x)")
        max_error = max([abs(fs[i] - interp_poly[i]) for i in range(len(fs))])
        print(f"n: {degree}, max_error: {max_error:.4f}")
    plt.plot(xspace, fs, label="f(x)")
    plt.grid()
    plt.legend(loc="best")
    plt.ylim((-3, 3))
    plt.show()


def question_3():
    a, b = -4, 4
    xspace = np.linspace(a, b, 200)
    f = lambda x: x**4
    fs = [f(x) for x in xspace]

    degrees = [1, 2, 3, 4]

    for degree in degrees:
        nodes = even_spaced_nodes(f, a, b, degree+1)
        interp_poly = newton_interpolating_poly(nodes, xspace)
        plt.plot(xspace, interp_poly, label=f"p{degree}(x)")
        max_error = max([abs(fs[i] - interp_poly[i]) for i in range(len(fs))])
        print(f"n: {degree}, max_error: {max_error:.4f}")
    plt.plot(xspace, fs, label="f(x)", linestyle="dashed")
    plt.grid()
    plt.legend(loc="best")
    plt.show()


def question_4():
    a, b = 0, 1
    xspace = np.linspace(a, b, 100)
    f = lambda x: np.sin(np.sqrt(x))
    fs = np.sin(np.sqrt(xspace))
    degree = 1
    max_error = float("inf")
    while max_error > 0.03:
        nodes = even_spaced_nodes(f, a, b, degree+1)
        i_poly = newton_interpolating_poly(nodes, xspace)
        max_error = np.max(np.abs(fs - np.array(i_poly)))
        print(f"n: {degree}, max_error: {max_error:.4f}")
        plt.plot(xspace, i_poly, label=f"p{degree}(x)")
        degree += 1
    plt.plot(xspace, fs, label="f(x)", linestyle="dashed")
    plt.grid()
    plt.legend(loc="best")
    plt.show()

def f_star(x):
    if x == 0:
        return 1
    else:
        return np.sin(np.sqrt(x))


def question_5():
    a, b = 0, 1
    xspace = np.linspace(a, b, 100)
    fstar = [f_star(x) for x in xspace]
    fs = np.sin(np.sqrt(xspace))
    degree = 1
    max_error = float("inf")
    while max_error > 0.03:
        nodes = even_spaced_nodes(f_star, a, b, degree+1)
        i_poly_star = newton_interpolating_poly(nodes, xspace)
        i_poly = np.sqrt(xspace) * np.array(i_poly_star)
        max_error = np.max(np.abs(fs - i_poly))
        plt.plot(xspace, i_poly, label=f"sqrt(x)p{degree}(x)")
        print(f"n: {max_error}, max_error: {max_error}")
        degree += 1
        
    plt.plot(xspace, fs, label="f(x)", linestyle="dashed")
    plt.grid()
    plt.legend(loc="best")
    plt.show()

question_5()

