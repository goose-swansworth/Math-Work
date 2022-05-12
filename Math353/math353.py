import matplotlib.pyplot as plt
import numpy as np

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


def newton_interpolating_poly(nodes, x):
    diffs = []
    for m in range(len(nodes)):
        diffs.append(newton_div_diff_wrapper(nodes, 0, m))
    return intrp_poly_with_diffs(nodes, diffs, x)