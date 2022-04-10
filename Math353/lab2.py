import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linalg, zeros 
import math

from numpy.lib.function_base import append

def numerical_jacobian(x, F):
    n = len(x)
    J = zeros((n, n))
    x_nh = x.copy()
    h = 1e-7
    for i in range(n):
        x_nh[i] = x[i] + h
        J[:,i] = (F(x_nh) - F(x)) / h
        x_nh = x.copy()
    return J


def newton_method_analytical_jac(F, Jf, x_0, tol, max_iterations):
    i = 0
    x_k = x_0
    x_ks = []
    while linalg.norm(F(x_k)) > tol and i < max_iterations:
        x_ks.append(x_k)
        delta = np.linalg.solve(Jf(x_k), -F(x_k))
        x_k = x_k + delta
        i += 1
    return x_ks, i


def newton_method_num_jac(F, x0, tol, max_iterations):
    i = 0
    xk = x0
    xk_s = []
    while linalg.norm(F(xk)) > tol and i < max_iterations:
        xk_s.append(xk)
        J = numerical_jacobian(xk, F)
        delta = np.linalg.solve(J, -F(xk))
        xk = xk + delta
        i += 1
    return xk_s, i

def q1_Jf(x):
    x, y  = x[0], x[1]
    df1dx = 2*x*np.exp(x**2 + y**2)
    df1dy = 2*y*np.exp(x**2 + y**2)
    df2dx = 2*x*np.exp(x**2 - y**2)
    df2dy = -2*y*np.exp(x**2 - y**2)
    J = [[df1dx, df1dy],
         [df2dx, df2dy]]
    return J

def question_1():
    F = lambda x: np.array([np.exp(x[0]**2 + x[1]**2) - 1, np.exp(x[0]**2 - x[1]**2) - 1])
    
    tol = 10**-4
    max_i = 500


    x_0 = [0.1, 0.1]
    x_ks, i = newton_method_analytical_jac(F, q1_Jf, x_0, tol, max_i)
    if i >= max_i:
        i = "Max Iterations Reached"
    root = x_ks[-1]
    print(f"x0 = {x_0}, r = {root}, ||F(r)|| = {np.linalg.norm(F(root))}, iterations: {i}")

    x_0 = [10, 10]
    x_ks, i = newton_method_analytical_jac(F, q1_Jf, x_0, tol, max_i)
    if i >= max_i:
        i = "Max Iterations Reached"
    root = x_ks[-1]
    print(f"x0 = {x_0}, r = {root}, ||F(r)|| = {np.linalg.norm(F(root))}, iterations: {i}")
    
    x_0 = [20, 20]
    x_ks, i = newton_method_analytical_jac(F, q1_Jf, x_0, tol, max_i)
    if i >= max_i:
        i = "Max Iterations Reached"
    root = x_ks[-1]
    print(f"x0 = {x_0}, r = {root}, ||F(r)|| = {np.linalg.norm(F(root))}, iterations: {i}")


def question_2():

    F = lambda x: array([2*x[0] + x[1] - 1, x[0]**2 + x[1]**2 - 1])
    Jf = lambda x: np.matrix([[2, 1],
                    [2*x[0], 2*x[1]
                   ]])

    tol = 10**-4
    max_i = 100
    x0 = [0, 0]

    #xk_s, i = newton_method_analytical_jac(F, Jf, x0, tol, max_i)
    #root = xk_s[-1]
    #print("Analytical Jacobian:")
    #print(f"x0 = {x0}, r = ({root[0]:.4f}, {root[1]:.4f}), ||F(r)|| = {np.linalg.norm(F(root)):.5f}, iterations: {i}")

    print("")
    
    xk_s, i = newton_method_num_jac(F, x0, tol, max_i)
    root = xk_s[-1]
    print("Numerical Jacobian:")
    print(f"x0 = {x0}, r = ({root[0]:.4f}, {root[1]:.4f}), ||F(r)|| = {np.linalg.norm(F(root)):.5f}, iterations: {i}")

    x0 = [-10, -10]

    xk_s, i = newton_method_analytical_jac(F, Jf, x0, tol, max_i)
    root = xk_s[-1]
    print("Analytical Jacobian:")
    print(f"x0 = {x0}, r = ({root[0]:.4f}, {root[1]:.4f}), ||F(r)|| = {np.linalg.norm(F(root)):.5f}, iterations: {i}")

    print("")
    
    xk_s, i = newton_method_num_jac(F, x0, tol, max_i)
    root = xk_s[-1]
    print("Numerical Jacobian:")
    print(f"x0 = {x0}, r = ({root[0]:.4f}, {root[1]:.4f}), ||F(r)|| = {np.linalg.norm(F(root)):.5f}, iterations: {i}")


def question_3():
    F = lambda x: np.array([(x[0] - 1)**2 + (x[1] - 1)**2 + x[2]**2 - 1,
                x[0]**2 + (x[1] - 1)**2 + (x[2] - 1)**2 - 1,
                (x[0] - 1)**2 + x[1]**2 + (x[2] - 1)**2 - 1])

    Jf = lambda x: np.matrix([[2*(x[0] - 1), 2*(x[1] - 1), 2*x[2]],
                              [2*x[0], 2*(x[1] - 1), 2*(x[2] - 1)],
                              [2*(x[0] - 1), 2*x[1], 2*(x[2] - 1)]])

    tol = 10**-4
    max_i = 100
    x0 = [0, 0, 0]
    xks, i = newton_method_analytical_jac(F, Jf, x0, tol , max_i)
    print(xks[-1])

    x0 = [4, 4, 4]
    xks, i = newton_method_analytical_jac(F, Jf, x0, tol , max_i)
    print(xks[-1])


def plot_convergence(xks, alpha):
    error_xk_plus1 = [np.log(np.linalg.norm(xks[k] - alpha)) for k in range(1, len(xks))]
    error_xk = [np.log(np.linalg.norm(xks[k] - alpha)) for k in range(len(xks) - 1)]
    plt.plot(error_xk, error_xk_plus1)
    plt.grid()
    plt.show()

def question_4():

    F = lambda x: np.array([(x[0] - 1)**2 + x[1]**2 + (x[2]-1)**2 - 8,
                x[0]**2 + (x[1] - 2)**2 + (x[2] - 2)**2 - 2,
                x[0]**2 + (x[1] - 3)**2 + (x[2] - 3)**2 - 2])

    Jf = lambda x: np.matrix([[2*(x[0] - 1), 2*x[1], 2*(x[2]-1)],
                              [2*x[0], 2*(x[1] - 2), 2*(x[2] - 2)],
                              [2*x[0], 3*(x[1] - 3), 3*(x[2] - 3)]])

    tol = 10**-4
    max_i = 100
    x0 = [1, 1, 1]
    xks, i = newton_method_num_jac(F, x0, tol , max_i)
    print(xks[-1], i)
    plot_convergence(xks, np.array([1, 2, 3]))



question_4()