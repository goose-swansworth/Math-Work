from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import math

def find_sign_changes(f, x_min, x_max, min_step):
    """Return intervals (a, b) where f changes sign"""
    found = False
    step = 1
    result_intervals = []
    while step > min_step:
        x = x_min
        while x <= x_max and not found:
            if f(x) * f(x + step) < 0:
                found = True
                result_intervals.append((x, x + step))
                x_min = x + step
            else:
                x += step
        if x > x_max and not found:
            step = step / 2
        else:
            found = False
    return result_intervals


def bisection_iterations(a, b, tol):
    res = math.log((b - a) / tol, 2)
    return math.ceil(res)


def det_func(x):
    A = np.matrix([[1, 2, 3, x],
                    [4, 5, x, 6],
                    [7, x, 8, 9],
                    [x, 10, 11, 12]])
    return np.linalg.det(A) - 1000


def question_1():
    zero_intervals = find_sign_changes(det_func, -25, 25, 0.1)
    roots = []
    for inter in zero_intervals:
        roots.append(bisection_method(det_func, inter, 10**-10)[0][-1])
    for i, x in enumerate(roots):
        det = det_func(x) + 1000
        print(f"x_{i + 1} = {x:.6f}, det(A) = {det:.6f}")


def bisection_method(f, interval, tol, max_iters):
    ak, bk = interval
    x_k =  (ak + bk) / 2
    x_ks = []
    i = 0
    while abs(f(x_k)) > tol and i < max_iters:
        x_ks.append(x_k)
        if f(ak) * f(x_k) < 0:
            bk = x_k
        else:
            ak = x_k
        x_k =  (ak + bk) / 2
        i += 1
    return x_ks, i


def chord_method(interval, f, tol, max_iters):
    a, b = interval
    x_k = a
    x_ks = []
    q = (f(b) - f(a)) / (b - a)
    i = 0
    while abs(f(x_k)) > tol and i < max_iters:
        x_ks.append(x_k)
        x_k = x_k - ((1/q) * f(x_k))
        i += 1
    return x_ks, i


def secant_method(interval, f, tol, max_iters):
    a, b = interval
    x_k = a
    x_km1 = b
    x_ks = []
    i = 0
    while abs(f(x_k)) > tol and i < max_iters:
        qk = (f(x_k) - f(x_km1)) / (x_k - x_km1)
        if abs(qk) < 10**-10: #stops underflows
            i = max_iters
        x_km1 = x_k
        x_k = x_k - ((1/qk) * f(x_k))
        x_ks.append(x_k)
        i += 1
    return x_ks, i


def falsi_helper(x_ks, f):
    """Return maximim x_k_prime such that f(x_k)f(x_k_prime) < 0"""
    i = len(x_ks) - 1
    x_k = x_ks[i]
    x_k_prime = x_ks[i - 1]
    while f(x_k)*f(x_k_prime) >= 0:
        i -= 1
        x_k = x_ks[i]
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
    return x_ks, i


def newton_method(f, f_prime, xk, tol, max_iters):
    x_ks = []
    x_ks.append(xk)
    i = 0
    while abs(f(xk)) > tol and i < max_iters:
        xk = xk - (f(xk) / f_prime(xk))
        i += 1
        x_ks.append(xk)
    return x_ks, i


def fixed_point_iteration(phi, x0, tol, max_iters, f=None):
    approxs = []
    i = 0
    xk = x0
    xk_m1 = float("inf")
    while abs(f(xk)) > tol and i < max_iters:
        approxs.append(xk)
        xk_m1 = xk
        xk = phi(xk)
        i += 1
    if i >= max_iters:
        print("Max Iterations Reached")
        return approxs, i
    else:
        return approxs, i
    
    


def question_3_a():

    f_q3 = lambda x: (np.cos(2*x))**2 - x**2 
    f_q3_prime = lambda x: -4 * np.sin(2*x) * np.cos(2*x) - 2*x
    tol = 10**-10
    max_iterations = 200

    x_is, i = bisection_method(f_q3, (0, 1.5), 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Bisection")
    plt.axvline(x=len(x_is), color='tab:blue', linestyle='--')
    print(f"Bisection Method took {i} iterations to reach {tol} tolerance with initial interval (0, 1.5)")

    x_is, i = chord_method((0, 1.5), f_q3, 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Chord")
    plt.axvline(x=len(x_is), color='tab:orange', linestyle='--')
    print(f"Chord Method took {i} iterations to reach {tol} tolerance with initial guess of 0")

    x_is, i = secant_method((0, 1.5), f_q3, 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Secant")
    plt.axvline(x=len(x_is), color='tab:green', linestyle='--')
    print(f"Secant Method took {i} iterations to reach {tol} tolerance with initial guess of 0")

    x_is, i = regular_falsi((0, 1.5), f_q3, 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Regular Falsi")
    plt.axvline(x=len(x_is), color='tab:red', linestyle='--')
    print(f"Regula Falsi Method took {i} iterations to reach {tol} tolerance with initial guess of 0")


    x_is, i = newton_method(f_q3, f_q3_prime, 0.75, 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Newton")
    plt.axvline(x=len(x_is), color='tab:purple', linestyle='--')
    print(f"Newton Method took {i} iterations to reach {tol} tolerance with initial guess of 0  ")


    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("Iterations")
    plt.ylabel("Absolute Error")
    plt.xlim((0, 30))
    plt.show()


def question_3_b():
    x0_array = np.linspace(-5, 5, 250)
    converged_x0 = []
    for x0 in x0_array:
        alpha = newton_method(f_q3, f_q3_prime, x0, 10*-10, 100)[-1]
        if alpha > 0:
            print(f"Converged for x0 = {x0:.6f}, alpha = {alpha:.4f}")
            converged_x0.append(x0)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x0_array, [f_q3(x) for x in x0_array], label="f(x)")
    plt.scatter(converged_x0, [0 for _ in range(len(converged_x0))], s=3, label="Converged x0's", color="tab:red")
    plt.ylim((-5, 5))
    plt.legend(loc="best")
    plt.show()




def question_5():
    g = lambda x: (3*x**2-3*x-2) / (x - 1)
    h = lambda x: x - 2 + x/(x - 1)

    max_i = 100
    alpha = 2
    xspace = np.linspace(1, 10, 100)
    x0 = 5
    xk_s, i = fixed_point_iteration(h, x0, 10**-10, max_i)
    alpha = xk_s[-1]



    #plt.plot(xspace, [g(x) for x in xspace], label="g(x)")
    plt.plot(xspace, [h(x) for x in xspace], label="h(x)")
    plt.scatter(xk_s, [h(xk) for xk in xk_s], color="tab:red", s=(5,))
    plt.grid()
    #plt.xlabel("x")
    #plt.ylabel("y")
    plt.legend(loc="best")
    plt.show()


def question_6():
    f = lambda x: np.exp(-x) - np.sin(x)
    phi_a = lambda x: -np.log(np.sin(x))
    phi_b = lambda x: np.arcsin(np.exp(-x))

    xk_s, i = fixed_point_iteration(phi_b, 0, 10**-8, 100)
    xspace = np.linspace(0, 2, 100)
    plt.plot(xspace, [f(x) for x in xspace])
    plt.scatter(xk_s, [0 for _ in range(len(xk_s))], color="tab:red", s=(3,))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc="best")
    plt.show()


f = lambda x: np.sin(x) + x - 1
phi = lambda x: 1 - np.sin(x)

print(fixed_point_iteration(phi, 0.5, 10**-10, 200, f))