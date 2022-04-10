import matplotlib.pyplot as plt
import numpy as np

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


f_q3 = lambda x: (np.cos(2*x))**2 - x**2 
f_q3_prime = lambda x: -4 * np.sin(2*x) * np.cos(2*x) - 2*x
tol = 10**-10
max_iterations = 200

def question_3_a():
    x_is, i = bisection_method(f_q3, (0, 1.5), 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Bisection")
    #plt.axvline(x=len(x_is), color='tab:blue', linestyle='--')
    print(f"Bisection Method took {i} iterations to reach {tol} \
    tolerance with initial interval (0, 1.5)")

    x_is, i = chord_method((0, 1.5), f_q3, 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Chord")
    #plt.axvline(x=len(x_is), color='tab:orange', linestyle='--')
    print(f"Chord Method took {i} iterations to reach {tol} tolerance with initial guess of 0")

    x_is, i = secant_method((0, 1.5), f_q3, 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Secant")
    #plt.axvline(x=len(x_is), color='tab:green', linestyle='--')
    print(f"Secant Method took {i} iterations to reach {tol} tolerance with initial guess of 0")

    x_is, i = regular_falsi((0, 1.5), f_q3, 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Regular Falsi")
    #plt.axvline(x=len(x_is), color='tab:red', linestyle='--')
    print(f"Regula Falsi Method took {i} iterations to reach {tol} tolerance with initial guess of 0")


    x_is, i = newton_method(f_q3, f_q3_prime, 0.75, 10**-10, 200)
    plt.plot(range(0, len(x_is)), [abs(f_q3(x)) for x in x_is], label="Newton")
    #plt.axvline(x=len(x_is), color='tab:purple', linestyle='--')
    print(f"Newton Method took {i} iterations to reach {tol} tolerance with initial guess of 0  ")


    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("Iterations")
    plt.ylabel("Absolute Error")
    plt.xlim((0, 25))
    plt.show()

    


question_3_a()