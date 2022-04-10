from audioop import lin2adpcm
import matplotlib.pyplot as plt
import numpy as np


def euler(f, y0, h, t0, tf):
    y_tilde = []
    t_step = (tf - t0) / h
    ti = t0
    yi = y0
    while ti <= tf:
        y_tilde.append(yi)
        yi = yi + f(ti, yi)*h
        ti += t_step
    return y_tilde

f = lambda t, y: 2/t*y + t**2*np.exp(t)
y = lambda t: t**2*(np.exp)

t_space = np.linspace(1, 2, 100)


def f(x, k):
    if 1/k <= x <= 1:
        return 1
    elif 0 <= x <= 1/k:
        return k*x
    else:
        return 0

x_space = np.linspace(-1, 1, 100)
for k in range(1, 20):
    fs = [f(x, k) for x in x_space]
    plt.plot(x_space, fs)
plt.grid()
plt.show()