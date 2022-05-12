import matplotlib.pyplot as plt
import numpy as np
from math353 import *

def euler(f, y0, h, t0, tf):
    y_tilde = []
    ts = []
    ti = t0
    yi = y0
    while ti <= tf + h:
        ts.append(ti)
        y_tilde.append(yi)
        yi = yi + f(ti, yi)*h
        ti += h
    return ts, y_tilde

f = lambda t, y: 2/t*y + t**2*np.exp(t)
y = lambda t: t**2*(np.exp(t)-np.exp(1))

L = 2
M = 2*2*np.exp(2)+4*np.exp(2)+2*np.exp(2)-2*np.exp(1)
epsilon = 0.1
h = lambda t: (2*L*epsilon/M)*(np.exp(L*(t - 1) - 1))

t_space = np.linspace(1, 2, 10)
ts, y_tilde = euler(f, 0, h(1), 1, 2)
ys = [y(t) for t in ts]
plot_values([(ts, y_tilde, "~y"), (ts, ys, "y")])
plt.show()
errors = np.abs(np.array(ys) - np.array(y_tilde))
print(max(errors))
