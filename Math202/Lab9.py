from re import S, X
from mpmath.functions.functions import arg, re
from numpy.lib.histograms import _histogramdd_dispatcher
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.plotting.textplot import linspace


def square_wave(t, p):
    t = t % p
    if 0 <= t <= np.pi:
        return 1
    else:
        return -1

def square_waveFS(t, n):
    sN = 0
    for k in range(1, n+1):
        bn = 2/(k*np.pi)*(1-(-1)**k)
        sN += bn*np.cos(k*t)
    return sN


tspace = linspace(-5, 5, 1000)
f = [square_wave(t, 2*np.pi) for t in tspace]
f.reverse()
n1 = [square_waveFS(t, 5) for t in tspace]
n5 = [square_waveFS(t, 10) for t in tspace]
n10 = [square_waveFS(t, 50) for t in tspace]
plt.plot(tspace, f, label="f(t)")
plt.plot(tspace, n1, label="n=5")
plt.plot(tspace, n5, label="n=10")
plt.plot(tspace, n10, label="n=50")

plt.grid()
plt.legend(loc="best")
plt.xlabel("t")
plt.ylabel("y")
plt.show()



