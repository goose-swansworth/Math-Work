import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.style.use('ggplot')

def heun_method(f, h, t0, t_final, y0, n):
    """Use heuns method to solve y' = f(w, t), where f: Rn -> Rn, 
    h is the step size, (t0, t_final) is the time interval to solve over,
    y0 is the initial condition"""
    t = np.arange(t0, t_final + h, h)
    w = np.zeros((len(t), n))
    w[0] = y0
    for i in range(len(t) - 1):
        w_tilde = w[i] + h * f(w[i], t[i])
        w[i + 1] = w[i] + h/2 * (f(w[i], t[i]) + f(w_tilde, t[i + 1]))
    return t, w

def SEIR_function(w, t):
    """Function used to model """
    print(w, t)

args =  ()
SEIR_function("w", "t", *args)