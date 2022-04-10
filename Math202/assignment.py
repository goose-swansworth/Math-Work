import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from sympy import *
 
def damped_spring(x, t, alpha):
    dxdt = x[1]
    dydt = -alpha*x[1] - x[0]
    return [dxdt, dydt]
 
def plot_phase_plane(de, alpha, clr):
    mids = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,0)]
    t = np.linspace(0, 10, 100)
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) not in mids:
                soln = odeint(de, [i, j], t, args=(alpha,))
                x = soln[:,0]
                y = soln[:,1]
                if i == 2 and j == 2:
                    plt.plot(x, y, color=clr, label=f"alpha={alpha}")
                    #plt.text(i, j, f"x0=({i}, {j})")
                else:
                    plt.plot(x, y, color=clr)
                    #plt.text(i, j, f"x0=({i}, {j})")
 
def assignemnt2_plot(a, clr):
    plot_phase_plane(damped_spring, a, clr)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best", prop={'size': 17})
    plt.show()
 
def SEIR_model(x, t, lam, gam, mu):
    S, E, I, R = x[0], x[1], x[2], x[3]
    dSdt = -lam*S
    dEdt = lam*S - gam*E
    dIdt = gam*E - mu*I
    dRdt = mu*I
    return [dSdt, dEdt, dIdt, dRdt]
 
def SEIR_model_b(x, t, lam, gam, mu):
    S, E, I, R = x[0], x[1], x[2], x[3]
 
    dSdt = -lam*I
    dEdt = lam*I-gam*E
    dIdt = gam*E-mu*I
    dRdt = mu*I
 
    return [dSdt, dEdt, dIdt, dRdt]
 
def SEIR_model_c(x, t, beta, gamma, mu):
    S, E, I, R = x[0], x[1], x[2], x[3]
 
    dSdt = -beta*I*(S/5_000_000)
    dEdt = beta*I*(S/5_000_000)-gamma*E
    dIdt = gamma*E-mu*I
    dRdt = mu*I
 
    return [dSdt, dEdt, dIdt, dRdt]
 
 
 
 
 
x0 = [5000000, 10000, 0, 0]
t = np.linspace(0, 275, 1500000)
soln = odeint(SEIR_model_c, x0, t, args=(0.25, 0.2, 0.1))
S, E, I, R = soln[:,0], soln[:,1], soln[:,2], soln[:,3]
plt.plot(t, S, label="[S]usceptible")
plt.plot(t, E, label="[E]xposed")
plt.plot(t, I, label="[I]nfectious")
plt.plot(t, R, label="[R]ecovered")
 
textstr = f"x0 = {x0}"
 
# these are matplotlib.patch.Patch properties
#props = dict(boxstyle='square', facecolor='white', alpha=0.3)
 
# place a text box in upper left in axes coords
#plt.text(0, -4000000, textstr, fontsize=14,
        #verticalalignment='top', bbox=props)
 
plt.legend(loc="best", prop={'size': 15})
plt.grid()
plt.xlabel("Time")
plt.ylabel("Population")
plt.show()