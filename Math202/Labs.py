from re import S
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def lead1(x, t):
     x1 = x[0]
     x2 = x[1]
     x3 = x[2]

     dx1dt = -0.0361*x1 + 0.0124*x2 + 0.000035*x3 + 49.3
     dx2dt = 0.0111*x1 - 0.0286*x2
     dx3dt = 0.0039*x1 - 0.000035*x3

     return [dx1dt, dx2dt, dx3dt]

def lead2(x, t):
     x1 = x[0]
     x2 = x[1]
     x3 = x[2]

     dx1dt = -0.0361*x1 + 0.0124*x2 + 0.000035*x3*2/3
     dx2dt = 0.0111*x1 - 0.0286*x2
     dx3dt = 0.0039*x1 - 0.000035*x3

     return [dx1dt, dx2dt, dx3dt]

def eg():
     x0 = [0, 0, 0]
     t = np.linspace(0, 10, 100)

     soln = odeint(lead1, x0, t)

     y10 = soln[-1,:]
     t = np.linspace(0, 100, 100)
     new_soln = odeint(lead2, y10, t)
     x1 = new_soln[:,0]
     x2 = new_soln[:,1]
     x3 = new_soln[:,2]
     plt.plot(t, x1, "tab:green", label="x_1: blood")
     plt.plot(t, x2, "tab:orange", label="x_2: hair etc")
     plt.plot(t, x3, "tab:blue", label="x_3: bones")
     plt.legend(loc="best")
     plt.xlabel("t")
     plt.grid()
     plt.show()

def des(x, t, a, b, c, d):
     x1 = x[0]
     x2 = x[1]

     dx1dt = a*x1 + b*x2
     dx2dt = c*x1 + d*x2

     return [dx1dt, dx2dt]

t = np.linspace(-10, 10, 1000)
sol = odeint(des, [1, 1], t, args=(-1, 5, -1, 1))
x = sol[:,0]
y = sol[:,1]
plt.plot(x, y, "tab:green")
plt.grid()
plt.show()