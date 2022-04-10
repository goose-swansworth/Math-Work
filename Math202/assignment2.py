from re import S, X
from mpmath.functions.functions import arg, re
from numpy.lib.histograms import _histogramdd_dispatcher
from scipy.integrate import odeint
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from sympy.plotting.textplot import linspace

def H(t):
     if t < 0:
          return 0
     else:
          return 1

def delta(a, t0, t):
     return 1/a*(H(t, t0)-H(t, t0+a))


def assignment2_de(U, x, k, w0, l):
     u1 = U[0]
     u2 = U[1]
     u3 = U[2]
     u4 = U[3]
     m = 2*w0/l
     du1dx = u2
     du2dx = u3
     du3dx = u4
     du4dx = -k*w(x, w0, l)
     return [du1dx, du2dx, du3dx, du4dx]

def w(x, w0, l):
     m = (2*w0)/l
     return w0 - w0*H(x-l/2)-m*x+m*(x-l/2)*H(x-l/2)+m*l/2*H(x-l/2)

def h(x, w0, l):
     m = (2*w0)/l
     return m*(x-(3/2)*l)*H(x-(3/2)*l)-m*(x-2*l)*H(x-2*l)-w0*H(x-2*l)

def g(x, w0, l):
     m = (2*w0)/l
     return w(x, w0, l) + h(x, w0, l)

def g_piecewise(x, w0, l):
     if 0 <= x <= l:
          return w(x, w0, l)
     elif l <= x < 2*l:
          return w(2*l-x, w0, l)
     else:
          return 0


def f_assignmentA_23(t):
     t = t % (2*np.pi)
     return t

def f_assignmentA_23_FS(t, N):
     pi = np.pi
     out_sum = pi
     for n in range(1, N + 1):
          bn = -2/n
          out_sum += bn*np.sin(n*t)
     return out_sum


def g_assignmentA_23(t):
     t = t % (2*np.pi)
     return t**2


def g_assignmentA_23_FS(t, N):
     pi = np.pi
     out_sum = (4/3)*pi**2
     for n in range(1, N + 1):
          an = 4/n**2
          bn = -4*pi/n
          out_sum += an*np.cos(n*t) + bn*np.sin(n*t)
     return out_sum

def b_limit_func_one(t):
     pi = np.pi
     t = t % (2*pi)
     return (pi - t)/2

def b_sum_one(t, N):
     sum_out = 0
     for n in range(1, N + 1):
          sum_out += np.sin(n*t)/n
     return sum_out

def b_limit_func_two(t):
     pi = np.pi
     t = t % (2*pi)
     return (1/4)*(t**2) - (1/3)*(pi**2) + pi*b_limit_func_one(t)

def b_sum_two(t, N):
     sum_out = 0
     for n in range(1, N + 1):
          sum_out += np.cos(n*t)/(n**2)
     return sum_out

def limit_error(limit_func_values, fs_values):
     return [limit_func_values[i] - fs_values[i] for i in range(len(limit_func_values))]
#
k = 1
w0 = 1
l = 2
a = (-k*w0*l**2)/24
b = (k*w0*l)/4
p = 2*np.pi    
tspace = linspace(0-0.01, 3*p+0.01, 1000)
#num_soln = odeint(assignment2_de, [0, 0, a, b], tspace, args=(k, w0, l) )
#num_soln = num_soln[:,0]



#plt.plot(tspace, [w(x, w0 ,l) for x in tspace], color="tab:blue", label="$f(x)$\n$w_0=1$\n$l=2$")
plt.xlabel("$t$", fontsize="x-large")
#plt.ylabel("$g(t)$",fontsize='x-large')
#plt.plot(tspace, [g(x, w0, l) for x in tspace],color="tab:green", label="$g(x)$\n$w_0=1$\n$l=2$")
#plt.plot(tspace, [g_piecewise(x, w0, l) for x in tspace], label="g piecewise")
#plt.plot(tspace, num_soln, label="odeint")
#plt.plot(tspace, [y(x, w0, l, k, a, b) for x in tspace], label="y(x)")


#plt.plot(tspace, [f_assignmentA_23(t) for t in tspace], label="$f(t)$")
#plt.plot(tspace, [f_assignmentA_23_FS(t, 2) for t in tspace], label="$s_2(t)$")
#plt.plot(tspace, [f_assignmentA_23_FS(t, 4) for t in tspace], label="$s_4(t)$")
#plt.plot(tspace, [f_assignmentA_23_FS(t, 6) for t in tspace], label="$s_6(t)$")
#plt.plot(tspace, [g_assignmentA_23(t) for t in tspace], label="$g(t)$")
#plt.plot(tspace, [g_assignmentA_23_FS(t, 2) for t in tspace], label="$s_2(t)$")
#plt.plot(tspace, [g_assignmentA_23_FS(t, 4) for t in tspace], label="$s_4(t)$")
#plt.plot(tspace, [g_assignmentA_23_FS(t, 6) for t in tspace], label="$s_6(t)$")
#limit_func_a = [b_limit_func_one(t) for t in tspace]
#plt.plot(tspace, limit_func, label=r"$\sum^\infty_{n=1}\frac{\sin(nt)}{n}$", color="tab:blue")
#s4 = [b_sum_one(t, 4) for t in tspace]
#s6 = [b_sum_one(t, 6) for t in tspace]
#s8 = [b_sum_one(t, 8) for t in tspace]
#plt.plot(tspace, limit_error(limit_func, s4), label=r"$\frac{t-\pi}{2}-s_4(t)$", color="tab:orange")
#plt.plot(tspace, limit_error(limit_func, s6), label=r"$\frac{t-\pi}{2}-s_6(t)$", color="tab:green")
#plt.plot(tspace, limit_error(limit_func, s8), label=r"$\frac{t-\pi}{2}-s_8(t)$", color="tab:red")

limit_func_b =[b_limit_func_two(t) for t in tspace]
#plt.plot(tspace, , label=r"$\sum^\infty_{n=1}\frac{\cos(nt)}{n^2}$", color="tab:green")
s4 = [b_sum_two(t, 4) for t in tspace]
s6 = [b_sum_two(t, 6) for t in tspace]
s8 = [b_sum_two(t, 8) for t in tspace]
plt.plot(tspace, limit_error(limit_func_b, s4), label=r"$\frac{1}{4}t^2-\frac{1}{3}\pi^2+\frac{\pi(\pi-t)}{2}-s_4(t)$", color="tab:red")
plt.plot(tspace, limit_error(limit_func_b, s6), label=r"$\frac{1}{4}t^2-\frac{1}{3}\pi^2+\frac{\pi(\pi-t)}{2}-s_6(t)$", color="tab:orange")
plt.plot(tspace, limit_error(limit_func_b, s8), label=r"$\frac{1}{4}t^2-\frac{1}{3}\pi^2+\frac{\pi(\pi-t)}{2}-s_8(t)$", color="tab:blue")




plt.grid()

plt.legend(loc="best", fontsize='large')
plt.show()