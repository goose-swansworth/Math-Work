import numpy as np

def f(x, y):
    u1, u2 = y
    return np.array([u2, -u1 - x*u2])

def euler(y0, t_final, steps):
    yi = y0
    xi = 0
    h = t_final / steps
    for _ in range(steps):
        yi = yi + f(xi, y0) * h
        xi += h
    return yi


u2 = euler([1, 2], 0.2, 100)
print(u2)