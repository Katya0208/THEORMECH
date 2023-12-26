import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

class ArrowedVector:
    def get_x(S):
        return np.concatenate([[S.x0, S.x1], S.x_arrow])

    def get_y(S):
        return np.concatenate([[S.y0, S.y1], S.y_arrow])

    def rotate_2d(S, X, Y, alph):
        R_X = X * np.cos(alph) - Y * np.sin(alph)
        R_Y = X * np.sin(alph) + Y * np.cos(alph)
        return R_X, R_Y

    def set(S, A, B):
        S.x0, S.y0 = A[0], A[1]
        S.x1, S.y1 = B[0], B[1]
        S.x_arrow = np.array([-0.05, 0, -0.05])
        S.y_arrow = np.array([0.025, 0, -0.025])
        S.x_arrow, S.y_arrow = S.rotate_2d(S.x_arrow, S.y_arrow, math.atan2(S.y1 - S.y0, S.x1 - S.x0))
        if A == B:
            S.x_arrow, S.y_arrow = np.zeros(3), np.zeros(3)
        S.x_arrow += B[0]
        S.y_arrow += B[1]

    def __init__(S, A, B) -> None:
        S.x0, S.y0 = A[0], A[1]
        S.x1, S.y1 = B[0], B[1]
        S.x_arrow = np.array([-0.05, 0, 0.05])
        S.y_arrow = np.array([0.025, 0, -0.25])
        S.x_arrow, S.y_arrow = S.rotate_2d(S.x_arrow, S.y_arrow, math.atan2(S.y1 - S.y0, S.x1 - S.x0))
        if A == B:
            S.x_arrow, S.y_arrow = np.zeros(3), np.zeros(3)
        S.x_arrow += B[0]
        S.y_arrow += B[1]

t = sp.Symbol("t")

r = 2 + sp.sin(6*t)
phi = 6.5 * t + 1.2 * sp.cos(6 * t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)

vx = sp.diff(x, t)
vy = sp.diff(y, t)

wx = sp.diff(vx, t)
wy = sp.diff(vy, t)

nvy = vy / (vx**2 + vy**2)**0.5
nwx = wx / (wx**2 + wy**2)**0.5
nwy = wy / (wx**2 + wy**2)**0.5
nvx = vx / (vx**2 + vy**2)**0.5


T = np.linspace(0, 10, 1000)

X, Y = np.zeros_like(T), np.zeros_like(T)
Vx, Vy, Wx, Wy = np.zeros_like(T), np.zeros_like(T), np.zeros_like(T), np.zeros_like(T)

V = ArrowedVector([0, 0], [0, 0])
W = ArrowedVector([0, 0], [0, 0])
R0 = ArrowedVector([0, 0], [0, 0])

for i, time in enumerate(T):
    X[i] = sp.Subs(x, t, time)
    Y[i] = sp.Subs(y, t, time)
    Wx[i] = sp.Subs(nwx, t, time)
    Wy[i] = sp.Subs(nwy, t, time)
    Vx[i] = sp.Subs(nvx, t, time)
    Vy[i] = sp.Subs(nvy, t, time)

fig, ax1 = plt.subplots(1, 1)
ax1.axis('equal')
ax1.plot(X, Y)

ax1.set(xlim = [-5, 5], ylim = [-5, 5])

P, = ax1.plot(X[0], Y[0], marker = 'o')
anim_R0, = ax1.plot(R0.get_x(), R0.get_y())
anim_V, = ax1.plot(V.get_x(), V.get_y())
anim_W, = ax1.plot(W.get_x(), W.get_y())

R0.set([0, 0], [Vx[0], Vy[0]])
anim_R0.set_data(R0.get_x(), R0.get_y())

def animate_frame(i):
    P.set_xdata(X[i])
    P.set_ydata(Y[i])


    R0.set([0, 0], [X[i], Y[i]])
    anim_R0.set_data(R0.get_x(), R0.get_y())

    V.set([X[i], Y[i]], [X[i] + Vx[i], Y[i] + Vy[i]])
    anim_V.set_data(V.get_x(), V.get_y())

    W.set([X[i], Y[i]], [X[i] + Wx[i], Y[i] + Wy[i]])
    anim_W.set_data(W.get_x(), W.get_y())

    return P, anim_R0, anim_V, anim_W

animation = FuncAnimation(fig, animate_frame, frames = 1000, interval = 2, repeat = True)
plt.show()