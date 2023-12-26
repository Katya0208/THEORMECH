import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Rot2D(X, Y, Alpha):#rotates point (X,Y) on angle alpha with respect to Origin
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

#defining parameters
R = 5 
l = 3
r = math.sqrt(R*R - l*l)

#defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')

#phi = (1/2)*math.pi + 0.3*t
#ksi = (1/2)*math.pi + 0.5*t
phi = 0.3*t
ksi = 0.5*t

x_O = -R*sp.sin(phi)
y_O = R*sp.cos(phi)

x_C = x_O-r*sp.sin(ksi)
y_C = y_O+r*sp.cos(ksi)

x_rel = -r*sp.sin(ksi)
y_rel = r*sp.cos(ksi)

Vx_C = sp.diff(x_C, t)
Vy_C = sp.diff(y_C, t)
V_mod_C = sp.sqrt(Vx_C**2 + Vy_C**2)

Ax_C = sp.diff(x_C, t, 2)
Ay_C = sp.diff(y_C, t, 2)
A_mod_C = sp.sqrt(Ax_C**2 + Ay_C**2)

#constructing corresponding arrays
T = np.linspace(0, 45, 1000)
X_O_def = sp.lambdify(t, x_O)
Y_O_def = sp.lambdify(t, y_O)
X_C_def = sp.lambdify(t, x_C)
Y_C_def = sp.lambdify(t, y_C)
X_REL_def = sp.lambdify(t, x_rel)
Y_REL_def = sp.lambdify(t, y_rel)
V_MOD_C_def = sp.lambdify(t, V_mod_C)
A_MOD_C_def = sp.lambdify(t, A_mod_C)

X_O = X_O_def(T)
Y_O = Y_O_def(T)
X_C = X_C_def(T)
Y_C = Y_C_def(T)
X_REL = X_REL_def(T)
Y_REL = Y_REL_def(T)
V_MOD_C = V_MOD_C_def(T)
A_MOD_C = A_MOD_C_def(T)

#here we start to plot
fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')
ax1.set(xlim=[-10, 10], ylim=[-20, 20])
ax1.set_xlabel('ось x')
ax1.set_ylabel('ось y')
ax1.invert_xaxis()
ax1.invert_yaxis()

#plotting initial positions
PointO1, = ax1.plot([0],[0],'bo')
Circ_Angle = np.linspace(0, 6.28, 100)
Circ, = ax1.plot(X_O[0]+R*np.cos(Circ_Angle), Y_O[0]+R*np.sin(Circ_Angle), 'g')

ArrowX = np.array([0, 0, 0])
ArrowY = np.array([l, 0, -l])
R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y_REL[0], X_REL[0]))
Stick_Arrow, = ax1.plot(R_Stick_ArrowX + X_C[0], R_Stick_ArrowY + Y_C[0])
O1O, = ax1.plot([0,X_O[0]],[0,Y_O[0]], 'b:')
OC, = ax1.plot([X_O[0],X_C[0]],[Y_O[0],Y_C[0]], 'b:')

ax2 = fig.add_subplot(4, 2, 2)
ax2.set(xlim=[0, 45], ylim=[V_MOD_C.min(), V_MOD_C.max()])
tv_x = [T[0]]
tv_y = [V_MOD_C[0]]
TV, = ax2.plot(tv_x, tv_y, '-')
ax2.set_xlabel('T')
ax2.set_ylabel('V')

ax3 = fig.add_subplot(4, 2, 4)
ax3.set(xlim=[0, 45], ylim=[A_MOD_C.min(), A_MOD_C.max()])
ta_x = [T[0]]
ta_y = [A_MOD_C[0]]
TA, = ax3.plot(ta_x, ta_y, '-')
ax3.set_xlabel('T')
ax3.set_ylabel('A')

plt.subplots_adjust(wspace=0.3, hspace=0.7)

#function for recounting the positions
def anima(i):
    O1O.set_data([0,X_O[i]],[0,Y_O[i]])
    OC.set_data([X_O[i],X_C[i]],[Y_O[i],Y_C[i]])
    Circ.set_data(X_O[i]+R*np.cos(Circ_Angle), Y_O[i]+R*np.sin(Circ_Angle))
    R_Stick_ArrowX, R_Stick_ArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y_REL[i], X_REL[i]))
    Stick_Arrow.set_data(R_Stick_ArrowX + X_C[i], R_Stick_ArrowY + Y_C[i])

    tv_x.append(T[i])
    tv_y.append(V_MOD_C[i])
    ta_x.append(T[i])
    ta_y.append(A_MOD_C[i])
    TV.set_data(tv_x, tv_y)
    TA.set_data(ta_x, ta_y)
    if i == 1000-1:
        tv_x.clear()
        tv_y.clear()
        ta_x.clear()
        ta_y.clear()
    
    return O1O, OC, Circ, Stick_Arrow, TV, TA

# animation function
anim = FuncAnimation(fig, anima,
                     frames=1000, interval=0.01, blit=True)

plt.show()