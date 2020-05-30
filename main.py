from __future__ import print_function
from numpy import sin,cos
import numpy as np
import matplotlib.pyplot as plt, random
import scipy.integrate as integrate
import matplotlib.animation as animation
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')

def show_entry_fields():
    global mass1, mass2, length1, length2, angle1, angle2, G
    length1 = float(e1.get())
    length2 = float(e2.get())
    mass1 = float(e3.get())
    mass2 = float(e4.get()) 
    angle1 = float(e5.get()) 
    angle2 = float(e6.get()) 
    G = float(e7.get())
    master.quit()
    master.destroy()
master = tk.Tk()
tk.Label(master, text="L1").grid(row=0)
tk.Label(master, text="L2").grid(row=1)
tk.Label(master, text="m1").grid(row=2)
tk.Label(master, text="m2").grid(row=3)
tk.Label(master, text="Teta1").grid(row=4)
tk.Label(master, text="Teta2").grid(row=5)
tk.Label(master, text="G").grid(row=6)
e1 = tk.Entry(master)
e2 = tk.Entry(master)
e3 = tk.Entry(master)
e4 = tk.Entry(master)
e5 = tk.Entry(master)
e6 = tk.Entry(master)
e7 = tk.Entry(master)
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)
e6.grid(row=5, column=1)
e7.grid(row=6, column=1)
tk.Button(master, 
          text='Show', command=show_entry_fields).grid(row=7, 
                                column=1,sticky=tk.W,pady=4)
master.mainloop() 

class DoublePendulum:
    def __init__(self,
                init_state = [120,0,-20,0],
                L1 = .5, #length of pendulum 1 in m
                L2 = .5, #length of pendulum 2 in m
                M1 = 1.0, #mass of pendulum 1 in kg
                M2 = 2.0, # mass of pendulum 2 in kg
                G = 9.8, # acceleration due to gravity, m/s^2
                origin=(0,0)):
        self.init_state = np.asarray(init_state,dtype='float')
        self.params = (L1,L2,M1,M2,G)
        self.origin = origin
        self.time_elapsed = 0
        self.state = self.init_state * np.pi/180

    def position(self):
        (L1, L2, M1, M2, G) = self.params
        x = np.cumsum([self.origin[0],
                        L1 * sin(self.state[0]),
                        L2 * sin(self.state[2])])
        y = np.cumsum([self.origin[1],
                        -L1 * cos(self.state[0]),
                        -L2 * cos(self.state[2])])
        return (x,y)
    
    def thetaandtheta(self):
        (L1, L2, M1, M2, G) = self.params
        
        theta1 = self.state[0]
        theta2 = self.state[2]
        mtot = M1+M2
        theta1dot = self.init_state[1] * np.pi/180
        theta2dot = self.init_state[3] * np.pi/180
        A = mtot * L1
        B = M2 * L2 * np.cos(theta1 - theta2)
        C = -M2 * L2 * theta2dot ** 2 * np.sin(theta1 - theta2) - mtot * G * np.sin(theta1)
        D = L1 / L2 * np.cos(theta1-theta2)
        E = (L1 * theta1dot ** 2 * np.sin(theta1 - theta2)- G * np.sin(theta2)) / L2

        theta1 = (C-B*E)/(A-B*D)
        theta2 = E - D * theta1dot
        x=theta1
        y=theta2
        return (x,y)

    def theta1(self,dt):
        (L1, L2, M1, M2, G) = self.params
        theta1 = self.state[0]
        theta2 = self.state[2]
        mtot = M1+M2
        theta1dot = self.init_state[1] * np.pi/180
        theta2dot = self.init_state[3] * np.pi/180
        A = mtot * L1
        B = M2 * L2 * np.cos(theta1 - theta2)
        C = -M2 * L2 * theta2dot ** 2 * np.sin(theta1 - theta2) - mtot * G * np.sin(theta1)
        D = L1 / L2 * np.cos(theta1-theta2)
        E = (L1 * theta1dot ** 2 * np.sin(theta1 - theta2)- G * np.sin(theta2)) / L2

        theta1 = (C-B*E)/(A-B*D)
        theta2 = E - D * theta1dot
        x=dt
        y=theta1
        return (x,y)

    def theta2(self,dt):
        (L1, L2, M1, M2, G) = self.params
        theta1 = self.state[0]
        theta2 = self.state[2]
        mtot = M1+M2
        theta1dot = self.init_state[1] * np.pi/180
        theta2dot = self.init_state[3] * np.pi/180
        A = mtot * L1
        B = M2 * L2 * np.cos(theta1 - theta2)
        C = -M2 * L2 * theta2dot ** 2 * np.sin(theta1 - theta2) - mtot * G * np.sin(theta1)
        D = L1 / L2 * np.cos(theta1-theta2)
        E = (L1 * theta1dot ** 2 * np.sin(theta1 - theta2)- G * np.sin(theta2)) / L2

        theta1 = (C-B*E)/(A-B*D)
        theta2 = E - D * theta1dot
        x=dt
        y=theta2
        return (x,y)

    def energy(self):
            (L1,L2,M1,M2,G) = self.params
            x = np.cumsum([L1 * sin(self.state[0]),
                        L2 * sin(self.state[2])])
            y = np.cumsum([-L1 * cos(self.state[0]),
                        -L2 * cos(self.state[2])])
            vx = np.cumsum([L1*self.state[1]*cos(self.state[0]),
                            L2*self.state[3]*cos(self.state[2])])
            vy = np.cumsum([L1*self.state[1]*cos(self.state[0]),
                            L2*self.state[3]*cos(self.state[2])])
            U = G * (M1 *y[0] +M2 * y[1])
            K = 0.5 * (M1 * np.dot(vx,vx) + M2 * np.dot(vy,vy))
            return U+K

    

    def dstate_dt(self,state,t):
        (M1,M2,L1,L2,G)=self.params
        dydx = np.zeros_like(state)
        dydx[0] = state[1]
        dydx[2] = state[3]

        cos_delta = cos(state[2] - state[0])
        sin_delta = sin(state[2] - state[0])

        den1 = (M1 + M2)  * L1 - M2 * L1 * cos_delta * cos_delta
        dydx[1] = (M2 * L1 * state[1] * state[1] * sin_delta * cos_delta
                + M2 * G * sin(state[2]) * cos_delta
                + M2 * L2 * state[3] * state[3] * sin_delta 
                - (M1+M2) * G * sin(state[0])) / den1
        den2 = (L2 / L1) * den1
        dydx[3] = (-M2 * L2 * state[3] * state[3] * sin_delta * cos_delta
                + (M1 + M2) * G * sin(state[0]) * cos_delta
                - (M1 + M2) * L1 * state[1] * state[1] * sin_delta
                - (M1 + M2) * G * sin(state[2])) / den2

        return dydx
    def step(self,dt):
        self.state = integrate.odeint(self.dstate_dt, self.state, [0,dt])[1]
        self.time_elapsed += dt

if length1 + length2 <=2:
    lim1 = -2
    lim2 = 2
elif length1 + length2 <=4:
    lim1 = -4
    lim2 = 4
elif length1 + length2 <=8:
    lim1 = -8
    lim2 = 8
elif length1 + length2 <=16:
    lim1 = -16
    lim2 = 16
elif length1 + length2 <=32:
    lim1 = -32
    lim2 = 32
elif length1 + length2 >32:
    lim1 = -100
    lim2 = 100

pendulum = DoublePendulum([angle1,0.0,angle2,0.0],length1,length2,mass1,mass2,G)
dt = 1./30 #fps
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
fig = plt.figure(1)

ax = fig.add_subplot(grid[0, 0],aspect='equal', autoscale_on=False,
                    xlim=(lim1,lim2),ylim=(lim1,lim2),alpha=0.5)

ax.grid()
line, = ax.plot([],[],'o-',lw=2)
time_text = ax.text(0.02,0.95,'', transform=ax.transAxes)
energy_text = ax.text(0.02,0.90,'', transform=ax.transAxes)
info_text = ax.text(0.8,0.8,'',transform=ax.transAxes)
path, = ax.plot([],[], color='C0')
tempx = []
tempy = []
def init():
    line.set_data([],[])
    time_text.set_text('')
    energy_text.set_text('')
    info_text.set_text('Double pendulum')
    path.set_data([], [])
    return line, time_text, energy_text

def animate(i):
    global pendulum, dt
    tempx.append(pendulum.position()[0][2])
    tempy.append(pendulum.position()[1][2])

    pendulum.step(dt)
    line.set_data(*pendulum.position())
    time_text.set_text('time = %.1f' % pendulum.time_elapsed)
    energy_text.set_text('energy = %.3f J' % pendulum.energy())
    path.set_data(tempx,tempy)
    return line, time_text, energy_text, path

from time import time
t0 = time()
animate(0)
t1 = time()
interval = 800 * dt - (t1-t0)

ani = animation.FuncAnimation(fig,animate,frames=150,
                            interval=interval, blit=True, init_func=init)



ax2 = fig.add_subplot(grid[1, 2],aspect='equal', autoscale_on=False,
                    xlim=(-30,30),ylim=(-30,30))


ax2.grid()
line2, = ax2.plot([],[],'o-')
path2, = ax2.plot([],[],color='C0')
info_text2 = ax2.text(0.8,0.8,'',transform=ax2.transAxes)
temp2x = []
temp2y = []
def init2():
    info_text2.set_text('Theta1/Theta2')
    line2.set_data([],[])
    path2.set_data([],[])
    return line2

def animate2(i):
    global pendulum, dt
    pendulum.step(dt)
    line2.set_data(*pendulum.thetaandtheta())
    temp2x.append(pendulum.thetaandtheta()[0])
    temp2y.append(pendulum.thetaandtheta()[1])
    path2.set_data(temp2x,temp2y)
    return line2, path2

animate2(0)
ani2 = animation.FuncAnimation(fig,animate2,frames=150,
                            interval=interval, init_func=init2)


ax3 = fig.add_subplot(grid[0, 1:],aspect='equal', autoscale_on=True,
                    xlim=(0,40),ylim=(-8,8))
info_text3 = ax3.text(0.8,0.8,'',transform=ax3.transAxes)
ax3.grid()
line3, = ax3.plot([],[],'o-')
path3, = ax3.plot([],[],color='C0')
temp3x = []
temp3y = []
def init3():
    line3.set_data([],[])
    path3.set_data([],[])
    info_text3.set_text('Theta1/time')
    return line3

def animate3(i):
    global pendulum, dt
    pendulum.step(dt)
    temp3x.append(pendulum.theta1((pendulum.time_elapsed))[0])
    temp3y.append(pendulum.theta1((pendulum.time_elapsed))[1])
    line3.set_data(*pendulum.theta1(pendulum.time_elapsed))
    path3.set_data(temp3x,temp3y)

    return line3, path3

animate3(0)

ani3 = animation.FuncAnimation(fig,animate3,frames=150,
                            interval=interval, init_func=init3)

dt = 1./30

ax4 = fig.add_subplot(grid[1, :2],aspect='equal', autoscale_on=True,
                    xlim=(0,40),ylim=(-8,8))

ax4.grid()
line4, = ax4.plot([],[],'o-')
path4, = ax4.plot([],[],color='C0')
info_text4 = ax4.text(0.4,0.2,'',transform=ax4.transAxes)
temp4x = []
temp4y = []


def init4():
    line4.set_data([],[])
    info_text4.set_text('Theta2/time')
    path4.set_data([],[])
    return line4

def animate4(i):
    global pendulum, dt
    pendulum.step(dt)
    temp4x.append(pendulum.theta2(pendulum.time_elapsed)[0])
    temp4y.append(pendulum.theta2(pendulum.time_elapsed)[1])
    print(pendulum.theta2(pendulum.time_elapsed))
    path4.set_data(temp4x,temp4y)

    
    line4.set_data(*pendulum.theta2(pendulum.time_elapsed))
    return line4, path4

animate4(0)

ani4 = animation.FuncAnimation(fig,animate4,frames=150,
                            interval=interval, init_func=init4)

fig.set_size_inches(13.5, 8.5) #13.5 inches = 1296 px and 8.5 inches = 816 px
plt.show()
