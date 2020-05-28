from __future__ import print_function
from numpy import sin,cos
import numpy as np
import matplotlib.pyplot as plt, random
import scipy.integrate as integrate
import matplotlib.animation as animation
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


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
    def test(self,state):
        (L1,L2,M1,M2,G) = self.params
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
        return [theta1,theta2]


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

        
        global temp
        temp = dydx
        return dydx

    def step(self,dt):
        self.state = integrate.odeint(self.dstate_dt, self.state, [0,dt])[1]
        self.time_elapsed += dt
    
pendulum = DoublePendulum([120.,0.0,10.,0.0])
dt = 1./30 #fps

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal', autoscale_on=False,
                    xlim=(-2,2),ylim=(-2,2))

ax.grid()
line, = ax.plot([],[],'o-',lw=2)
time_text = ax.text(0.02,0.95,'', transform=ax.transAxes)
energy_text = ax.text(0.02,0.90,'', transform=ax.transAxes)

def init():
    line.set_data([],[])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text

def animate(i):
    global pendulum, dt
    pendulum.step(dt)
    line.set_data(*pendulum.position())
    time_text.set_text('time = %.1f' % pendulum.time_elapsed)
    energy_text.set_text('energy = %.3f J' % pendulum.energy())
    pendulum.test(dt)
    return line, time_text, energy_text









from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1-t0)

ani = animation.FuncAnimation(fig,animate,frames=150,
                            interval=interval, blit=True, init_func=init)



class DoublePendulum2:
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
        print(x)
        print(y)
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
    def test(self,state):
        (L1,L2,M1,M2,G) = self.params
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
        return [theta1,theta2]


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

        
        global temp
        temp = dydx
        return dydx

    def step(self,dt):
        self.state = integrate.odeint(self.dstate_dt, self.state, [0,dt])[1]
        self.time_elapsed += dt
    
pendulum2 = DoublePendulum2([120.,0.0,10.,0.0])
dt = 1./30 #fps

fig2 = plt.figure()
ax2 = fig2.add_subplot(111,aspect='equal', autoscale_on=False,
                    xlim=(-40,40),ylim=(-40,40))

ax2.grid()
line2, = ax2.plot([],[],'o-',lw=2)
time_text2 = ax2.text(0.02,0.95,'', transform=ax2.transAxes)
energy_text2 = ax2.text(0.02,0.90,'', transform=ax2.transAxes)

def init2():
    line2.set_data([],[])
    time_text2.set_text('')
    energy_text2.set_text('')
    return line2, time_text2, energy_text2

def animate2(i):
    global pendulum2, dt
    pendulum2.step(dt)
    line2.set_data(*pendulum2.position())
    time_text2.set_text('time = %.1f' % pendulum2.time_elapsed)
    energy_text2.set_text('energy = %.3f J' % pendulum2.energy())
    pendulum2.test(dt)
    return line2, time_text2, energy_text2









from time import time
t0 = time()
animate2(0)
t1 = time()
interval = 1000 * dt - (t1-t0)

ani2 = animation.FuncAnimation(fig2,animate2,frames=150,
                            interval=interval, blit=True, init_func=init2)

plt.show()


