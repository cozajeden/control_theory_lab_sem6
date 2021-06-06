# -*- coding: utf-8 -*-
"""
@author: Radoslaw Patelski
"""

from numpy import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

class Solver:
    # initialize constant values
    m = 1
    l = 0.5
    J = 0.05
    g = 9.81
    b = [0, 0.1, 0.5]
    p1 = 1/(m*l**2 + J)
    p2 = -m*g*l*p1
    fs = [2, 0.65, 0.2]


    #define time-dependent input function. Here, unit step is defined
    def sin(self, t):
        return 0.1*sin([[2*pi*self.f*t]])

    def zero(self, t):
        return zeros((1,1))

    #define dynamic system.
    def deg1(self, t, x):
        x = array([x]).T #for compatibility with a solver, convert x to matrix and get transpose
        Ax = None
        if self.state == 2:
            Ax = array([
                [x[1][0]],
                [self.p2*x[0][0]-self.p1*self.current_b*x[1][0]]
            ], dtype=float)
        else:
            Ax = array([
                [x[1][0]],
                [self.p2*sin(x[0][0])-self.p1*self.current_b*x[1][0]]
            ], dtype=float)
        
        dx = Ax + self.B @ self.fun(t); #calculate state equation
        #note: matrix multiplication is done using @ operator
        
        return ndarray.tolist(dx.T[0]) #for compatibility with a solver, transpose x and convert into list

    def resolve(self, b, fun, title, legend, f, start=(0, 0), state=0):
        self.state = state
        self.fun = fun
        self.f = f
        self.current_b = b
        # macierze stanu
        #define arrays B and C of a linear system

        self.B = array([[0],
                    [self.p1]])

        C = array([[1, 0]])

        #simulate the dynamic system, pass a system deg1, time of simulation and initial conditions
        res = solve_ivp(self.deg1, [0,60], start, rtol=1e-10) #arguments rtol and atol sets calculation tolerance

        #calculate output based on the obtained state
        y = array([
            sin(res.y[0]),
            -cos(res.y[0])
        ])

        # plot results
        
        if not state:
            plt.plot(res.y[0],res.y[1])

        if state:
            last = res.y[0][0]
            zero_cross = 1
            for pos in res.y[0][1:]:
                if last < 0 and pos > 0:
                    zero_cross += 1
                last = pos
            print(f'freq={zero_cross/60}')
            plt.plot(res.t, y[0])
            plt.plot(res.t, y[1])
            plt.title(title)
            plt.legend(legend)
            plt.show()

    def loop_b(self):

        for b in self.b:
            self.resolve(b, self.zero, None, None, None, (pi*0.5, 0))
        plt.title(f"$\\displaystyle{{\\dot{{\\Theta}}=f(\\Theta)}} | {{\\Theta(0)={{\\pi}}/{{2}}}}$")
        plt.legend([f'b={b}' for b in self.b])
        plt.show()

        legend = ['x', 'y']
        for b in self.b:
            self.resolve(b, self.zero, f"$\\displaystyle{{u=0}} | {{b={b}}}$", legend, None, (pi*0.5, 0), 1)

        self.resolve(0, self.zero, f"$\\displaystyle{{u=0}} | {{b={0}}} | \\sin{{\\Theta}}\\approx{{\\Theta}}$", legend, None, state=2)

        for f in self.fs:
            self.resolve(0.1, self.sin, f"$\\displaystyle{{u=0}} | {{b={0.1}}} | {{f={f}}}$", legend, f, state=1)

if __name__ == '__main__':
    solver = Solver()
    solver.loop_b()