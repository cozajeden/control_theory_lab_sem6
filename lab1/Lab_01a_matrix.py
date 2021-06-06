# -*- coding: utf-8 -*-
"""
@author: Radoslaw Patelski
"""

import numpy as np, os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Solver:
    # initialize constant values
    b = [0, 0.5, 2] # tłumienie
    k = 1 # współczynnik spręzystoścki
    m = 1 # masa
    index = 1


    #define time-dependent input function. Here, unit step is defined
    def force(self, t):
        return np.array([[1]]) #return as an array with single element

    def sin(self, t):
        return np.sin([[2*t]])

    #define dynamic system.
    def deg1(self, t, x):
        x = np.array([x]).T #for compatibility with a solver, convert x to matrix and get transpose
        
        dx = self.A @ x + self.B @ self.fun(t); #calculate state equation
        #note: matrix multiplication is done using @ operator
        
        return np.ndarray.tolist(dx.T[0]) #for compatibility with a solver, transpose x and convert into list

    def resolve(self, b, fun, title, legend, fName):
        self.fun = fun
        # macierze stanu
        #define arrays A, B and C of a linear system
        self.A = np.array([[0, 1],
                    [-self.k/self.m, -b/self.m]])

        self.B = np.array([[0],
                    [1/self.m]])

        C = np.array([[1, 0]])

        #simulate the dynamic system, pass a system deg1, time of simulation and initial conditions
        res = solve_ivp(self.deg1, [0,10], [0,0], rtol=1e-10) #arguments rtol and atol sets calculation tolerance

        #calculate output based on the obtained state
        y = C @ np.array(res.y)
        y = np.ndarray.tolist(y[0].T)

        #plot results
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(res.t, res.y[0])
        plt.plot(res.t, res.y[1])
        plt.plot(res.t, [fun(t)[0][0] for t in res.t], alpha=0.2)
        plt.title(title)
        plt.legend(legend)
       # plt.show()
        plt.savefig(fName)

    def loop_b(self):
        legend = ['x1', 'x2', 'u']
        for b in self.b:
            self.resolve(b, self.force, f"$\\displaystyle{{u=1(t)}} | {{b={b}}}$", legend, f'1_b-{b}.png')
        for b in self.b:
            self.resolve(b, self.sin, f"$\\displaystyle{{u=sin(2*t)}} | {{b={b}}}$", legend, f'sin_b-{b}.png')

if __name__ == '__main__':
    solver = Solver()
    solver.loop_b()