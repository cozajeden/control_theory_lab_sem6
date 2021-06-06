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
    R1=0.2
    L=0.1
    C=0.05
    A1=[-10, 2, 5, 10]
    A2=[2, 10]


    #define time-dependent input function. Here, unit step is defined
    def sin(self, t):
        return self.Amp*sin([[10*t]])

    def force(self, t):
        return array([[t]])

    def R2(self, u):
        return 0.25*u/(5-u)

    #define dynamic system.
    def deg1(self, t, x):
        x = array([x]).T #for compatibility with a solver, convert x to matrix and get transpose
        
        A = array([
            [-self.R1/self.L, -1/self.L],
            [1/self.C, 0.25/(self.C*(self.fun(t) - 5))]
        ])
        dx = A @ x + self.B @ self.fun(t); #calculate state equation
        #note: matrix multiplication is done using @ operator
        
        return ndarray.tolist(dx.T[0]) #for compatibility with a solver, transpose x and convert into list

    def resolve(self, a, fun, title, legend):
        self.fun = fun
        self.Amp = a
        # macierze stanu
        #define arrays B and C of a linear system

        self.B = array([[1/self.L],
                    [0]])

        C = array([0.5, 0.5])

        #simulate the dynamic system, pass a system deg1, time of simulation and initial conditions
        res = solve_ivp(self.deg1, [1e-10,2], [0,0], rtol=1e-10) #arguments rtol and atol sets calculation tolerance

        #calculate output based on the obtained state
        y = C @ array(res.y)

        #plot results
        plt.plot(res.t, res.y[0])
        plt.plot(res.t, res.y[1])
        plt.plot(res.t, y)
        plt.title(title)
        plt.legend(legend)
        plt.show()

    def loop_b(self):
        legend = [r'${\dot{x_1}}$', r'${\dot{x_2}}$', r'${i_2}$']
        for A in self.A1:
            self.resolve(A, self.force, f"${{u={A}*1(t)}}$", legend)
        for A in self.A2:
            self.resolve(A, self.sin, f"$\\displaystyle{{u={A}*\\sin(10*t)}}$", legend)

if __name__ == '__main__':
    solver = Solver()
    solver.loop_b()