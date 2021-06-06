# -*- coding: utf-8 -*-
"""
@author: Radoslaw Patelski
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#initialize constant values
b = 0
k = 1
m = 1

#define arrays A, B and C of a linear system
A = [[0, 1],
    [-k/m, -b/m]]

B = [0,
    1/m]

C = [1, 0]

#define time-dependent input function. Here, unit step is defined
def force(t):
    return 1

#define dynamic system.
def deg1(t, x):
    dx1 = x[1]
    dx2 = x[0] * A[1][0] + x[1] * A[1][1] + B[1] * force(t)
       
    return [dx1, dx2]

#simulate the dynamic system, pass a system deg1, time of simulation and initial conditions
res = solve_ivp(deg1, [0,10], [0,0], rtol=1e-10) #arguments rtol and atol sets calculation tolerance

#calculate output based on the obtained state
y = res.y[0]

#plot results
plt.plot(res.t, res.y[0])
plt.plot(res.t, res.y[1])
plt.show()
#plt.savefig("filepath.pdf", format = 'pdf')
