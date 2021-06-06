from numpy import *
from scipy import signal as sig


if __name__ == '__main__':
    # równanie 1
    a = array([
        [-4, -1],
        [-2, -1]
    ])
    b = array([
        [2],
        [1]
    ])
    c = array([3, -4])
    d = array([1])
    ss = sig.StateSpace(a, b, c, d)
    tf = ss.to_tf()
    print('równanie 1')
    print(tf.num)
    print(tf.den)
    
    # równanie 2
    a = array([
        [-1, 0, 1],
        [-6, -3, 5],
        [-5, -2, 4]
    ])
    b = array([
        [0],
        [1],
        [1]
    ])
    c = array([1, 1, 1])
    d = array([0])
    ss = sig.StateSpace(a, b, c, d)
    tf = ss.to_tf()
    print('równanie 2')
    print(tf.num)
    print(tf.den)
    
    # równanie 3
    a = array([
        [-3, 1.25, -0.75, -2.75],
        [-6, 3, -3.5, -6],
        [0, -1, 0, 1],
        [-6, 5, -4.5, -6]
    ])
    b = array([
        [0.5],
        [1],
        [0],
        [1]
    ])
    c = array([2, 0, 0, 0])
    d = array([0])
    ss = sig.StateSpace(a, b, c, d)
    tf = ss.to_tf()
    print('równanie 3')
    print(tf.num)
    print(tf.den)
    