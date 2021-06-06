from numpy import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal as sig


ticks = {
    'axis':'x',
    'bottom':False,
    'top':False,
    'labelbottom':False,
    'labeltop':False}

def step(tf, ssm, ssa):
    t, y = sig.step2(tf)
    plt.subplot(311)
    plt.plot(t, y)
    plt.tick_params(**ticks)
    plt.title('Transmitancja')
    t, y = sig.step2(ssm)
    plt.subplot(312)
    plt.plot(t, y)
    plt.tick_params(**ticks)
    plt.title('Przestrzeń stanu manualnie')
    t, y = sig.step2(ssa)
    plt.subplot(313)
    plt.plot(t, y)
    plt.title('Przestrzeń stanu scipy')
    plt.show()

def lti(ssm, ssa, init):
    ltim = sig.lti(ssm.A, ssm.B, ssm.C, ssm.D)
    ltia = sig.lti(ssa.A, ssa.B, ssa.C, ssa.D)
    ta, ya, xa = ltia.output(None, linspace(0, 7, 1000), init)
    tm, ym, xm = ltim.output(None, linspace(0, 7, 1000), init)
    
    plt.subplot(211)
    plt.plot(tm, ym)
    for i in range(len(init)):
        plt.plot(tm, xm if len(init) == 1 else xm[:, i])
    plt.legend(['y'] + [f'x{i}' for i in range(len(init))])
    plt.title('Przestrzeń stanu manualnie')
    plt.tick_params(**ticks)

    plt.subplot(212)
    plt.plot(ta, ya)
    for i in range(len(init)):
        plt.plot(ta, xa if len(init) == 1 else xa[:, i])
    plt.legend(['y'] + [f'x{i}' for i in range(len(init))])
    plt.title('Przestrzeń stanu automatycznie')
    plt.show()

def sequence(num, den, a, b, c, d, init):
    tf = sig.TransferFunction(num, den)
    ssManual = sig.StateSpace(a, b, c, d)
    ssAuto = tf.to_ss()
    print(ssAuto)
    step(tf, ssManual, ssAuto)
    lti(ssManual, ssAuto, init)

if __name__ == '__main__':
    # równanie 1
    num = [10.]
    den = [1., 2.]
    a = [[-2.]]
    b = [[1.]]
    c = [[10.]]
    d = [[0.]]
    sequence(num, den, a, b, c, d, [2.])

    # równanie 2
    num = [4.]
    den = [2., 0., 1.]
    a = [
        [0., 1.],
        [-0.5, 0.]]

    b = [
        [0.],
        [0.5]]

    c = [[4., 0.]]
    d = [[0.]]
    sequence(num, den, a, b, c, d, [2., 1.])

    # równanie 3
    num = [-2., 6.]
    den = [1., 5., 10., 12.]
    a = [
        [0., 1., 0.],
        [0., 0., 1.],
        [-12., -10., -5.]]
        
    b = [
        [0.],
        [0],
        [1.]]

    c = [[6., -2., 0.]]
    d = [[0.]]
    sequence(num, den, a, b, c, d, [2., 1., 0.5])