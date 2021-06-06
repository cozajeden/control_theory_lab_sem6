import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg.linalg import solve
from scipy import integrate as itg
from itertools import cycle

class Solver:

    def define_matrices(self, b, m, k):
        A, B, C, D = self.get_matrices(b, m, k)

        self.matrices = [A, B, C, D, None]

        # Badanie oberwowalności
        G = np.concatenate([C, C @ A])
        detG = np.linalg.det(G)
        print('Układ obserwowalny' if detG != 0  else 'Układ nieobserowalny')

    def get_matrices(self, b, m, k):
        A = np.array([
            [0, 1],
            [-k/m, -b/m]
        ])
        B = np.array([
            [0],
            [1/m]
        ])
        C = np.array([
            [1, 0]
        ])
        D = np.array([[0]])

        return (A, B, C, D)

    def define_L(self, omega0):
        self.matrices[4] = self.get_L(omega0)

    def get_L(self, omega0):
        return np.array([
            [2*omega0-0.5],
            [omega0**2 - omega0 - 0.75]
        ])

    def sin(self, t):
        return np.sin([[t]])

    def step(self, t):
        return np.array([[1]])

    def solve_system(self, t, x, A, B, C, D, L, u, noise):
        x = x.reshape((x.shape[0], 1))

        dx1 = A @ x[:2,:] + B @ u(t)
        if noise is None:
            dx2 = A @ x[2:,:] + B @ u(t) + L @ C @ (dx1 +  x[2:,:])
        else:
            dx2 = A @ x[2:,:] + B @ u(t) + L @ C @ (dx1 + noise() +  x[2:,:])
        dx = np.concatenate([dx1, dx2])

        return dx.T[0]

    def solve_different_omega(self, fun, uptime, start, end, amount, noise=None):
        rows = 0
        columns = 0
        for i in range(amount+1):
            columns = i + 1
            rows = int((amount+1)/columns)
            if amount + 1 == columns*rows and (columns > rows or columns == rows):
                break
        else:
            print(f"Wrong amount ({amount}). The (amount + 1) has to be a number that allow plotter to place all plots in a grid.")
            pass

        results = []
        omegas = np.linspace(start, end, amount)
        print(omegas)
        print(start, end, amount)
        for i, omega0 in enumerate(omegas):
            print(f'{i}: Solving for omega_0={omega0:.2f}')
            solver.define_L(omega0)
            res = itg.solve_ivp(self.solve_system, [0, uptime], [0, 0, 0, 0], rtol=1e-10, args=(*self.matrices, fun, noise))
            results.append(res)

        plt.subplot(rows, columns, 1)
        plt.plot(results[0].t, results[0].y[0])
        plt.legend(['y'])

        for i, omega0 in enumerate(omegas):
            print(f'{i} Plotting for omega_0={omega0:.2f}')
            plt.subplot(rows, columns, 2+i)
            plt.plot(results[i].t, results[i].y[2])
            plt.legend([f'omega_0={omega0:.2f}'])

        plt.show()

    def noise(self):
        noise = np.array([
            [next(self.normal)]
            [0]
        ])
        return noise
    
    def generate_noise(self, loc, dev,  samples):
        self.normal = cycle(np.random.normal(loc, dev, samples))


if __name__ == '__main__':
    b = 0.5
    m = 1
    k = 1
    solver = Solver()
    solver.define_matrices(b, m, k)
    solver.solve_different_omega(solver.sin, 100, -2, 2, 11)
    solver.solve_different_omega(solver.step, 100, -2, 2, 11)
    solver.generate_noise(0., 0.01, 1000)
    solver.solve_different_omega(solver.sin, 100, -2, 2, 11, solver.noise)
    solver.generate_noise(0., 0.1, 1000)
    solver.solve_different_omega(solver.sin, 100, -2, 2, 11, solver.noise)
    solver.generate_noise(0., 0.01, 1000)
    solver.solve_different_omega(solver.step, 100, -2, 2, 11, solver.noise)
    solver.generate_noise(0., 0.1, 1000)
    solver.solve_different_omega(solver.step, 100, -2, 2, 11, solver.noise)
    b = 0.2
    m = 0.7
    k = 5
    solver.define_matrices(b, m, k)
    solver.solve_different_omega(solver.sin, 100, -1, -1, 1)
    solver.solve_different_omega(solver.step, 100, -1, -1, 1)
    b = 0.5
    m = 1
    k = 1
    solver.define_matrices(b, m, k)
    solver.define_L(0.4)

    res = itg.solve_ivp(solver.solve_system, [0, 100], [0, 0, 0, 0], rtol=1e-10, args=(*solver.matrices, solver.step, None))
    t = res.t
    x1 = res.y[2]
    x2 = res.y[3]
    x2_calc = (x1[1:] - x1[:-1])/(t[1:] - t[:-1])
    plt.subplot(121)
    plt.plot(t, x2)
    plt.plot(t[1:], x2_calc)
    plt.legend(['x2 symulowane', 'x2 obliczone'])
    plt.title('Wymuszenie skokowe')

    res = itg.solve_ivp(solver.solve_system, [0, 100], [0, 0, 0, 0], rtol=1e-10, args=(*solver.matrices, solver.sin, None))
    t = res.t
    x1 = res.y[2]
    x2 = res.y[3]
    x2_calc = (x1[1:] - x1[:-1])/(t[1:] - t[:-1])
    plt.subplot(122)
    plt.plot(t, x2)
    plt.plot(t[1:], x2_calc)
    plt.legend(['x2 symulowane', 'x2 obliczone'])
    plt.title('Wymuszenie sinusoidalne')
    plt.show()