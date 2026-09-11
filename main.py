import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

M1 = 52969000.0
K1 = 42250000.0
C1 = 1897000.0
OMEGA1= np.sqrt(K1/M1)
M2 = 660000
K2 = (M1**2 * M2 * OMEGA1**2)/((M1+M2)**2)
C2 = np.sqrt(1.5) * (M1**2 * M2**(1.5) * OMEGA1)/((M1 + M2)**(2.5))


def main():
    t0, tf, t_steps = 0.0, 200.0, 500
    y0 = [0.1, 0.0, 0.0, 0.0]

    sol = solve_ivp(deriv, (t0, tf), y0, t_eval=np.linspace(t0, tf, t_steps), max_step=1e-3)
    
    t = sol.t
    x1, x2, v1, v2 = sol.y
    a1 = (disturbance(t) - C1*v1 - K1*x1 - C2*(v1 - v2) - K2*(x1-x2))/M1

    print(f"The Mass of the TMD: {int(M2/1000)}mt")
    print(f"RMS Building Acceleration: {rms(a1):.04f}m/s^2")
    print(f"Peak Building Acceleration: {np.max(np.abs(a1)):.04f}m/s^2")
    
    plt.plot(t, x1)
    plt.show()

def deriv(t,y):
    x1, x2, v1, v2 = y
    a1 = (disturbance(t) - C1*v1 - K1*x1 - C2*(v1 - v2) - K2*(x1-x2))/M1
    a2 = (-C2*(v2-v1) - K2*(x2 -x1))/M2
    return [v1, v2, a1, a2]

    
def disturbance(t):
    return 0


def rms(x):
    return np.sqrt(np.mean(x**2))

if __name__=="__main__":
    main()