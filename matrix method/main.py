import numpy as np
import matplotlib.pyplot as plt
# from sympy.abc import *
# import sympy as sp

def s21(w, H, **kwargs):
    gyro1 = kwargs.get('gyro1', 2.94e-3)
    gyro2 = kwargs.get('gyro2', 1.76e-2)
    M1 = kwargs.get('M1', 10900) # Py
    M2 = kwargs.get('M2', 1750) # YIG

    gamma_1 = kwargs.get('gamma_1', .0001)
    gamma_2 = kwargs.get('gamma_2', .008)
    gamma_r = kwargs.get('gamma_r', .02)

    alpha_1 = kwargs.get('alpha_1', 0)
    alpha_2 = kwargs.get('alpha_2', 0)
    alpha_r = kwargs.get('alpha_r', 0)

    omega_1 = gyro1*np.sqrt(H*(H+M1))
    omega_2 = gyro2*np.sqrt(H*(H+M2))
    omega_r = kwargs.get('omega_r', 5.3)

    g1 = kwargs.get('g1', 1)
    g2 = kwargs.get('g2', 1)

    tomega_1 = omega_1 - 1j*(alpha_1+gamma_1)
    tomega_2 = omega_2 - 1j*(alpha_2+gamma_2)
    tomega_r = omega_r - 1j*(alpha_r+gamma_r)


    M = np.array([[w-tomega_1,-g1+1j*np.sqrt(gamma_1*gamma_r),1j*np.sqrt(gamma_1*gamma_2)],[-g1+1j*np.sqrt(gamma_1*gamma_r),w-tomega_r,-g2+1j*np.sqrt(gamma_2*gamma_r)],[1j*np.sqrt(gamma_1*gamma_2),-g2+1j*np.sqrt(gamma_2*gamma_r),w-tomega_2]]) * 1j
    B = np.array([[np.sqrt(gamma_1), np.sqrt(gamma_r), np.sqrt(gamma_2)]]).T * np.sqrt(2)

    return B.T @ np.linalg.inv(M) @ B
    # return M

# print(s21(5.4, 1))#, gamma_1=a, gamma_2=b, gamma_r=c,)

def plot_s21(g1=.1,g2=.1,H=0):
    w = np.linspace(.1,10,500)
    # H = H
    s21s = [s21(ww, H, g1=g1, g2=g2)[0,0] for ww in w]
    plt.plot(w, np.abs(s21s))
    plt.show()

def plot_2d_s21(g1=.1,g2=.1):
    w = np.linspace(.1,10,500)
    H = np.linspace(0, 1.6, 150)*1e3
    s21s = np.array([[s21(ww, hh, g1=g1, g2=g2)[0,0] for hh in H] for ww in w])
    plt.pcolormesh(H, w, np.abs(s21s))#, extent=[w[0], w[-1], H[0], H[-1]], aspect='auto')
    plt.xlabel('H (Oe)')
    plt.ylabel('$\\omega$ (GHz)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # plot_s21()
    # plot_s21(10)
    plot_2d_s21()