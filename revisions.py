import numpy as np
import matplotlib.pyplot as plt

default_values = {
    'g1': 0.2,
    'g2': 0.1,
    'g3': 0.0,
    'alpha_1': 1e-2,
    'alpha_2': 1e-4,
    'lambda_1': 1e-2,
    'lambda_2': 1e-4,
    'lambda_r': .08,
    'beta': 1e-4,
}

def s21_theoretical(
    w, H,
    g1=default_values['g1'],
    g2=default_values['g2'],
    g3=default_values['g3'],
    alpha_1=default_values['alpha_1'],
    alpha_2=default_values['alpha_2'],
    lambda_1=default_values['lambda_1'],
    lambda_2=default_values['lambda_2'],
    lambda_r=default_values['lambda_r'],
    beta=default_values['beta']
):
    # Constants
    gyro1 = 2.94e-3
    gyro2 = 1.76e-2 / (2 * np.pi)
    M1 = 10900.0  # Py
    M2 = 1750.0   # YIG

    gamma_1 = 2 * np.pi * lambda_1**2
    gamma_2 = 2 * np.pi * lambda_2**2
    gamma_r = 2 * np.pi * lambda_r**2

    alpha_r = beta

    omega_1 = gyro1 * np.sqrt(H * (H + M1))
    omega_2 = gyro2 * np.sqrt(H * (H + M2))
    omega_r = 5.0 # Assuming this was a constant, in Julia it's 5

    tomega_1 = omega_1 - 1j * (alpha_1 + gamma_1)
    tomega_2 = omega_2 - 1j * (alpha_2 + gamma_2)
    tomega_r = omega_r - 1j * (alpha_r + gamma_r)

    # Construct the matrix M
    M = np.array([
        [w - tomega_1,                             -g1 + 1j * np.sqrt(gamma_1 * gamma_r), -g3 + 1j * np.sqrt(gamma_1 * gamma_2)],
        [-g1 + 1j * np.sqrt(gamma_1 * gamma_r),    w - tomega_r,                          -g2 + 1j * np.sqrt(gamma_2 * gamma_r)],
        [-g3 + 1j * np.sqrt(gamma_1 * gamma_2),    -g2 + 1j * np.sqrt(gamma_2 * gamma_r),   w - tomega_2]
    ], dtype=complex)

    B = np.array([np.sqrt(gamma_1), np.sqrt(gamma_r), np.sqrt(gamma_2)], dtype=complex)
    B_col = B[:, np.newaxis] 

    try:
        inv_M = np.linalg.inv(M)
        B_row_real = np.array([np.sqrt(gamma_1), np.sqrt(gamma_r), np.sqrt(gamma_2)]) 
        result = B_row_real @ inv_M @ B_row_real
    except np.linalg.LinAlgError:
        print(f"Singular matrix for w={w}, H={H}. Returning NaN.")
        return np.nan

    return np.abs(result)


omega = np.linspace(4.3,6.1, 1000)  # Example frequency range
# omega = np.linspace(0,10, 1000)  # Example frequency range
H = np.linspace(0, 1600, 100)  # Example magnetic field range

s21 = np.zeros((omega.size, H.size))

for i, w in enumerate(omega):
    for j, h in enumerate(H):
        s21[i,j] = s21_theoretical(w,h)

plt.pcolormesh(H, omega, s21)
plt.show()