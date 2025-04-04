import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data from CSV file
filename = 'combined_plots_params.csv'
data = pd.read_csv(filename)

# Extract columns
t = data['t'].values
g1 = data['g1'].values
g2 = data['g2'].values

# Define saturation model: y = a * (1 - exp(-b * x)) + c
def saturation_model(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c

# Set initial guesses for g1 and g2 fits
# a_g1_guess, b_g1_guess, c_g1_guess = 1, 1, 0
# a_g2_guess, b_g2_guess, c_g2_guess = 1, -36, 0

# # Fit g1 data
# params_g1, _ = curve_fit(saturation_model, t, g1, p0=[a_g1_guess, b_g1_guess, c_g1_guess])

# # Fit g2 data
# params_g2, _ = curve_fit(saturation_model, t, g2, p0=[a_g2_guess, b_g2_guess, c_g2_guess])

# # Compute R-squared for g1
# g1_fit_vals = saturation_model(t, *params_g1)
# residuals_g1 = g1 - g1_fit_vals
# ss_res_g1 = np.sum(residuals_g1**2)
# ss_tot_g1 = np.sum((g1 - np.mean(g1))**2)
# r2_g1 = 1 - (ss_res_g1 / ss_tot_g1)

# # Compute R-squared for g2
# g2_fit_vals = saturation_model(t, *params_g2)
# residuals_g2 = g2 - g2_fit_vals
# ss_res_g2 = np.sum(residuals_g2**2)
# ss_tot_g2 = np.sum((g2 - np.mean(g2))**2)
# r2_g2 = 1 - (ss_res_g2 / ss_tot_g2)

# # Generate fit values for plotting
# t_fit = np.linspace(min(t), max(t), 100)
# g1_fit_plot = saturation_model(t_fit, *params_g1)
# g2_fit_plot = saturation_model(t_fit, *params_g2)

# Plot data and fits
# plt.figure(figsize=(8, 5))

# g1 data and fit
plt.scatter(t, g1, color='blue', label='g1 data')
# plt.plot(t_fit, g1_fit_plot, 'b--', label=f'g1 fit: y = {params_g1[0]:.2f} * (1 - exp(-{params_g1[1]:.2f} * x)) + {params_g1[2]:.2f}')

# g2 data and fit
plt.scatter(t, g2, color='red', label='g2 data')
# plt.plot(t_fit, g2_fit_plot, 'r--', label=f'g2 fit: y = {params_g2[0]:.2f} * (1 - exp(-{params_g2[1]:.2f} * x)) + {params_g2[2]:.2f}')

# Customize plot
plt.xlabel('t')
plt.ylabel('g values')
# plt.title(f'Saturation Fits for g1 and g2 (R^2: g1 = {r2_g1:.2f}, g2 = {r2_g2:.2f})')
plt.legend()
# plt.grid()
# plt.show()
plt.savefig('images\\g_vs_t.png', dpi=300, bbox_inches='tight')
