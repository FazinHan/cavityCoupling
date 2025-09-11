import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

fontsize = 25#int(sys.argv[1])
labelsize = 33#fontsize + int(sys.argv[2])

# Load data from CSV file
filename = 'combined_plots_params.csv'
data = pd.read_csv(filename)

# Extract columns
t = data['t'].values
g1 = data['g1'].values
g2 = data['g2'].values
g3 = data['g3'].values
g1_min = data['g1_min'].values
g1_max = data['g1_max'].values
g2_min = data['g2_min'].values
g2_max = data['g2_max'].values

g1_errors = g1_max - g1_min
g2_errors = g2_max - g2_min

if t[0]== 0:
    t = t[1:]
    g1 = g1[1:]
    g2 = g2[1:]
    g3 = g3[1:]
    g1_errors = g1_errors[1:]
    g2_errors = g2_errors[1:]



# Plot data and fits
plt.figure(figsize=(9, 6))

# Perform linear regression for g1
g1_model = LinearRegression()
g1_model.fit(t.reshape(-1, 1), g1)
g1_fit_vals = g1_model.predict(t.reshape(-1, 1))

# Calculate R^2 for g1
r2_g1 = g1_model.score(t.reshape(-1, 1), g1)

# Perform linear regression for g2
g2_model = LinearRegression()
g2_model.fit(t.reshape(-1, 1), g2)
g2_fit_vals = g2_model.predict(t.reshape(-1, 1))

# Calculate R^2 for g2
r2_g2 = g2_model.score(t.reshape(-1, 1), g2)

g3_model = LinearRegression()
g3_model.fit(t.reshape(-1, 1), g3)
g3_fit_vals = g3_model.predict(t.reshape(-1, 1))
r2_g3 = g3_model.score(t.reshape(-1, 1), g3)

# g1 data and fit
plt.plot(t, g1, 'ro', label='$g_{Py}$ data',markersize=15)
plt.errorbar(t, g1, yerr=g1_errors, fmt='none',ecolor='r',capsize=10)#, markersize=15, capsize=5)
plt.plot(t, g1_fit_vals, 'r-', label=f'$g_{{\\text{{Py}}}} = {g1_model.coef_[0]:.2f}t + {g1_model.intercept_:.2f}$')# (R^2 = {r2_g1:.2f})$')
# Print the slope and intercept of the fit for g1
print(f"Slope (g1): {g1_model.coef_[0]:.2f}, Intercept (g1): {g1_model.intercept_:.2f}")

# g2 data and fit
plt.plot(t, g2, 'bx', label='$g_{YIG}$ data (Py present)',markersize=15)
plt.errorbar(t, g2, yerr=g2_errors, fmt='none',ecolor='b',capsize=10)#, markersize=15, capsize=5)
plt.plot(t, g2_fit_vals, 'b-', label=f'$g_{{\\text{{YIG}}}} = {g2_model.coef_[0]:.2f}t + {g2_model.intercept_:.2f}$')# (R^2 = {r2_g2:.2f})$')
# Print the slope and intercept of the fit for g2
print(f"Slope (g2): {g2_model.coef_[0]:.2f}, Intercept (g2): {g2_model.intercept_:.2f}")
print()
# plt.plot(t_fit, g2_fit_plot, 'r--', label=f'g2 fit: y = {params_g2[0]:.2f} * (1 - exp(-{params_g2[1]:.2f} * x)) + {params_g2[2]:.2f}')

# plt.plot(t, g3, 'y^', label='$g_{YIG}$ data (Py absent)',markersize=15)
# plt.plot(t, g3_fit_vals, 'y-', label=f'$g_{{\\text{{int}}}} = {g3_model.coef_[0]:.2f}t + {g3_model.intercept_:.2f}$')# (R^2 = {r2_g3:.2f})$')
# plt.xticks([.02,.06,.1])
# plt.yticks([.05,.15,.25])
# Customize plot
plt.xlabel('t',fontsize=labelsize)
plt.ylabel('$g$',fontsize=labelsize)
plt.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
plt.tick_params(axis='both', which='minor', labelsize=8, direction='in')
# plt.title(f'Saturation Fits for g1 and g2 (R^2: g1 = {r2_g1:.2f}, g2 = {r2_g2:.2f})')
# plt.legend(fontsize=15)
# plt.grid()
# plt.show()
plt.savefig(os.path.join("tentative","images",'g1g2_vs_t.png'), dpi=300, bbox_inches='tight')
