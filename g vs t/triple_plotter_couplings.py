import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

env_name = "basic"

fontsize = 25
label_surp = 8

labelsize = fontsize + label_surp

PYTHON_PATH = f"/home/fazinhan/miniconda3/envs/{env_name}/bin/python"
# PYTHON_PATH = "C:/Users/freak/miniforge3/envs/cavityCoupling/python.exe"
DIR = os.path.dirname(os.path.abspath(__file__))

fig, axs = plt.subplots(1,3, figsize=(30,9))#, sharey=True)

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
# plt.figure(figsize=(9, 6))

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
axs[0].plot(t, g1, 'ro', label='$g_{Py}$ data',markersize=15)
axs[0].plot(t, g1_fit_vals, 'r-', label=f'$g_{{\\text{{Py}}}} = {g1_model.coef_[0]:.2f}t + {g1_model.intercept_:.2f}$')# (R^2 = {r2_g1:.2f})$')
axs[0].errorbar(t, g1, yerr=g1_errors, fmt='none',ecolor='r',capsize=10)#, markersize=15, capsize=5)
# Print the slope and intercept of the fit for g1
print(f"Slope (g1): {g1_model.coef_[0]:.2f}, Intercept (g1): {g1_model.intercept_:.2f}")

# g2 data and fit
axs[0].plot(t, g2, 'bx', label='$g_{YIG}$ data (Py present)',markersize=15)
axs[0].plot(t, g2_fit_vals, 'b-', label=f'$g_{{\\text{{YIG}}}} = {g2_model.coef_[0]:.2f}t + {g2_model.intercept_:.2f}$')# (R^2 = {r2_g2:.2f})$')
axs[0].errorbar(t, g2, yerr=g2_errors, fmt='none',ecolor='b',capsize=10)#, markersize=15, capsize=5)
# Print the slope and intercept of the fit for g2
print(f"Slope (g2): {g2_model.coef_[0]:.2f}, Intercept (g2): {g2_model.intercept_:.2f}")
print()

# Customize plot
axs[0].set_xlabel('t',fontsize=labelsize)
axs[0].set_ylabel('$g$',fontsize=labelsize)
axs[0].tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
axs[0].tick_params(axis='both', which='minor', labelsize=8, direction='in')
axs[0].text(.005,.24,'(a)',fontsize=labelsize)

# Fit g1 vs g2 to a line

slope, intercept, r_value, p_value, std_err = linregress(g2, g1)

# Print the fit parameters
print("Fit parameters for g1 vs g2:")
print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")
print()

# Plot the fitted line

axs[2].plot(g2,g1,'go',markersize=16,label='Data')
axs[2].errorbar(g2, g1, xerr=g2_errors, yerr=g1_errors, fmt='none', ecolor='g', capsize=10)#, markersize=15, capsize=5)
axs[2].plot(g2, slope * g2 + intercept, 'g-', label=f'Fit: $g_{{Py}} = {slope:.2f} g_{{YIG}} + {intercept:.2f}$')
axs[2].set_xlabel("$g_2$",fontsize=labelsize)
axs[2].set_ylabel("$g_1$",fontsize=labelsize)
axs[2].tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
axs[2].tick_params(axis='both', which='minor', labelsize=8, direction='in')
axs[2].text(.003,.195,'(c)',fontsize=labelsize)
plt.tight_layout()

root = os.path.join(os.getcwd(),"data","lone_t_sweep_yig")
# J_values = np.linspace(0.09, 0.237, 7)
file_path = 'g_yig_vs_t.csv'
data = pd.read_csv(file_path,header=None)

t_values = data[0]
J_values = data[1]

# Fit the data to a line

# Extract t and J_values for fitting
# t_values = [float('0.' + file.split('.')[1]) for file in files if len(file.split('.')[1]) == 3]
slope, intercept, r_value, p_value, std_err = linregress(t_values, J_values[:len(t_values)])

# plt.figure(figsize=(9, 6))

# Plot the fitted line
t_fit = np.linspace(min(t_values), max(t_values), 100)
J_fit = slope * t_fit + intercept
axs[1].plot(t_values, J_values[:len(t_values)], 'bx', label='$g_{YIG}$ data (Py absent)',markersize=15)
axs[1].plot(t_fit, J_fit, 'b-', label=f'$g_{{YIG}} = {slope:.3f}t {'+'*(int(intercept>=0))} {intercept:.3f}$',markersize=15)
axs[1].text(.02,.29,'(b)',fontsize=labelsize)

# Print the slope and intercept of the fit
print("Lone YIG fit:")
print(f"Slope: {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print()

axs[1].set_xlabel('t',fontsize=labelsize)
axs[1].set_ylabel('$g_2$',fontsize=labelsize)
# plt.grid()
axs[1].tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
axs[1].tick_params(axis='both', which='minor', labelsize=8, direction='in')
plt.tight_layout()
# plt.savefig(os.path.join("tentative","images","Jvst.png"),dpi=300, bbox_inches='tight')

plt.savefig(os.path.join("tentative","images","triple_plotter_couplings.png"), dpi=300, bbox_inches='tight')