import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

fontsize = 5
label_surp = 2
markersize1 = 7
markersize2 = 5
linewidth = 0.8

labelsize = fontsize + label_surp

PYTHON_PATH = "/home/fazinhan/miniconda3/envs/cavityCoupling/bin/python"
# PYTHON_PATH = "C:/Users/freak/miniforge3/envs/cavityCoupling/python.exe"
DIR = os.path.dirname(os.path.abspath(__file__))

# os.system(f'{PYTHON_PATH} "{os.path.join(DIR,"g1g2_vs_t_plotter_saturation.py")}" {fontsize} {label_surp}')
# os.system(f'{PYTHON_PATH} "{os.path.join(DIR,"g2_vs_g1.py")}" {fontsize} {label_surp}')
# os.system(f'{PYTHON_PATH} "{os.path.join(DIR,"gyig_vs_t.py")}" {fontsize} {label_surp}')


fig, axs = plt.subplots(1, 1, figsize=(5.5 / 2.54, 5.0 / 2.54))#, sharey=True)

# Load data from CSV file
filename = 'combined_plots_params.csv'
data = pd.read_csv(filename)

# Extract columns
t = data['t'].values
g1 = data['g1'].values
g2 = data['g2'].values
g3 = data['g3'].values

if t[0]== 0:
    t = t[1:]
    g1 = g1[1:]
    g2 = g2[1:]
    g3 = g3[1:]



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
# axs.plot(t, g1, 'ro', label='$g_{Py}$ data',markersize=15)
# axs.plot(t, g1_fit_vals, 'r-', label=f'$g_{{\\text{{Py}}}} = {g1_model.coef_[0]:.2f}t + {g1_model.intercept_:.2f}$')# (R^2 = {r2_g1:.2f})$')
# Print the slope and intercept of the fit for g1
print(f"Slope (g1): {g1_model.coef_[0]:.2f}, Intercept (g1): {g1_model.intercept_:.2f}")

# plt.xlabel('t',fontsize=16)
# plt.ylabel('$g_{Py}$',fontsize=16)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.tick_params(axis='both', which='minor', labelsize=8)
# plt.title(f'Saturation Fits for g1 and g2 (R^2: g1 = {r2_g1:.2f}, g2 = {r2_g2:.2f})')
# plt.legend(fontsize=15)
# plt.grid()
# plt.show()
# plt.savefig('tentative\\images\\gpy_vs_t.png', dpi=300, bbox_inches='tight')
# plt.close()

# plt.figure(figsize=(9, 6))

# g2 data and fit
axs.plot(t, g2, 'g2', label='Py present',markersize=markersize1)
axs.plot(t, g2_fit_vals, 'g-', linewidth=linewidth)#, label=f'$g_{{\\text{{YIG}}}} = {g2_model.coef_[0]:.2f}t + {g2_model.intercept_:.2f}$')# (R^2 = {r2_g2:.2f})$')
# Print the slope and intercept of the fit for g2
print(f"Slope (g2): {g2_model.coef_[0]:.2f}, Intercept (g2): {g2_model.intercept_:.2f}")
print()
# plt.plot(t_fit, g2_fit_plot, 'r--', label=f'g2 fit: y = {params_g2[0]:.2f} * (1 - exp(-{params_g2[1]:.2f} * x)) + {params_g2[2]:.2f}')

# plt.plot(t, g3, 'y^', label='$g_{YIG}$ data (Py absent)',markersize=markersize)
# plt.plot(t, g3_fit_vals, 'y-', label=f'$g_{{\\text{{int}}}} = {g3_model.coef_[0]:.2f}t + {g3_model.intercept_:.2f}$')# (R^2 = {r2_g3:.2f})$')
# plt.xticks([.02,.06,.1])
# plt.yticks([.05,.15,.25])
# Customize plot
axs.set_xlabel('t',fontsize=labelsize)
axs.set_ylabel('$g$',fontsize=labelsize)
axs.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
axs.tick_params(axis='both', which='minor', labelsize=8, direction='in')
# axs[0].text(.005,.24,'(a)',fontsize=labelsize)
# plt.title(f'Saturation Fits for g1 and g2 (R^2: g1 = {r2_g1:.2f}, g2 = {r2_g2:.2f})')
# plt.legend(fontsize=15)
# plt.grid()
# plt.show()
# plt.savefig(os.path.join("tentative","images",'g1g2_vs_t.png'), dpi=300, bbox_inches='tight')

filename = 'combined_plots_params.csv'
data = pd.read_csv(filename)

# Extract columns
t = data['t'].values
g1 = data['g1'].values
g2 = data['g2'].values

# Fit g1 vs g2 to a line

slope, intercept, r_value, p_value, std_err = linregress(g2, g1)

# Print the fit parameters
print("Fit parameters for g1 vs g2:")
print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")
print()

# Plot the fitted line


# plt.figure(figsize=(9, 6))

# axs.plot(g2,g1,'go',markersize=markersize,label='Data')
# axs.plot(g2, slope * g2 + intercept, 'g-', label=f'Fit: $g_{{Py}} = {slope:.2f} g_{{YIG}} + {intercept:.2f}$')
# axs.set_xlabel("$g_2$",fontsize=labelsize)
# axs.set_ylabel("$g_1$",fontsize=labelsize)
# # plt.xticks([.05,.15,.25])
# # plt.yticks([.12,.15,.2])
# axs.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
# axs.tick_params(axis='both', which='minor', labelsize=8, direction='in')
# axs.text(.003,.195,'(c)',fontsize=labelsize)
# # plt.legend(fontsize=12)
# # plt.legend(fontsize=15)
# plt.tight_layout()
# plt.savefig(os.path.join("tentative","images","g2_vs_g1.png"),dpi=300,bbox_inches='tight')
# plt.close()

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
axs.plot(t_values, J_values[:len(t_values)], 'bx', label='Py absent',markersize=markersize2)
axs.plot(t_fit, J_fit, 'b-', linewidth=linewidth)  # Reduced line weight
# axs.text(.02,.29,'(b)',fontsize=labelsize)
axs.tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False, labelbottom=False, labelleft=False)

# Print the slope and intercept of the fit
print("Lone YIG fit:")
print(f"Slope: {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print()
#     # file_path_full = os.path.join(root,f"{type}.csv")
#     idx = 0
#     for _, file in enumerate(files):
#         if len(file.split('.')) == 3:
#             t_val = float('0.' + file.split('.'))
#             plt.plot(t_val, J_values[idx], 'ro')
#             idx += 1


# plt.legend(fontsize=15)
# plt.xticks([.02,.06,.1])
# plt.yticks([.13,.2,.3])
# plt.title('$g_{YIG}$ vs t')
# axs.set_xlabel('t',fontsize=labelsize)
# axs.set_ylabel('$g_2$',fontsize=labelsize)
# plt.grid()
# axs.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
# axs.tick_params(axis='both', which='minor', labelsize=8, direction='in')
plt.tight_layout()
# plt.savefig(os.path.join("tentative","images","Jvst.png"),dpi=300, bbox_inches='tight')
plt.legend(fontsize=fontsize)
plt.savefig(os.path.join("tentative","images","toc-image.png"), dpi=300, bbox_inches='tight')