# Reload the necessary libraries and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the uploaded CSV file
file_path = 'combined_plots_params.csv'
data = pd.read_csv(file_path)

# Extract columns for fitting and plotting
t = data['t']
g1 = data['g1']
g2 = data['g2']

# Define the function to fit: y = a * log(x) + b
def fit_function(x, a, b):
    return a * np.log(x) + b

# def fit_function(x,a,b,c):
#     return a*(1-np.exp(-x/b))+c

# Fit g1 and g2 data
params_log_g1, _ = curve_fit(fit_function, t, g1)
params_log_g2, _ = curve_fit(fit_function, t, g2)

# Generate fitted values for plotting
t_fit = np.linspace(t.min(), t.max(), 100)
g1_log_fit = fit_function(t_fit, *params_log_g1)
g2_log_fit = fit_function(t_fit, *params_log_g2)

# Plot the data and logarithmic fits
plt.figure(figsize=(8, 5))

# Plot g1 data and log fit
plt.scatter(t, g1, label='g1 data', marker='o', color='blue')
plt.plot(t_fit, g1_log_fit, label=f'g1 log fit: a={round(params_log_g1[0],3)} b={round(params_log_g1[1],3)}', color='blue', linestyle='--')

# Plot g2 data and log fit
plt.scatter(t, g2, label='g2 data', marker='x', color='orange')
plt.plot(t_fit, g2_log_fit, label=f'g2 log fit: a={round(params_log_g2[0],3)} b={round(params_log_g2[1],3)}', color='orange', linestyle='--')

# Add labels, legend, and grid
plt.xlabel('t ($\\mu$m)')
plt.ylabel('g values (arb. units)')
plt.title('g1 and g2 vs t with logarithmic fits')
plt.legend()
plt.tight_layout()
plt.savefig("g vs t.png")
