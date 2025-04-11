import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

filename = 'combined_plots_params.csv'
data = pd.read_csv(filename)

# Extract columns
t = data['t'].values
g1 = data['g1'].values
g2 = data['g2'].values

# Fit g1 vs g2 to a line

slope, intercept, r_value, p_value, std_err = linregress(g2, g1)

# Print the fit parameters
print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")

# Plot the fitted line


plt.figure(figsize=(9, 6))

plt.plot(g2,g1,'r.',markersize=16,label='Data')
plt.xlabel("$g_{YIG}$",fontsize=16)
plt.ylabel("$g_{Py}$",fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.plot(g2, slope * g2 + intercept, 'b-', label=f'Fit: $g_{{Py}} = {slope:.2f} g_{{YIG}} + {intercept:.2f}$')
# plt.legend(fontsize=12)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("tentative\\images\\g2_vs_g1.png",dpi=300,bbox_inches='tight')
plt.close()

# exit()
