import sys, os
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

fontsize = int(sys.argv[1])
labelsize = fontsize + int(sys.argv[2])

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


plt.figure(figsize=(9, 6))

plt.plot(g2,g1,'go',markersize=16,label='Data')
plt.plot(g2, slope * g2 + intercept, 'g-', label=f'Fit: $g_{{Py}} = {slope:.2f} g_{{YIG}} + {intercept:.2f}$')
plt.xlabel("$g_2$",fontsize=labelsize)
plt.ylabel("$g_1$",fontsize=labelsize)
# plt.xticks([.05,.15,.25])
# plt.yticks([.12,.15,.2])
plt.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
plt.tick_params(axis='both', which='minor', labelsize=8, direction='in')
# plt.legend(fontsize=12)
# plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join("tentative","images","g2_vs_g1.png"),dpi=300,bbox_inches='tight')
plt.close()

# exit()
