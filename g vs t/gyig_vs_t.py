import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import pandas as pd
import sys

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

plt.figure(figsize=(9, 6))

# Plot the fitted line
t_fit = np.linspace(min(t_values), max(t_values), 100)
J_fit = slope * t_fit + intercept
plt.plot(t_values, J_values[:len(t_values)], 'bx', label='$g_{YIG}$ data (Py absent)',markersize=15)
plt.plot(t_fit, J_fit, 'b-', label=f'$g_{{YIG}} = {slope:.3f}t {'+'*(int(intercept>=0))} {intercept:.3f}$',markersize=15)

# Print the slope and intercept of the fit
print("Lone YIG fit:")
print(f"Slope: {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print()
#     # file_path_full = os.path.join(root,f"{type}.csv")
#     idx = 0
#     for _, file in enumerate(files):
#         if len(file.split('.')[1]) == 3:
#             t_val = float('0.' + file.split('.')[1])
#             plt.plot(t_val, J_values[idx], 'ro')
#             idx += 1


# plt.legend(fontsize=15)

# plt.title('$g_{YIG}$ vs t')
plt.xlabel('t',fontsize=sys.argv[1])
plt.ylabel('$g_2$',fontsize=sys.argv[1])
# plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20, direction='in')
plt.tick_params(axis='both', which='minor', labelsize=8, direction='in')
plt.tight_layout()
plt.savefig('tentative\\images\\Jvst.png',dpi=300, bbox_inches='tight')