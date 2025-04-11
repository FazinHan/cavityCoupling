import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import pandas as pd

root = os.path.join(os.getcwd(),"data","lone_t_sweep_yig")
# J_values = np.linspace(0.09, 0.237, 7)
file_path = 'g_yig_vs_t.csv'
data = pd.read_csv(file_path)

t_values = data['t']
J_values = data['g']

# Fit the data to a line

# Extract t and J_values for fitting
# t_values = [float('0.' + file.split('.')[1]) for file in files if len(file.split('.')[1]) == 3]
slope, intercept, r_value, p_value, std_err = linregress(t_values, J_values[:len(t_values)])

plt.figure(figsize=(9, 6))

# Plot the fitted line
t_fit = np.linspace(min(t_values), max(t_values), 100)
J_fit = slope * t_fit + intercept
plt.plot(t_values, J_values[:len(t_values)], 'ro', label='Data')
plt.plot(t_fit, J_fit, 'b-', label=f'$g_{{YIG}}$ = {slope:.3f}t + {intercept:.3f}')
# for roots,dirs,files in os.walk(root):
#     # file_path_full = os.path.join(root,f"{type}.csv")
#     idx = 0
#     for _, file in enumerate(files):
#         if len(file.split('.')[1]) == 3:
#             t_val = float('0.' + file.split('.')[1])
#             plt.plot(t_val, J_values[idx], 'ro')
#             idx += 1


plt.legend(fontsize=15)

# plt.title('$g_{YIG}$ vs t')
plt.xlabel('t',fontsize=16)
plt.ylabel('$g_{YIG}$',fontsize=16)
# plt.grid()
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.tight_layout()
plt.savefig('tentative\\images\\Jvst.png',dpi=300, bbox_inches='tight')