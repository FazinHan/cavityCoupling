import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = 'combined_plots_params.csv'
data = pd.read_csv(filename)

# Extract columns
t = data['t'].values
g1 = data['g1'].values
g2 = data['g2'].values

plt.figure(figsize=(9, 6))

plt.plot(g2,g1,'r.',markersize=16)
plt.xlabel("$g_{YIG}$",fontsize=16)
plt.ylabel("$g_{Py}$",fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.tight_layout()
plt.savefig("images\\g2_vs_g1.png",dpi=300,bbox_inches='tight')
plt.close()

# exit()
