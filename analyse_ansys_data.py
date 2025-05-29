import os, csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

second_variable = sys.argv[1] if len(sys.argv) > 1 else 'lt[mm]'

# file = os.path.join('data', 'S Parameter 300 oe thickness sweep auto causal dielectrics.csv')
file = "position vary.csv"

data = pd.read_csv(file, header=0)

# unique_yig_t_values = data['yig_t [mm]'].unique()
# dataframes = {value: data[data['yig_t [mm]'] == value].drop(columns=['yig_t [mm]']) for value in unique_yig_t_values}

# df_100 = data[data['bias []'] == 100].drop(columns=['bias []'])
# df_23873 = data[data['bias []'] == 23873].drop(columns=['bias []'])
df_23873 = data

# pivoted_100 =  df_100.pivot(columns='yig_t [mm]', index='Freq [GHz]', values='dB(S(2,1)) []')
pivoted_23873 =  df_23873.pivot(columns=second_variable, index='Freq[GHz]', values='S21')

# print(pivoted_100)

# s21_array_100 = pivoted_100.to_numpy()
s21_array_23873 = pivoted_23873.to_numpy()

# s21_array_23873 = np.log(-s21_array_23873)

# fig, acs = plt.subplots(1,2, figsize=(10,5),sharey=True)
# acs[0].pcolormesh(pivoted_100.columns, pivoted_100.index, s21_array_100, cmap='inferno')
# acs[0].set_xlabel('YIG Thickness [mm]')
# acs[0].set_ylabel('Frequency [GHz]')
# acs[0].colorbar()
plt.pcolormesh(pivoted_23873.columns, pivoted_23873.index, s21_array_23873, cmap='inferno')
plt.xlabel(second_variable)
plt.ylabel('Frequency [GHz]')
# plt.colorbar()
# plt.yaxis.tick_right()
# plt.yaxis.set_label_position("right")

plt.savefig("ansys_sweep_uday.png")