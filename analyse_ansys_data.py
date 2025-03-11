import os, csv
import pandas as pd
import matplotlib.pyplot as plt

file = os.path.join('data', 'S Parameter Plot 2.csv')

data = pd.read_csv(file, header=0)

pivoted =  data.pivot(columns='bias []', index='Freq [GHz]', values='dB(S(2,1)) []')

# print(pivoted)

s21_array = pivoted.to_numpy()

plt.pcolormesh(pivoted.columns*1e-3, pivoted.index, s21_array, cmap='inferno')
plt.xlabel('Bias [kA/m]')
plt.ylabel('Frequency [GHz]')
plt.colorbar()
plt.savefig("results\\ansys_sweep.png")