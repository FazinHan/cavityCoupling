import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# List of CSV file paths
for roots, dirs, files in os.walk("data\\yig_t_sweep_outputs\\intermediaries"):
    csv_files = [os.path.join(roots, file) for file in files if file.endswith('.csv')]

for file in csv_files:
    # Load the data
    data = pd.read_csv(file)

    # Extract yig_t value from the filename
    yig_t_value = file.split('_')[-1].replace('.csv', '')

    # Pivot the data to create a grid for plotting
    pivot_table = data.pivot_table(values='S21', index='Frequency', columns='Hdc', dropna=False)

    # Calculate the mean and standard deviation of the data
    mean_value = np.mean(pivot_table.values)
    std_value = np.std(pivot_table.values)

    # Define a threshold to discard values far from the mean
    threshold = 8 * std_value

    # Discard values far from the mean
    pivot_table.values[np.abs(pivot_table.values - mean_value) > threshold] = 0
    pivot_table.values[pivot_table.values > 0] = 0

    freq = np.array(pivot_table.index)
    hdc = np.array(pivot_table.columns)
    s21 = np.array(pivot_table.values)

    # print(freq.reshape(1,-1).shape, hdc.shape, s21.shape)
    # print(s21)
    # exit()

    csv_file = open(f"data\\yig_t_sweep_outputs\\yig_t_{yig_t_value}.csv", "w",newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([0]+list(hdc))
    freq_s21 = np.hstack((freq.reshape(-1, 1), s21))
    csv_writer.writerows(freq_s21)
    csv_file.close()

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(pivot_table.columns, pivot_table.index, pivot_table.values, cmap='inferno_r', levels=100)
    plt.colorbar(label="S21 (dB)")
    plt.title(f"S21 Heatmap (yig_t={yig_t_value}mm)")
    plt.xlabel("Hdc")
    plt.ylabel("Frequency (GHz)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs("results\\yig_t_sweep_plots", exist_ok=True)
    plt.savefig(f"results\\yig_t_sweep_plots\\yig_t_sweep_plot_{int(float(yig_t_value)*1e3)}um.png")
    print(f"Plot saved for yig_t={yig_t_value}mm")