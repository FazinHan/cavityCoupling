import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

csv_files = []

# List of CSV file paths
for roots, dirs, files in os.walk("data\\lone_t_sweep_yig"):
    # if dirs == ['intermediaries', 'peaks_widths']:
        # csv_files = [os.path.join(roots, file) for file in files if file.endswith('.csv')]# if dirs == ['intermediaries', 'peaks_widths']]
    csv_files = [os.path.join(roots, file) for file in files if file.endswith('.csv')]# if dirs == ['intermediaries', 'peaks_widths']]

for file in csv_files:
    # Load the data
    pivot_table = pd.read_csv(file)

    # Extract yig_t value from the filename
    yig_t_value = os.path.basename(file).split('_')[-1].replace('.csv', '')
    print(f"Processing yig_t={yig_t_value}mm")


    freq = np.array(pivot_table.index)
    hdc = np.array(pivot_table.columns)[1:].astype(float) # Skip the first column which is 'Frequency'
    s21 = pivot_table.to_numpy()[:,1:] # Skip the first column which is 'Frequency'
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(hdc, freq, s21, cmap='inferno_r', levels=100)
    plt.colorbar(label="S21 (dB)")
    plt.title(f"S21 Heatmap (yig_t={yig_t_value}mm)")
    plt.xlabel("Hdc")
    plt.ylabel("Frequency (GHz)")
    plt.xlim((1000,1600))
    plt.ylim((1250,1750))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs("results\\lone_yig_sweep", exist_ok=True)
    plt.savefig(f"results\\lone_yig_sweep\\yig_t_sweep_plot_{int(float(yig_t_value)*1e3)}um.png")
    print(f"Plot saved for yig_t={yig_t_value}mm")