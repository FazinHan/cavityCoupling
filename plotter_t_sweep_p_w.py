import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from get_peaks_widths_of_sweep import get_peaks_widths_of_sweep

get_peaks_widths_of_sweep(0.5,.3,.1)

csv_files = []

# List of CSV file paths
for roots, dirs, files in os.walk("data\\yig_t_sweep_outputs\\peaks_widths"):
    csv_files = [os.path.join(roots, file) for file in files if file.endswith('.csv')]# if dirs == ['intermediaries', 'peaks_widths']]

os.makedirs("results\\yig_t_sweep_plots\\peaks_widths", exist_ok=True)

for file in csv_files:
    # Load the data
    pivot_table = pd.read_csv(file)

    # Extract yig_t value from the filename
    yig_t_value = os.path.basename(file).lstrip('yig_t_').rstrip('_peaks_widths.csv')
    print(f"Processing yig_t={yig_t_value}mm")

    plt.figure()
    plt.plot(pivot_table['xc1'],'o',markersize=2)
    plt.plot(pivot_table['xc2'],'o',markersize=2)
    plt.savefig("results\\yig_t_sweep_plots\\peaks_widths\\"+f"peaks_widths_{int(float(yig_t_value)*1000)}um.png")

    print(f"Plot saved for yig_t={yig_t_value}mm")