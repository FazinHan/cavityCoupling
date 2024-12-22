import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from get_peaks_widths_of_sweep import get_peaks_widths_of_sweep
from scipy.interpolate import CubicSpline

csv_files = []

# List of CSV file paths
for roots, dirs, files in os.walk("data\\yig_t_sweep_outputs\\peaks_widths"):
    csv_files = [os.path.join(roots, file) for file in files if file.endswith('.csv')]# if dirs == ['intermediaries', 'peaks_widths']]

os.makedirs("results\\yig_t_sweep_plots\\peaks_widths", exist_ok=True)

for file in csv_files:
    # Load the data
    pivot_table = pd.read_csv(file)
    # Replace 'xc1' and 'xc2' with their cubic spline
    # cs_xc1 = CubicSpline(pivot_table.index, pivot_table['xc1'])
    # cs_xc2 = CubicSpline(pivot_table.index, pivot_table['xc2'])
    
    # pivot_table['xc1'] = cs_xc1(pivot_table.index)
    # pivot_table['xc2'] = cs_xc2(pivot_table.index)

    # Extract yig_t value from the filename
    yig_t_value = os.path.basename(file).lstrip('yig_t_').rstrip('_peaks_widths.csv')
    print(f"Processing yig_t={yig_t_value}mm")

    plt.figure()
    plt.plot(pivot_table['xc1'],'o',markersize=2)
    plt.plot(pivot_table['xc2'],'o',markersize=2)
    name = "results\\yig_t_sweep_plots\\peaks_widths\\"+f"peaks_widths_{int(float(yig_t_value)*1000)}um.png"
    plt.savefig(name)

    print(f"Plot saved to {name}")