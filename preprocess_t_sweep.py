# Re-importing necessary libraries for processing
import re
import pandas as pd
import numpy as np
import os
import csv

# Define the file path again
file_path = "data\\raw\\yig lone t sweep.txt"

# Reinitialize datasets storage
datasets = {}

# Read and parse the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Regex patterns for extracting parameters and data
param_pattern = re.compile(r"#Parameters = {(.*?)}")
data_pattern = re.compile(r"([\d\.E\-\+]+)\s+([\d\.E\-\+]+)")

# Initialize variables for processing
current_params = {}
current_data = []

for line in lines:
    # Match parameter lines to extract values
    param_match = param_pattern.match(line)
    if param_match:
        # Save the dataset from the previous parameter block
        if current_params and current_data:
            yig_t = current_params['yig_t']
            Hdc = current_params['Hdc']
            df = pd.DataFrame(current_data, columns=["Frequency", "S21"])
            df["Hdc"] = Hdc
            if yig_t not in datasets:
                datasets[yig_t] = []
            datasets[yig_t].append(df)
        # Update parameters for the new block
        params = param_match.group(1)
        current_params = {key: float(value) for key, value in (item.split('=') for item in params.split('; '))}
        current_data = []
    # Match and append data rows
    elif data_match := data_pattern.match(line):
        current_data.append([float(data_match.group(1)), float(data_match.group(2))])

# Save the last dataset if it exists
if current_params and current_data:
    yig_t = current_params['yig_t']
    Hdc = current_params['Hdc']
    df = pd.DataFrame(current_data, columns=["Frequency", "S21"])
    df["Hdc"] = Hdc
    if yig_t not in datasets:
        datasets[yig_t] = []
    datasets[yig_t].append(df)

# Save each yig_t's data into separate CSV files
output_paths = []
for yig_t, data_list in datasets.items():
    combined_df = pd.concat(data_list, ignore_index=True)

    output_path = f"data\\lone_t_sweep_yig\\yig_t_{yig_t}.csv"
    
    pt = combined_df.pivot(index='Frequency', columns='Hdc', values='S21')
    pt.dropna(inplace=True)
    # Find all rows that have at least one empty value

    # Calculate the mean and standard deviation of the data
    mean_value = np.mean(pt.values)
    std_value = np.std(pt.values)

    # Define a threshold to discard values far from the mean
    threshold = 10 * std_value

    # Discard values far from the mean
    pt.values[np.abs(pt.values - mean_value) > threshold] = 0
    pt.values[pt.values > 0] = 0

    pt.index.name = '0'

    pt.to_csv(output_path)

    print(f"Data preprocessed for yig_t={yig_t}mm")
    output_paths.append(output_path)

print(output_paths)
