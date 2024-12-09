# Re-importing necessary libraries for processing
import re
import pandas as pd
import os

# Define the file path again
file_path = "data\\raw\\yig_t_sweep_final.txt"

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
    output_dir = "data\\yig_t_sweep_outputs\\intermediaries"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}\\yig_t_{yig_t}.csv"
    combined_df.to_csv(output_path, index=False)
    output_paths.append(output_path)

output_paths
