import numpy as np
import re
import matplotlib.pyplot as plt
import os, sys
from statsmodels.nonparametric.smoothers_lowess import lowess

file_path = sys.argv[1]
# file_path = os.path.join('data','raw',file_name)
# file_path = "m=4 to.txt"

def read_parameters(file_path):
    with open(file_path, 'r') as file:
        # Read the entire file content
        content = file.read()

        # Use regex to extract the parameters inside the curly braces
        match = re.search(r'Parameters\s*=\s*{(.+?)}', content)
        if match:
            param_str = match.group(1)
            
            # Split the parameters into key-value pairs
            param_pairs = param_str.split(';')
            
            # Create a dictionary by splitting each pair at the '=' sign
            param_dict = {}
            for pair in param_pairs:
                if pair.strip():  # Ignore empty strings
                    key, value = pair.split('=')
                    param_dict[key.strip()] = float(value.strip())
                    
            return param_dict
        else:
            raise ValueError("No parameters found in the file.")

# Example usage

def extract_data(file_path):
    H_params = []  # To store Hdc values
    frequency = None  # To store the frequency values (we only need one set)
    s21 = []  # To store all S21 values for each Hdc

    with open(file_path, 'r') as file:
        lines = file.readlines()
        param_read = False  # To track if the parameters have been read
        current_s21 = []  # To temporarily store the current S21 values
        current_frequency = []  # To temporarily store the frequency values
        
        for line in lines:
            # Ignore comment lines
            # if line.startswith('#'):
                # continue

            # Extract the first set of parameters
            if 'Parameters =' in line and not param_read:
                match = re.search(r'Hdc=([0-9.]+)', line)
                if match:
                    # First set of parameters
                    Hdc_value = float(match.group(1))
                    H_params.append(Hdc_value)
                    param_read = True  # Mark that parameters are read
                    continue

            # For subsequent sets of parameters, only extract Hdc
            if 'Parameters =' in line and param_read:
                match = re.search(r'Hdc=([0-9.]+)', line)
                if match:
                    Hdc_value = float(match.group(1))
                    H_params.append(Hdc_value)
                    s21.append(current_s21)
                    current_s21 = []
                    continue

            # Handle frequency and S21 data lines
            data = line.strip().split()
            if len(data) == 2:
                freq_value = float(data[0])
                s21_value = float(data[1])

                if frequency is None:
                    current_frequency.append(freq_value)

                current_s21.append(s21_value)

        # Add the last block of S21 data
        if current_s21:
            # s21.append(smoothed_s21)
            s21.append(current_s21)

    # Final processing
    if frequency is None:
        frequency = np.array(current_frequency)

    H_params = np.array(H_params)
    s21 = np.array(s21).T  # Convert S21 to a 2D array (transpose for correct dimensions)

    return frequency, H_params, s21

def sorter(H_params, s21):
    # Sort the data based on Hdc values
    sorted_indices = np.argsort(H_params)
    H_params = H_params[sorted_indices]
    s21 = s21[:, sorted_indices]

    return H_params, s21

if __name__=="__main__":
    name = os.path.basename(file_path).split('.')[0]
    # file_path = f'{name}.png'  # Replace with your actual file path
    frequency, H_params, s21 = extract_data(file_path)
    # Apply LOWESS smoothing to each column of s21
    frequency = frequency[:s21.shape[0]] # Trim the frequency array to match the S21 data
    # H_params, s21 = sorter(H_params, s21)
    params = read_parameters(file_path)

    # print("Frequency:", frequency)
    # print("Hdc parameters:", H_params)
    # print("S21:", s21.shape)
    # print(params)

    plt.pcolormesh(H_params, frequency, s21, cmap='jet')
    plt.colorbar()
    plt.title(f'|$S_{{21}}$| for {name}')
    plt.xlabel('Hdc (Oe)')
    plt.ylabel('Frequency (GHz)')
    plt.ylim(4.3,6.3)
    plt.tight_layout()
    plt.savefig(os.path.join('results',f'{name}.png'))
    # plt.show()