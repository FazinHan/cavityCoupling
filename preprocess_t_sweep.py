import sys, csv, os
import numpy as np
import re

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

def extract_data(file_path, yig_t):
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
                match2 = re.search(r'yig_t=([0-9.]+)', line)
                if float(match2.group(1)) != yig_t:
                    print(f'YIG thickness: {float(match2.group(1))}')
                    # return frequency, H_params, s21, float(match2.group(1))
                    continue
                if match:
                    # First set of parameters
                    Hdc_value = float(match.group(1))
                    H_params.append(Hdc_value)
                    param_read = True  # Mark that parameters are read
                    continue

            # For subsequent sets of parameters, only extract Hdc
            if 'Parameters =' in line and param_read:
                match = re.search(r'Hdc=([0-9.]+)', line)
                match2 = re.search(r'yig_t=([0-9.]+)', line)
                if float(match2.group(1)) != yig_t:
                    # return frequency, H_params, s21, float(match2.group(1))
                    continue
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

file_path = 'data\\raw\\yig_t_sweep.txt'  # Replace with your actual file path
# file_path = os.path.join('data', 'raw', file_name)  
def find_last_yig_t(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'yig_t=' in line:
                match = re.search(r'yig_t=([0-9.]+)', line)
                if match:
                    # print(f'Last YIG thickness: {float(match.group(1))}')
                    return float(match.group(1))
    raise ValueError("No yig_t value found in the file.")

def find_last_Hdc(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'Hdc=' in line:
                match = re.search(r'Hdc=([0-9.]+)', line)
                if match:
                    print(f'Last Hdc value: {float(match.group(1))}')
    raise ValueError("No Hdc value found in the file.")

yig_t_start = 0.02
yig_t_end = find_last_yig_t(file_path)
yig_t_list = np.arange(yig_t_start,yig_t_end,0.005)

for yig_t in yig_t_list:
    frequency, H_params, s21 = extract_data(file_path, yig_t)
    print(H_params.shape,s21.shape)
    H_params, s21 = sorter(H_params, s21)
    frequency = frequency[:s21.shape[0]] # Trim the frequency array to match the S21 data
    params = read_parameters(file_path)

    H_params1 = np.concatenate((np.array([0]),H_params))
    # H_params1[1:] = H_params

    data1 = np.concatenate((frequency.reshape(frequency.size,1),s21),axis=1)
    data = np.concatenate((H_params1.reshape(1,H_params1.size),data1),axis=0)

    # print(frequency.shape,H_params1.shape,s21.shape)
    print(data.shape)

    os.mkdir(os.path.join('data',os.path.basename(file_path).split('.')[0]))

    with open(os.path.join('data',os.path.basename(file_path).split('.')[0],f'_{yig_t}'+'.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f'File {yig_t} saved')