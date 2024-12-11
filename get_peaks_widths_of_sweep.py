import matlab.engine
import os

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define the argument as a string

cwd = os.getcwd()

eng.addpath(cwd, nargout=0)

# Extract all filenames from 'data\yig_t_sweep_outputs'
directory = os.path.join(cwd, 'data', 'yig_t_sweep_outputs')
filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
data_files = [i.strip('.csv') for i in filenames]

# print(data_files)

os.makedirs(os.path.join(directory,'peaks_widths'), exist_ok=True)

for data in data_files:
    eng.s21_optimiser(data, nargout=0)
    print(f"Processed {data}.")
# Close the MATLAB engine
eng.quit()

