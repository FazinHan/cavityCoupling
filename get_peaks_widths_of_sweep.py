import matlab.engine
import os
from plotter_t_sweep_p_w import plotter

# Start MATLAB engine
def get_peaks_widths_of_sweep(prominence, fine_smoothing, coarse_smoothing):
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

    for file in data_files:
        eng.s21_optimiser(file, prominence, fine_smoothing, coarse_smoothing, nargout=0)
        print(f"Processed {file}.")
    # Close the MATLAB engine
    eng.quit()

if __name__ == "__main__":
    get_peaks_widths_of_sweep(0.2,.1, 0) # ref line 24
    plotter()