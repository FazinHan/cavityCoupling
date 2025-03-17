import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from parameters import *

target_shape = (axis_resolution, axis_resolution)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_dir = os.path.join(parent_dir,'data','yig_t_sweep_outputs')
file_paths = [os.path.join(source_dir,f) for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and 'csv' in f]

os.makedirs(os.path.join(parent_dir,'data','reduced_data'), exist_ok=True)
target_dir = os.path.join(parent_dir,'data','reduced_data')

for file in file_paths:
    filename = os.path.basename(file)
    data = pandas.read_csv(file)
    full_array = data.to_numpy()
    freq = full_array[:,0]
    freq = np.linspace(freq[0],freq[-1],axis_resolution)
    s21_array = full_array[:,1:]
    Hdc = data.columns[1:].to_numpy(dtype=float)
    Hdc = np.linspace(Hdc[0],Hdc[-1],axis_resolution)

    factor_x = s21_array.shape[0] // axis_resolution
    factor_y = s21_array.shape[1] // axis_resolution
    reduced = block_reduce(s21_array, (factor_x, factor_y), np.mean)
    # freq = fft2(s21_array)
    # freq_cropped = freq[:axis_resolution, :axis_resolution]  # Crop high frequencies
    # arr_reduced = np.abs(ifft2(freq_cropped))

    plt.pcolormesh(reduced, cmap='jet')
    plt.show()