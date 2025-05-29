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
    Hdc = np.array([0] + Hdc.tolist())

    factor_x = s21_array.shape[0] // axis_resolution
    factor_y = s21_array.shape[1] // axis_resolution
    reduced = block_reduce(s21_array, (factor_x, factor_y), np.mean)
    reduced = reduced[:axis_resolution,:axis_resolution] # reduce size 251 to 250
    full_array = np.concatenate((freq[:,np.newaxis], reduced), axis=1)
    full_array = np.concatenate((Hdc[np.newaxis,:], full_array), axis=0)

    df = pandas.DataFrame(full_array)
    df.to_csv(os.path.join(target_dir,filename), index=False, header=False)