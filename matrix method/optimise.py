from s21 import s21
from scipy.optimize import minimize 
import numpy as np
import os
import pandas

observation_data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data','yig_t_sweep_outputs', 'yig_t_0.06.csv')

observation_data = pandas.read_csv(observation_data_file,header=None).to_numpy()
observation_s21 = observation_data[1:,1:]

hdc = np.linspace(observation_data[0,1], observation_data[0,-1], observation_data.shape[1]-1)
freq = np.linspace(observation_data[1,0], observation_data[-1,0], observation_data.shape[0]-1)

print(hdc)

caluclated_s2 = pandas.read_csv