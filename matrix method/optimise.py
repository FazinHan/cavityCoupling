from functions import s21
from scipy.optimize import minimize 
import numpy as np
import os
import pandas
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

observation_data_file_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data','yig_t_sweep_outputs')# 'yig_t_0.06.csv')

observation_data_files = [os.path.join(observation_data_file_dir,f) for f in os.listdir(observation_data_file_dir) if f.endswith('.csv')]
observation_data_files.sort()

observation_data = pandas.read_csv(observation_data_files[rank],header=None).to_numpy()
observation_s21 = observation_data[1:,1:]

hdc = np.linspace(observation_data[0,1], observation_data[0,-1], observation_data.shape[1]-1)
freq = np.linspace(observation_data[1,0], observation_data[-1,0], observation_data.shape[0]-1)

# print(hdc)

def loss(params):
    gamma_1, gamma_2, gamma_r, alpha_1, alpha_2, alpha_r, g1, g2 = params
    s21_arr = np.array([[s21(w, h, gamma_1=gamma_1, gamma_2=gamma_2, gamma_r=gamma_r, alpha_1=alpha_1, alpha_2=alpha_2, alpha_r=alpha_r, g1=g1, g2=g2)[0,0] for h in hdc] for w in freq])
    return np.sum(np.abs(s21_arr + observation_s21))

init_guess = [0.0001, 0.008, 0.02, 1e-2, 1e-5, 1e-4, 0.1, 0.1]

# print(s21(freq[0], hdc[0], gamma_1=0.0001, gamma_2=0.008, gamma_r=0.02))
# import time; t0 = time.time()
# print(loss([0.0001, 0.008, 0.02, 0, 0, 0, 0.1, 0.1]))
# print(f"time taken: {time.time() - t0:.2f} seconds")

bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
res = minimize(loss, init_guess, method='L-BFGS-B', bounds=bounds)
print(observation_data_files[rank+1],":")
print(res.x)
print("fun:", res.fun)