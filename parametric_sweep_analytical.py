import numpy as numpy
from mpi4py import MPI
from triple_mode_sweeps import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

v1_array = np.linspace(0, .001, 10)
v2_array = np.linspace(0, .01, 10)

print(f"computing s21 for rank {rank} : v1 = {v1_array[rank]} and v2 = {v2_array[rank]}")

compute_s21(v1_array[rank],v2_array[rank])

print(f"rank {rank} done : v1 = {v1_array[rank]} and v2 = {v2_array[rank]}")