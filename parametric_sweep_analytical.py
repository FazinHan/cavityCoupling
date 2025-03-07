import numpy as numpy
from mpi4py import MPI
from triple_mode_sweeps import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

array_density = 10

v1_array = np.linspace(0, .001, array_density)
v2_array = np.linspace(0, .01, array_density)

v1_value = v1_array[rank%array_density]
v2_value = v2_array[rank//array_density]

print(f"computing s21 for rank {rank} : v1 = {v1_value} and v2 = {v2_value}")

compute_s21(v1_value,v2_value)

print(f"rank {rank} done : v1 = {v1_value} and v2 = {v2_value}")