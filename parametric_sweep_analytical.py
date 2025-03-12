import numpy as np
from mpi4py import MPI
from triple_mode_sweeps import *
import sympy as sp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

array_density = 10

v1_array = np.linspace(0, .001, array_density)
v2_array = np.linspace(0, .01, array_density)

v1_value = v1_array[rank%array_density]
v2_value = v2_array[rank//array_density]

print(f"computing s21 for rank {rank} : v1 = {v1_value} and v2 = {v2_value}")

# S = compute_s21(v1_value,v2_value)
S = sp.sin(sp.sqrt(2))


print(f"rank {rank} done : v1 = {v1_value} and v2 = {v2_value}")

# if rank != 0:
comm.send(S, dest=0, tag=rank)
if rank == 0:
    result_dict = dict()
    print("Gathering results...")
    # gathered_S = comm.recv()
    # result_dict = {f'({v1_value},{v2_value})': s for i, s in enumerate(gathered_S)}
    for sender in range(1,size):
        S = comm.recv(source=sender, tag=sender)
        result_dict[f'({v1_array[sender%array_density]},{v2_array[sender//array_density]})'] = S
    import pickle
    with open('s21_analytical_gathered.pkl', 'wb') as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Results gathered and saved to s21_analytical_gathered.pkl")
