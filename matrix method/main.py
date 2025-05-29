import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import a,b,c,w,x,y,z,g,h
import sympy as sp
from mpi4py import MPI
import os
from parameters import *
from functions import *
from pprint import pprint

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

try:
    chunk_size = axis_resolution//(size-1)
except ZeroDivisionError:
    chunk_size = 1

def parameter_sweep_plots():

    gamma1_arr = np.linspace(0,2*np.pi,axis_resolution)
    gamma2_arr = np.linspace(0,.12,axis_resolution)

    diffs = np.zeros((axis_resolution, axis_resolution))

    if rank == 0:
        for i in range(1, size):
            diffs += comm.Recv(source=i, tag=13)
    else:
        for i in range((rank-1)*chunk_size, rank*chunk_size):
            for j in range(axis_resolution):
                gamma1 = gamma1_arr[i]
                gamma2 = gamma2_arr[j]
                peak1, peak2 = s21_couplings(gamma_1=gamma1, gamma_2=gamma2)
                diffs[i,j] = np.diff([peak1, peak2])[0]
        comm.Send(np.array(diffs), dest=0, tag=13)

    for i in range(size * chunk_size, axis_resolution):
        for j in range(axis_resolution):
            gamma1 = gamma1_arr[i]
            gamma2 = gamma2_arr[j]
            peak1, peak2 = s21_couplings(gamma_1=gamma1, gamma_2=gamma2)
            diffs[i,j] = np.diff([peak1, peak2])[0]

    with open(os.path.join(os.path.dirname(__file__), 'diffs.npy'), "wb") as f:
        np.save(f, diffs)

    fig, axs = plt.subplots(1,2,sharey=False,figsize=(10,5))
    axs[0].pcolormesh(gamma1_arr, gamma2_arr, diffs[...,0])
    axs[0].set_xlabel('$2\\pi\\lambda_1^2$')
    axs[0].set_ylabel('$2\\pi\\lambda_2^2$')
    # axs[0].title('Difference between peaks in s21_1')
    # axs[0].colorbar()
    # plt.show()
    # print(diffs)
    axs[1].pcolormesh(gamma1_arr, gamma2_arr, diffs[...,1])
    axs[1].set_xlabel('$2\\pi\\lambda_1^2$')
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[1].set_ylabel('$2\\pi\\lambda_2^2$')
    # axs[1].title('Difference between peaks in s21_2')
    # axs[1].colorbar()
    axs[0].set_title('Py Coupling')
    # axs[0].plot(.08,1.9,'r9.')
    axs[1].set_title('YIG Coupling')
    # fig.suptitle("Peak separation not affected by damping")
    fig.tight_layout()
    fig.colorbar(axs[0].collections[0], ax=axs, location="right", use_gridspec=False)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'couplings.png'))
    plt.close()

    # diffs = np.array([[s21_couplings(gamma_1=gamma1, gamma_2=gamma2) for gamma1 in gamma1_arr] for gamma2 in gamma2_arr])[...,0]


def s21_theoretical_calculations():

    x_vals = np.linspace(0,2*np.pi,axis_resolution)
    y_vals = np.linspace(0,.12,axis_resolution)

    # x_vals = np.linspace(0, 10, resolution)  # Example x-axis values
    # y_vals = np.linspace(0, 10, resolution)  # Example y-axis values

    # Create the meshgrid and flatten into pairs
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    pairs = np.column_stack((X.ravel(), Y.ravel()))  # Shape (resolution^2, 2)

    # Split pairs into nearly equal chunks
    chunks = np.array_split(pairs, size)

    # Scatter chunks to each process
    local_chunk = np.zeros_like(chunks)

    comm.Scatter(chunks, local_chunk, root=0)

    print(s21())
    exit()

    # Compute on assigned pairs
    local_results = np.array([s21(gamma_1=x, gamma_2=y) for x, y in local_chunk])

    # Gather results at root
    gathered_results = comm.Gather(local_results, root=0)

    if rank == 0:

        ''' RETHINK THIS ENTIRE PART
        final_result = np.concatenate(gathered_results).reshape(axis_resolution, axis_resolution)  # Reshape for plotting
        save_array = np.zeros((final_result.shape[0]+1,final_result.shape[1]+1))
        save_array[1:,0] = y_vals
        save_array[0,1:] = x_vals
        with open('gamma1_gamma2_sweep.npy', 'wb') as file:
            np.save(file,save_array)
        '''    

if __name__ == "__main__":
    s21_theoretical_calculations()
