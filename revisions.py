import numpy as np
import matplotlib.pyplot as plt
import os, time

parallel = False

os.makedirs(os.path.join('new','figures'),exist_ok=True)
os.makedirs(os.path.join('new','data'), exist_ok=True)

grid_resolution = 100
sweep_resolution = 100

default_values = {
    'g1': 0.2,
    'g2': 0.1,
    'g3': 0.0,
    'alpha_1': 1e-2,
    'alpha_2': 1e-4,
    'lambda_1': 1e-2,
    'lambda_2': 1e-4,
    'lambda_r': .08,
    'beta': 1e-4,
}

def s21_theoretical(
    w, H,
    g1=default_values['g1'],
    g2=default_values['g2'],
    g3=default_values['g3'],
    alpha_1=default_values['alpha_1'],
    alpha_2=default_values['alpha_2'],
    lambda_1=default_values['lambda_1'],
    lambda_2=default_values['lambda_2'],
    lambda_r=default_values['lambda_r'],
    beta=default_values['beta']
):
    
    matrices = []

    for i in range(3):
        for j in range(3):
            mat = np.zeros((3, 3), dtype=int)
            mat[i, j] = 1
            matrices.append(mat)
    
    # Constants
    gyro1 = 2.94e-3
    gyro2 = 1.76e-2 / (2 * np.pi)
    M1 = 10900.0  # Py
    M2 = 1750.0   # YIG

    gamma_1 = 2 * np.pi * lambda_1**2
    gamma_2 = 2 * np.pi * lambda_2**2
    gamma_r = 2 * np.pi * lambda_r**2

    alpha_r = beta

    omega_1 = gyro1 * np.sqrt(H * (H + M1))
    omega_2 = gyro2 * np.sqrt(H * (H + M2))
    omega_r = 5.0 # Assuming this was a constant, in Julia it's 5

    tomega_1 = omega_1 - 1j * (alpha_1 + gamma_1)
    tomega_2 = omega_2 - 1j * (alpha_2 + gamma_2)
    tomega_r = omega_r - 1j * (alpha_r + gamma_r)

    # Construct the matrix M
    M = np.array([
        [w - tomega_1,                             -g1 + 1j * np.sqrt(gamma_1 * gamma_r), -g3 + 1j * np.sqrt(gamma_1 * gamma_2)],
        [-g1 + 1j * np.sqrt(gamma_1 * gamma_r),    w - tomega_r,                          -g2 + 1j * np.sqrt(gamma_2 * gamma_r)],
        [-g3 + 1j * np.sqrt(gamma_1 * gamma_2),    -g2 + 1j * np.sqrt(gamma_2 * gamma_r),   w - tomega_2]
    ], dtype=complex)

    B = np.array([np.sqrt(gamma_1), np.sqrt(gamma_r), np.sqrt(gamma_2)], dtype=complex)

    try:
        inv_M = np.linalg.inv(M)
        B_row_real = np.array([np.sqrt(gamma_1), np.sqrt(gamma_r), np.sqrt(gamma_2)]) 
        result = B_row_real @ inv_M @ B_row_real
    except np.linalg.LinAlgError:
        print(f"Singular matrix for w={w}, H={H}. Returning NaN.")
        return np.nan

    return np.abs(result)

def coupling(plot=True,**parameters):
    
    def analyze_array(arr, slope, name):
        # Length
        length = len(arr)
        # Number of zeros at the end
        num_zeros_end = 0
        for val in arr[::-1]:
            if val == 0:
                num_zeros_end += 1
            else:
                break
        # Largest two elements
        if np.count_nonzero(arr) >= 2:
            largest_indices = np.argpartition(arr, -2)[-2:]
            largest_values = arr[largest_indices]
            largest_values = np.sort(largest_values)[::-1]
        else:
            largest_values = arr[np.nonzero(arr)]
        # print(f"{name}:")
        # print(f"  Length: {length}")
        # print(f"  Number of zeros at the end: {num_zeros_end}")
        # print(f"  Largest two elements: {largest_values}")
        # print(f"  Difference between largest two elements: {np.diff(largest_values).__abs__()}")
        coupling = np.diff(omega[largest_indices]).__abs__()[0]/np.sin(-np.arctan(slope))
        # print(f"  Coupling gap: {coupling:.5f}\n")

        return coupling, largest_indices

    omega = np.linspace(4.3,6.1, grid_resolution)  # Example frequency range
    # omega = np.linspace(0,10, grid_resolution)  # Example frequency range
    H = np.linspace(0, 1600, grid_resolution)  # Example magnetic field range

    s21 = np.zeros((omega.size, H.size))

    for i, w in enumerate(omega):
        for j, h in enumerate(H):
            s21[i,j] = s21_theoretical(w,h,**parameters)

    line_a = [(238,5.063),(290,4.917)]
    line_b = [(1093,5.076),(1137,4.931)]

    slope_a, intercept_a = np.polyfit(*zip(*line_a), 1)
    slope_b, intercept_b = np.polyfit(*zip(*line_b), 1)

    lina = lambda x: slope_a * x + intercept_a
    linb = lambda x: slope_b * x + intercept_b

    s21_a = np.zeros_like(lina(H))
    s21_b = np.zeros_like(linb(H))
    idx=0
    for hdx, h in enumerate(H):
        if lina(h) > omega.min() and lina(h) < omega.max():
            atol = .5*np.min(np.diff(omega))
            closeness_array = np.isclose(omega, lina(h), atol=atol)
            z=np.count_nonzero(closeness_array)
            if z == 2:
                true_indices = np.where(closeness_array)[0]
                idx_to_switch = np.random.choice(true_indices)
                closeness_array[idx_to_switch] = False
            s21_a[idx] = s21[np.where(closeness_array),hdx]
            idx+=1
    idx=0
    for hdx, h in enumerate(H):
        if linb(h) > omega.min() and linb(h) < omega.max():
            atol = .5 * np.min(np.diff(omega))
            closeness_array = np.isclose(omega, linb(h), atol=atol)
            z = np.count_nonzero(closeness_array)
            if z == 2:
                true_indices = np.where(closeness_array)[0]
                idx_to_switch = np.random.choice(true_indices)
                closeness_array[idx_to_switch] = False
            s21_b[idx] = s21[np.where(closeness_array), hdx]
            idx+=1

    
    c1, largest_indices_a = analyze_array(s21_a, slope_a, "s21_a")
    c2, largest_indices_b = analyze_array(s21_b, slope_b, "s21_b")

    if plot:
        plt.pcolormesh(H,omega,s21)
        plt.plot(H, lina(H), 'r--', label=f'coupling = {c1:.5f}')
        plt.plot(H, linb(H), 'g--', label=f'coupling = {c2:.5f}')
        plt.plot(H[largest_indices_a], s21_a[largest_indices_a], 'ko', markersize=5)
        plt.xlabel("H (kOe)")
        plt.ylabel("$\\omega$")
        plt.ylim(omega.min(), omega.max())
        plt.legend()
        plt.title(parameters)
        plt.tight_layout()
        plt.colorbar(label='$|S_{21}|$')
        plt.show()

    return c1, c2

if __name__ == "__main__":
    # coupling(g1=.1)

    sweep_array = np.linspace(0,1, sweep_resolution)

    params = default_values.keys()

    total_time = 0
    counter = 0
    total_count = len(params) * (len(params) - 1)

    # if not parallel:
    #     print("Running sequentially...")
    #     for param1 in params:
    #         for param2 in params:
    #             if param1 != param2:
    #                 t0 = time.time()
    #                 print(f"Analyzing {param1} and {param2}")
    #                 c1_array = np.zeros((sweep_resolution, sweep_resolution))
    #                 c2_array = np.zeros((sweep_resolution, sweep_resolution))
    #                 for idx,value1 in enumerate(sweep_array):
    #                     for jdx,value2 in enumerate(sweep_array):
    #                         parameters = default_values.copy()
    #                         parameters[param1] = value1
    #                         parameters[param2] = value2
    #                         c1_array[idx,jdx], c2_array[idx,jdx] = coupling(plot=False, **parameters)
    #                 print(f"Finished analyzing {param1} and {param2}\n")
    #                 plt.pcolormesh(sweep_array, sweep_array, c1_array, shading='auto')
    #                 plt.colorbar(label=f'coupling strength')
    #                 plt.xlabel(param1)
    #                 plt.ylabel(param2)
    #                 plt.title(f'Coupling {param1} vs {param2}')
    #                 plt.tight_layout()
    #                 plt.savefig(os.path.join('new','figures',f'coupling_{param1}_vs_{param2}.png'))
    #                 print(f"Saved figure for {param1} vs {param2}")
    #                 plt.close()
    #                 file = os.path.join('new','data',f'coupling_{param1}_vs_{param2}.npz')
    #                 np.savez(file, {'c1': c1_array, 'c2': c2_array, 'sweep_array': sweep_array})
    #                 print(f"Saved data for {param1} vs {param2} to {file}")
    #                 t1 = time.time()
    #                 print(f"Time taken for {param1} and {param2}: {t1 - t0:.2f} seconds\n")
    #                 counter += 1
    #                 total_time += (t1 - t0)
    #                 print("Estimated time remaining: {:.2f} seconds".format((total_count - counter) * (total_time / counter)))
    # else: # written by copilot, dunno if it works
    #     print("Running in parallel mode...")
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
    #     size = comm.Get_size()
    #     indices = np.arange(len(params),dtype=int)
    #     param_pairs = [(params[i], params[j]) for i in indices for j in indices if i != j]
    #     total_count = len(param_pairs)
    #     local_pairs = np.array_split(param_pairs, size)[rank]
    #     local_results = []
    #     for param1, param2 in local_pairs:
    #         t0 = time.time()
    #         print(f"Rank {rank} analyzing {param1} and {param2}")
    #         c1_array = np.zeros((sweep_resolution, sweep_resolution))
    #         c2_array = np.zeros((sweep_resolution, sweep_resolution))
    #         for idx, value1 in enumerate(sweep_array):
    #             for jdx, value2 in enumerate(sweep_array):
    #                 parameters = default_values.copy()
    #                 parameters[param1] = value1
    #                 parameters[param2] = value2
    #                 c1_array[idx, jdx], c2_array[idx, jdx] = coupling(plot=False, **parameters)
    #         print(f"Rank {rank} finished analyzing {param1} and {param2}\n")
    #         plt.pcolormesh(sweep_array, sweep_array, c1_array, shading='auto')
    #         plt.colorbar(label=f'coupling strength')
    #         plt.xlabel(param1)
    #         plt.ylabel(param2)
    #         plt.title(f'Coupling {param1} vs {param2}')
    #         plt.tight_layout()
    #         plt.savefig(os.path.join('new', 'figures', f'coupling_{param1}_vs_{param2}_rank{rank}.png'))
    #         print(f"Rank {rank} saved figure for {param1} vs {param2}")
    #         plt.close()
    #         file = os.path.join('new', 'data', f'coupling_{param1}_vs_{param2}_rank{rank}.npz')
    #         np.savez(file, {'c1': c1_array, 'c2': c2_array, 'sweep_array': sweep_array})
    #         print(f"Rank {rank} saved data for {param1} vs {param2} to {file}")
    #         t1 = time.time()
    #         # print(f"Rank {rank} time taken for {param1} and {param2}: {t1 - t0:.2f} seconds\n")
    #         local_results.append((param1, param2, c1_array, c2_array))
    #         # print(f"Rank {rank} estimated time remaining: {((total_count - len(local_pairs)) * (t1 - t0)) / size:.2f} seconds")
    #     # Gather results from all ranks
    #     all_results = comm.gather(local_results, root=0)
    #     if rank == 0:
    #         for result in all_results:
    #             for param1, param2, c1_array, c2_array in result:
    #                 print(f"Rank {rank} processed {param1} and {param2}")
    #                 plt.pcolormesh(sweep_array, sweep_array, c1_array, shading='auto')
    #                 plt.colorbar(label=f'coupling strength')
    #                 plt.xlabel(param1)
    #                 plt.ylabel(param2)
    #                 plt.title(f'Coupling {param1} vs {param2}')
    #                 plt.tight_layout()
    #                 plt.savefig(os.path.join('new', 'figures', f'coupling_{param1}_vs_{param2}.png'))
    #                 print(f"Saved figure for {param1} vs {param2}")
    #                 plt.close()
    #                 file = os.path.join('new', 'data', f'coupling_{param1}_vs_{param2}.npz')
    #                 np.savez(file, {'c1': c1_array, 'c2': c2_array, 'sweep_array': sweep_array})
    #                 print(f"Saved data for {param1} vs {param2} to {file}")

    for param in params:
        t0 = time.time()
        print(f"Analyzing {param} sweep")
        c1_array = np.zeros(sweep_resolution)
        c2_array = np.zeros(sweep_resolution)
        for idx, value1 in enumerate(sweep_array):
            parameters = default_values.copy()
            parameters[param] = value1
            c1_array[idx], c2_array[idx] = coupling(plot=False, **parameters)
        print(f"Finished analyzing {param} sweep\n")
        # plt.pcolormesh(sweep_array, sweep_array, c1_array, shading='auto')
        # plt.colorbar(label=f'coupling strength')
        plt.plot(sweep_array, c1_array, label='c1')
        plt.plot(sweep_array, c2_array, label='c2')
        plt.legend()
        plt.xlabel(param)
        plt.ylabel('Coupling strength')
        plt.title(f'Coupling strength vs {param}')
        plt.tight_layout()
        plt.savefig(os.path.join('new', 'figures', f'coupling_vs_{param}.png'))
        print(f"Saved figure for {param}")
        plt.close()
        file = os.path.join('new', 'data', f'coupling_vs_{param}.npz')
        np.savez(file, {'c1': c1_array, 'c2': c2_array, 'sweep_array': sweep_array})
        print(f"Saved data for {param} to {file}")
        t1 = time.time()
        total_time += (t1 - t0)
        counter += 1
        print(f"Time taken for {param} sweep: {t1 - t0:.2f} seconds\n")

    print("All analyses completed.")
    print(f"Total time taken: {total_time:.2f} seconds")
