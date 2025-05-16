import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats # For potential use if np.std isn't sufficient (e.g. for degrees of freedom)

# --- Placeholder for the missing Julia function ---
# You will need to implement the Python equivalent of your 'main_calc_real_part' function.
# Based on its usage, it likely takes Hlist, ωcn, and a variable number of other parameters,
# and returns a NumPy array (e.g., with shape (len(Hlist), 3) based on the loop in 'inter').
def main_calc_real_part(Hlist, omega_cn, *params):
    """
    Placeholder for the main_calc_real_part function.
    This function needs to be implemented based on your original Julia code.
    It should return a NumPy array. For example:
    return np.random.rand(len(Hlist), 3) # Replace with actual calculation
    """
    print(f"Warning: 'main_calc_real_part' is a placeholder and needs proper implementation.")
    # Example: Assuming it returns something with 3 columns as iterated in 'inter'
    # This is a dummy implementation.
    if len(params) > 0 : # Just to use params to avoid lint error
      pass
    return np.zeros((len(Hlist), 3)) # Adjust dimensions as per actual function

def main_plotter(type_str, optimized_params_input, lone=False):
    print(f"Running main_plotter for {type_str}")

    root_dir_parts = ["data"]
    if lone:
        root_dir_parts.append("lone_t_sweep_yig")
    else:
        root_dir_parts.append("yig_t_sweep_new")
    
    root = os.path.join(os.getcwd(), *root_dir_parts)
    file_path_full = os.path.join(root, f"{type_str}.csv")

    # Read the CSV file. Assuming the structure based on Julia's indexing:
    # First row is Hlist values (after first column)
    # First column is frequencies (after first row)
    # Data starts from the second row and second column.
    try:
        full_data = np.loadtxt(file_path_full, delimiter=',', dtype=float, skiprows=1) # Skip header row
        
        # To get Hlist from the first row of the CSV (which was skipped by np.loadtxt with skiprows=1)
        # We need to read it separately or assume a different structure if data files are structured differently.
        # For this translation, I'll assume Hlist might be part of the loaded data or needs a separate read.
        # Let's re-read the header for Hlist:
        with open(file_path_full, 'r') as f:
            header_line = f.readline().strip()
            Hlist = np.array(header_line.split(',')[1:], dtype=float)

        frequencies = full_data[:, 0]  # First column of data (after header)
        s21 = full_data[:, 1:]       # Rest of the data
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path_full}")
        return None, None, None
    except Exception as e:
        print(f"Error reading or processing file {file_path_full}: {e}")
        return None, None, None


    # Outlier removal, similar to Julia: s21_H[ (s21_H .- mean(s21_H)) ./ std(s21_H) .> 3 ] .= 0
    # Applying to a copy to avoid modifying original s21 if it's used elsewhere before this step.
    s21_processed = np.copy(s21)
    # Perform operations column-wise if that was the intent for each H field
    for i in range(s21_processed.shape[1]):
        column = s21_processed[:, i]
        if np.std(column) > 1e-9: # Avoid division by zero or tiny std dev
            condition = (column - np.mean(column)) / np.std(column) > 3
            column[condition] = 0 # Or np.nan, or re-assign column if needed
            s21_processed[:, i] = column
        else: # If std is zero (all values same), no outliers by this definition
            pass 
    s21 = s21_processed


    # The 'inter' function was defined inside 'main' in Julia.
    # It seemed related to an optimization objective which was commented out.
    # If 'main_calc_real_part' is not available, 'inter' cannot be fully implemented.
    # I'll include its structure for completeness if you need to uncomment/use it.
    
    # def inter(Hlist_local, omega_cn_local, params_local_inter):
    #     n_field = len(Hlist_local)
    #     n_freq = len(frequencies)
    #     array = np.zeros((n_field, n_freq))
        
    #     # This is the critical call to the function you need to define
    #     data_points = main_calc_real_part(Hlist_local, omega_cn_local, *params_local_inter)
        
    #     for i in range(data_points.shape[1]): # Assuming data_points has columns to iterate (e.g., 3)
    #         for idx, point in enumerate(data_points[:, i]):
    #             if not np.isnan(point): # Check for NaN to prevent errors with argmin
    #                 index = np.argmin(np.abs(frequencies - point))
    #                 if idx < n_field and index < n_freq : # Boundary check
    #                     array[idx, index] += 1
            
    #     array = array.T  # Transpose
        
    #     sq_error = (s21 + array)**2 # Note: original was s21 .+ array, if array is meant to be subtractive, adjust
    #     return np.sum(sq_error)

    # omega_cn = optimized_params_input[0]
    # current_optimized_params = optimized_params_input[1:]

    # objective = lambda p: inter(Hlist, omega_cn, p)

    # Optimization part was commented out in Julia
    # lower = [5.06, 0, 0]
    # upper = [3.3, 1, 1]
    # initial_params = ... # Need to define
    # from scipy.optimize import minimize, LBFGSB # Example
    # result = minimize(objective, initial_params, method='L-BFGS-B', bounds=list(zip(lower, upper)))
    # optimized_params_result = result.x
    # print("Optimized parameters: ", omega_cn, optimized_params_result)
    
    # occupationList_calc = main_calc_real_part(Hlist, omega_cn, *current_optimized_params)

    return Hlist, frequencies, s21 #, occupationList_calc #, current_optimized_params

def s21_theoretical(w, H, g1, g2, g3, alpha_1, alpha_2, lambda_1, lambda_2, lambda_r, beta):
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
    # B needs to be a column vector for B.T @ inv(M) @ B if B.T is row vector
    B_col = B[:, np.newaxis] 

    try:
        inv_M = np.linalg.inv(M)
        result = B_col.T.conj() @ inv_M @ B_col # Using B.T.conj() for Hermitian transpose, then matmul
                                              # Or (B.T @ inv_M @ B) if complex conjugate transpose not intended for B.
                                              # Julia's la.transpose(B) * la.inv(M) * B for complex vectors
                                              # might imply element-wise transpose, not hermitian.
                                              # If B is real, B.T is fine. Given sqrt, it should be real.
                                              # Let's stick to simple transpose for B if it's intended as real components.
        B_row_real = np.array([np.sqrt(gamma_1), np.sqrt(gamma_r), np.sqrt(gamma_2)]) 
        result_val = B_row_real @ inv_M @ B_row_real.T # Or B_row_real @ inv_M @ B_row_real if B is treated as 1D array for matmul
                                                        # The result should be a scalar.
        # The operation in Julia: la.transpose(B) * la.inv(M) * B
        # B is a 1D vector. transpose(B) would be a row vector.
        # So it's row_vector * matrix * column_vector (if B is treated as column vector by default in M*B)
        # Or row_vector * matrix * vector (if B is 1D array)
        # result = B_row_real.dot(inv_M.dot(B_row_real)) # This should give scalar
        result = B_row_real @ inv_M @ B_row_real
    except np.linalg.LinAlgError:
        print(f"Singular matrix for w={w}, H={H}. Returning NaN.")
        return np.nan


    return np.abs(result) # In Julia it was abs(result[1,1]), so result must be a 1x1 matrix or scalar
                           # If result from B @ inv(M) @ B is already scalar, then just np.abs(result)

def plot_multiple_calculations(params_dict, save_file, plot_size=3, width_excess=0, lone=False, nrows=2, theo=True):
    files = sorted(params_dict.keys())
    num_actual_plots = len(files)
    
    if theo:
        num_subplots_per_file = 2 # One for experimental, one for theoretical
    else:
        num_subplots_per_file = 1

    total_subplots_needed = num_actual_plots * num_subplots_per_file
    
    # Ensure ncols is at least 1, even if total_subplots_needed is less than nrows
    if total_subplots_needed == 0:
        print("No plots to generate.")
        return
        
    ncols = math.ceil(total_subplots_needed / nrows)
    if ncols == 0 and total_subplots_needed > 0: # Ensure at least one column if plots exist
        ncols = 1
    
    # If theo is False, and we want all plots in 'nrows', then ncols needs to be calculated based on num_actual_plots
    if not theo:
        ncols = math.ceil(num_actual_plots / nrows)
        if ncols == 0 and num_actual_plots > 0:
             ncols = 1


    print(f"nrows: {nrows}, ncols: {ncols}")

    fig, axes = plt.subplots(nrows, ncols, figsize=(plot_size * ncols + width_excess, plot_size * nrows), 
                             sharey=theo, sharex=True, squeeze=False) # squeeze=False ensures axes is always 2D
    axes_flat = axes.flatten() # Flatten to make indexing easier

    plot_idx_counter = 0 # For placing plots into the flattened axes array

    for file_key in files:
        param_values = params_dict[file_key]
        
        # Call the main data processing function (renamed from 'main' to avoid confusion)
        main_result = main_plotter(file_key, param_values, lone=lone)
        if main_result[0] is None: # Error in main_plotter
            print(f"Skipping plot for {file_key} due to data loading/processing error.")
            continue
            
        Hlist, frequencies, s21_exp = main_result #, occupationList, _ = main_result
        # occupationList is not fully handled as main_calc_real_part is a placeholder

        Hlist_kOe = Hlist / 1e3  # Convert to kOe

        # Unpack parameters for theoretical calculation if needed
        # ωcn is param_values[0], rest are for s21_theoretical
        # g1n,g2n,g3n,λ1n,λ2n,λcn,α1n,α2n,βn = param_values
        # For s21_theoretical, it seems it needs: g1, g2, g3, alpha_1, alpha_2, lambda_1, lambda_2, lambda_r, beta
        # Mapping based on typical physics params:
        # param_values = [ωcn, g1n, g2n, g3n, λ1n, λ2n, λcn, α1n, α2n, βn]
        #                  0    1    2    3    4    5    6    7    8   9
        if len(param_values) < 10:
            print(f"Warning: Not enough parameters for {file_key}. Expected 10, got {len(param_values)}")
            # Pad with defaults or skip, here skipping theoretical if not enough params
            current_theo = False
        else:
            current_theo = theo # Use the global theo flag
            g1n, g2n, g3n = param_values[1], param_values[2], param_values[3]
            alpha1n, alpha2n = param_values[7], param_values[8]
            lambda1n, lambda2n, lambdacn = param_values[4], param_values[5], param_values[6]
            betan = param_values[9]


        s21_theoretical_array = np.zeros((len(frequencies), len(Hlist)))

        if current_theo:
            for i_h, H_val in enumerate(Hlist): # Use original Hlist for calculation
                for j_f, freq_val in enumerate(frequencies):
                    s21_theoretical_array[j_f, i_h] = s21_theoretical(freq_val, H_val, 
                                                                      g1n, g2n, g3n, 
                                                                      alpha1n, alpha2n, 
                                                                      lambda1n, lambda2n, lambdacn, betan)
            print(f"Size of s21_theoretical_array for {file_key}: {s21_theoretical_array.shape}")


        # Normalization of experimental s21
        # In Julia: if theo || idx==1 (where idx is 1-based plot index)
        # Here, let's normalize if 'theo' is true, or if it's the first file being processed when 'theo' is false
        # This logic was per H column in Julia:
        # s21[:,h] .= (s21[:,h] .- minimum(s21[:,h])) ./ (maximum(s21[:,h]) .- minimum(s21[:,h]))
        if current_theo or (not current_theo and files.index(file_key) == 0):
             for h_col in range(s21_exp.shape[1]):
                col_data = s21_exp[:, h_col]
                min_val = np.min(col_data)
                max_val = np.max(col_data)
                if (max_val - min_val) > 1e-9: # Avoid division by zero
                    s21_exp[:, h_col] = (col_data - min_val) / (max_val - min_val)
                else: # If all values are the same, set to 0 or 0.5 or leave as is
                    s21_exp[:, h_col] = 0.0 


        # --- Plotting Experimental Data ---
        if plot_idx_counter < len(axes_flat):
            ax_exp = axes_flat[plot_idx_counter]
            plot_idx_counter += 1

            im_exp = ax_exp.pcolormesh(Hlist_kOe, frequencies, s21_exp, cmap='inferno_r', shading='auto')
            # fig.colorbar(im_exp, ax=ax_exp) # Optional: add colorbar for each plot

            t_val_str = file_key.split("_")[-1] # Assuming format "yig_t_0.005" -> "0.005"
            try:
                t_val = round(float(t_val_str), 3)
                tt_val = int(t_val * 1e3)
                # Text placement in Julia: ax.text(1150, 5.5, ...), Hlist was original Oe values
                # Need to adjust coordinates if Hlist_kOe is used or refer to original Hlist scale
                # For now, using relative position or a fixed kOe value
                ax_exp.text(0.95, 0.90, f"t = {tt_val} μm (Exp)", color="white", fontsize=10, 
                            ha="right", va="top", transform=ax_exp.transAxes)
            except ValueError:
                print(f"Could not parse t_value from filename: {file_key}")
                ax_exp.set_title(f"{file_key} (Exp)", fontsize=10, color="white", backgroundcolor='gray')


            # Y-axis limits (consistent with Julia code's different conditions)
            if lone and file_key == files[1]: # Corresponds to idx==2 in 1-based Julia
                lower, upper = 4.5, 6.3
            else:
                lower, upper = 4.3, 5.8
            ax_exp.set_ylim(lower, upper)
            ax_exp.set_yticks(np.linspace(lower, upper, 4)) # Similar to 3 divisions

            if not current_theo: # If only experimental plots, this is the main title for the file key
                 ax_exp.set_title(f"{file_key}", fontsize=10)


        # --- Plotting Theoretical Data (if theo is True) ---
        if current_theo:
            if plot_idx_counter < len(axes_flat):
                ax_theo = axes_flat[plot_idx_counter]
                plot_idx_counter += 1

                im_theo = ax_theo.pcolormesh(Hlist_kOe, frequencies, s21_theoretical_array, cmap='inferno', shading='auto')
                # fig.colorbar(im_theo, ax=ax_theo) # Optional

                # In Julia, experimental text was on the first of the pair, theoretical didn't add new t-value text
                # If you want title for theoretical:
                ax_theo.set_title(f"{file_key} (Theo)", fontsize=10)
                
                # Plotting occupationList (commented out lines from Julia)
                # if occupationList is not None:
                #     ax_exp.plot(Hlist_kOe, occupationList[:,0], "w", alpha=0.5)
                #     ax_exp.plot(Hlist_kOe, occupationList[:,1], "w", alpha=0.5)
                #     ax_exp.plot(Hlist_kOe, occupationList[:,2], "w", alpha=0.5)
                
                # Y-axis limits for theoretical plot (if sharey=True, this is set by the first plot in the row)
                if not theo: # if not sharey
                    if lone and file_key == files[1]: 
                        lower, upper = 4.5, 6.3
                    else:
                        lower, upper = 4.3, 5.8
                    ax_theo.set_ylim(lower, upper)
                    ax_theo.set_yticks(np.linspace(lower, upper, 4))
            else:
                print(f"Warning: Not enough subplots for theoretical data of {file_key}")


        # Specific text annotations from Julia for 'lone' case (adjust indices for flattened axes)
        # This part needs careful mapping of Julia's axes[1], axes[2], axes[3] to Python's axes_flat
        # This depends on how 'lone' case populates the plots (is it always 3 specific files?)
        # Assuming 'lone' implies specific files are at the beginning of `params_dict` if this text is desired
        if lone and len(files) >=3 : # Check if enough files for these annotations
            if file_key == files[0]: # First plot
                 axes_flat[0].text(0.1, 0.1, "(a) Py", color="white", fontsize=12, ha="center", transform=axes_flat[0].transAxes)
            if file_key == files[1]: # Second plot
                 axes_flat[plot_idx_counter - (2 if theo else 1)].text(0.5, 0.9, "YIG", color="white", fontsize=12, ha="center", transform=axes_flat[plot_idx_counter - (2 if theo else 1)].transAxes)
                 axes_flat[plot_idx_counter - (2 if theo else 1)].text(0.1, 0.1, "(b)", color="white", fontsize=12, ha="center", transform=axes_flat[plot_idx_counter - (2 if theo else 1)].transAxes)
            if file_key == files[2]: # Third plot
                 axes_flat[plot_idx_counter - (2 if theo else 1)].text(0.5, 0.9, "Py+YIG", color="white", fontsize=12, ha="center", transform=axes_flat[plot_idx_counter - (2 if theo else 1)].transAxes)
                 axes_flat[plot_idx_counter - (2 if theo else 1)].text(0.1, 0.1, "(c)", color="white", fontsize=12, ha="center", transform=axes_flat[plot_idx_counter - (2 if theo else 1)].transAxes)


    # Hide any unused subplots
    for i in range(plot_idx_counter, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.supxlabel("Magnetic Field (kOe)", fontsize=12)
    fig.supylabel("Frequency (GHz)", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle if any, or general spacing
    
    # Ensure the target directory exists
    save_dir = os.path.join(os.getcwd(), "tentative", "images")
    os.makedirs(save_dir, exist_ok=True)
    full_save_path = os.path.join(save_dir, save_file)

    plt.savefig(full_save_path, dpi=300, bbox_inches="tight", transparent=True) # backend="QtAgg" is not a standard Matplotlib savefig param
    print(f"Saved figure to {full_save_path}")
    plt.close(fig)


if __name__ == '__main__':
    # Example params dictionary (matching Julia structure)
    params1 = { # ωcn  g1n  g2n g3n λ1n  λ2n  λcn  α1n  α2n  βn
        #  "yig_t_0.000": [5.09, .11, 0.0, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
        "yig_t_0.005": [5.06, .12, 0.04, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
        #  "yig_t_0.013": [5.01, .13, 0.075, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5], 
        "yig_t_0.027": [5.01, .14, 0.12, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
        #  "yig_t_0.040": [5.04, .155, 0.13, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5], 
        "yig_t_0.053": [5.01, .16, 0.15, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5], 
        #  "yig_t_0.067": [5.02, .18, 0.18, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
        "yig_t_0.100": [5.01, .2,  0.25, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5], 
    }

    print(f"CPU cores available (similar to Threads.nthreads()): {os.cpu_count()}")

    # Create dummy data files for testing if they don't exist
    # Example: ./data/yig_t_sweep_new/yig_t_0.005.csv
    # Header: ,H1,H2,H3,...
    # Freq1,val11,val12,val13,...
    # Freq2,val21,val22,val23,...
    
    # Check and create dummy files
    for p_key in params1.keys():
        path_parts = ["data"]
        if "lone" in p_key: # Simplified check for lone based on key name for dummy data
             path_parts.append("lone_t_sweep_yig")
        else:
             path_parts.append("yig_t_sweep_new")
        
        dummy_file_dir = os.path.join(os.getcwd(), *path_parts)
        os.makedirs(dummy_file_dir, exist_ok=True)
        dummy_file_path = os.path.join(dummy_file_dir, f"{p_key}.csv")

        if not os.path.exists(dummy_file_path):
            print(f"Creating dummy data file: {dummy_file_path}")
            H_example = np.linspace(0, 1500, 10)
            Freq_example = np.linspace(4, 6, 20)
            header = "," + ",".join([str(h) for h in H_example])
            dummy_s21_data = np.random.rand(len(Freq_example), len(H_example)) * 0.1
            
            with open(dummy_file_path, 'w') as f:
                f.write(header + "\n")
                for i_freq, freq_val in enumerate(Freq_example):
                    f.write(f"{freq_val}," + ",".join([f"{dummy_s21_data[i_freq, j_h]:.4f}" for j_h in range(len(H_example))]) + "\n")


    plot_multiple_calculations(params1, "combined_plots.png")

    params2 = { # ωcn  g1n  g2n g3n λ1n  λ2n  λcn  α1  α2  β
        "yig_t_0.000": [5.09, .11, 0.0, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
        "yig_t_0.100_lone": [5.2, .11, 0.0, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5],  
        "yig_t_0.100_z": [5.01, .2,  0.25, .001, .01, .01, .07, 2e-2, 1e-5, 1e-5], 
    }
    # Create dummy files for params2 as well
    for p_key in params2.keys():
        path_parts = ["data"]
        if "lone" in p_key:
             path_parts.append("lone_t_sweep_yig")
        else:
             path_parts.append("yig_t_sweep_new")
        
        dummy_file_dir = os.path.join(os.getcwd(), *path_parts)
        os.makedirs(dummy_file_dir, exist_ok=True)
        dummy_file_path = os.path.join(dummy_file_dir, f"{p_key}.csv")
        if not os.path.exists(dummy_file_path):
            print(f"Creating dummy data file: {dummy_file_path}")
            # Simplified dummy data creation
            with open(dummy_file_path, 'w') as f:
                f.write(",100,200,300\n4.0,0.1,0.2,0.1\n5.0,0.15,0.25,0.15\n6.0,0.12,0.22,0.12\n")


    plot_multiple_calculations(params2, "combined_plots_isolate.png", plot_size=3, width_excess=0.5, lone=True, nrows=3, theo=False)