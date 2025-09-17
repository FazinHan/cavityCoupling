
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from scipy.optimize import curve_fit
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, find_peaks
from sklearn.metrics import mean_squared_error

def peak_pos(data, freqs):
    """Return the index of the maximum value in a 1D array."""
    peak = np.argmax(data)
    plt.plot(freqs, data)
    plt.annotate('', xy=(freqs[peak], data[peak]), xytext=(freqs[peak], data[peak]+0.1*max(data)), 
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=8, color='green')
    plt.tight_layout()
    plt.savefig(os.path.join('tentative','int_plots', f'peak_pos_{fig_count}.png'))
    plt.close()
    return peak

def std_dev(data, filename):
    """Return the standard deviation of peaks of a 1D array."""
    
    mags = data.columns.values.astype(float)
    freqs = data.index.values.astype(float)

    peak_poses = []
    for i in range(data.values.shape[1]):
        col = -data.values[:, i]
        peak = np.argmax(col)
        peak_poses.append(freqs[peak])
    
    peak_poses = np.array(peak_poses)

    error = np.std(peak_poses[np.where((mags > 400) & (mags < 1000))])

    plt.plot(mags*1e-3, peak_poses, '.', label='Error = {error:.4f} GHz'.format(error=error))
    plt.xlabel('Magnitude (kOe)')
    plt.ylabel('Peak Position (GHz)')
    plt.title(f'Peak Positions vs Magnitude for {os.path.basename(filename).rstrip(".csv")}')
    plt.ylim(min(freqs), max(freqs))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('tentative','int_plots', 'std_devs', f'std_dev_{fig_count}.png'))
    plt.close()

    return error
        
def fit_and_plot_lorentzian(freq, data, prominence_scaler=0.1, xtol=1e-10):
    """
    Fits up to two Lorentzian peaks with a y-offset to the provided data,
    plots the result, saves the plot, and returns the fitting error.

    Args:
        freq (np.ndarray): A 1D array of frequency values (x-axis).
        data (np.ndarray): A 1D array of data values (y-axis).
        output_filename (str): The path to save the output plot file.

    Returns:
        float: The Root Mean Squared Error (RMSE) of the fit. Returns np.inf
               if the fitting process fails.
    """

    def lorentzian_peak(x, amplitude, center, hwhm):
        """
        Returns the value of a single Lorentzian peak component at a given point x.
        This function does not include a y-offset.
        
        Args:
            x (float or np.ndarray): The independent variable.
            amplitude (float): The peak amplitude of the Lorentzian.
            center (float): The center (peak location) of the Lorentzian.
            hwhm (float): The half-width at half-maximum.
            
        Returns:
            float or np.ndarray: The value of the Lorentzian peak function.
        """
        return amplitude * (hwhm**2 / ((x - center)**2 + hwhm**2))

    def one_lorentzian(x, amp1, center1, hwhm1, y_offset):
        """Model for a single Lorentzian peak with a y-offset."""
        return lorentzian_peak(x, amp1, center1, hwhm1) + y_offset

    def two_lorentzian(x, amp1, center1, hwhm1, amp2, center2, hwhm2, y_offset):
        """Model for the sum of two Lorentzian peaks with a y-offset."""
        return lorentzian_peak(x, amp1, center1, hwhm1) + lorentzian_peak(x, amp2, center2, hwhm2) + y_offset

    output_filename = os.path.join('tentative', 'int_plots', f'lorentzian_fit_{fig_count}.png')

    # --- 1. Find peaks for initial parameter guesses ---
    # The prominence is set to 10% of the data's dynamic range
    peaks, properties = find_peaks(data, prominence=(np.max(data) - np.min(data)) * prominence_scaler)

    # Sort peaks by prominence in descending order
    if len(peaks) > 1:
        prominence_sorted_indices = np.argsort(properties['prominences'])[::-1]
        peaks = peaks[prominence_sorted_indices]

    # --- 2. Attempt to fit the data based on the number of peaks found ---
    popt = None
    num_peaks_fitted = 0
    
    # Try to fit with a two-peak model if two or more peaks are detected
    if len(peaks) >= 2:
        try:
            # Initial guesses [amp1, center1, hwhm1, amp2, center2, hwhm2, y_offset]
            p0 = [
                data[peaks[0]] - np.min(data), freq[peaks[0]], (freq[-1] - freq[0]) * 0.05,
                data[peaks[1]] - np.min(data), freq[peaks[1]], (freq[-1] - freq[0]) * 0.05,
                np.min(data)
            ]
            # Parameter bounds to ensure physical relevance
            bounds = (
                [0, freq.min(), 1e-6, 0, freq.min(), 1e-6, -np.inf],
                [np.inf, freq.max(), np.inf, np.inf, freq.max(), np.inf, np.inf]
            )
            popt, _ = curve_fit(two_lorentzian, freq, data, p0=p0, bounds=bounds, maxfev=10000, xtol=xtol)
            num_peaks_fitted = 2
        except RuntimeError:
            print("Two-peak fit failed. Falling back to a one-peak fit.")
            popt = None # Reset popt to trigger one-peak fit

    # Fit with a one-peak model if fewer than two peaks were found, or if the two-peak fit failed
    if popt is None:
        peak_index = peaks[0] if len(peaks) > 0 else np.argmax(data)
        try:
            # Initial guesses [amp1, center1, hwhm1, y_offset]
            p0 = [data[peak_index] - np.min(data), freq[peak_index], (freq[-1] - freq[0]) * 0.05, np.min(data)]
            bounds = (
                [0, freq.min(), 1e-6, -np.inf],
                [np.inf, freq.max(), np.inf, np.inf]
            )
            popt, _ = curve_fit(one_lorentzian, freq, data, p0=p0, bounds=bounds, maxfev=10000)
            num_peaks_fitted = 1
        except RuntimeError:
            print("FATAL: One-peak fit also failed. Could not fit the data.")
            return np.inf

    # --- 3. Calculate the fitting error ---
    if num_peaks_fitted == 2:
        fit_curve = two_lorentzian(freq, *popt)
    elif num_peaks_fitted == 1:
        fit_curve = one_lorentzian(freq, *popt)
    else: # Should not be reached
        return np.inf
        
    rmse = np.sqrt(mean_squared_error(data, fit_curve))

    # --- 4. Generate and save the plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(freq, data, 'o', color='gray', markersize=5, alpha=0.6, label='Original Data')

    # Create a smooth frequency array for plotting the fit curve
    freq_smooth = np.linspace(freq.min(), freq.max(), 1000)
    
    err_arr = np.full_like(freq_smooth, rmse)

    if num_peaks_fitted == 2:
        fit_smooth = two_lorentzian(freq_smooth, *popt)
        peak1_component = lorentzian_peak(freq_smooth, *popt[0:3])
        peak2_component = lorentzian_peak(freq_smooth, *popt[3:6])
        y_offset_val = popt[6]
        
        ax.plot(freq_smooth, fit_smooth, color='crimson', linewidth=2, label='Total Fit')
        ax.plot(freq_smooth, peak1_component + y_offset_val, '--', color='dodgerblue', linewidth=1.5, label='Peak 1 Component')
        ax.plot(freq_smooth, peak2_component + y_offset_val, '--', color='orchid', linewidth=1.5, label='Peak 2 Component')
        # ax.errorbar(freq_smooth, fit_smooth, yerr=err_arr, fmt='none', ecolor='green', alpha=1, label='Fit Error')
        ax.axhline(y_offset_val, color='green', linestyle=':', linewidth=1.5, label=f'Y-Offset: {y_offset_val:.2f}')
        ax.set_title(f'Two Lorentzian Peaks Fit (RMSE: {rmse:.4f})', fontsize=16)
        
    elif num_peaks_fitted == 1:
        fit_smooth = one_lorentzian(freq_smooth, *popt)
        y_offset_val = popt[3]
        ax.plot(freq_smooth, fit_smooth, color='crimson', linewidth=2, label='Fit')
        # ax.errorbar(freq_smooth, fit_smooth, yerr=err_arr, fmt='none', ecolor='green', alpha=1, label='Fit Error')
        ax.axhline(y_offset_val, color='green', linestyle=':', linewidth=1.5, label=f'Y-Offset: {y_offset_val:.2f}')
        ax.set_title(f'One Lorentzian Peak Fit (RMSE: {rmse:.4f})', fontsize=16)

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Signal Intensity', fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()

    plt.savefig(output_filename, dpi=300)
    plt.close(fig)
    print(f"Plot saved to '{output_filename}'")
    # plt.show()

    return rmse

def std_in_bbox(data_array, x_array, y_array, min_y, max_y, min_x, max_x):
    """
    Plot data with pcolormesh + bounding box, return σ of the sub‑region.
    x_array: 1D array for rows (vertical axis)
    y_array: 1D array for columns (horizontal axis)
    min_x, max_x: bounds for x (vertical)
    min_y, max_y: bounds for y (horizontal)
    """
    # Find indices within bounds
    row_mask = (x_array >= min_x) & (x_array <= max_x)
    col_mask = (y_array >= min_y) & (y_array <= max_y)

    # print(col_mask)

    if not np.any(row_mask) or not np.any(col_mask):
        print("Warning: Bounding box does not intersect the grid. Using closest extant values.")
        # Find closest indices
        row_mask = (x_array >= min(x_array)) & (x_array <= max(x_array))
        col_mask = (y_array >= min(y_array)) & (y_array <= max(y_array))

    subregion = data_array[np.ix_(row_mask, col_mask)]
    std_val = np.std(subregion)
    mu_val = np.mean(subregion)

    # Plotting
    X, Y = np.meshgrid(y_array, x_array)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(X, Y, data_array, cmap='inferno_r', shading='auto')
    fig.colorbar(pcm, ax=ax)

    rect = Rectangle((min_y, min_x), width=max_y - min_y, height=max_x - min_x,
                     linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Frequency')
    plt.title('Data + bounding box')
    plt.savefig(os.path.join('tentative','int_plots', f'std_in_bbox.png'))
    plt.close()

    return std_val, mu_val

def peak_detection(data, std_dev, mu):

    def lorentzian(x, wid, cen):
        return wid / np.pi * ((x - cen)**2 + wid**2) + mu/2

    def double_lorentzian(x, amp1, cen1, amp2, cen2):
        return lorentzian(x, amp1, cen1) + lorentzian(x, amp2, cen2)

    x = freqs
    y = data
    # y_smoothed = gaussian_filter1d(data.copy(), sigma=4)

    # Initial guess: amplitudes=max(y)/2, centers at two largest peaks, widths=10
    # Find indices of two largest extrema (peaks or troughs)

    # Find local maxima and minima
    max_indices = argrelextrema(y, np.greater)[0]
    min_indices = argrelextrema(y, np.less)[0]
    extrema_indices = np.concatenate((max_indices, min_indices))

    # If less than two extrema found, fallback to global extrema
    if len(extrema_indices) < 2:
        extrema_indices = np.argsort(np.abs(y - mu))[-2:]

    # Sort by prominence (distance from mean)
    prominences = np.abs(y[extrema_indices] - mu)
    peak_indices = extrema_indices[np.argsort(prominences)[-2:]] 
    
    p0 = [y[peak_indices[0]]-mu, x[peak_indices[0]], 
          y[peak_indices[1]]-mu, x[peak_indices[1]],]
        #   mu/2, mu/2]

    try:
        popt, pcov = curve_fit(double_lorentzian, x, y, p0=p0, xtol=1e-10)
        fit_y = double_lorentzian(x, *popt)
        fit_error = np.sqrt(np.mean((y - fit_y)**2))
        print(popt)
    except Exception as e:
        print(f"Fit failed: {e}")
        fit_y = np.zeros_like(y) + mu
        fit_error = np.nan

    plt.figure()
    # Add arrows to indicate peak positions
    for cen in [x[peak_indices[0]], x[peak_indices[1]]]:
        print('annotating peak at', cen)
        peak_y = double_lorentzian(cen, *popt)
        plt.annotate('', xy=(cen, peak_y), xytext=(cen, peak_y + 0.1 * np.max(y)),
                     arrowprops=dict(facecolor='green', shrink=0.05), fontsize=8, color='green') 
    plt.plot(x, y, 'b.', label='Data', markersize=.4)
    # plt.plot(x, y_smoothed, 'g', label='Data', markersize=.4)
    plt.plot(x, fit_y, 'r-', label='Double Lorentzian Fit')
    plt.xlabel('Frequency')
    plt.ylabel('Value')
    plt.title('Data and Double Lorentzian Fit')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join('tentative', 'int_plots', 'col0_peak_fit.png'))
    plt.close()

    return fit_error

def load_data(filename):
    try:
        data = pd.read_csv(filename, index_col=0, dtype=float)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None, None

    # freqs = data.index.values.astype(float)
    # mags = data.columns.values.astype(float)

    return data

# Example usage:
if __name__ == "__main__":
    fig_count = 0
    os.makedirs(os.path.join('tentative','int_plots'), exist_ok=True)

    # Load the CSV file safely
    dirname = os.path.join('data', 'yig_t_sweep_new')

    filenames = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith('.csv')]

    stds = np.zeros_like(filenames, dtype=float)
    
    for i in range(len(filenames)):

        os.makedirs(os.path.join('tentative','int_plots','std_devs'), exist_ok=True)

        print(f"Processing file {i+1}/{len(filenames)}: {filenames[i]}")
        data = load_data(filenames[i])
        if data is None:
            continue

        full_std = std_dev(data, filenames[i])
        # print(f"Standard deviation of peak positions for file {filenames[i]}: {full_std}")

        fig_count += 1

        stds[i] = full_std

    print("\nAll standard deviations:", stds)
    print()
    print("Mean standard deviation:", np.mean(stds))