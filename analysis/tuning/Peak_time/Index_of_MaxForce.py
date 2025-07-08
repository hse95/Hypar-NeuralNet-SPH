import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set default plotting styles
plt.rcParams.update({
    'font.size': 24,
    'axes.grid': True,
    'grid.alpha': 0.10,
})

def read_and_prepare_data(pressure_file, normals_file):
    """Read and preprocess the data."""
    # Reading pressure data
    pressure_df = pd.read_csv(pressure_file, sep=';', skiprows=3)
    pressure_df = pressure_df.drop(pressure_df.columns[[0, 1]], axis=1)
    pressure_matrix = pressure_df.to_numpy(dtype=float)

    # Reading normals data
    normals = pd.read_csv(normals_file, sep=' ', header=None)
    normals = normals.replace({'}': '', '{': ''}, regex=True)
    normals = normals[0].str.split(',', expand=True)
    normals.columns = ['NormalX', 'NormalY', 'NormalZ']
    normals_matrix = normals.astype(float).to_numpy()

    return pressure_matrix, normals_matrix

def calculate_force(pressure_matrix, normals_matrix, area_per_quad=0.035588):
    """Calculate force from pressure and normals."""
    force_matrix = np.dot(pressure_matrix, normals_matrix) * area_per_quad / 1000  # Convert to kN
    force_magnitude = np.linalg.norm(force_matrix, axis=1)
    # Calcualte the force magnitude average
    force_magnitude_avg = np.mean(force_magnitude)
    print("Average force magnitude: ", force_magnitude_avg)
    return force_magnitude

def filter_high_frequencies(signal, threshold=200):
    """Filter high frequencies using FFT."""
    fft_signal = np.fft.rfft(signal)
    fft_signal[threshold:] = 0
    return np.fft.irfft(fft_signal)

def find_maximum_after_midpoint(signal):
    """Find maximum value and its index after a certain % of the signal."""
    start_index = len(signal) // 3
    subset_signal = signal[start_index:]
    max_value = np.max(subset_signal)
    max_index = np.argmax(subset_signal) + start_index
    return max_value, max_index

def plot_force(signal, max_index, max_value):
    """Plot force over time with annotations."""
    plt.figure(figsize=(10, 8), dpi=100)
    line1, = plt.plot(signal, color='black')
    line2 = plt.axvline(x=max_index, color='red', linestyle='--', linewidth=2)
    scatter = plt.scatter(max_index, max_value, color='green', s=100, zorder=5)
    plt.xlabel('Time Step')
    # Limit x-limt to 200 time steps before and after the maximum
    plt.xlim(max_index - 100, max_index + 100)
    plt.ylim(0, 800)
    plt.ylabel('Force Magnitude (kN)')
    plt.grid(True)
    plt.show()

    plot_legend = False
    if plot_legend == True:
        # Create a separate figure for the legend
        plt.figure(figsize=(2, 2))
        plt.legend([line1, line2, scatter], ['Calculated Force from Pressure', 'Time of Maximum Force', 'Maximum Force'], loc='center')
        plt.axis('off')
        plt.show()


def main():
    os.chdir(os.path.dirname(__file__))
    pressure_matrix, normals_matrix = read_and_prepare_data('PointsPressureOut_Press_dr250_H1.2m.csv', 'Normals_Rn0500.txt')
    force_magnitude = calculate_force(pressure_matrix, normals_matrix)
    force_magnitude = filter_high_frequencies(force_magnitude)
    max_force, max_index = find_maximum_after_midpoint(force_magnitude)
    print("Maximum force magnitude after midpoint: ", max_force)
    print("Index of maximum force magnitude: ", max_index)
    plot_force(force_magnitude, max_index, max_force)

if __name__ == "__main__":
    main()
