import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt


def process_and_plot_spectrogram(folder_path):
    processed_files = set()

    while True:
        # Get all CSV files in the folder
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        for file in files:
            if file in processed_files:
                continue

            file_path = os.path.join(folder_path, file)

            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)

            # Extract triggered time (first line)
            triggered_time = lines[0][0].replace("Triggered Time: ", "")

            # Extract wavelengths (line after "Wavelengths:")
            wavelengths_index = next(i for i, line in enumerate(lines) if "Wavelengths:" in line)
            wavelengths = np.array([float(x) for x in lines[wavelengths_index + 1]])

            # Extract intensities (line after "Intensities:")
            intensities_index = next(i for i, line in enumerate(lines) if "Intensities:" in line)
            intensities = np.array([[float(x) for x in line] for line in lines[intensities_index + 1:]])

            # Compute average intensities
            avg_amplitudes = intensities.mean(axis=1)

            # Find the spectra with the highest and lowest average amplitude
            max_amplitude_index = np.argmax(avg_amplitudes)
            min_amplitude_index = np.argmin(avg_amplitudes)

            # Subtract the lowest from the highest
            difference_spectrum = intensities[max_amplitude_index] - intensities[min_amplitude_index]

            # Plot the spectrogram
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, difference_spectrum, label="Difference Spectrum")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity (a.u.)")
            plt.title(f"Spectrogram Difference\n{file}")
            plt.legend()

            # Save the plot
            image_filename = file.replace('.csv', '.png')
            image_path = os.path.join(folder_path, image_filename)
            plt.savefig(image_path)
            plt.close()

            print(f"Processed and saved spectrogram for {file} as {image_filename}")

            # Mark the file as processed
            processed_files.add(file)

        # Wait before checking again
        time.sleep(2)


# Folder path to monitor
folder_path = "/air_breakdown/104"
os.makedirs(folder_path, exist_ok=True)

# Start monitoring and processing
process_and_plot_spectrogram(folder_path)
