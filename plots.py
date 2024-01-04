import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_audio_signal(audio_signal: np.ndarray, sampling_rate: int, title: str = "Audio Signal") -> None:
    """
    Plots the audio signal
    """
    # Create a time array in seconds
    time_array = np.arange(0, len(audio_signal) / sampling_rate, 1 / sampling_rate)

    # Plot the audio signal
    plt.plot(time_array, audio_signal, alpha=0.5, color='blue')

    N_dict = {1: 0.2, 10: 1, 30: 5, 60: 10, 120: 30, 600: 60, 1800: 300, 3600: 600}
    
    for key in N_dict.keys():
        if max(time_array) < key:
            N = N_dict[key]
            break
    
    # Set the x-axis ticks and labels
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(N))  # Set tick every N seconds
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Rotate and set font size for readability

    # Set the title and labels
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # add grid on major ticks only
    plt.grid(which='major', axis='both')
    
    plt.show()


def plot_spectrogram(spectrogram: np.ndarray, sampling_rate: int, title: str = "Spectrogram") -> None:
    """
    Plots the spectrogram
    """
    # Create a time array in seconds
    time_array = np.arange(0, len(spectrogram) / sampling_rate, 1 / sampling_rate)

    # Plot the spectrogram
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap="gray", interpolation='none', extent=[0, max(time_array), 0, sampling_rate / 2])

    # Set the title and labels
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    # add grid on major ticks only
    plt.grid(which='major', axis='both')

    plt.show()