# audiovis2.py
# Created by Jack Tobia
# February 2025
#
# Takes a WAV file and visualizes the audio with bars representing different
# audio frequencies.


import numpy as np
import pygame
import wave
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Plays the audio file using pygame.
def play_audio(audio_file):
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

# Reads WAV file and returns normalized audio data and frame rate.
def read_wav(audio_file):
    with wave.open(audio_file, 'rb') as wav:
        params = wav.getparams()
        num_channels, sample_width, frame_rate, num_frames, _, _ = params
        frames = wav.readframes(num_frames)
        data = np.frombuffer(frames, dtype=np.int16)

        # Convert stereo to mono if needed
        if num_channels == 2:
            data = data[::2]

        # Normalize the data (-1 to 1)
        data = data / 32768.0

    return data, frame_rate

# Calculates the frequency spectrum using FFT.
def calculate_spectrum(data, chunk_size):
    # Apply FFT to the audio data and take the magnitude of the result, and
    # only use the first half of the spectrum
    spectrum = np.abs(np.fft.fft(data))[:chunk_size//2]
    # Normalize the spectrum
    spectrum = spectrum / np.max(spectrum)

    return spectrum

# Splits the spectrum into defined frequency bands between low_freq and high_freq.
def split_spectrum(spectrum, frame_rate, bands, low_freq=20, high_freq=20000):
    # Calculate the frequency bin width (Hz per bin)
    bin_width = frame_rate / len(spectrum)

    # Calculate the indices corresponding to low and high frequency bounds
    low_bin = int(low_freq / bin_width)
    high_bin = int(high_freq / bin_width)

    # Create bands with their corresponding frequency ranges
    band_heights = []
    for i in range(bands):
        # Divide the frequency range between low_freq and high_freq evenly into 'bands'
        start_idx = low_bin + (i * (high_bin - low_bin)) // bands
        end_idx = low_bin + ((i + 1) * (high_bin - low_bin)) // bands
        band_heights.append(np.mean(spectrum[start_idx:end_idx]))

    return band_heights

# Updates the bar heights dynamically based on audio spectrum.
def update_bars(frame, bars, data, frame_rate, chunk_size, start_time, bands, low_freq, high_freq):
    # Time in seconds
    elapsed_time = pygame.mixer.music.get_pos() / 1000.0
    # Convert seconds to samples
    current_frame = int(elapsed_time * frame_rate)
    # Keep within bounds
    start = max(current_frame - chunk_size, 0)
    end = start + chunk_size

    # Stop updating when data ends
    if end > len(data):
        return bars

    # Get the spectrum for this chunk of data
    spectrum = calculate_spectrum(data[start:end], chunk_size)

    # Split the spectrum into frequency bands (low, mid, high)
    band_heights = split_spectrum(spectrum, frame_rate, bands, low_freq, high_freq)

    # Update the bar heights
    for i, bar in enumerate(bars):
        # Scale by 10 for better visibility
        bar.set_height(band_heights[i] * 10)

    return bars

# Animates the bars in response to audio spectrum.
def visualize_audio(audio_file, bands=50, low_freq=20, high_freq=20000):
    data, frame_rate = read_wav(audio_file)
    # Number of samples per update for smoother visual
    chunk_size = 2048

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(np.arange(bands), np.zeros(bands), color='cyan', width=0.8)

    # Adjust to fit the max bar height
    ax.set_ylim(0, 7)
    # Align with the frequency bands
    ax.set_xlim(-0.5, bands - 0.5)

    # Adjust the x-axis labels to display frequency ranges
    ax.set_xticks(np.arange(bands))
    xticks = [f'{low_freq + i * (high_freq - low_freq) // bands} Hz' for i in range(bands)]
    ax.set_xticklabels(xticks, rotation=90)

    # Start playing audio
    play_audio(audio_file)

    # Create an animation with smoother updates
    # Set the interval to synchronize with the audio playback rate
    ani = animation.FuncAnimation(fig, update_bars, fargs=(bars, data, frame_rate, chunk_size, None, bands, low_freq, high_freq),
                                  interval=(1000 * chunk_size / frame_rate), blit=False, cache_frame_data=False)

    plt.show()

# Run the visualization
if __name__ == "__main__":
    # Replace with your audio file path
    visualize_audio("Atlantis.wav")
