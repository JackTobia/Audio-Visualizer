import numpy as np
import pygame
import wave
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize Pygame
pygame.init()
pygame.mixer.init()

def play_audio(audio_file):
    """Plays the audio file using pygame."""
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

def read_wav(audio_file):
    """Reads WAV file and returns normalized audio data and frame rate."""
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

def calculate_spectrum(data, chunk_size):
    """Calculates the frequency spectrum using FFT."""
    # Apply FFT to the audio data and take the magnitude of the result
    spectrum = np.abs(np.fft.fft(data))[:chunk_size//2]  # Only use the first half of the spectrum
    spectrum = spectrum / np.max(spectrum)  # Normalize the spectrum
    return spectrum

def split_spectrum(spectrum, frame_rate, bands, low_freq=20, high_freq=20000):
    """Splits the spectrum into defined frequency bands between low_freq and high_freq."""
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

def update_bars(frame, bars, data, frame_rate, chunk_size, start_time, bands, low_freq, high_freq):
    """Updates the bar heights dynamically based on audio spectrum."""
    elapsed_time = pygame.mixer.music.get_pos() / 1000.0  # Time in seconds
    current_frame = int(elapsed_time * frame_rate)  # Convert seconds to samples
    start = max(current_frame - chunk_size, 0)  # Keep within bounds
    end = start + chunk_size

    if end > len(data):
        return bars  # Stop updating when data ends

    # Get the spectrum for this chunk of data
    spectrum = calculate_spectrum(data[start:end], chunk_size)

    # Split the spectrum into frequency bands (low, mid, high)
    band_heights = split_spectrum(spectrum, frame_rate, bands, low_freq, high_freq)

    # Update the bar heights
    for i, bar in enumerate(bars):
        bar.set_height(band_heights[i] * 10)  # Scale by 10 for better visibility

    return bars

def visualize_audio(audio_file, bands=50, low_freq=20, high_freq=20000):
    """Animates the bars in response to audio spectrum."""
    data, frame_rate = read_wav(audio_file)
    chunk_size = 2048  # Number of samples per update for smoother visual

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(np.arange(bands), np.zeros(bands), color='cyan', width=0.8)

    ax.set_ylim(0, 7)  # Adjust to fit the max bar height
    ax.set_xlim(-0.5, bands - 0.5)  # Align with the frequency bands

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
    visualize_audio("Atlantis.wav")  # Replace with your audio file path
