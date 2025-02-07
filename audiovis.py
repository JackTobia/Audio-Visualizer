# audiovis.py
# Created by Jack Tobia
# February 2025
#
# Takes a WAV file and visualizes the audio using a sine wave.


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

# Updates the sine wave plot dynamically based on audio level.
def update(frame, line, data, frame_rate, chunk_size, start_time):
    # Time in milliseconds
    elapsed_time = pygame.time.get_ticks() - start_time
    # Convert ms to samples
    current_frame = int((elapsed_time / 1000.0) * frame_rate)
    # Keep within bounds
    start = max(current_frame - chunk_size, 0)
    end = start + chunk_size

    # Stop updating when data ends
    if end > len(data):
        return line,

    # Update the sine wave with audio data
    x = np.linspace(0, chunk_size, chunk_size)
    # Interpolation for smoothness
    y = np.interp(x, np.arange(chunk_size), data[start:end])

    # Update the sine wave
    line.set_ydata(y)

    return line,

# Animates a smooth sine wave visualization that responds to audio.
def visualize_audio(audio_file):
    data, frame_rate = read_wav(audio_file)
    # Number of samples per update for smoother visual
    chunk_size = 2048

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, chunk_size, chunk_size)
    line, = ax.plot(x, np.zeros(chunk_size), lw=2)

    # Normalized amplitude range
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, chunk_size)
    ax.set_xticks([])
    ax.set_yticks([])

    # Start playing audio
    play_audio(audio_file)
    # Record start time
    start_time = pygame.time.get_ticks()

    # Create an animation
    ani = animation.FuncAnimation(fig, update, fargs=(line, data, frame_rate, chunk_size, start_time),
                                  interval=30, blit=True)

    plt.show()

# Run the visualization
if __name__ == "__main__":
    # audio_file = input("Provide a WAV filepath to visualize and play: ")
    visualize_audio("Atlantis.wav")
