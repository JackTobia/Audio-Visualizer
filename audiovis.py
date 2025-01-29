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

def update(frame, line, data, frame_rate, chunk_size, start_time):
    """Updates the sine wave plot dynamically based on audio level."""
    elapsed_time = pygame.time.get_ticks() - start_time  # Time in milliseconds
    current_frame = int((elapsed_time / 1000.0) * frame_rate)  # Convert ms to samples
    start = max(current_frame - chunk_size, 0)  # Keep within bounds
    end = start + chunk_size

    if end > len(data):
        return line,  # Stop updating when data ends

    # Update the sine wave with audio data
    x = np.linspace(0, chunk_size, chunk_size)
    y = np.interp(x, np.arange(chunk_size), data[start:end])  # Interpolation for smoothness

    line.set_ydata(y)  # Update the sine wave

    return line,

def visualize_audio(audio_file):
    """Animates a smooth sine wave visualization that responds to audio."""
    data, frame_rate = read_wav(audio_file)
    chunk_size = 2048  # Number of samples per update for smoother visual

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, chunk_size, chunk_size)
    line, = ax.plot(x, np.zeros(chunk_size), lw=2)

    ax.set_ylim(-1, 1)  # Normalized amplitude range
    ax.set_xlim(0, chunk_size)
    ax.set_xticks([])
    ax.set_yticks([])

    # Start playing audio
    play_audio(audio_file)
    start_time = pygame.time.get_ticks()  # Record start time

    # Create an animation
    ani = animation.FuncAnimation(fig, update, fargs=(line, data, frame_rate, chunk_size, start_time),
                                  interval=30, blit=True)

    plt.show()

# Run the visualization
if __name__ == "__main__":
    # audio_file = input("Provide a WAV filepath to visualize and play: ")
    visualize_audio("Atlantis.wav")
