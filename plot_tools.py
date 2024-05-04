# This script is used to plot some figures
import numpy as np
import matplotlib.pyplot as plt
import librosa


def plot_waveform_td(waveform,sr,title='Waveform in time domain'):
    """
    This function is used to plot the waveform in time domain
    """
    waveform = np.array(waveform)
    samples = waveform.size
    time_scale = np.linspace(0, len(waveform)/samples,num=samples)
    plt.figure(figsize=(10,5))
    plt.plot(time_scale,waveform,linewidth=1)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_waveform_fd(waveform,sr,n_fft,title="Waveform in frequency domain"):
    """
    This function is used to plot the waveform in frequency domain
    """
    waveform = np.array(waveform)
    samples = waveform.size
    waveform_fd = np.fft.rfft(waveform, n_fft)
    freq_scale = np.linspace(0,sr/2,num=int(n_fft/2)+1)
    plt.figure(figsize=(10,5))
    plt.plot(freq_scale, waveform_fd, linewidth=1)
    plt.title(title)
    plt.show()

def plot_waveform_stft(spectrogram, title="Waveform in STFT domain (dB)"):
    # Convert amplitude spectrum to dB-scaled spectrum
    db_spectrogram = librosa.amplitude_to_db(spectrogram)
    # Plot using imshow with adjusted aspect ratio
    plt.figure(figsize=(10, 4))  
    plt.imshow(db_spectrogram, aspect='auto', origin='lower')
    plt.title(title)  
    plt.xlabel('Frames')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.show()
