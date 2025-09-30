import numpy as np
import math

def dft(x):
    """
    Discrete Fourier Transform implementation from first principles.
    Input: x - time domain signal
    Output: X - frequency domain signal
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
            
    return X

def fft(x):
    """
    Fast Fourier Transform implementation using Cooley-Tukey algorithm.
    Input: x - time domain signal
    Output: X - frequency domain signal
    """
    N = len(x)
    

    if N <= 1:
        return x
    

    if N & (N-1) != 0:

        next_power = 2**math.ceil(math.log2(N))
        x_padded = np.zeros(next_power)
        x_padded[:N] = x
        x = x_padded
        N = next_power
    

    even = fft(x[0::2])
    odd = fft(x[1::2])
    

    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    first_half = even + factor[:N//2] * odd
    second_half = even + factor[N//2:] * odd
    
    return np.concatenate([first_half, second_half])

def ifft(X):
    """
    Inverse Fast Fourier Transform implementation.
    Input: X - frequency domain signal
    Output: x - time domain signal
    """
    N = len(X)
    

    X_conjugate = np.conjugate(X)
    

    x_conjugate = fft(X_conjugate)
    

    x = np.conjugate(x_conjugate) / N
    
    return x

def stft(x, frame_size=1024, hop_size=256):
    """
    Short-Time Fourier Transform implementation.
    Input: x - time domain signal
    Output: X - spectrogram (time-frequency representation)
    """

    window = np.hanning(frame_size)
    

    num_frames = 1 + (len(x) - frame_size) // hop_size
    

    X = np.zeros((num_frames, frame_size), dtype=complex)
    

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        

        frame = x[start:end] * window
        

        X[i] = fft(frame)
    
    return X

def istft(X, hop_size=256):
    """
    Inverse Short-Time Fourier Transform implementation.
    Input: X - spectrogram
    Output: x - time domain signal
    """
    num_frames, frame_size = X.shape
    

    output_length = (num_frames - 1) * hop_size + frame_size
    

    x = np.zeros(output_length)
    norm = np.zeros(output_length)
    

    window = np.hanning(frame_size)
    

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        

        frame = np.real(ifft(X[i]))
        
 
        windowed_frame = frame * window
        

        x[start:end] += windowed_frame
        norm[start:end] += window**2
    

    idx = norm > 1e-10
    x[idx] /= norm[idx]
    
    return x