import numpy as np
from fft import stft

def estimate_noise_profile(audio_data, frame_size=1024, hop_size=256, noise_frames=10):
    """
    Estimate noise profile from the beginning of the audio.
    Assumes the first few frames contain only noise.
    """

    spectrogram = stft(audio_data, frame_size, hop_size)
    

    noise_spectrogram = spectrogram[:noise_frames, :]
    

    noise_profile = np.mean(np.abs(noise_spectrogram), axis=0)
    
    return noise_profile

def detect_voice_activity(frame_spectrum, noise_profile, threshold_factor=2.5):
    """
    Detect voice activity in a frame using adaptive thresholding.
    Returns a mask of frequency bins with voice activity.
    """
    frame_magnitude = np.abs(frame_spectrum)
    

    snr = np.zeros_like(frame_magnitude)
    nonzero_mask = noise_profile > 1e-10
    snr[nonzero_mask] = frame_magnitude[nonzero_mask] / noise_profile[nonzero_mask]
    

    voice_mask = snr > threshold_factor
    
    return voice_mask

def update_noise_profile(noise_profile, frame_spectrum, voice_mask, alpha=0.1):
    """
    Update noise profile using non-voice frames.
    Uses exponential averaging for smooth updates.
    """
    frame_magnitude = np.abs(frame_spectrum)
    

    non_voice_mask = ~voice_mask
    

    if np.any(non_voice_mask):
        noise_profile[non_voice_mask] = (1 - alpha) * noise_profile[non_voice_mask] + \
                                       alpha * frame_magnitude[non_voice_mask]
    
    return noise_profile