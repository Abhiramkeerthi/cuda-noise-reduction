import numpy as np
from fft import stft, istft
from noise_analysis import estimate_noise_profile, detect_voice_activity, update_noise_profile

def spectral_subtraction(audio_data, noise_profile, frame_size=1024, hop_size=256, 
                        subtraction_factor=2.0, noise_floor=0.01):
    """
    Basic spectral subtraction algorithm for noise reduction.
    """

    spectrogram = stft(audio_data, frame_size, hop_size)
    num_frames, num_bins = spectrogram.shape
    

    for i in range(num_frames):
        frame_spectrum = spectrogram[i]
        frame_magnitude = np.abs(frame_spectrum)
        frame_phase = np.angle(frame_spectrum)
        

        clean_magnitude = np.maximum(
            frame_magnitude - subtraction_factor * noise_profile,
            noise_floor * frame_magnitude
        )
        

        spectrogram[i] = clean_magnitude * np.exp(1j * frame_phase)
    

    clean_audio = istft(spectrogram, hop_size)
    

    if len(clean_audio) > len(audio_data):
        clean_audio = clean_audio[:len(audio_data)]
    
    return clean_audio

def wiener_filter(audio_data, noise_profile, frame_size=1024, hop_size=256):
    """
    Wiener filter implementation for noise reduction.
    """

    spectrogram = stft(audio_data, frame_size, hop_size)
    num_frames, num_bins = spectrogram.shape
    

    enhanced_spectrogram = np.zeros_like(spectrogram)
    

    for i in range(num_frames):
        frame_spectrum = spectrogram[i]
        frame_magnitude_squared = np.abs(frame_spectrum)**2
        frame_phase = np.angle(frame_spectrum)
        

        noise_magnitude_squared = noise_profile**2
        priori_snr = np.maximum(frame_magnitude_squared / (noise_magnitude_squared + 1e-10) - 1, 0)
        

        gain = priori_snr / (priori_snr + 1)
        

        enhanced_magnitude = np.sqrt(frame_magnitude_squared) * gain
        

        enhanced_spectrogram[i] = enhanced_magnitude * np.exp(1j * frame_phase)
    

    enhanced_audio = istft(enhanced_spectrogram, hop_size)
    

    if len(enhanced_audio) > len(audio_data):
        enhanced_audio = enhanced_audio[:len(audio_data)]
    
    return enhanced_audio

def adaptive_noise_reduction(audio_data, frame_size=1024, hop_size=256, 
                            noise_frames=10, alpha=0.05):
    """
    Adaptive noise reduction that updates the noise profile over time.
    """

    spectrogram = stft(audio_data, frame_size, hop_size)
    num_frames, num_bins = spectrogram.shape
    

    noise_profile = estimate_noise_profile(audio_data, frame_size, hop_size, noise_frames)
    

    enhanced_spectrogram = np.zeros_like(spectrogram)
    

    for i in range(num_frames):
        frame_spectrum = spectrogram[i]
        frame_magnitude = np.abs(frame_spectrum)
        frame_phase = np.angle(frame_spectrum)
        

        voice_mask = detect_voice_activity(frame_spectrum, noise_profile)
        

        if i >= noise_frames:  
            noise_profile = update_noise_profile(noise_profile, frame_spectrum, voice_mask, alpha)
        

        subtraction_factor = 2.0
        noise_floor = 0.01
        
        clean_magnitude = np.maximum(
            frame_magnitude - subtraction_factor * noise_profile,
            noise_floor * frame_magnitude
        )
        

        enhanced_spectrogram[i] = clean_magnitude * np.exp(1j * frame_phase)
    

    enhanced_audio = istft(enhanced_spectrogram, hop_size)
    

    if len(enhanced_audio) > len(audio_data):
        enhanced_audio = enhanced_audio[:len(audio_data)]
    
    return enhanced_audio