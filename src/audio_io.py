import numpy as np
from scipy.io import wavfile  
import os

def read_audio(file_path):
    """Read various audio file formats and convert to standard format."""
    file_extension = os.path.splitext(file_path)[1].lower()
    

    if file_extension == '.wav':
        sample_rate, audio_data = wavfile.read(file_path)
    elif file_extension in ['.mp3', '.m4a']:

        import soundfile as sf  
        audio_data, sample_rate = sf.read(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    

    if audio_data.dtype == np.int16:
        audio_data = audio_data / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data / 2147483648.0
        
    return sample_rate, audio_data

def write_audio(file_path, sample_rate, audio_data):
    """Write audio data to file in various formats."""
    file_extension = os.path.splitext(file_path)[1].lower()
    

    dir_path = os.path.dirname(file_path)
    if dir_path:  
        os.makedirs(dir_path, exist_ok=True)
    
    
    if file_extension == '.wav':
        
        scaled_data = np.int16(audio_data * 32767)
        wavfile.write(file_path, sample_rate, scaled_data)
    elif file_extension in ['.mp3', '.m4a']:
        
        import soundfile as sf
        sf.write(file_path, audio_data, sample_rate, format=file_extension[1:])
    else:
        raise ValueError(f"Unsupported output format: {file_extension}")