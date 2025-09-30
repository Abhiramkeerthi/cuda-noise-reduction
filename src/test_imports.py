try:
    import numpy as np
    import scipy
    import soundfile as sf
    import matplotlib.pyplot as plt
    import librosa
    print("Tüm gerekli kütüphaneler başarıyla yüklendi!")
    

    import sys
    import os
    sys.path.append('src')
    
    from audio_io import read_audio, write_audio
    from noise_reduction import adaptive_noise_reduction
    from real_time import RealTimeNoiseReducer
    print("Proje modülleri başarıyla import edildi!")
    
except ImportError as e:
    print(f"Import hatası: {e}")