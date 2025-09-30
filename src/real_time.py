import numpy as np
import queue
import threading
import time
from fft import fft, ifft
from noise_analysis import update_noise_profile, detect_voice_activity

class RealTimeNoiseReducer:
    def __init__(self, sample_rate=44100, frame_size=1024, hop_size=256):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        

        self.noise_profile = np.ones(frame_size) * 1e-3
        

        self.input_buffer = np.zeros(frame_size)
        self.output_buffer = np.zeros(frame_size)
        self.buffer_position = 0
        

        self.overlap_buffer = np.zeros(frame_size)
        

        self.window = np.hanning(frame_size)
        

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        

        self.is_noise_estimation = True
        self.noise_frames_collected = 0
        self.required_noise_frames = 10
        

        self.subtraction_factor = 2.0
        self.noise_floor = 0.01
        self.adaptation_rate = 0.02
        

        self.running = False
        self.processing_thread = None
    
    def start(self):
        """Start the real-time processing thread."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        """Stop the real-time processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def _processing_loop(self):
        """Main processing loop for real-time noise reduction."""
        while self.running:

            try:
                input_frame = self.input_queue.get(block=True, timeout=0.1)
            except queue.Empty:

                continue
            

            output_frame = self._process_frame(input_frame)
            

            self.output_queue.put(output_frame)
    
    def _process_frame(self, input_frame):
        """Process a single audio frame with noise reduction."""

        windowed_frame = input_frame * self.window
        

        frame_spectrum = fft(windowed_frame)
        frame_magnitude = np.abs(frame_spectrum)
        frame_phase = np.angle(frame_spectrum)
        

        if self.is_noise_estimation and self.noise_frames_collected < self.required_noise_frames:

            self.noise_profile = (self.noise_frames_collected * self.noise_profile + frame_magnitude) / (self.noise_frames_collected + 1)
            self.noise_frames_collected += 1
            

            if self.noise_frames_collected >= self.required_noise_frames:
                self.is_noise_estimation = False
            

            return input_frame
        

        voice_mask = detect_voice_activity(frame_spectrum, self.noise_profile, threshold_factor=2.5)
        

        update_noise_profile(self.noise_profile, frame_spectrum, voice_mask, alpha=self.adaptation_rate)
        

        clean_magnitude = np.maximum(
            frame_magnitude - self.subtraction_factor * self.noise_profile,
            self.noise_floor * frame_magnitude
        )
        

        clean_spectrum = clean_magnitude * np.exp(1j * frame_phase)
        

        clean_frame = np.real(ifft(clean_spectrum))
        

        clean_frame = clean_frame * self.window
        

        output_frame = self.overlap_buffer + clean_frame[:self.frame_size]
        

        self.overlap_buffer = np.zeros(self.frame_size)
        if self.frame_size > self.hop_size:
            self.overlap_buffer[:self.frame_size-self.hop_size] = clean_frame[self.hop_size:]
        
        return output_frame
    
    def process_chunk(self, input_chunk):
        """Process a chunk of audio data in a blocking manner."""

        chunk_length = len(input_chunk)
        

        if chunk_length < self.hop_size:
            padded_chunk = np.zeros(self.hop_size)
            padded_chunk[:chunk_length] = input_chunk
            input_chunk = padded_chunk
        

        num_frames = (chunk_length - self.frame_size) // self.hop_size + 1
        output_chunk = np.zeros(chunk_length)
        
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            
            if end > chunk_length:

                frame = np.zeros(self.frame_size)
                frame[:chunk_length-start] = input_chunk[start:]
            else:
                frame = input_chunk[start:end]
            

            output_frame = self._process_frame(frame)
            

            if i == 0:
                output_chunk[start:end] = output_frame
            else:

                output_chunk[start:end] += output_frame
        
        return output_chunk[:chunk_length]