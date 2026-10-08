
import time
import numpy as np
import cupy as cp
from scipy.io import wavfile


FRAME_SIZE = 1024
HOP_SIZE = 256
NOISE_FRAMES = 10


def check_cuda():
    if not cp.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)

    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()

    print("=" * 60)
    print("CUDA DEVICE")
    print("=" * 60)
    print("GPU:", name)
    print("Device ID:", device.id)
    print("CuPy:", cp.__version__)
    print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
    print("=" * 60)


def gpu_stft(audio_gpu, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):

    audio_length = len(audio_gpu)

    if audio_length < frame_size:
        padded = cp.zeros(frame_size, dtype=audio_gpu.dtype)
        padded[:audio_length] = audio_gpu
        audio_gpu = padded
        audio_length = frame_size

    num_frames = 1 + (audio_length - frame_size) // hop_size

    starts = cp.arange(num_frames, dtype=cp.int64) * hop_size
    offsets = cp.arange(frame_size, dtype=cp.int64)

    indices = starts[:, None] + offsets[None, :]

    frames = audio_gpu[indices]

    window = cp.hanning(frame_size)

    frames = frames * window

    # CUDA FFT
    spectrogram = cp.fft.fft(frames, axis=1)

    return spectrogram


def gpu_istft(
    spectrogram,
    original_length,
    frame_size=FRAME_SIZE,
    hop_size=HOP_SIZE
):

    num_frames = spectrogram.shape[0]

    # CUDA inverse FFT
    frames = cp.fft.ifft(
        spectrogram,
        axis=1
    ).real

    window = cp.hanning(frame_size)
    frames = frames * window

    output_length = (
        (num_frames - 1) * hop_size
        + frame_size
    )

    output = cp.zeros(
        output_length,
        dtype=cp.float64
    )

    normalization = cp.zeros(
        output_length,
        dtype=cp.float64
    )

    starts = cp.arange(num_frames, dtype=cp.int64) * hop_size
    offsets = cp.arange(frame_size, dtype=cp.int64)

    indices = starts[:, None] + offsets[None, :]

    # GPU overlap-add
    cp.add.at(
        output,
        indices.ravel(),
        frames.ravel()
    )

    window_squared = window ** 2

    norm_values = cp.broadcast_to(
        window_squared,
        (num_frames, frame_size)
    )

    cp.add.at(
        normalization,
        indices.ravel(),
        norm_values.ravel()
    )

    output /= cp.maximum(
        normalization,
        1e-10
    )

    return output[:original_length]


def estimate_noise_profile_gpu(
    spectrogram,
    noise_frames=NOISE_FRAMES
):

    noise_frames = min(
        noise_frames,
        spectrogram.shape[0]
    )

    noise = spectrogram[:noise_frames]

    magnitude = cp.abs(noise)

    profile = cp.mean(
        magnitude,
        axis=0
    )

    return cp.maximum(
        profile,
        1e-8
    )


def gpu_wiener_filter(
    audio,
    frame_size=FRAME_SIZE,
    hop_size=HOP_SIZE,
    noise_frames=NOISE_FRAMES
):

    original_length = len(audio)

    # CPU → GPU
    audio_gpu = cp.asarray(
        audio,
        dtype=cp.float64
    )

    # STFT
    spectrogram = gpu_stft(
        audio_gpu,
        frame_size,
        hop_size
    )

    # Noise estimate
    noise_profile = estimate_noise_profile_gpu(
        spectrogram,
        noise_frames
    )

    magnitude = cp.abs(
        spectrogram
    )

    signal_power = magnitude ** 2

    noise_power = noise_profile ** 2

    # Wiener SNR estimate
    snr = (
        signal_power /
        (noise_power[None, :] + 1e-10)
    ) - 1.0

    snr = cp.maximum(
        snr,
        0.0
    )

    # Wiener gain
    gain = snr / (snr + 1.0)

    # Prevent total removal
    gain = cp.maximum(
        gain,
        0.05
    )

    # Apply filter
    enhanced_spectrogram = (
        spectrogram * gain
    )

    # Inverse STFT
    enhanced_gpu = gpu_istft(
        enhanced_spectrogram,
        original_length,
        frame_size,
        hop_size
    )

    # GPU → CPU
    return cp.asnumpy(
        enhanced_gpu
    )


def process_file_cuda(
    input_file,
    output_file
):

    check_cuda()

    sample_rate, audio = wavfile.read(
        input_file
    )

    print("\nInput file:", input_file)
    print("Sample rate:", sample_rate)
    print("Samples:", len(audio))

    duration = len(audio) / sample_rate

    print(
        f"Duration: {duration:.2f} seconds"
    )

    # Convert to floating point
    if np.issubdtype(
        audio.dtype,
        np.integer
    ):

        info = np.iinfo(audio.dtype)

        scale = max(
            abs(info.min),
            info.max
        )

        audio = (
            audio.astype(np.float64)
            / scale
        )

    else:

        audio = audio.astype(
            np.float64
        )

    # Stereo → mono
    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    print("\nRunning CUDA Wiener filter...")

    # Warm-up
    warmup = audio[:min(
        len(audio),
        sample_rate
    )]

    gpu_wiener_filter(warmup)

    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.perf_counter()

    enhanced = gpu_wiener_filter(
        audio
    )

    cp.cuda.Stream.null.synchronize()

    elapsed = (
        time.perf_counter() - start
    )

    # Normalize
    peak = np.max(
        np.abs(enhanced)
    )

    if peak > 1.0:
        enhanced /= peak

    output_audio = np.int16(
        np.clip(
            enhanced,
            -1.0,
            1.0
        ) * 32767
    )

    wavfile.write(
        output_file,
        sample_rate,
        output_audio
    )

    print("\n" + "=" * 60)
    print("CUDA PROCESSING COMPLETE")
    print("=" * 60)
    print(
        f"Processing time : {elapsed:.6f} sec"
    )
    print(
        f"Audio duration  : {duration:.6f} sec"
    )
    print(
        f"Real-time factor: {duration / elapsed:.2f}x"
    )
    print(
        f"Output          : {output_file}"
    )
    print("=" * 60)

    return elapsed
