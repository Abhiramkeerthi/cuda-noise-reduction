
import time
import numpy as np
import cupy as cp
from scipy.io import wavfile

FRAME_SIZE = 1024
HOP_SIZE = 256


def check_cuda():
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()

    print("=" * 70)
    print("CUDA AUDIO NOISE REDUCTION")
    print("=" * 70)
    print("GPU:", name)
    print("CuPy:", cp.__version__)
    print("=" * 70)


def load_audio(path):
    sr, audio = wavfile.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32)
        audio /= max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    return sr, audio


# ============================================================
# MULTI-SCALE IMPULSE DETECTION
# ============================================================

def detect_impulses(audio, sample_rate):

    n = len(audio)

    # --------------------------------------------------------
    # First derivative
    # --------------------------------------------------------

    d1 = cp.abs(audio[1:] - audio[:-1])
    d1 = cp.pad(d1, (0, 1))

    # Second derivative catches sharp attacks even better.
    d2 = cp.abs(d1[1:] - d1[:-1])
    d2 = cp.pad(d2, (0, 1))

    # --------------------------------------------------------
    # Robust statistics
    # --------------------------------------------------------

    d1_med = cp.median(d1)
    d2_med = cp.median(d2)

    d1_mad = cp.median(
        cp.abs(d1 - d1_med)
    ) + 1e-12

    d2_mad = cp.median(
        cp.abs(d2 - d2_med)
    ) + 1e-12

    d1_score = (
        d1 - d1_med
    ) / d1_mad

    d2_score = (
        d2 - d2_med
    ) / d2_mad

    # --------------------------------------------------------
    # Multi-scale local median residual
    # --------------------------------------------------------

    impulse_masks = []

    for radius in [4, 8, 16, 32]:

        shifted = []

        for k in range(
            -radius,
            radius + 1
        ):

            if k < 0:
                x = cp.concatenate([
                    cp.full(
                        -k,
                        audio[0]
                    ),
                    audio[:n + k]
                ])

            elif k > 0:
                x = cp.concatenate([
                    audio[k:],
                    cp.full(
                        k,
                        audio[-1]
                    )
                ])

            else:
                x = audio

            shifted.append(x)

        local_median = cp.median(
            cp.stack(shifted),
            axis=0
        )

        residual = cp.abs(
            audio - local_median
        )

        threshold = cp.percentile(
            residual,
            99.0
        )

        impulse_masks.append(
            residual > threshold
        )

    # --------------------------------------------------------
    # Combine scales
    # --------------------------------------------------------

    impulse_mask = (
        impulse_masks[0]
        |
        impulse_masks[1]
        |
        impulse_masks[2]
        |
        impulse_masks[3]
    )

    # Strong derivative condition catches impulses
    # that are not extreme in amplitude.

    impulse_mask |= (
        (d1_score > 10)
        &
        (d2_score > 6)
    )

    # --------------------------------------------------------
    # Ignore tiny numerical detections
    # --------------------------------------------------------

    # Require an actual cluster of impulse samples.
    # This prevents isolated numerical spikes from being
    # treated as noise.

    cluster = cp.zeros(
        n,
        dtype=cp.int32
    )

    original = impulse_mask.copy()

    for shift in range(1, 25):

        cluster += (
            original
        )

        original = cp.roll(
            original,
            1
        )

    impulse_mask &= (
        cluster >= 2
    )

    # --------------------------------------------------------
    # Expand detected impulse
    # --------------------------------------------------------

    original = impulse_mask.copy()

    expansion = min(
        int(0.006 * sample_rate),
        300
    )

    for shift in range(
        1,
        expansion + 1
    ):

        impulse_mask[shift:] |= (
            original[:-shift]
        )

        impulse_mask[:-shift] |= (
            original[shift:]
        )

    return impulse_mask


# ============================================================
# REPAIR IMPULSES
# ============================================================

def repair_impulses(
    audio,
    impulse_mask
):

    cleaned = audio.copy()

    indices = cp.where(
        impulse_mask
    )[0].get()

    if len(indices) == 0:
        return cleaned

    # Group consecutive samples.
    groups = []

    start = indices[0]
    previous = indices[0]

    for idx in indices[1:]:

        if idx > previous + 1:
            groups.append(
                (start, previous)
            )
            start = idx

        previous = idx

    groups.append(
        (start, previous)
    )

    print(
        "Impulse groups detected:",
        len(groups)
    )

    # --------------------------------------------------------
    # Interpolate across each detected tap.
    # --------------------------------------------------------

    for a, b in groups:

        length = b - a + 1

        # Ignore extremely tiny groups.
        if length < 2:
            continue

        # Don't let one accidental classification wipe
        # a long section of speech.
        if length > int(0.020 * 48000):
            continue

        left = max(
            0,
            a - int(0.004 * 48000)
        )

        right = min(
            len(audio) - 1,
            b + int(0.004 * 48000)
        )

        left_value = audio[left]
        right_value = audio[right]

        count = b - a + 1

        interpolation = cp.linspace(
            left_value,
            right_value,
            count + 2,
            dtype=cp.float32
        )[1:-1]

        cleaned[a:b + 1] = interpolation

    return cleaned


# ============================================================
# STFT
# ============================================================

def gpu_stft(audio):

    pad = FRAME_SIZE // 2

    padded = cp.pad(
        audio,
        (pad, pad)
    )

    num_frames = (
        1
        + (len(padded) - FRAME_SIZE)
        // HOP_SIZE
    )

    frames = cp.stack([
        padded[
            i * HOP_SIZE:
            i * HOP_SIZE + FRAME_SIZE
        ]
        for i in range(num_frames)
    ])

    window = cp.hanning(
        FRAME_SIZE
    ).astype(cp.float32)

    frames *= window[None, :]

    spectrum = cp.fft.rfft(
        frames,
        axis=1
    )

    return spectrum, window


# ============================================================
# ISTFT
# ============================================================

def gpu_istft(
    spectrum,
    window,
    original_length
):

    frames = cp.fft.irfft(
        spectrum,
        n=FRAME_SIZE,
        axis=1
    )

    frames *= window[None, :]

    num_frames = frames.shape[0]

    output_length = (
        (num_frames - 1)
        * HOP_SIZE
        + FRAME_SIZE
    )

    output = cp.zeros(
        output_length,
        dtype=cp.float32
    )

    window_sum = cp.zeros(
        output_length,
        dtype=cp.float32
    )

    for i in range(num_frames):

        start = i * HOP_SIZE
        end = start + FRAME_SIZE

        output[start:end] += frames[i]

        window_sum[start:end] += (
            window ** 2
        )

    output /= cp.maximum(
        window_sum,
        1e-8
    )

    pad = FRAME_SIZE // 2

    return output[
        pad:
        pad + original_length
    ]


# ============================================================
# WIENER
# ============================================================

def estimate_noise(
    magnitude,
    frame_energy
):

    threshold = cp.percentile(
        frame_energy,
        20
    )

    quiet = (
        frame_energy <= threshold
    )

    return cp.median(
        magnitude[quiet],
        axis=0
    )


def wiener_filter(
    spectrum,
    noise_profile
):

    magnitude = cp.abs(
        spectrum
    )

    power = magnitude ** 2

    noise_power = (
        noise_profile ** 2
        + 1e-10
    )

    snr = cp.maximum(
        power - noise_power,
        0
    ) / noise_power[None, :]

    gain = (
        snr /
        (snr + 1.0)
    )

    return cp.clip(
        gain,
        0.40,
        1.0
    )


# ============================================================
# BLOW DETECTOR
# ============================================================

def detect_blow(
    spectrum,
    frame_energy,
    sample_rate
):

    power = cp.abs(
        spectrum
    ) ** 2

    frequencies = cp.fft.rfftfreq(
        FRAME_SIZE,
        d=1.0 / sample_rate
    )

    low_bins = frequencies < 300

    low_energy = cp.sum(
        power[:, low_bins],
        axis=1
    )

    total_energy = cp.sum(
        power,
        axis=1
    ) + 1e-12

    low_ratio = (
        low_energy /
        total_energy
    )

    median_energy = cp.median(
        frame_energy
    )

    blow = (
        (low_ratio > 0.45)
        &
        (
            frame_energy
            > median_energy * 8
        )
    )

    original = blow.copy()

    for shift in range(1, 7):

        blow[shift:] |= (
            original[:-shift]
        )

        blow[:-shift] |= (
            original[shift:]
        )

    return blow


# ============================================================
# BLOW SUPPRESSION
# ============================================================

def suppress_blow(
    gain,
    blow,
    sample_rate
):

    frequencies = cp.fft.rfftfreq(
        FRAME_SIZE,
        d=1.0 / sample_rate
    )

    blow_gain = cp.interp(

        frequencies,

        cp.array([
            0,
            100,
            200,
            300,
            500,
            800,
            1200,
            2000,
            4000,
            sample_rate / 2
        ], dtype=cp.float32),

        cp.array([
            0.002,
            0.002,
            0.003,
            0.005,
            0.01,
            0.02,
            0.04,
            0.10,
            0.35,
            0.70
        ], dtype=cp.float32)
    )

    gain[blow] *= (
        blow_gain[None, :]
    )

    return gain


# ============================================================
# MAIN
# ============================================================

def process_file_cuda(
    input_path,
    output_path
):

    check_cuda()

    sample_rate, audio = load_audio(
        input_path
    )

    duration = (
        len(audio)
        / sample_rate
    )

    print(
        "\nDuration:",
        round(duration, 3),
        "seconds"
    )

    audio_gpu = cp.asarray(
        audio,
        dtype=cp.float32
    )

    cp.fft.fft(
        cp.zeros(
            1024,
            dtype=cp.float32
        )
    )

    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()

    # ========================================================
    # 1. FIND ALL IMPULSIVE NOISE
    # ========================================================

    print(
        "\nScanning entire recording for taps/impacts..."
    )

    impulse_mask = detect_impulses(
        audio_gpu,
        sample_rate
    )

    impulse_count = int(
        cp.sum(impulse_mask)
    )

    print(
        "Impulse samples detected:",
        impulse_count
    )

    # ========================================================
    # 2. REPAIR IMPULSES
    # ========================================================

    cleaned_audio = repair_impulses(
        audio_gpu,
        impulse_mask
    )

    # ========================================================
    # 3. STFT
    # ========================================================

    spectrum, window = gpu_stft(
        cleaned_audio
    )

    magnitude = cp.abs(
        spectrum
    )

    frame_energy = cp.mean(
        magnitude ** 2,
        axis=1
    )

    print(
        "STFT frames:",
        spectrum.shape[0]
    )

    # ========================================================
    # 4. WIENER
    # ========================================================

    noise_profile = estimate_noise(
        magnitude,
        frame_energy
    )

    gain = wiener_filter(
        spectrum,
        noise_profile
    )

    # ========================================================
    # 5. BLOW
    # ========================================================

    blow = detect_blow(
        spectrum,
        frame_energy,
        sample_rate
    )

    print(
        "Blow frames detected:",
        int(cp.sum(blow))
    )

    gain = suppress_blow(
        gain,
        blow,
        sample_rate
    )

    # ========================================================
    # 6. ISTFT
    # ========================================================

    enhanced_gpu = gpu_istft(
        spectrum * gain,
        window,
        len(audio)
    )

    cp.cuda.Stream.null.synchronize()

    processing_time = (
        time.perf_counter()
        - start
    )

    enhanced = cp.asnumpy(
        enhanced_gpu
    )

    # ========================================================
    # 7. OUTPUT LEVEL
    # ========================================================

    original_rms = np.sqrt(
        np.mean(
            audio ** 2
        )
    )

    enhanced_rms = np.sqrt(
        np.mean(
            enhanced ** 2
        )
    )

    scale = (
        original_rms /
        (enhanced_rms + 1e-12)
    )

    scale = np.clip(
        scale,
        0.90,
        1.10
    )

    enhanced *= scale

    enhanced = np.clip(
        enhanced,
        -1.0,
        1.0
    )

    # ========================================================
    # 8. SAVE
    # ========================================================

    output = (
        enhanced * 32767
    ).astype(
        np.int16
    )

    wavfile.write(
        output_path,
        sample_rate,
        output
    )

    print("\n" + "=" * 70)
    print("FINAL CUDA PROCESSING COMPLETE")
    print("=" * 70)
    print("Output:", output_path)
    print(
        "Processing time:",
        round(processing_time, 4),
        "seconds"
    )
    print(
        "Real-time factor:",
        round(
            duration /
            processing_time,
            2
        ),
        "x"
    )
    print("=" * 70)

    return output_path
