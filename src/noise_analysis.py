import numpy as np
from fft import stft


def estimate_noise_profile(
    audio_data,
    frame_size=1024,
    hop_size=256,
    percentile=20
):
    """
    Estimate a frequency-dependent noise magnitude profile
    using the lower-energy portion of the complete recording.

    Does not assume that the beginning of the recording
    contains only noise.
    """

    spectrogram = stft(
        audio_data,
        frame_size,
        hop_size
    )

    magnitude = np.abs(spectrogram)

    frame_power = np.mean(
        magnitude ** 2,
        axis=1
    )

    noise_threshold = np.percentile(
        frame_power,
        percentile
    )

    noise_frames = (
        frame_power <= noise_threshold
    )

    if np.sum(noise_frames) < 5:

        count = max(
            5,
            len(frame_power) // 20
        )

        count = min(
            count,
            len(frame_power)
        )

        indices = np.argsort(
            frame_power
        )[:count]

        noise_frames = np.zeros(
            len(frame_power),
            dtype=bool
        )

        noise_frames[indices] = True

    noise_profile = np.median(
        magnitude[noise_frames],
        axis=0
    )

    noise_profile = np.maximum(
        noise_profile,
        1e-8
    )

    return noise_profile


def frame_is_speech(
    frame_spectrum,
    noise_profile,
    energy_threshold=2.5
):
    """
    Frame-level voice activity detector.

    Returns True for likely speech and False for
    likely non-speech/noise.
    """

    magnitude = np.abs(
        frame_spectrum
    )

    frame_power = np.mean(
        magnitude ** 2
    )

    noise_power = (
        np.mean(
            noise_profile ** 2
        )
        + 1e-12
    )

    energy_ratio = (
        frame_power /
        noise_power
    )

    return (
        energy_ratio >
        energy_threshold
    )


def detect_voice_activity(
    frame_spectrum,
    noise_profile,
    threshold_factor=2.5
):
    """
    Frequency-bin voice activity detector.

    Kept for compatibility with real_time.py.
    """

    magnitude = np.abs(
        frame_spectrum
    )

    ratio = (
        magnitude /
        (noise_profile + 1e-10)
    )

    return (
        ratio >
        threshold_factor
    )


def update_noise_profile(
    noise_profile,
    frame_spectrum,
    voice_activity,
    alpha=0.01
):
    """
    Slowly update the noise profile.

    Noise is updated only when a frame is classified
    as non-speech.
    """

    if isinstance(
        voice_activity,
        (bool, np.bool_)
    ):

        if voice_activity:
            return noise_profile

        magnitude = np.abs(
            frame_spectrum
        )

        return (
            (1.0 - alpha)
            * noise_profile
            +
            alpha
            * magnitude
        )

    magnitude = np.abs(
        frame_spectrum
    )

    non_voice = ~voice_activity

    updated = noise_profile.copy()

    updated[non_voice] = (
        (1.0 - alpha)
        * noise_profile[non_voice]
        +
        alpha
        * magnitude[non_voice]
    )

    return updated