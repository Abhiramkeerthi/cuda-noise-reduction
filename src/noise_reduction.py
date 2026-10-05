import numpy as np

from fft import stft, istft

from noise_analysis import (
    estimate_noise_profile,
    frame_is_speech,
    update_noise_profile
)


# ============================================================
# FREQUENCY SMOOTHING
# ============================================================

def smooth_frequency_mask(mask):
    """
    Smooth neighboring frequency bins.

    This reduces musical/metallic artifacts.
    """

    if len(mask) < 3:
        return mask

    result = mask.copy()

    result[1:-1] = (
        0.25 * mask[:-2]
        +
        0.50 * mask[1:-1]
        +
        0.25 * mask[2:]
    )

    return result


# ============================================================
# SPECTRAL SUBTRACTION
# ============================================================

def spectral_subtraction(
    audio_data,
    noise_profile,
    frame_size=1024,
    hop_size=256,
    subtraction_factor=1.5,
    noise_floor=0.10
):
    """
    Spectral subtraction.
    """

    spectrogram = stft(
        audio_data,
        frame_size,
        hop_size
    )

    for i in range(
        len(spectrogram)
    ):

        spectrum = spectrogram[i]

        magnitude = np.abs(
            spectrum
        )

        phase = np.angle(
            spectrum
        )

        cleaned = np.maximum(
            magnitude
            -
            subtraction_factor
            * noise_profile,

            noise_floor
            * magnitude
        )

        spectrogram[i] = (
            cleaned
            *
            np.exp(
                1j * phase
            )
        )

    output = istft(
        spectrogram,
        hop_size
    )

    return output[
        :len(audio_data)
    ]


# ============================================================
# WIENER FILTER
# ============================================================

def wiener_filter(
    audio_data,
    noise_profile,
    frame_size=1024,
    hop_size=256
):
    """
    Standard conservative Wiener filter.
    """

    spectrogram = stft(
        audio_data,
        frame_size,
        hop_size
    )

    enhanced = np.zeros_like(
        spectrogram
    )

    noise_power = (
        noise_profile ** 2
    )

    for i in range(
        len(spectrogram)
    ):

        spectrum = spectrogram[i]

        magnitude = np.abs(
            spectrum
        )

        signal_power = (
            magnitude ** 2
        )

        snr = np.maximum(
            signal_power /
            (noise_power + 1e-12)
            -
            1.0,
            0.0
        )

        gain = (
            snr /
            (snr + 1.0)
        )

        gain = np.maximum(
            gain,
            0.30
        )

        gain = smooth_frequency_mask(
            gain
        )

        enhanced[i] = (
            magnitude
            *
            gain
            *
            np.exp(
                1j *
                np.angle(spectrum)
            )
        )

    output = istft(
        enhanced,
        hop_size
    )

    return output[
        :len(audio_data)
    ]


# ============================================================
# TRANSIENT DETECTION
# ============================================================

def detect_transients(
    spectrogram,
    threshold=8.0
):
    """
    Detect strong sudden broadband events.

    Intended for:
        microphone hits
        knocks
        clicks
        handling noise
    """

    magnitude = np.abs(
        spectrogram
    )

    num_frames = (
        magnitude.shape[0]
    )

    flux = np.zeros(
        num_frames
    )

    energy = np.mean(
        magnitude ** 2,
        axis=1
    )

    if num_frames > 1:

        change = np.maximum(
            magnitude[1:]
            -
            magnitude[:-1],
            0.0
        )

        flux[1:] = np.mean(
            change,
            axis=1
        )

    flux_median = np.median(
        flux
    )

    flux_mad = (
        np.median(
            np.abs(
                flux -
                flux_median
            )
        )
        +
        1e-10
    )

    flux_score = (
        flux -
        flux_median
    ) / flux_mad

    energy_median = (
        np.median(
            energy
        )
        +
        1e-12
    )

    energy_ratio = (
        energy /
        energy_median
    )

    transient_mask = (
        (flux_score > threshold)
        &
        (energy_ratio > 4.0)
    )

    if num_frames:
        transient_mask[0] = False

    if num_frames > 1:
        transient_mask[-1] = False

    return transient_mask


# ============================================================
# TRANSIENT SUPPRESSION
# ============================================================

def suppress_transients(
    spectrogram,
    transient_mask,
    strength=0.10
):
    """
    Suppress detected impulse frames.

    Only frames already identified as strong transients
    are modified.

    strength:
        0.0 = maximum suppression
        1.0 = no suppression
    """

    output = spectrogram.copy()

    magnitude = np.abs(
        spectrogram
    )

    phase = np.angle(
        spectrogram
    )

    count = 0

    for i in range(
        1,
        len(spectrogram) - 1
    ):

        if not transient_mask[i]:
            continue

        count += 1

        surrounding = (
            magnitude[i - 1]
            +
            magnitude[i + 1]
        ) / 2.0

        cleaned = (
            strength * magnitude[i]
            +
            (1.0 - strength)
            * surrounding
        )

        output[i] = (
            cleaned
            *
            np.exp(
                1j *
                phase[i]
            )
        )

    print(
        f"   - Transient frames detected: "
        f"{count}"
    )

    return output


# ============================================================
# ADAPTIVE SPECTRAL GATE
# ============================================================

def adaptive_noise_reduction(
    audio_data,
    frame_size=1024,
    hop_size=256,
    alpha=0.01
):
    """
    Main noise-reduction algorithm.

    Uses a soft spectral gate instead of relying solely
    on the Wiener equation.

    Pipeline:

        Audio
          ↓
        STFT
          ↓
        Noise floor estimation
          ↓
        VAD
          ↓
        Adaptive noise tracking
          ↓
        Per-bin SNR
          ↓
        Soft spectral gate
          ↓
        Frequency smoothing
          ↓
        Time smoothing
          ↓
        Transient suppression
          ↓
        iSTFT
    """

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    print(
        "   - STFT hesaplanıyor..."
    )

    spectrogram = stft(
        audio_data,
        frame_size,
        hop_size
    )

    num_frames = (
        len(spectrogram)
    )

    print(
        f"   - Toplam ses çerçevesi: "
        f"{num_frames}"
    )

    # --------------------------------------------------------
    # INITIAL NOISE PROFILE
    # --------------------------------------------------------

    print(
        "   - Dosyanın tamamından "
        "gürültü profili hesaplanıyor..."
    )

    noise_profile = (
        estimate_noise_profile(
            audio_data,
            frame_size,
            hop_size,
            percentile=20
        )
    )

    noise_profile = (
        noise_profile.copy()
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    enhanced = np.zeros_like(
        spectrogram
    )

    speech_frames = 0
    noise_frames = 0

    previous_mask = None

    # --------------------------------------------------------
    # PROCESS EACH FRAME
    # --------------------------------------------------------

    print(
        "   - VAD + adaptif spectral gating..."
    )

    for i in range(
        num_frames
    ):

        spectrum = (
            spectrogram[i]
        )

        magnitude = np.abs(
            spectrum
        )

        phase = np.angle(
            spectrum
        )

        # ----------------------------------------------------
        # VAD
        # ----------------------------------------------------

        is_speech = frame_is_speech(
            spectrum,
            noise_profile,
            energy_threshold=2.0
        )

        if is_speech:

            speech_frames += 1

        else:

            noise_frames += 1

            noise_profile = (
                update_noise_profile(
                    noise_profile,
                    spectrum,
                    False,
                    alpha
                )
            )

        # ----------------------------------------------------
        # PER-FREQUENCY SNR
        # ----------------------------------------------------

        ratio = (
            magnitude /
            (noise_profile + 1e-10)
        )

        # ----------------------------------------------------
        # SOFT SPECTRAL GATE
        # ----------------------------------------------------
        #
        # ratio < 1.0
        #     signal is approximately at/below noise
        #
        # ratio ~ 2.0
        #     transition region
        #
        # ratio >= 3.0
        #     preserve signal
        #
        # ----------------------------------------------------

        gate_start = 1.20
        gate_end = 2.50

        mask = (
            ratio - gate_start
        ) / (
            gate_end - gate_start
        )

        mask = np.clip(
            mask,
            0.0,
            1.0
        )

        # Keep a small amount of low-level signal
        # to avoid hard digital artifacts.
        minimum_gain = 0.12

        mask = (
            minimum_gain
            +
            (
                1.0 -
                minimum_gain
            )
            * mask
        )

        # ----------------------------------------------------
        # SPEECH BOOST
        # ----------------------------------------------------

        if is_speech:

            # During speech, use a more conservative
            # minimum gain to preserve intelligibility.
            mask = np.maximum(
                mask,
                0.30
            )

        # ----------------------------------------------------
        # FREQUENCY SMOOTHING
        # ----------------------------------------------------

        mask = (
            smooth_frequency_mask(
                mask
            )
        )

        # ----------------------------------------------------
        # TIME SMOOTHING
        # ----------------------------------------------------

        if previous_mask is not None:

            mask = (
                0.20 *
                previous_mask
                +
                0.80 *
                mask
            )

        previous_mask = mask

        # ----------------------------------------------------
        # APPLY MASK
        # ----------------------------------------------------

        enhanced[i] = (
            magnitude
            *
            mask
            *
            np.exp(
                1j *
                phase
            )
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print(
        f"   - Speech frames: "
        f"{speech_frames}"
    )

    print(
        f"   - Non-speech frames: "
        f"{noise_frames}"
    )

    # --------------------------------------------------------
    # TRANSIENT DETECTION
    # --------------------------------------------------------

    print(
        "   - Impulse/transient "
        "noise detection..."
    )

    transient_mask = (
        detect_transients(
            enhanced,
            threshold=8.0
        )
    )

    # --------------------------------------------------------
    # TRANSIENT SUPPRESSION
    # --------------------------------------------------------

    enhanced = (
        suppress_transients(
            enhanced,
            transient_mask,
            strength=0.10
        )
    )

    # --------------------------------------------------------
    # ISTFT
    # --------------------------------------------------------

    print(
        "   - iSTFT..."
    )

    output = istft(
        enhanced,
        hop_size
    )

    # --------------------------------------------------------
    # EXACT LENGTH
    # --------------------------------------------------------

    if len(output) > len(audio_data):

        output = output[
            :len(audio_data)
        ]

    elif len(output) < len(audio_data):

        output = np.pad(
            output,
            (
                0,
                len(audio_data)
                -
                len(output)
            )
        )

    # --------------------------------------------------------
    # SAFE NORMALIZATION
    # --------------------------------------------------------

    peak = np.max(
        np.abs(output)
    )

    if peak > 0.99:

        output = (
            output / peak
        ) * 0.99

    return output