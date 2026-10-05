import os
import time
import argparse
import logging
import numpy as np

from audio_io import read_audio, write_audio

from noise_reduction import (
    spectral_subtraction,
    wiener_filter,
    adaptive_noise_reduction
)

from noise_analysis import (
    estimate_noise_profile
)

from real_time import (
    RealTimeNoiseReducer
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


# ============================================================
# AUDIO INFORMATION
# ============================================================

def print_audio_info(
    sample_rate,
    audio_data,
    file_path
):

    duration = (
        len(audio_data) /
        sample_rate
    )

    peak = np.max(
        np.abs(audio_data)
    )

    rms = np.sqrt(
        np.mean(
            audio_data ** 2
        )
    )

    dynamic_range = (
        20 *
        np.log10(
            peak /
            (rms + 1e-10)
        )
    )

    print(
        "\n" + "=" * 60
    )

    print(
        f"SES DOSYASI ANALİZİ: "
        f"{os.path.basename(file_path)}"
    )

    print(
        "=" * 60
    )

    print(
        f"Dosya Yolu: {file_path}"
    )

    print(
        f"Örnekleme Hızı: "
        f"{sample_rate:,} Hz"
    )

    print(
        f"Süre: "
        f"{duration:.2f} saniye "
        f"({len(audio_data):,} örnek)"
    )

    print(
        f"Maksimum Genlik: "
        f"{peak:.4f}"
    )

    print(
        f"RMS Seviyesi: "
        f"{rms:.4f}"
    )

    print(
        f"Dinamik Aralık: "
        f"{dynamic_range:.2f} dB"
    )

    print(
        "=" * 60
    )


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(
    input_file,
    output_file,
    method='adaptive',
    real_time=False
):

    logger.info(
        f"Input: {input_file}"
    )

    logger.info(
        f"Output: {output_file}"
    )

    logger.info(
        f"Method: {method}"
    )

    print(
        "\n🎵 GÜRÜLTÜ AZALTMA SİSTEMİ"
    )

    print(
        "=" * 60
    )

    print(
        f"📁 Giriş: {input_file}"
    )

    print(
        f"📁 Çıkış: {output_file}"
    )

    print(
        f"⚙️ Yöntem: {method.upper()}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "\n🔍 AŞAMA 1: SES DOSYASI YÜKLEME"
    )

    sample_rate, audio_data = (
        read_audio(
            input_file
        )
    )

    print(
        "✅ Ses dosyası başarıyla yüklendi"
    )

    print_audio_info(
        sample_rate,
        audio_data,
        input_file
    )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    print(
        "\n🔧 AŞAMA 2: GÜRÜLTÜ AZALTMA"
    )

    processing_start = (
        time.time()
    )

    if real_time:

        print(
            "🕒 Gerçek zamanlı simülasyon..."
        )

        reducer = (
            RealTimeNoiseReducer(
                sample_rate=sample_rate
            )
        )

        chunk_size = 1024

        output = np.zeros_like(
            audio_data
        )

        for start_idx in range(
            0,
            len(audio_data),
            chunk_size
        ):

            end_idx = min(
                start_idx +
                chunk_size,
                len(audio_data)
            )

            chunk = audio_data[
                start_idx:end_idx
            ]

            processed = (
                reducer.process_chunk(
                    chunk
                )
            )

            output[
                start_idx:end_idx
            ] = processed[
                :end_idx - start_idx
            ]

    else:

        if method == 'adaptive':

            output = (
                adaptive_noise_reduction(
                    audio_data
                )
            )

        elif method == 'wiener':

            print(
                "   - Noise profile "
                "oluşturuluyor..."
            )

            profile = (
                estimate_noise_profile(
                    audio_data
                )
            )

            output = (
                wiener_filter(
                    audio_data,
                    profile
                )
            )

        elif method == 'spectral':

            print(
                "   - Noise profile "
                "oluşturuluyor..."
            )

            profile = (
                estimate_noise_profile(
                    audio_data
                )
            )

            output = (
                spectral_subtraction(
                    audio_data,
                    profile
                )
            )

        else:

            raise ValueError(
                f"Unknown method: "
                f"{method}"
            )

    processing_time = (
        time.time()
        -
        processing_start
    )

    # --------------------------------------------------------
    # QUALITY METRICS
    # --------------------------------------------------------

    print(
        "\n📊 AŞAMA 3: SES KALİTESİ ANALİZİ"
    )

    n = min(
        len(audio_data),
        len(output)
    )

    original = (
        audio_data[:n]
    )

    enhanced = (
        output[:n]
    )

    original_rms = np.sqrt(
        np.mean(
            original ** 2
        )
    )

    enhanced_rms = np.sqrt(
        np.mean(
            enhanced ** 2
        )
    )

    try:

        correlation = np.corrcoef(
            original,
            enhanced
        )[0, 1]

        if np.isnan(
            correlation
        ):

            correlation = 0.0

    except Exception:

        correlation = 0.0

    rms_change = (
        (
            enhanced_rms /
            (original_rms + 1e-12)
        )
        -
        1.0
    ) * 100

    residual_rms = np.sqrt(
        np.mean(
            (
                original -
                enhanced
            ) ** 2
        )
    )

    print(
        f"   - Original RMS: "
        f"{original_rms:.6f}"
    )

    print(
        f"   - Enhanced RMS: "
        f"{enhanced_rms:.6f}"
    )

    print(
        f"   - Correlation: "
        f"{correlation:.4f}"
    )

    print(
        f"   - RMS change: "
        f"{rms_change:+.2f}%"
    )

    print(
        f"   - Residual RMS: "
        f"{residual_rms:.6f}"
    )

    print(
        "   - Note: True SNR requires "
        "a clean reference recording."
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print(
        "\n💾 AŞAMA 4: ÇIKTI DOSYASI"
    )

    write_audio(
        output_file,
        sample_rate,
        output
    )

    print(
        f"✅ Kaydedildi: "
        f"{output_file}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_time = (
        time.time()
        -
        processing_start
    )

    duration = (
        len(audio_data) /
        sample_rate
    )

    real_time_factor = (
        duration /
        total_time
    )

    print(
        "\n🎉 İŞLEM TAMAMLANDI"
    )

    print(
        "=" * 60
    )

    print(
        f"Processing time: "
        f"{total_time:.2f} s"
    )

    print(
        f"Audio duration: "
        f"{duration:.2f} s"
    )

    print(
        f"Real-time factor: "
        f"{real_time_factor:.2f}x"
    )

    print(
        f"Algorithm: "
        f"{method.upper()}"
    )

    print(
        "Status: SUCCESS"
    )

    print(
        "=" * 60
    )

    return total_time


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Adaptive Audio Noise Reduction"
    )

    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='Input audio file'
    )

    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='Output audio file'
    )

    parser.add_argument(
        '-m',
        '--method',
        choices=[
            'adaptive',
            'wiener',
            'spectral'
        ],
        default='adaptive',
        help='Noise reduction algorithm'
    )

    parser.add_argument(
        '-r',
        '--real-time',
        action='store_true',
        help='Enable real-time simulation'
    )

    args = parser.parse_args()

    if not os.path.exists(
        args.input
    ):

        print(
            f"❌ Input not found: "
            f"{args.input}"
        )

        return

    process_file(
        args.input,
        args.output,
        args.method,
        args.real_time
    )


if __name__ == "__main__":
    main()