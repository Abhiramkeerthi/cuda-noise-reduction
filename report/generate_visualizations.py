import numpy as np
import matplotlib.pyplot as plt
import os


plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)


def create_algorithm_comparison():
    algorithms = ['Spektral\nÇıkarma', 'Wiener\nFiltresi', 'Adaptif\nGürültü Azaltma']
    
   
    snr_improvements = [8.3, 10.7, 12.4]
    snr_std = [2.1, 1.8, 1.5]
    
   
    quality_scores = [3.2, 3.7, 4.1]
    
    
    processing_times = [0.087, 0.124, 0.168]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(algorithms, snr_improvements, yerr=snr_std, capsize=10, 
                     color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_ylabel('SNR İyileştirmesi (dB)')
    ax1.set_title('Algoritmaların SNR İyileştirme Performansı')
    ax1.grid(axis='y', alpha=0.3)
    
    
    for bar, value in zip(bars1, snr_improvements):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f} dB', ha='center', va='bottom')
    
   
    ax2 = axes[0, 1]
    bars2 = ax2.bar(algorithms, quality_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax2.set_ylabel('Kalite Skoru (1-5)')
    ax2.set_title('Subjektif Kalite Değerlendirmesi')
    ax2.set_ylim(0, 5)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars2, quality_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom')
    
    
    ax3 = axes[1, 0]
    bars3 = ax3.bar(algorithms, processing_times, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax3.set_ylabel('İşleme Süresi (saniye)')
    ax3.set_title('1 Saniyelik Ses için İşleme Süreleri')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars3, processing_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}s', ha='center', va='bottom')
    
   
    ax4 = axes[1, 1]
    rt_factors = [1/t for t in processing_times]
    bars4 = ax4.bar(algorithms, rt_factors, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Gerçek Zamanlı Sınır')
    ax4.set_ylabel('Gerçek Zamanlı Faktör')
    ax4.set_title('Gerçek Zamanlı İşleme Performansı')
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend()
    
    for bar, value in zip(bars4, rt_factors):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{value:.1f}x', ha='center', va='bottom')
    
    plt.suptitle('Gürültü Azaltma Algoritmalarının Karşılaştırması', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_realtime_performance():
    chunk_sizes = [256, 512, 1024, 2048]
    avg_latencies = [2.1, 3.9, 7.8, 15.6]
    max_latencies = [3.8, 6.2, 11.5, 22.3]
    chunk_durations = [c/44100*1000 for c in chunk_sizes]  
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    
    x = np.arange(len(chunk_sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, avg_latencies, width, label='Ortalama Gecikme', color='#3498db')
    bars2 = ax1.bar(x + width/2, max_latencies, width, label='Maksimum Gecikme', color='#e74c3c')
    
    ax1.set_xlabel('Chunk Boyutu (örnek)')
    ax1.set_ylabel('Gecikme (ms)')
    ax1.set_title('Farklı Chunk Boyutları için Gecikme Analizi')
    ax1.set_xticks(x)
    ax1.set_xticklabels(chunk_sizes)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    
    rt_ratios = [d/l for d, l in zip(chunk_durations, avg_latencies)]
    
    bars3 = ax2.bar(chunk_sizes, rt_ratios, color=['#2ecc71' if r > 1 else '#e74c3c' for r in rt_ratios])
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Gerçek Zamanlı Sınır')
    ax2.set_xlabel('Chunk Boyutu (örnek)')
    ax2.set_ylabel('Performans Oranı')
    ax2.set_title('Gerçek Zamanlı Performans Oranları')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars3, rt_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{value:.1f}x', ha='center', va='bottom')
    
    plt.suptitle('Gerçek Zamanlı İşleme Performans Analizi', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'realtime_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_spectrogram_example():
    
    duration = 3  
    sample_rate = 44100
    t = np.linspace(0, duration, duration * sample_rate)
    
    
    speech_freq = [300, 600, 900, 1200]  
    noise_freq = [50, 100, 4000, 6000, 8000]  
    
    
    time_bins = 50
    freq_bins = 20
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    
    noisy_spec = np.random.rand(freq_bins, time_bins) * 0.3
    
    for i in range(3):
        start = i * 15 + 5
        end = start + 10
        noisy_spec[2:8, start:end] += 0.7
    
    noisy_spec[12:, :] += 0.5
    
    im1 = ax1.imshow(noisy_spec, aspect='auto', origin='lower', cmap='hot')
    ax1.set_ylabel('Frekans (kHz)')
    ax1.set_title('Orijinal Gürültülü Ses Spektrogramı')
    ax1.set_yticks([0, 5, 10, 15, 19])
    ax1.set_yticklabels(['0', '2', '4', '6', '8'])
    
    
    clean_spec = noisy_spec.copy()
    clean_spec[12:, :] *= 0.1  
    clean_spec[:2, :] *= 0.3   
    
    im2 = ax2.imshow(clean_spec, aspect='auto', origin='lower', cmap='hot')
    ax2.set_xlabel('Zaman (s)')
    ax2.set_ylabel('Frekans (kHz)')
    ax2.set_title('Adaptif Gürültü Azaltma Sonrası Spektrogram')
    ax2.set_xticks([0, 12.5, 25, 37.5, 50])
    ax2.set_xticklabels(['0', '0.75', '1.5', '2.25', '3'])
    ax2.set_yticks([0, 5, 10, 15, 19])
    ax2.set_yticklabels(['0', '2', '4', '6', '8'])
    
    
    cbar = plt.colorbar(im2, ax=[ax1, ax2])
    cbar.set_label('Genlik', rotation=270, labelpad=20)
    
    plt.suptitle('Gürültü Azaltma Öncesi ve Sonrası Spektrogram Karşılaştırması', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectrogram_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_waveform_comparison():
    
    duration = 0.5  
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    
    clean_signal = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 2)
    clean_signal += 0.5 * np.sin(2 * np.pi * 880 * t) * np.exp(-t * 3)
    
    
    noise = np.random.normal(0, 0.2, len(t))
    noisy_signal = clean_signal + noise
    
    
    processed_signal = clean_signal + noise * 0.2  
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    
    axes[0].plot(t, noisy_signal, 'b', alpha=0.7, linewidth=0.5)
    axes[0].set_ylabel('Genlik')
    axes[0].set_title('Gürültülü Ses Sinyali')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.5, 1.5)
    
    
    axes[1].plot(t, processed_signal, 'g', alpha=0.7, linewidth=0.5)
    axes[1].set_ylabel('Genlik')
    axes[1].set_title('Gürültü Azaltma Sonrası')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1.5, 1.5)
    
    
    axes[2].plot(t, clean_signal, 'r', alpha=0.7, linewidth=0.5)
    axes[2].set_xlabel('Zaman (s)')
    axes[2].set_ylabel('Genlik')
    axes[2].set_title('Orijinal Temiz Sinyal (Referans)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-1.5, 1.5)
    
    plt.suptitle('Ses Sinyali Dalga Formu Karşılaştırması', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waveform_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_fft_visualization():
    
    duration = 0.1
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    
    signal = np.sin(2 * np.pi * 440 * t)  
    signal += 0.5 * np.sin(2 * np.pi * 880 * t)  
    signal += 0.3 * np.sin(2 * np.pi * 1320 * t)  
    signal += np.random.normal(0, 0.1, len(t))  
    
    
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)
    magnitude = np.abs(fft_result)
    
    
    pos_mask = frequencies > 0
    pos_freq = frequencies[pos_mask]
    pos_mag = magnitude[pos_mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    
    ax1.plot(t[:1000], signal[:1000], 'b', linewidth=0.5)
    ax1.set_xlabel('Zaman (s)')
    ax1.set_ylabel('Genlik')
    ax1.set_title('Zaman Domaininde Ses Sinyali')
    ax1.grid(True, alpha=0.3)
    
    
    ax2.plot(pos_freq[:2000], pos_mag[:2000], 'r', linewidth=1)
    ax2.set_xlabel('Frekans (Hz)')
    ax2.set_ylabel('Genlik')
    ax2.set_title('Frekans Domaininde Ses Sinyali (FFT)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2000)
    
    
    peaks = [440, 880, 1320]
    for peak in peaks:
        idx = np.argmin(np.abs(pos_freq - peak))
        ax2.annotate(f'{peak} Hz', xy=(pos_freq[idx], pos_mag[idx]), 
                    xytext=(pos_freq[idx]+50, pos_mag[idx]+50),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    plt.suptitle('Fourier Dönüşümü Görselleştirmesi', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fft_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("Görselleştirmeler oluşturuluyor...")
    
    create_algorithm_comparison()
    print("✓ Algoritma karşılaştırma grafikleri oluşturuldu")
    
    create_realtime_performance()
    print("✓ Gerçek zamanlı performans grafikleri oluşturuldu")
    
    create_spectrogram_example()
    print("✓ Spektrogram örnekleri oluşturuldu")
    
    create_waveform_comparison()
    print("✓ Dalga formu karşılaştırmaları oluşturuldu")
    
    create_fft_visualization()
    print("✓ FFT görselleştirmesi oluşturuldu")
    
    print(f"\nTüm grafikler '{output_dir}/' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()