import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_io import read_audio, write_audio
from noise_reduction import spectral_subtraction, wiener_filter, adaptive_noise_reduction
from real_time import RealTimeNoiseReducer

class PerformanceEvaluator:
    
    def __init__(self):
        self.results = {}
        
    def calculate_snr(self, clean_signal, noisy_signal):
        min_length = min(len(clean_signal), len(noisy_signal))
        clean_trimmed = clean_signal[:min_length]
        noisy_trimmed = noisy_signal[:min_length]
        
        signal_power = np.mean(clean_trimmed**2)
        noise_power = np.mean((noisy_trimmed - clean_trimmed)**2)
        
        if noise_power == 0 or signal_power == 0:
            return float('inf')
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def calculate_pesq_estimate(self, original, enhanced):
        min_length = min(len(original), len(enhanced))
        original_trimmed = original[:min_length]
        enhanced_trimmed = enhanced[:min_length]
        
        correlation = np.corrcoef(original_trimmed, enhanced_trimmed)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        mse = np.mean((original_trimmed - enhanced_trimmed)**2)
        
        quality_score = 1 + 4 * np.clip(correlation, 0, 1) * np.exp(-mse * 10)
        return quality_score
    
    def measure_processing_time(self, algorithm_func, audio_data, *args):
        start_time = time.time()
        result = algorithm_func(audio_data, *args)
        end_time = time.time()
        
        processing_time = end_time - start_time
        return result, processing_time
    
    def evaluate_algorithm(self, algorithm_name, algorithm_func, test_files, *args):
        print(f"\n{algorithm_name} algoritması değerlendiriliyor...")
        
        algorithm_results = {
            'snr_improvements': [],
            'quality_scores': [],
            'processing_times': [],
            'file_names': []
        }
        
        for test_file in test_files:
            if not os.path.exists(test_file):
                print(f"Uyarı: {test_file} dosyası bulunamadı.")
                continue
                
            print(f"İşleniyor: {os.path.basename(test_file)}")
            
            sample_rate, audio_data = read_audio(test_file)
            
            enhanced_audio, processing_time = self.measure_processing_time(
                algorithm_func, audio_data, *args
            )
            
            min_length = min(len(audio_data), len(enhanced_audio))
            audio_trimmed = audio_data[:min_length]
            enhanced_trimmed = enhanced_audio[:min_length]
            
            baseline_noise = np.random.normal(0, 0.01, len(audio_trimmed))
            snr_original = self.calculate_snr(audio_trimmed, audio_trimmed + baseline_noise)
            
            snr_enhanced = self.calculate_snr(enhanced_trimmed, audio_trimmed)
            
            if np.isfinite(snr_original) and np.isfinite(snr_enhanced):
                snr_improvement = snr_enhanced - snr_original
            else:
                original_rms = np.sqrt(np.mean(audio_trimmed**2))
                enhanced_rms = np.sqrt(np.mean(enhanced_trimmed**2))
                residual_rms = np.sqrt(np.mean((audio_trimmed - enhanced_trimmed)**2))
                
                if residual_rms > 0:
                    snr_improvement = 20 * np.log10(original_rms / residual_rms)
                else:
                    snr_improvement = 20.0
            
            quality_score = self.calculate_pesq_estimate(audio_trimmed, enhanced_trimmed)
            
            algorithm_results['snr_improvements'].append(snr_improvement)
            algorithm_results['quality_scores'].append(quality_score)
            algorithm_results['processing_times'].append(processing_time)
            algorithm_results['file_names'].append(os.path.basename(test_file))
            
            print(f"  SNR iyileştirmesi: {snr_improvement:.2f} dB")
            print(f"  Kalite skoru: {quality_score:.2f}")
            print(f"  İşleme süresi: {processing_time:.3f} saniye")
        
        self.results[algorithm_name] = algorithm_results
        return algorithm_results
    
    def test_real_time_performance(self, test_file, chunk_size=1024):
        print("\nGerçek zamanlı performans testi...")
        
        sample_rate, audio_data = read_audio(test_file)
        
        rt_processor = RealTimeNoiseReducer(sample_rate=sample_rate)
        
        num_chunks = len(audio_data) // chunk_size
        processing_times = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = audio_data[start_idx:end_idx]
            
            start_time = time.time()
            processed_chunk = rt_processor.process_chunk(chunk)
            end_time = time.time()
            
            chunk_time = end_time - start_time
            processing_times.append(chunk_time)
        
        avg_processing_time = np.mean(processing_times)
        real_time_factor = (chunk_size / sample_rate) / avg_processing_time
        
        print(f"Ortalama chunk işleme süresi: {avg_processing_time:.4f} saniye")
        print(f"Gerçek zamanlı faktör: {real_time_factor:.2f}x")
        print(f"Gerçek zamanlı uygunluk: {'EVET' if real_time_factor >= 1.0 else 'HAYIR'}")
        
        return {
            'avg_processing_time': avg_processing_time,
            'real_time_factor': real_time_factor,
            'is_real_time_capable': real_time_factor >= 1.0
        }
    
    def generate_comparison_plots(self, output_dir='report/figures'):
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        algorithms = list(self.results.keys())
        snr_data = [self.results[alg]['snr_improvements'] for alg in algorithms]
        
        plt.subplot(2, 2, 1)
        plt.boxplot(snr_data, labels=algorithms)
        plt.title('SNR İyileştirme Karşılaştırması')
        plt.ylabel('SNR İyileştirmesi (dB)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        quality_data = [self.results[alg]['quality_scores'] for alg in algorithms]
        plt.boxplot(quality_data, labels=algorithms)
        plt.title('Kalite Skoru Karşılaştırması')
        plt.ylabel('Kalite Skoru (1-5)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        time_data = [self.results[alg]['processing_times'] for alg in algorithms]
        plt.boxplot(time_data, labels=algorithms)
        plt.title('İşleme Süresi Karşılaştırması')
        plt.ylabel('İşleme Süresi (saniye)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        table_data = []
        for alg in algorithms:
            avg_snr = np.mean(self.results[alg]['snr_improvements'])
            avg_quality = np.mean(self.results[alg]['quality_scores'])
            avg_time = np.mean(self.results[alg]['processing_times'])
            table_data.append([alg, f"{avg_snr:.2f}", f"{avg_quality:.2f}", f"{avg_time:.3f}"])
        
        table = plt.table(cellText=table_data,
                         colLabels=['Algoritma', 'Ort. SNR (dB)', 'Ort. Kalite', 'Ort. Süre (s)'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.title('Ortalama Performans Metrikleri')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Karşılaştırma grafikleri {output_dir}/algorithm_comparison.png dosyasına kaydedildi.")
    
    def save_results_report(self, output_file='report/performance_report.txt'):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("GÜRÜLTÜ AZALTMA SİSTEMİ PERFORMANS RAPORU\n")
            f.write("=" * 50 + "\n\n")
            
            for algorithm, results in self.results.items():
                f.write(f"{algorithm.upper()} ALGORİTMASI SONUÇLARI\n")
                f.write("-" * 30 + "\n")
                
                f.write(f"Ortalama SNR İyileştirmesi: {np.mean(results['snr_improvements']):.2f} dB\n")
                f.write(f"Ortalama Kalite Skoru: {np.mean(results['quality_scores']):.2f}\n")
                f.write(f"Ortalama İşleme Süresi: {np.mean(results['processing_times']):.3f} saniye\n")
                f.write(f"Test Edilen Dosya Sayısı: {len(results['file_names'])}\n\n")
                
                f.write("Detaylı Sonuçlar:\n")
                for i, filename in enumerate(results['file_names']):
                    f.write(f"  {filename}: SNR={results['snr_improvements'][i]:.2f}dB, "
                           f"Kalite={results['quality_scores'][i]:.2f}, "
                           f"Süre={results['processing_times'][i]:.3f}s\n")
                f.write("\n")
        
        print(f"Performans raporu {output_file} dosyasına kaydedildi.")


def main():
    test_files = [
        'test_recordings/car.wav',
        'test_recordings/car1.wav', 
        'test_recordings/h_orig.wav',
        'test_recordings/nsa_st.wav',
        'test_recordings/nsa_wbn.wav'
    ]
    
    evaluator = PerformanceEvaluator()
    
    print("Algoritma performans testleri başlatılıyor...")
    
    evaluator.evaluate_algorithm(
        'Spektral Çıkarma',
        lambda audio, noise_prof: spectral_subtraction(audio, noise_prof),
        test_files,
        0.01
    )
    
    evaluator.evaluate_algorithm(
        'Wiener Filtre',
        lambda audio, noise_prof: wiener_filter(audio, noise_prof),
        test_files,
        0.01
    )
    
    evaluator.evaluate_algorithm(
        'Uyarlamalı Gürültü Azaltma',
        adaptive_noise_reduction,
        test_files
    )
    
    if test_files and os.path.exists(test_files[0]):
        rt_results = evaluator.test_real_time_performance(test_files[0])
    
    evaluator.generate_comparison_plots()
    evaluator.save_results_report()
    
    print("\nPerformans testleri tamamlandı.")
    print("Sonuçlar report/ klasöründe bulunabilir.")


if __name__ == "__main__":
    main()