import numpy as np
import os
import sys
import time
import threading
import queue
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_io import read_audio, write_audio
from real_time import RealTimeNoiseReducer

class RealTimeTestSuite:
    """Gerçek zamanlı gürültü azaltma özelliğini test eden kapsamlı test paketi."""
    
    def __init__(self):
        self.test_results = {}
        
    def test_latency_performance(self, audio_file, sample_rate=44100, chunk_sizes=[256, 512, 1024, 2048]):
        """Farklı chunk boyutları için gecikme performansını test eder."""
        print("Gecikme performans testi başlatılıyor...")
        
        sample_rate, audio_data = read_audio(audio_file)
        latency_results = {}
        
        for chunk_size in chunk_sizes:
            print(f"  Chunk boyutu {chunk_size} test ediliyor...")
            
            processor = RealTimeNoiseReducer(sample_rate=sample_rate, frame_size=chunk_size)
            latencies = []
            

            num_test_chunks = min(100, len(audio_data) // chunk_size)
            
            for i in range(num_test_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                if end_idx > len(audio_data):
                    break
                    
                chunk = audio_data[start_idx:end_idx]
                
                start_time = time.perf_counter()
                processed_chunk = processor.process_chunk(chunk)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            
            avg_latency = np.mean(latencies)
            max_latency = np.max(latencies)
            chunk_duration_ms = (chunk_size / sample_rate) * 1000
            real_time_ratio = chunk_duration_ms / avg_latency
            
            latency_results[chunk_size] = {
                'avg_latency_ms': avg_latency,
                'max_latency_ms': max_latency,
                'chunk_duration_ms': chunk_duration_ms,
                'real_time_ratio': real_time_ratio,
                'is_real_time': real_time_ratio >= 1.0
            }
            
            print(f"    Ortalama gecikme: {avg_latency:.2f} ms")
            print(f"    Maksimum gecikme: {max_latency:.2f} ms")
            print(f"    Chunk süresi: {chunk_duration_ms:.2f} ms")
            print(f"    Gerçek zamanlı oran: {real_time_ratio:.2f}x")
            print(f"    Gerçek zamanlı uygunluk: {'EVET' if real_time_ratio >= 1.0 else 'HAYIR'}")
            
        self.test_results['latency'] = latency_results
        return latency_results
    
    def test_streaming_simulation(self, audio_file, chunk_size=1024, duration_seconds=10):
        """Sürekli akış simülasyonu testi gerçekleştirir."""
        print(f"\nSürekli akış simülasyonu testi ({duration_seconds} saniye)...")
        
        sample_rate, audio_data = read_audio(audio_file)
        

        required_samples = int(duration_seconds * sample_rate)
        if len(audio_data) < required_samples:

            repeats = (required_samples // len(audio_data)) + 1
            audio_data = np.tile(audio_data, repeats)[:required_samples]
        else:
            audio_data = audio_data[:required_samples]
        
        processor = RealTimeNoiseReducer(sample_rate=sample_rate, frame_size=chunk_size)
        

        input_queue = queue.Queue()
        output_queue = queue.Queue()
        

        processing_times = []
        queue_sizes = []
        dropped_chunks = 0
        
        def audio_producer():
            """Ses verisi üreten thread."""
            chunk_duration = chunk_size / sample_rate
            num_chunks = len(audio_data) // chunk_size
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunk = audio_data[start_idx:end_idx]
                
                input_queue.put(chunk)
                time.sleep(chunk_duration)  
        
        def audio_processor():
            """Ses verisi işleyen thread."""
            nonlocal dropped_chunks
            
            while True:
                try:
                    chunk = input_queue.get(timeout=1.0)
                    queue_sizes.append(input_queue.qsize())
                    
                    start_time = time.perf_counter()
                    processed_chunk = processor.process_chunk(chunk)
                    end_time = time.perf_counter()
                    
                    processing_time = (end_time - start_time) * 1000
                    processing_times.append(processing_time)
                    
                    output_queue.put(processed_chunk)
                    
                    
                    if input_queue.qsize() > 10:
                        dropped_chunks += 1
                        input_queue.get_nowait()
                    
                except queue.Empty:
                    break
        
        
        producer_thread = threading.Thread(target=audio_producer)
        processor_thread = threading.Thread(target=audio_processor)
        
        start_time = time.time()
        producer_thread.start()
        processor_thread.start()
        
        producer_thread.join()
        processor_thread.join()
        end_time = time.time()
        
        
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        avg_queue_size = np.mean(queue_sizes)
        total_duration = end_time - start_time
        
        streaming_results = {
            'total_duration': total_duration,
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'avg_queue_size': avg_queue_size,
            'dropped_chunks': dropped_chunks,
            'total_chunks': len(processing_times),
            'success_rate': (len(processing_times) - dropped_chunks) / len(processing_times) * 100
        }
        
        print(f"  Toplam süre: {total_duration:.2f} saniye")
        print(f"  Ortalama işleme süresi: {avg_processing_time:.2f} ms")
        print(f"  Maksimum işleme süresi: {max_processing_time:.2f} ms")
        print(f"  Ortalama queue boyutu: {avg_queue_size:.1f}")
        print(f"  Düşürülen chunk sayısı: {dropped_chunks}")
        print(f"  Başarı oranı: {streaming_results['success_rate']:.1f}%")
        
        self.test_results['streaming'] = streaming_results
        return streaming_results
    
    def test_noise_adaptation(self, audio_file, add_artificial_noise=True):
        """Gürültü profiline uyum performansını test eder."""
        print("\nGürültü uyum performans testi...")
        
        sample_rate, audio_data = read_audio(audio_file)
        
        if add_artificial_noise:
            
            noise = np.random.normal(0, 0.1, len(audio_data))
            noisy_audio = audio_data + noise
        else:
            noisy_audio = audio_data
        
        processor = RealTimeNoiseReducer(sample_rate=sample_rate)
        
        chunk_size = 1024
        num_chunks = len(noisy_audio) // chunk_size
        
        
        first_chunks = []
        last_chunks = []
        
        for i in range(min(10, num_chunks)):  
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = noisy_audio[start_idx:end_idx]
            processed = processor.process_chunk(chunk)
            first_chunks.append(processed)
        
        
        for i in range(10, num_chunks - 10):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = noisy_audio[start_idx:end_idx]
            processor.process_chunk(chunk)
        
        for i in range(num_chunks - 10, num_chunks):  
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            if end_idx > len(noisy_audio):
                break
            chunk = noisy_audio[start_idx:end_idx]
            processed = processor.process_chunk(chunk)
            last_chunks.append(processed)
        
        
        first_snr = self._calculate_snr(np.concatenate(first_chunks), 
                                       noisy_audio[:len(first_chunks)*chunk_size])
        last_snr = self._calculate_snr(np.concatenate(last_chunks),
                                      noisy_audio[-len(last_chunks)*chunk_size:])
        
        adaptation_improvement = last_snr - first_snr
        
        adaptation_results = {
            'first_chunks_snr': first_snr,
            'last_chunks_snr': last_snr,
            'adaptation_improvement_db': adaptation_improvement,
            'adaptation_success': adaptation_improvement > 0.5
        }
        
        print(f"  İlk chunk'ların SNR'si: {first_snr:.2f} dB")
        print(f"  Son chunk'ların SNR'si: {last_snr:.2f} dB")
        print(f"  Uyum iyileştirmesi: {adaptation_improvement:.2f} dB")
        print(f"  Uyum başarısı: {'EVET' if adaptation_improvement > 0.5 else 'HAYIR'}")
        
        self.test_results['adaptation'] = adaptation_results
        return adaptation_results
    
    def _calculate_snr(self, signal, reference):
        """SNR hesaplama yardımcı fonksiyonu."""
        signal_power = np.mean(signal**2)
        noise_power = np.mean((signal - reference[:len(signal)])**2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    def generate_realtime_report(self, output_dir='report'):
        """Gerçek zamanlı test sonuçlarını rapor halinde oluşturur."""
        os.makedirs(output_dir, exist_ok=True)
        
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        
        if 'latency' in self.test_results:
            latency_data = self.test_results['latency']
            chunk_sizes = list(latency_data.keys())
            latencies = [latency_data[cs]['avg_latency_ms'] for cs in chunk_sizes]
            ratios = [latency_data[cs]['real_time_ratio'] for cs in chunk_sizes]
            
            axes[0, 0].bar(range(len(chunk_sizes)), latencies)
            axes[0, 0].set_title('Ortalama Gecikme Süreleri')
            axes[0, 0].set_xlabel('Chunk Boyutu')
            axes[0, 0].set_ylabel('Gecikme (ms)')
            axes[0, 0].set_xticks(range(len(chunk_sizes)))
            axes[0, 0].set_xticklabels(chunk_sizes)
            
            axes[0, 1].bar(range(len(chunk_sizes)), ratios)
            axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Gerçek Zamanlı Sınır')
            axes[0, 1].set_title('Gerçek Zamanlı Performans Oranları')
            axes[0, 1].set_xlabel('Chunk Boyutu')
            axes[0, 1].set_ylabel('Performans Oranı')
            axes[0, 1].set_xticks(range(len(chunk_sizes)))
            axes[0, 1].set_xticklabels(chunk_sizes)
            axes[0, 1].legend()
        
        
        if 'streaming' in self.test_results:
            streaming_data = self.test_results['streaming']
            metrics = ['Başarı Oranı (%)', 'Ortalama İşleme (ms)', 'Düşürülen Chunk']
            values = [streaming_data['success_rate'], 
                     streaming_data['avg_processing_time_ms'],
                     streaming_data['dropped_chunks']]
            
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_title('Sürekli Akış Performansı')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        
        if 'adaptation' in self.test_results:
            adaptation_data = self.test_results['adaptation']
            
            categories = ['İlk Chunks', 'Son Chunks']
            snr_values = [adaptation_data['first_chunks_snr'], 
                         adaptation_data['last_chunks_snr']]
            
            axes[1, 1].bar(categories, snr_values)
            axes[1, 1].set_title('Gürültü Uyum Performansı')
            axes[1, 1].set_ylabel('SNR (dB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'realtime_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        
        report_path = os.path.join(output_dir, 'realtime_test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("GERÇEK ZAMANLI GÜRÜLTÜ AZALTMA TEST RAPORU\n")
            f.write("=" * 50 + "\n\n")
            
            if 'latency' in self.test_results:
                f.write("GECİKME PERFORMANS TESTİ\n")
                f.write("-" * 25 + "\n")
                for chunk_size, data in self.test_results['latency'].items():
                    f.write(f"Chunk Boyutu {chunk_size}:\n")
                    f.write(f"  Ortalama Gecikme: {data['avg_latency_ms']:.2f} ms\n")
                    f.write(f"  Gerçek Zamanlı Oran: {data['real_time_ratio']:.2f}x\n")
                    f.write(f"  Gerçek Zamanlı Uygunluk: {'EVET' if data['is_real_time'] else 'HAYIR'}\n\n")
            
            if 'streaming' in self.test_results:
                f.write("SÜREKLİ AKIŞ TESTİ\n")
                f.write("-" * 17 + "\n")
                data = self.test_results['streaming']
                f.write(f"Başarı Oranı: {data['success_rate']:.1f}%\n")
                f.write(f"Ortalama İşleme Süresi: {data['avg_processing_time_ms']:.2f} ms\n")
                f.write(f"Düşürülen Chunk Sayısı: {data['dropped_chunks']}\n\n")
            
            if 'adaptation' in self.test_results:
                f.write("GÜRÜLTÜ UYUM TESTİ\n")
                f.write("-" * 17 + "\n")
                data = self.test_results['adaptation']
                f.write(f"Uyum İyileştirmesi: {data['adaptation_improvement_db']:.2f} dB\n")
                f.write(f"Uyum Başarısı: {'EVET' if data['adaptation_success'] else 'HAYIR'}\n")
        
        print(f"Gerçek zamanlı test raporu {report_path} dosyasına kaydedildi.")


def main():
    """Ana test fonksiyonu."""
    
    test_file = 'test_recordings/car.wav'
    
    if not os.path.exists(test_file):
        print(f"Hata: Test dosyası {test_file} bulunamadı.")
        print("Lütfen test_recordings/ klasöründe ses dosyalarının bulunduğundan emin olun.")
        return
    
    
    test_suite = RealTimeTestSuite()
    
    print("Gerçek zamanlı gürültü azaltma sistemi test ediliyor...\n")
    
    
    test_suite.test_latency_performance(test_file)
    test_suite.test_streaming_simulation(test_file, duration_seconds=5)
    test_suite.test_noise_adaptation(test_file)
    
    
    test_suite.generate_realtime_report()
    
    print("\nTüm gerçek zamanlı testler tamamlandı.")
    print("Sonuçlar report/ klasöründe bulunabilir.")


if __name__ == "__main__":
    main()