import os
import numpy as np
import argparse
import time
import logging
from audio_io import read_audio, write_audio
from noise_reduction import spectral_subtraction, wiener_filter, adaptive_noise_reduction
from real_time import RealTimeNoiseReducer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_audio_info(sample_rate, audio_data, file_path):
    """Detaylı ses dosyası bilgilerini yazdırır."""
    duration = len(audio_data) / sample_rate
    max_amplitude = np.max(np.abs(audio_data))
    rms_level = np.sqrt(np.mean(audio_data**2))
    
    print(f"\n{'='*60}")
    print(f"SES DOSYASI ANALİZİ: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    print(f"Dosya Yolu: {file_path}")
    print(f"Örnekleme Hızı: {sample_rate:,} Hz")
    print(f"Süre: {duration:.2f} saniye ({len(audio_data):,} örnek)")
    print(f"Maksimum Genlik: {max_amplitude:.4f}")
    print(f"RMS Seviyesi: {rms_level:.4f}")
    print(f"Dinamik Aralık: {20 * np.log10(max_amplitude / (rms_level + 1e-10)):.2f} dB")
    print(f"Dosya Boyutu: {os.path.getsize(file_path) / 1024:.1f} KB")
    print(f"{'='*60}\n")

def print_algorithm_info(method):
    """Algoritma özel bilgilerini yazdırır."""
    algorithm_info = {
        'spectral': {
            'name': 'Spektral Çıkarma',
            'description': 'Frekans domaininde genlik spektrum modifikasyonu kullanan klasik gürültü azaltma',
            'advantages': 'Hızlı işleme, sabit gürültü için etkili',
            'parameters': 'Çıkarma faktörü: 2.0, Gürültü tabanı: 0.01'
        },
        'wiener': {
            'name': 'Wiener Filtresi',
            'description': 'İstatistiksel sinyal ve gürültü özelliklerine dayalı optimal doğrusal filtre',
            'advantages': 'Matematiksel olarak optimal, iyi SNR iyileştirmesi',
            'parameters': 'A priori SNR tahmini, frekansa bağımlı kazanç hesaplama'
        },
        'adaptive': {
            'name': 'Uyarlamalı Gürültü Azaltma',
            'description': 'Ses aktivitesi algılama ile dinamik gürültü profili uyarlaması',
            'advantages': 'Değişen gürültü koşullarına uyum sağlar, konuşma kalitesini korur',
            'parameters': 'Uyarlama oranı: 0.05, Ses eşiği: 2.5, İlk gürültü çerçeveleri: 10'
        }
    }
    
    info = algorithm_info.get(method, {})
    print(f"\n{'='*60}")
    print(f"ALGORİTMA KONFIGÜRASYONU: {info.get('name', method.upper())}")
    print(f"{'='*60}")
    print(f"Açıklama: {info.get('description', 'Özel algoritma')}")
    print(f"Avantajları: {info.get('advantages', 'Özelleştirilmiş işleme')}")
    print(f"Parametreler: {info.get('parameters', 'Varsayılan ayarlar')}")
    print(f"{'='*60}\n")

def print_processing_progress(current, total, start_time, stage="İşleniyor"):
    """İşleme ilerlemesini zaman tahmini ile yazdırır."""
    elapsed = time.time() - start_time
    progress = current / total
    if progress > 0:
        estimated_total = elapsed / progress
        remaining = estimated_total - elapsed
        print(f"\r{stage}: %{progress*100:.1f} ({current}/{total}) - "
              f"Geçen: {elapsed:.1f}s - Kalan: {remaining:.1f}s", end='', flush=True)

def process_file(input_file, output_file, method='adaptive', real_time=False):
    """Detaylı günlükleme ile tek ses dosyası gürültü azaltma işlemi."""
    logger.info(f"Gürültü azaltma işlemi başlatılıyor")
    logger.info(f"Giriş dosyası: {input_file}")
    logger.info(f"Çıkış dosyası: {output_file}")
    logger.info(f"İşleme yöntemi: {method}")
    logger.info(f"Gerçek zamanlı mod: {real_time}")
    
    print(f"\n🎵 GÜRÜLTÜ AZALTMA SİSTEMİ - İŞLEM BAŞLATILDI")
    print(f"📁 Giriş: {input_file}")
    print(f"📁 Çıkış: {output_file}")
    print(f"⚙️  Yöntem: {method.upper()}")
    print(f"⏱️  Mod: {'Gerçek Zamanlı Simülasyon' if real_time else 'Standart İşleme'}")
    
    overall_start_time = time.time()
    

    print(f"\n🔍 AŞAMA 1: SES DOSYASI YÜKLEME")
    print("-" * 40)
    read_start = time.time()
    
    try:
        sample_rate, audio_data = read_audio(input_file)
        read_time = time.time() - read_start
        print(f"✅ Ses dosyası {read_time:.3f} saniyede başarıyla yüklendi")
        logger.info(f"Ses dosyası {read_time:.3f} saniyede okundu")
        
    except Exception as e:
        print(f"❌ Ses dosyası okuma hatası: {e}")
        logger.error(f"Ses dosyası okuma başarısız: {e}")
        return None
    

    print_audio_info(sample_rate, audio_data, input_file)
    

    print(f"⚙️  AŞAMA 2: ALGORİTMA KONFIGÜRASYONU")
    print("-" * 40)
    print_algorithm_info(method)
    

    print(f"🔧 AŞAMA 3: GÜRÜLTÜ AZALTMA İŞLEMİ")
    print("-" * 40)
    processing_start = time.time()
    
    if real_time:
        print("🕒 Gerçek zamanlı işleme simülasyonu başlatılıyor...")
        logger.info("Gerçek zamanlı işleme simülasyonu başlatılıyor")
        

        reducer = RealTimeNoiseReducer(sample_rate=sample_rate)
        chunk_size = 1024
        num_chunks = len(audio_data) // chunk_size
        enhanced_audio = np.zeros_like(audio_data)
        
        print(f"📊 İşleme parametreleri:")
        print(f"   - Parça boyutu: {chunk_size} örnek ({chunk_size/sample_rate*1000:.1f} ms)")
        print(f"   - Toplam parça sayısı: {num_chunks}")
        print(f"   - Çerçeve boyutu: {reducer.frame_size}")
        print(f"   - Adım boyutu: {reducer.hop_size}")
        
        chunk_times = []
        
        for i in range(num_chunks):
            chunk_start = time.time()
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = audio_data[start_idx:end_idx]
            enhanced_chunk = reducer.process_chunk(chunk)
            enhanced_audio[start_idx:end_idx] = enhanced_chunk
            
            chunk_time = time.time() - chunk_start
            chunk_times.append(chunk_time)
            
            if i % max(1, num_chunks // 20) == 0:  
                print_processing_progress(i + 1, num_chunks, processing_start, "Gerçek Zamanlı İşleme")
        

        if len(audio_data) % chunk_size != 0:
            start_idx = num_chunks * chunk_size
            chunk = audio_data[start_idx:]
            enhanced_chunk = reducer.process_chunk(chunk)
            enhanced_audio[start_idx:start_idx+len(enhanced_chunk)] = enhanced_chunk
        
        print()  
        

        avg_chunk_time = np.mean(chunk_times)
        max_chunk_time = np.max(chunk_times)
        real_time_factor = (chunk_size / sample_rate) / avg_chunk_time
        
        print(f"\n📈 Gerçek Zamanlı Performans Analizi:")
        print(f"   - Ortalama parça işleme süresi: {avg_chunk_time*1000:.2f} ms")
        print(f"   - Maksimum parça işleme süresi: {max_chunk_time*1000:.2f} ms")
        print(f"   - Gerçek zamanlı faktör: {real_time_factor:.2f}x")
        print(f"   - Gerçek zamanlı uygunluk: {'✅ EVET' if real_time_factor >= 1.0 else '❌ HAYIR'}")
        
        logger.info(f"Gerçek zamanlı işleme: ortalama={avg_chunk_time*1000:.2f}ms, faktör={real_time_factor:.2f}x")
        
    else:
        print("🔧 Standart işleme başlatılıyor...")
        logger.info(f"{method} gürültü azaltma algoritması başlatılıyor")
        
        if method == 'spectral':
            print("🎛️  İlk çerçevelerden gürültü profili hesaplanıyor...")
            noise_frames = int(0.5 * sample_rate)
            noise_profile = np.mean(np.abs(audio_data[:noise_frames]))
            print(f"   - Gürültü tahmin süresi: 0.5 saniye ({noise_frames} örnek)")
            print(f"   - Hesaplanan gürültü seviyesi: {noise_profile:.6f}")
            
            print("🎛️  Spektral çıkarma uygulanıyor...")
            enhanced_audio = spectral_subtraction(audio_data, noise_profile)
            
        elif method == 'wiener':
            print("🎛️  Wiener filtresi için gürültü profili hesaplanıyor...")
            noise_frames = int(0.5 * sample_rate)
            noise_profile = np.mean(np.abs(audio_data[:noise_frames]))
            print(f"   - Gürültü tahmin süresi: 0.5 saniye ({noise_frames} örnek)")
            print(f"   - Hesaplanan gürültü seviyesi: {noise_profile:.6f}")
            
            print("🎛️  Wiener filtresi uygulanıyor...")
            enhanced_audio = wiener_filter(audio_data, noise_profile)
            
        elif method == 'adaptive':
            print("🎛️  Uyarlamalı gürültü azaltma uygulanıyor...")
            print("   - İlk gürültü profili tahmin ediliyor...")
            print("   - Ses aktivitesi algılama etkinleştiriliyor...")
            print("   - Uyarlamalı gürültü profili güncellemeleri konfigüre ediliyor...")
            enhanced_audio = adaptive_noise_reduction(audio_data)
            
        else:
            raise ValueError(f"Bilinmeyen yöntem: {method}")
    
    processing_time = time.time() - processing_start
    print(f"\n✅ Gürültü azaltma {processing_time:.2f} saniyede tamamlandı")
    logger.info(f"Gürültü azaltma işlemi {processing_time:.2f} saniyede tamamlandı")
    

    print(f"\n📊 AŞAMA 4: SES KALİTESİ ANALİZİ")
    print("-" * 40)
    

    min_length = min(len(audio_data), len(enhanced_audio))
    audio_data_trimmed = audio_data[:min_length]
    enhanced_audio_trimmed = enhanced_audio[:min_length]
    

    original_rms = np.sqrt(np.mean(audio_data_trimmed**2))
    enhanced_rms = np.sqrt(np.mean(enhanced_audio_trimmed**2))
    

    try:
        correlation = np.corrcoef(audio_data_trimmed, enhanced_audio_trimmed)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    

    noise_estimate = np.sqrt(np.mean((audio_data_trimmed - enhanced_audio_trimmed)**2))
    signal_power = np.mean(enhanced_audio_trimmed**2)
    if noise_estimate > 0:
        snr_estimate = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
    else:
        snr_estimate = float('inf')
    
    print(f"📈 Kalite Metrikleri:")
    print(f"   - Orijinal RMS seviyesi: {original_rms:.6f}")
    print(f"   - İyileştirilmiş RMS seviyesi: {enhanced_rms:.6f}")
    print(f"   - Sinyal korelasyonu: {correlation:.4f}")
    print(f"   - RMS değişimi: {((enhanced_rms/original_rms - 1) * 100):+.2f}%")
    print(f"   - Tahmini SNR: {snr_estimate:.2f} dB")
    print(f"   - İşlenen örnek uzunluğu: {min_length:,} (orijinal: {len(audio_data):,})")
    

    print(f"\n💾 AŞAMA 5: ÇIKTI DOSYASI OLUŞTURMA")
    print("-" * 40)
    output_start = time.time()
    
    try:
        write_audio(output_file, sample_rate, enhanced_audio)
        output_time = time.time() - output_start
        output_size = os.path.getsize(output_file) / 1024
        print(f"✅ Çıktı dosyası {output_time:.3f} saniyede başarıyla kaydedildi")
        print(f"📁 Çıktı dosyası boyutu: {output_size:.1f} KB")
        logger.info(f"Çıktı dosyası {output_time:.3f} saniyede yazıldı")
        
    except Exception as e:
        print(f"❌ Çıktı dosyası yazma hatası: {e}")
        logger.error(f"Çıktı dosyası yazma başarısız: {e}")
        return None
    

    total_time = time.time() - overall_start_time
    print(f"\n🎉 İŞLEM TAMAMLANDI - ÖZET")
    print("=" * 50)
    print(f"📊 Toplam işleme süresi: {total_time:.2f} saniye")
    print(f"⚡ İşleme hızı: {len(audio_data)/sample_rate/total_time:.2f}x gerçek zamanlı")
    print(f"🎵 Ses süresi: {len(audio_data)/sample_rate:.2f} saniye")
    print(f"💻 İşlenen örnek sayısı: {len(audio_data):,}")
    print(f"🔧 Kullanılan algoritma: {method.upper()}")
    print(f"✅ Durum: BAŞARILI")
    print("=" * 50)
    
    logger.info(f"İşlem başarıyla tamamlandı. Toplam süre: {total_time:.2f}s")
    
    return total_time

def main():
    """Gürültü azaltma sistemi ana giriş noktası."""
    print("🎵 GELİŞMİŞ GÜRÜLTÜ AZALTMA SİSTEMİ")
    print("=" * 60)
    print("🔧 Sistem parametreleri başlatılıyor...")
    
    parser = argparse.ArgumentParser(
        description='Ses Kayıtları için Gelişmiş Gürültü Azaltma Sistemi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py -i ses.wav -o temiz.wav -m adaptive
  python main.py -i giris/ -o cikis/ -m spectral
  python main.py -i ses.wav -o temiz.wav -r
        """
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='Giriş ses dosyası veya klasör yolu')
    parser.add_argument('--output', '-o', required=True, 
                       help='Çıkış dosyası veya klasör yolu')
    parser.add_argument('--method', '-m', 
                       choices=['spectral', 'wiener', 'adaptive'], 
                       default='adaptive', 
                       help='Gürültü azaltma algoritması (varsayılan: adaptive)')
    parser.add_argument('--real-time', '-r', action='store_true', 
                       help='Gerçek zamanlı işleme simülasyonunu etkinleştir')
    
    args = parser.parse_args()
    
    print(f"⚙️  Konfigürasyon yüklendi:")
    print(f"   - Giriş: {args.input}")
    print(f"   - Çıkış: {args.output}")
    print(f"   - Yöntem: {args.method}")
    print(f"   - Gerçek zamanlı: {args.real_time}")
    
    logger.info(f"Sistem parametrelerle başlatıldı: giriş={args.input}, çıkış={args.output}, yöntem={args.method}, gerçek_zamanlı={args.real_time}")
    

    if os.path.isfile(args.input):
        print(f"\n🎯 Tekli dosya işleme modu")
        process_file(args.input, args.output, args.method, args.real_time)
        
    elif os.path.isdir(args.input):
        print(f"\n🎯 Toplu işleme modu")
        print(f"📁 Klasör taranıyor: {args.input}")
        

        os.makedirs(args.output, exist_ok=True)
        

        audio_extensions = ('.wav', '.mp3', '.m4a')
        audio_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(audio_extensions)]
        
        if not audio_files:
            print(f"❌ {args.input} klasöründe ses dosyası bulunamadı")
            logger.warning(f"Giriş klasöründe ses dosyası bulunamadı: {args.input}")
            return
        
        print(f"🔍 {len(audio_files)} ses dosyası bulundu:")
        for i, filename in enumerate(audio_files, 1):
            print(f"   {i}. {filename}")
        

        total_time = 0.0
        successful_files = 0
        
        for i, filename in enumerate(audio_files, 1):
            print(f"\n{'='*80}")
            print(f"🎵 DOSYA İŞLENİYOR {i}/{len(audio_files)}: {filename}")
            print(f"{'='*80}")
            
            input_path = os.path.join(args.input, filename)
            output_path = os.path.join(args.output, filename)
            
            processing_time = process_file(input_path, output_path, 
                                         args.method, args.real_time)
            
            if processing_time is not None:
                total_time += processing_time
                successful_files += 1
                print(f"✅ Dosya {i}/{len(audio_files)} başarıyla tamamlandı")
            else:
                print(f"❌ Dosya {i}/{len(audio_files)} işlenirken hata oluştu")
        

        print(f"\n🎉 TOPLU İŞLEM TAMAMLANDI")
        print("=" * 60)
        print(f"📊 İşlenen dosyalar: {successful_files}/{len(audio_files)}")
        print(f"⏱️  Toplam işleme süresi: {total_time:.2f} saniye")
        if successful_files > 0:
            print(f"📈 Dosya başına ortalama süre: {total_time/successful_files:.2f} saniye")
        print(f"✅ Başarı oranı: %{successful_files/len(audio_files)*100:.1f}")
        print("=" * 60)
        
        logger.info(f"Toplu işlem tamamlandı: {successful_files}/{len(audio_files)} dosya başarılı")
        
    else:
        print(f"❌ Hata: Giriş yolu '{args.input}' mevcut değil.")
        logger.error(f"Giriş yolu mevcut değil: {args.input}")

if __name__ == "__main__":
    main()