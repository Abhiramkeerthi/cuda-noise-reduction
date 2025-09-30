Gürültülü Ortamlarda Ses Temizleme Sistemi
==========================================

Bu proje, gürültülü ortamlarda kaydedilen sesleri temizlemeye yönelik bir sistem sunmaktadır. 
Fourier Dönüşümü kullanarak, farklı ortamlarda kaydedilen seslerdeki gürültüyü analiz eder, 
ayrıştırır ve temizler. Hem teorik bilgi hem de pratik uygulama içerir.

Özellikler:
-----------
- Çeşitli ses formatlarını destekler (.wav, .mp3, .m4a)
NOT: Veri kaybına bağlı ses işleme performansının düşmemesi için .wav tipindeki dosyalar kullanmanızı öneririm.
- Spektral Çıkarma, Wiener Filtresi ve Adaptif Gürültü Temizleme yöntemleri
- Gerçek zamanlı gürültü temizleme simülasyonu
- Sıfırdan uygulanmış FFT algoritması
- Kapsamlı analiz ve raporlama araçları

SİSTEM GEREKSİNİMLERİ:
=====================
- Python 3.7 veya daha yeni
- İşletim Sistemi: Windows, macOS veya Linux
- En az 4GB RAM
- Ses kayıtları için mikrofonlu bir bilgisayar (opsiyonel)

HIZLI KURULUM REHBERİ:
=====================

ADIM 1: Python Kurulumu
-----------------------
1. https://www.python.org/downloads/ adresine gidin
2. Python 3.7 veya daha yeni sürümü indirin
3. İndirdiğiniz dosyayı çalıştırın
4. "Add Python to PATH" seçeneğini işaretleyin
5. "Install Now" butonuna tıklayın

ADIM 2: Proje Dosyalarının Çıkarılması
--------------------------------------
1. Proje zip dosyasını masaüstüne çıkarın
2. Çıkarılan klasörü açın

ADIM 3: Gerekli Kütüphanelerin Kurulumu
---------------------------------------
Windows için:
1. Başlat menüsüne sağ tıklayın ve "Command Prompt" veya "Komut İstemi" seçin
2. Açılan pencerede şu komutu yazın ve Enter'a basın:
   cd Desktop\noise-reduction-system
3. Ardından şu komutu yazın ve Enter'a basın:
   pip install -r requirements.txt

macOS/Linux için:
1. Terminal uygulamasını açın
2. Şu komutu yazın ve Enter'a basın:
   cd ~/Desktop/noise-reduction-system
3. Ardından şu komutu yazın ve Enter'a basın:
   pip install -r requirements.txt

PROGRAMIN KULLANIMI:
===================

TEMEL KULLANIM - Tek Dosya İşleme:
----------------------------------
Windows için:
   python src\main.py -i tests\test_recordings\car.wav -o tests\test_results\car_clean.wav -m adaptive

macOS/Linux için:
   python src/main.py -i tests/test_recordings/car.wav -o tests/test_results/car_clean.wav -m adaptive

TOPLU DOSYA İŞLEME:
-------------------
Windows için:
   python src\main.py -i tests\test_recordings -o tests\test_results -m adaptive

macOS/Linux için:
   python src/main.py -i tests/test_recordings -o tests/test_results -m adaptive

GERÇEK ZAMANLI İŞLEME:
---------------------
Windows için:
   python src\main.py -i tests\test_recordings\car.wav -o tests\test_results\car_realtime.wav -m adaptive --real-time

macOS/Linux için:
   python src/main.py -i tests/test_recordings/car.wav -o tests/test_results/car_realtime.wav -m adaptive --real-time

PARAMETRE AÇIKLAMALARI:
======================
--input veya -i  : İşlenecek ses dosyası veya klasör yolu
--output veya -o : Temizlenmiş ses dosyasının kaydedileceği yer
--method veya -m : Gürültü temizleme yöntemi
                   spectral : Spektral Çıkarma (hızlı, basit gürültüler için)
                   wiener   : Wiener Filtresi (orta seviye)
                   adaptive : Adaptif Temizleme (en iyi sonuç, varsayılan)
--real-time veya -r : Gerçek zamanlı işleme modunu etkinleştirir

HAZIR ÖRNEKLER - KOPYALA/YAPIŞTIR:
==================================

Örnek 1: Araç içi kayıt temizleme (Windows)
   python src\main.py -i tests\test_recordings\car.wav -o tests\test_results\car_clean.wav -m adaptive

Örnek 2: Tüm test dosyalarını temizleme (Windows)
   python src\main.py -i tests\test_recordings -o tests\test_results -m adaptive

Örnek 3: Hızlı temizleme (daha düşük kalite) (Windows)
   python src\main.py -i tests\test_recordings\car.wav -o tests\test_results\car_fast.wav -m spectral

Örnek 4: Gerçek zamanlı simülasyon (Windows)
   python src\main.py -i tests\test_recordings\car.wav -o tests\test_results\car_rt.wav -m adaptive -r

macOS/Linux kullanıcıları: Yukarıdaki örneklerde \ işaretlerini / ile değiştirin

KENDİ SESİNİZİ KAYDETMEK VE TEMİZLEMEK:
=======================================
1. Windows Ses Kaydedici veya başka bir program ile ses kaydedin
2. Kaydı .wav formatında kaydedin (örnek: benim_sesim.wav)
3. Dosyayı proje klasörüne kopyalayın
4. Şu komutu çalıştırın:
   Windows: python src\main.py -i benim_sesim.wav -o benim_sesim_temiz.wav -m adaptive
   macOS/Linux: python src/main.py -i benim_sesim.wav -o benim_sesim_temiz.wav -m adaptive

PERFORMANS TESTLERİ:
===================
Algoritmalar arası karşılaştırma için:
   Windows: python tests\performance_evaluation.py
   macOS/Linux: python tests/performance_evaluation.py

Gerçek zamanlı performans testi için:
   Windows: python tests\test_realtime.py
   macOS/Linux: python tests/test_realtime.py

SORUN GİDERME:
=============
1. "Python bulunamadı" hatası:
   - Python'u yeniden kurun ve PATH'e eklemeyi unutmayın
   
2. "pip bulunamadı" hatası:
   - python -m pip install -r requirements.txt komutunu deneyin
   
3. Ses dosyası okunamıyor hatası:
   - Dosya yolunun doğru olduğundan emin olun
   - .wav, .mp3 veya .m4a formatında olduğundan emin olun

4. İşlem çok uzun sürüyor:
   - spectral yöntemini kullanarak daha hızlı sonuç alabilirsiniz
   - Daha küçük ses dosyaları ile test edin

ÇIKTI DOSYALARINI DİNLEME:
=========================
Temizlenmiş ses dosyaları test_results klasöründe bulunur.
Windows Media Player, VLC veya herhangi bir ses oynatıcı ile dinleyebilirsiniz.

PROJE İÇERİĞİ:
=============
- src/             : Kaynak kodlar
  - main.py        : Ana program
  - fft.py         : Fourier dönüşümü algoritmaları
  - noise_reduction.py : Gürültü temizleme algoritmaları
  - real_time.py   : Gerçek zamanlı işleme
  - audio_io.py    : Ses dosyası okuma/yazma
  
- tests/           : Test dosyaları ve sonuçları
  - test_recordings/ : Örnek gürültülü ses kayıtları
  - test_results/    : Temizlenmiş ses dosyaları
  
- report/          : Proje raporu ve görseller
- requirements.txt : Gerekli Python kütüphaneleri

DESTEK:
======
Herhangi bir sorun yaşarsanız, proje klasöründeki report/proje_raporu.pdf 
dosyasını inceleyebilir veya hata mesajını not alarak bana ulaşabilirsiniz.
enes.uca@bil.omu.edu.tr
