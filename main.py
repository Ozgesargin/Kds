import subprocess
import time
import sys
import os

# --- Sabitler ---
# Flask uygulamasının bulunduğu dosya (Bu projede ana uygulama bu dosyada)
FLASK_API_FILE = "app/app.py"
FLASK_PORT = "5000"

# --- Uygulamayı Çalıştırma Fonksiyonu ---
def run_project():
    """
    Ana Flask uygulamasını (app.py) arka planda başlatır ve
    terminal açık kaldığı sürece çalıştırır.
    """

    print("--- Eğitmen Destek Sistemi Flask Uygulaması Başlatılıyor ---")
    api_process = None # Flask sürecini tutacak değişken

    # 1. Flask Uygulamasını Başlatma (Arka plan süreci)
    # Flask uygulamasının aynı dizinde olduğunu varsayıyoruz (FLASK_API_FILE = "app.py").
    print(f"1/1: Flask Uygulaması başlatılıyor: {FLASK_API_FILE} (Port: {FLASK_PORT})")

    # Flask dosyasının varlığını kontrol et
    if not os.path.exists(FLASK_API_FILE):
        print(f"\nHATA: Flask uygulama dosyası bulunamadı: '{FLASK_API_FILE}'")
        print("Lütfen uygulamanızın dosyasının 'app.py' adında olduğundan emin olun.")
        return

    try:
        # API sürecini başlatma. Flask'ı direkt Python betiği olarak çalıştırıyoruz.
        # Kullanılan komut: python app.py
        api_process = subprocess.Popen([sys.executable, FLASK_API_FILE])

        # Uygulamanın yüklenmesi için yeterli bir süre bekleyelim
        time.sleep(3)

        print(f"✅ Flask Uygulaması başarılı bir şekilde başlatıldı ve çalışıyor (http://127.0.0.1:{FLASK_PORT})")
        print("Uygulamayı durdurmak için lütfen terminalde CTRL+C tuşlarına basın.")

        # Uygulamanın çalışmaya devam etmesini sağlamak için bir döngüde bekleyelim
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        # Kullanıcı CTRL+C yaptığında buraya düşer
        pass
    except Exception as e:
        print(f"\nHATA: Flask Uygulaması başlatılamadı veya çalışırken bir hata oluştu: {e}")
    finally:
        # Program sonlandığında API sürecini sonlandır
        if api_process:
            print("\n--- Kullanıcı isteği üzerine Flask Uygulaması sonlandırılıyor... ---")
            try:
                api_process.terminate()
            except ProcessLookupError:
                # Süreç zaten kapanmış olabilir
                pass
            print("--- Proje Tamamlandı. ---")


# ----------------------------------------------------------------------
# ANA ÇALIŞTIRMA BLOĞU
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_project()