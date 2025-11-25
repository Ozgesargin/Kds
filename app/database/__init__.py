import os
# database.py dosyasındaki fonksiyonu içe aktarın
from database import init_db

if __name__ == "__main__":
    # Önemli Not: Bu betiği, database.py ve schema.sql dosyalarıyla aynı dizinde çalıştırın.
    # Flask uygulamasının olduğu dizinde (örn: 'app' klasöründe) çalıştırmanız idealdir.

    # Veritabanını başlat ve tabloları oluştur
    init_db(schema_path='schema.sql')