-- Öğretmen Bilgileri Tablosu
CREATE TABLE IF NOT EXISTS teachers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL, -- Gerçekte hash'lenmiş şifre saklanmalıdır!
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Öğrenci Bilgileri Tablosu (Opsiyonel: Detaylı Öğrenci yönetimi için)
CREATE TABLE IF NOT EXISTS students (
    id TEXT PRIMARY KEY,
    teacher_id TEXT,
    name TEXT NOT NULL,
    class_name TEXT,
    FOREIGN KEY (teacher_id) REFERENCES teachers (id)
);

-- Konu Başarı Analizi Tablosu (KDS'den gelen veriyi simüle eder)
-- Bu tablo, ML modelinizden veya daha karmaşık SQL sorgularından elde edilen
-- öğretmen/sınıf seviyesi analizleri için kullanılabilir.
CREATE TABLE IF NOT EXISTS kds_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id TEXT NOT NULL,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    class_accuracy REAL, -- Sınıfın genel başarı yüzdesi
    struggling_topic_json TEXT, -- JSON formatında zorlanılan konular listesi
    risk_students_json TEXT,     -- JSON formatında riskli öğrenciler listesi
    FOREIGN KEY (teacher_id) REFERENCES teachers (id)
);

-- Örnek Öğretmen Verisi Ekleme
INSERT OR IGNORE INTO teachers (id, name, email, password) VALUES
('TCHR_001', 'Ayşe Yılmaz', 'ayse.yilmaz@okul.edu', 'sifre');
-- NOT: OR IGNORE, zaten varsa tekrar eklemeyi engeller.