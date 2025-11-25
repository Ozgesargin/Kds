import sqlite3
import os

# Veri Tabanı Yolu
# Projenin kök dizininde 'kds_data.db' adında bir veritabanı dosyası oluşturulacak.
DB_PATH = os.path.join(os.path.dirname(__file__), 'kds_data.db')


class DatabaseManager:
    """
    SQLite veritabanı bağlantılarını ve temel işlemlerini yöneten sınıf.
    """

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def __enter__(self):
        """Context manager girişi: Bağlantı açma."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Sütun isimleriyle erişim için
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager çıkışı: Değişiklikleri kaydet ve bağlantıyı kapat."""
        if exc_type is None:
            # Hata yoksa commit et
            self.conn.commit()
        else:
            # Hata varsa rollback yap
            self.conn.rollback()
        self.conn.close()
        return False  # Hatanın tekrar yükseltilmesine izin ver


def init_db(schema_path='schema.sql'):
    """
    Veritabanını başlatır ve tabloları oluşturur.
    """
    print("--- Veritabanı Başlatılıyor ---")

    # Schema dosyasının yolunu düzeltme (aynı dizinde varsayılıyor)
    base_dir = os.path.dirname(__file__)
    full_schema_path = os.path.join(base_dir, schema_path)

    if not os.path.exists(full_schema_path):
        print(f"HATA: Schema dosyası bulunamadı: {full_schema_path}")
        return

    with open(full_schema_path, 'r', encoding='utf-8') as f:
        schema = f.read()

    try:
        # Veritabanı yöneticisi ile bağlantı kurma ve işlemleri yapma
        with DatabaseManager() as cur:
            cur.executescript(schema)
        print(f"✅ Veritabanı ve tablolar başarıyla oluşturuldu: {DB_PATH}")
    except sqlite3.Error as e:
        print(f"Veritabanı başlatılırken hata oluştu: {e}")


# Diğer CRUD (Create, Read, Update, Delete) fonksiyonları buraya eklenebilir.
# Örneğin, Flask uygulamasında kullanılmak üzere:

def get_all_users():
    """Tüm kullanıcıları (öğretmenler) çeker."""
    with DatabaseManager() as cur:
        # Şifre alanını güvenlik nedeniyle çekmekten kaçının (ya da hash'lenmiş veriyi çekin)
        cur.execute("SELECT id, name, email FROM teachers")
        # row_factory sayesinde sütun isimleriyle dict listesi olarak döner
        return [dict(row) for row in cur.fetchall()]


def get_user_by_email(email):
    """E-posta ile öğretmen bilgilerini çeker."""
    with DatabaseManager() as cur:
        cur.execute("SELECT * FROM teachers WHERE email = ?", (email,))
        row = cur.fetchone()
        return dict(row) if row else None


def add_new_user(user_id, name, email, password):
    """Yeni bir öğretmen kaydeder."""
    # NOT: Gerçek uygulamada password'ün hash'lenmesi GEREKİR!
    sql = "INSERT INTO teachers (id, name, email, password) VALUES (?, ?, ?, ?)"
    try:
        with DatabaseManager() as cur:
            cur.execute(sql, (user_id, name, email, password))
        return True
    except sqlite3.IntegrityError:
        # E-posta zaten varsa
        return False
    except sqlite3.Error as e:
        print(f"Kullanıcı eklenirken hata: {e}")
        return False


# --- ÖRNEK ANALİZ VERİLERİNİ ÇEKME FONKSİYONU ---
# Flask uygulamasında simüle edilmiş TEACHER_DATA'nın yerini alacak fonksiyon.

def get_teacher_dashboard_data(teacher_id):
    """
    Öğretmen paneli için gerekli analiz verilerini (simülasyon) veritabanından çeker.
    """
    # Bu fonksiyon, asıl uygulamanızdaki analiz mantığını simüle eder.
    # Gerçek KDS'de bu veriler, ML model çıktılarından ve analiz sorgularından gelmelidir.

    # Simülasyon verileri: Gerçek veritabanınızda ayrı tablolardan JOIN ile gelecektir.
    # Veritabanınızda öğrencilerin, konuların, denemelerin ve skorların olduğu varsayılır.

    # Örnek 1: Sınıf Genel Başarısı
    class_accuracy = 70.8

    # Örnek 2: En Çok Zorlanılan Konular (Başarı yüzdesi düşükten yükseğe)
    struggling_topics = [
        {"name": "Dönme Geometrisi", "struggle_level": 95},  # %95 zorlanma = %5 başarı
        {"name": "Silindir Alanı", "struggle_level": 88},
        {"name": "Prizma Hacmi", "struggle_level": 76},
    ]

    # Örnek 3: En Riskli Öğrenciler (Son skorları düşük olanlar)
    top_struggling_students = [
        {"id": 92007, "name": "Eren Yılmaz", "recent_score": 55, "hint_avg": 4.1},
        {"id": 88904, "name": "Defne Demir", "recent_score": 62, "hint_avg": 3.5},
        {"id": 78647, "name": "Can Kaya", "recent_score": 70, "hint_avg": 2.8},
    ]

    # Öğretmenin adını veritabanından çekme
    teacher_info = get_user_by_email("ayse.yilmaz@okul.edu")  # Örnek e-posta
    teacher_name = teacher_info['name'] if teacher_info else "Bilinmiyor"

    return {
        "class_name": "9/A Matematik Sınıfı",
        "teacher_name": teacher_name,
        "class_overview_accuracy": class_accuracy,
        "struggling_topics": struggling_topics,
        "top_struggling_students": top_struggling_students
    }