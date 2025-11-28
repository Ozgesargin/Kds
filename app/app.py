# app/app.py

from flask import Flask, render_template, request, redirect, url_for, session, g
from functools import wraps
import os
import hashlib

from database.data_service import DataService
from database.db import get_db_connection
from model_inference import predict_difficulty_for_file

# ---------------------------------------------------------
# TEMPLATE KLASÖRÜ
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "ui", "templates")

app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.environ.get(
    "SECRET_KEY", "cok_gizli_ve_uzun_bir_flask_session_anahtari"
)



# ---------------------------------------------------------
# Öğretmen Login (DB üzerinden)
# ---------------------------------------------------------
def fetch_teacher_by_credentials(username, password):
    """
    Egitim.Ogretmenler tablosundan KullaniciAdi ve düz Sifre ile öğretmen bulur.
    (Proje için sade/guvensiz versiyon)
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                ogretmen_id,
                okul_id,
                KullaniciAdi,
                Sifre
            FROM Egitim.Ogretmenler
            WHERE KullaniciAdi = ?
            """,
            (username,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        ogretmen_id = row.ogretmen_id
        okul_id = row.okul_id
        kullanici_adi = row.KullaniciAdi
        sifre_db = row.Sifre  # NVARCHAR sütun

        # Düz karşılaştırma
        if sifre_db is None or sifre_db != password:
            return None

        display_name = f"Öğretmen {ogretmen_id}"

        return {
            "id": ogretmen_id,
            "school_id": okul_id,
            "username": kullanici_adi,
            "name": display_name,
            "email": kullanici_adi,
        }

    except Exception as e:
        print(f"DB Login Error: {e}")
        return None
    finally:
        if conn is not None:
            conn.close()
# ---------------------------------------------------------
# Decorator: Giriş Zorunlu
# ---------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_email" not in session:
            return redirect(url_for("login", next=request.url))

        g.user = {
            "email": session.get("user_email"),
            "name": session.get("user_name", "Bilinmiyor"),
            "teacher_id": session.get("teacher_id"),
            "school_id": session.get("school_id"),
        }

        if not g.user["email"]:
            session.clear()
            return redirect(url_for("login", next=request.url))

        return f(*args, **kwargs)

    return decorated_function


# ---------------------------------------------------------
# Karar Destek Önerileri (UI'da gösterilen 3 kutu)
# ---------------------------------------------------------
def generate_recommendations(data):
    """
    Dashboard verisine göre 3 adet öneri objesi üretir.
    """
    if (
        not data
        or not data.get("struggling_topics")
        or not data.get("top_struggling_students")
    ):
        return [
            {
                "type": "Hata",
                "text": "KDS verisi yüklenemediği için öneri üretilemiyor.",
                "action": "Kontrol Et",
            }
        ]

    top_struggle_topic = data["struggling_topics"][0]
    top_struggle_student = data["top_struggling_students"][0]

    recommendations = []

    recommendations.append(
        {
            "type": "Konu Odaklı",
            "text": (
                f"Sınıfın en çok zorlandığı konu olan **{top_struggle_topic['name']}** için "
                f"ek alıştırmalar veya video kaynakları atayın."
            ),
            "action": "Ders Materyali Ekle",
        }
    )

    recommendations.append(
        {
            "type": "Bireysel Öğrenci",
            "text": (
                f"**{top_struggle_student['name']}** ({top_struggle_student['recent_score']}%) "
                f"son denemelerinde düşük performans gösteriyor. Bire bir görüşme planlayın."
            ),
            "action": "Görüşme Planla",
        }
    )

    recommendations.append(
        {
            "type": "Grup Çalışması",
            "text": (
                "Ortalama zorluk seviyesindeki öğrencileri birbirleriyle eşleştirerek "
                "akran desteği grubu oluşturun."
            ),
            "action": "Grup Oluştur",
        }
    )

    return recommendations


# ---------------------------------------------------------
# Rotalar
# ---------------------------------------------------------
@app.route("/")
def index():
    if "user_email" in session:
        return redirect(url_for("teacher_dashboard"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["email"]
        password = request.form["password"]

        user_data = fetch_teacher_by_credentials(username, password)

        if user_data:
            session.clear()
            session["user_email"] = user_data["email"]
            session["user_name"] = user_data["name"]
            session["teacher_id"] = user_data["id"]
            session["school_id"] = user_data["school_id"]

            return redirect(url_for("teacher_dashboard"))
        else:
            error = "Yanlış kullanıcı adı veya şifre."

    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = "Kayıt işlemi veritabanı simülasyonunda devre dışıdır."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route('/dashboard')
@login_required
def teacher_dashboard():
    teacher_id = session.get('teacher_id')   # login sırasında set etmiştik
    okul_id = session.get('okul_id')

    kds_service = DataService(teacher_id=teacher_id, okul_id=okul_id)
    dashboard_data = kds_service.predict_and_aggregate_data()
    recommendations = generate_recommendations(dashboard_data)

    context = {
        "user": g.user,
        "data": dashboard_data,
        "recommendations": recommendations,
    }
    return render_template("teacher.html", **context)

@app.route('/students')
@login_required
def students():
    """Öğrenciler listesi sayfası."""
    teacher_id = session.get('teacher_id')
    okul_id = session.get('okul_id')

    try:
        kds_service = DataService(teacher_id=teacher_id, okul_id=okul_id)
        # Dashboard’ta kullandığın özet veri:
        dashboard_data = kds_service.predict_and_aggregate_data()
        # Öğrenciler tablosu:
        students_list = kds_service.get_student_performance()
    except Exception as e:
        print(f"HATA: Öğrenci listesi alınırken hata oluştu: {e}")
        dashboard_data = {
            "class_name": "Veri Yok",
            "class_overview_accuracy": 0.0,
            "total_students": 0,
        }
        students_list = []

    context = {
        "user": g.user,
        "data": dashboard_data,   # 🔴 students.html içinde kullanacağımız değişken
        "students": students_list
    }
    return render_template("students.html", **context)


@app.route("/evaluation", methods=["GET", "POST"])
@login_required
def evaluation():
    """
    Başarı değerlendirme sayfası.
    Öğretmen CSV/Excel dosyası yükler, model tahmin eder,
    biz de evaluation.html’de gösteririz.
    """
    message = None
    headers = []
    rows = []

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            message = "Lütfen bir CSV veya Excel dosyası seçin."
        else:
            try:
                # Model ile tahmin yap
                result_df = predict_difficulty_for_file(file)

                # Ekranda ilk 100 satırı gösterelim
                preview_df = result_df.head(100)

                headers = list(preview_df.columns)
                rows = preview_df.to_dict(orient="records")

                message = (
                    f"{len(result_df)} satır için değerlendirme yapıldı. "
                    f"İlk {len(preview_df)} satır gösteriliyor."
                )

            except Exception as e:
                print(f"HATA: Dosya değerlendirme başarısız oldu: {e}")
                message = (
                    "Dosya okunurken veya model tahmini yapılırken bir hata oluştu. "
                    "Lütfen dosya kolonlarının eğitimde kullandığın formatla uyumlu olduğundan emin ol."
                )

    return render_template(
        "evaluation.html",
        user=g.user,
        message=message,
        headers=headers,
        rows=rows,
    )



# ---------------------------------------------------------
# Lokal Çalıştırma
# ---------------------------------------------------------
if __name__ == "__main__":
    print("WARNING: Flask app is running in debug mode.")
    print("Örnek giriş bilgileri:")
    print("  Kullanıcı Adı: ogr_pk22059_1  | Şifre: sifre")
    print("  Kullanıcı Adı: ogr_pk10023_5  | Şifre: sifre")
    app.run(debug=True, port=5000)
