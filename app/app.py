# app/app.py

from flask import Flask, render_template, request, redirect, url_for, session, g
from functools import wraps
import os
import json  # Ödev sorularını JSON olarak almak için

from database.data_service import DataService
from database.db import (
    get_db_connection,
    insert_evaluation_results,      # İstersen kullanılmayabilir ama dursun
    save_homework_questions_for_student,
    fetch_homework_for_student,
)
from model_inference import predict_difficulty_for_file
from question_service import QuestionService  # DeepMind Math tabanlı soru servisi

# DeepMind Math soru servisini tek örnek olarak oluşturuyoruz
q_service = QuestionService()

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
        sifre_db = row.Sifre

        # Basit şifre kontrolü (hash yok, proje için sade hali)
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
# Karar Destek Önerileri (dashboard için)
# ---------------------------------------------------------
def generate_recommendations(data):
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
# Rotalar (Auth + Dashboard)
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
    error = "Kayıt işlemi bu demo sürümde devre dışıdır."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def teacher_dashboard():
    teacher_id = session.get("teacher_id")
    school_id = session.get("school_id")

    kds_service = DataService(teacher_id=teacher_id, okul_id=school_id)
    dashboard_data = kds_service.predict_and_aggregate_data()
    recommendations = generate_recommendations(dashboard_data)

    context = {
        "user": g.user,
        "data": dashboard_data,
        "recommendations": recommendations,
    }
    return render_template("teacher.html", **context)


# ---------------------------------------------------------
# Öğrenciler listesi
# ---------------------------------------------------------
@app.route("/students")
@login_required
def students():
    teacher_id = session.get("teacher_id")
    school_id = session.get("school_id")

    try:
        kds_service = DataService(teacher_id=teacher_id, okul_id=school_id)
        dashboard_data = kds_service.predict_and_aggregate_data()
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
        "data": dashboard_data,
        "students": students_list,
    }
    return render_template("students.html", **context)


# ---------------------------------------------------------
# Evaluation (Dosya yükleyip modelle değerlendirme)
# ---------------------------------------------------------
@app.route("/evaluation", methods=["GET", "POST"])
@login_required
def evaluation():
    from database.db import save_evaluation_results_to_db  # döngü import olmaması için

    message = None
    headers = []
    rows = []

    if request.method == "POST":
        action = request.form.get("action", "preview")  # "preview" veya "save"
        file = request.files.get("file")

        if not file or file.filename == "":
            message = "Lütfen bir CSV veya Excel dosyası seçin."
        else:
            try:
                # Her iki durumda da önce modeli çalıştırıp önizlemeyi üretelim
                result_df = predict_difficulty_for_file(file)

                # Öğretmene gösterilecek sade tablo
                display_cols = [
                    "user_id",
                    "problem_id",
                    "skill_name",
                    "correct",
                    "p_correct",
                    "difficulty_level",
                ]
                display_cols = [c for c in display_cols if c in result_df.columns]

                preview_df = result_df[display_cols].copy()

                if "p_correct" in preview_df.columns:
                    preview_df["p_correct"] = (preview_df["p_correct"] * 100).round(1)

                headers = list(preview_df.columns)
                rows = preview_df.to_dict(orient="records")

                if action == "save":
                    # Kullanıcı önce zaten önizleyip sonra bu butona basacak
                    save_evaluation_results_to_db(result_df)
                    message = (
                        f"{len(result_df)} satırlık değerlendirme sonucu sisteme kaydedildi. "
                        "Dashboard ve Öğrenciler ekranı bu verileri de kullanacak."
                    )
                else:
                    message = (
                        f"{len(result_df)} satır için değerlendirme yapıldı. "
                        f"İlk {len(preview_df)} satır gösteriliyor."
                    )

            except Exception as e:
                print(f"HATA: Dosya değerlendirme başarısız oldu: {e}")
                message = (
                    "Dosya okunurken, veri ön işleme yapılırken veya model tahmini sırasında bir hata oluştu. "
                    "Çoğunlukla dosyanın encoding'i (UTF-8 vs.) ya da zorunlu kolonlardan birinin eksik olması "
                    "bu hataya sebep olur. En az şu kolonlar bulunmalı: "
                    "user_id, problem_id, skill_id, skill_name, correct, ms_first_response."
                )

    return render_template(
        "evaluation.html",
        user=g.user,
        message=message,
        headers=headers,
        rows=rows,
    )


# ---------------------------------------------------------
# YENİ: Öğrenci Profil Sayfası + Soru Önerileri + Ödevler
# ---------------------------------------------------------
@app.route("/students/<int:student_id>")
@login_required
def student_profile(student_id: int):
    teacher_id = session.get("teacher_id")
    school_id = session.get("school_id")

    kds_service = DataService(teacher_id=teacher_id, okul_id=school_id)
    profile = kds_service.get_student_profile(student_id)

    # En çok zorlanılan / çalışılan skil
    primary_skill = None
    has_any_activity = False

    topics = profile.get("topics") or []
    if topics:
        topics_sorted = sorted(topics, key=lambda t: t.get("accuracy", 100.0))
        primary_skill = topics_sorted[0].get("name")
        has_any_activity = True
    else:
        has_any_activity = False

    # DeepMind Math'ten soru öner
    recommendations = q_service.recommend_for_student_accuracy(
        overall_accuracy=profile.get("overall_accuracy", 0.0),
        n=8,
        primary_skill_name=primary_skill,
        has_any_activity=has_any_activity,
    )

    # Bu öğrenciye daha önce verilmiş ödevler
    assigned_homework = fetch_homework_for_student(student_id)

    context = {
        "user": g.user,
        "profile": profile,
        "recommendations": recommendations,
        "assigned_homework": assigned_homework,
    }
    return render_template("student_profile.html", **context)


# ---------------------------------------------------------
# YENİ: Bu öğrencinin önerilen sorularını ödev olarak kaydet
# ---------------------------------------------------------
@app.route("/students/<int:student_id>/assign_homework", methods=["POST"])
@login_required
def assign_homework(student_id: int):
    teacher_id = session.get("teacher_id")

    questions_json = request.form.get("questions_json", "[]")
    try:
        all_questions = json.loads(questions_json)
    except Exception:
        all_questions = []

    # Seçili index değerlerini al
    selected_indices = request.form.getlist("selected_idx")
    try:
        selected_indices = [int(i) for i in selected_indices]
    except Exception:
        selected_indices = []

    if selected_indices:
        questions = [
            q for idx, q in enumerate(all_questions) if idx in selected_indices
        ]
    else:
        questions = []

    if not questions:
        print("INFO: assign_homework - Seçili soru yok, ödev kaydedilmedi.")
        return redirect(url_for("student_profile", student_id=student_id))

    try:
        save_homework_questions_for_student(
            teacher_id=teacher_id,
            student_id=student_id,
            questions=questions,
        )
    except Exception as e:
        print(f"HATA: Ödev kaydedilirken hata oluştu: {e}")

    return redirect(url_for("student_profile", student_id=student_id))

# ---------------------------------------------------------
# Lokal Çalıştırma
# ---------------------------------------------------------
if __name__ == "__main__":
    print("WARNING: Flask app is running in debug mode.")
    print("Örnek giriş bilgileri:")
    print("  Kullanıcı Adı: ogr_pk22059_1  | Şifre: sifre")
    print("  Kullanıcı Adı: ogr_pk10023_5  | Şifre: sifre")
    app.run(debug=True, port=5000)
