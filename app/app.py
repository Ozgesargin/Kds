from flask import Flask, render_template, request, redirect, url_for, session, g
from functools import wraps
import random
import os

# Flask uygulamasını başlatma
app = Flask(__name__)
# Oturum yönetimi için gizli anahtar
# GÜVENLİK NOTU: Buradaki anahtarı gerçek uygulamada daha güvenli ve uzun yapın
app.secret_key = os.environ.get('SECRET_KEY', 'cok_gizli_ve_uzun_bir_flask_session_anahtari')

# --- KULLANICI YÖNETİMİ İÇİN STATİK (SİMÜLE) VERİ TABANI ---
# Gerçek uygulamada bu veritabanı (örneğin Firestore) veya bir SQL/NoSQL veritabanı olmalıdır.
USERS = {
    # Varsayılan öğretmen hesabı (demo amaçlı)
    "ayse.yilmaz@okul.edu": {
        "id": "TCHR_001",
        "name": "Ayşe Yılmaz",
        "password": "sifre"  # DİKKAT: Gerçekte şifre hash'lenmelidir!
    }
}

# --- KARAR DESTEK SİSTEMİ ANALİZ VERİLERİ (VERİ SETİNİZDEN GELEN BİLGİ) ---
# Bu veri, veri setinizden analiz edilerek çıkarılmış özet bilgiyi temsil eder.
TEACHER_DATA = {
    "class_name": "9/A Matematik Sınıfı",
    "teacher_name": "Ayşe Yılmaz",
    "class_overview_accuracy": 70.8,  # Genel başarı yüzdesi
    # Konu Zorlukları: Başarı yüzdesi düşükten yükseğe sıralanmalıdır.
    "struggling_topics": [
        {"name": "Rotations", "struggle_level": 95},
        {"name": "Surface Area Cylinder", "struggle_level": 88},
        {"name": "Volume Cylinder", "struggle_level": 76},
        {"name": "Angles in Triangle", "struggle_level": 55},
    ],
    # En çok zorlanan öğrenciler: Skorları düşükten yükseğe sıralanmalıdır.
    "top_struggling_students": [
        {"id": 92007, "name": "Eren Yılmaz", "recent_score": 55, "hint_avg": 4.1},
        {"id": 88904, "name": "Defne Demir", "recent_score": 62, "hint_avg": 3.5},
        {"id": 78647, "name": "Selin Kaya", "recent_score": 70, "hint_avg": 2.8}
    ]
}


# --- DECORATORS VE YARDIMCI FONKSİYONLAR ---

def login_required(f):
    """Giriş yapılmasını zorunlu kılan decorator."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            # Oturum yoksa, kullanıcıyı giriş sayfasına yönlendir.
            return redirect(url_for('login', next=request.url))

        # Kullanıcı verilerini g global nesnesine yükle
        g.user = USERS.get(session.get('user_email'))
        if not g.user:
            # Kullanıcı veritabanında bulunamazsa oturumu temizle ve giriş sayfasına gönder
            session.clear()
            return redirect(url_for('login', next=request.url))

        return f(*args, **kwargs)

    return decorated_function


def generate_recommendations(data):
    """
    Simüle edilmiş KDS verilerine dayanarak 3 adet karar destek önerisi üretir.
    Gerçek KDS'de bu kısım, API hizmeti (api_service.py) ile etkileşime girmelidir.
    """

    # 1. En çok zorlanılan konu
    top_struggle_topic = data['struggling_topics'][0]['name']

    # 2. En düşük skorlu öğrenci
    top_struggle_student = data['top_struggling_students'][0]

    # 3. İpucu Bağımlılığı Yüksek Olan Öğrenci (Örneğin, ipucu ortalaması 3.5'tan büyük olan ilk öğrenci)
    hint_dependent_student = next((s for s in data['top_struggling_students'] if s['hint_avg'] > 3.5), None)

    recommendations = []

    # Öneri 1: Konu Odaklı (En çok zorlanılan konuya ek materyal)
    recommendations.append({
        "type": "Konu Odaklı",
        "text": f"Sınıfın en çok zorlandığı konu olan **{top_struggle_topic}** için ek alıştırmalar veya video kaynakları atayın.",
        "action": "Ders Materyali Ekle"
    })

    # Öneri 2: Bireysel Öğrenci (Düşük skora bireysel görüşme)
    recommendations.append({
        "type": "Bireysel Öğrenci",
        "text": f"**{top_struggle_student['name']}** ({top_struggle_student['recent_score']}%) son denemelerinde düşük performans gösteriyor. Bire bir görüşme planlayın.",
        "action": "Görüşme Planla"
    })

    # Öneri 3: Davranışsal Öneri (İpucu bağımlılığı)
    if hint_dependent_student:
        recommendations.append({
            "type": "Davranışsal",
            "text": f"**{hint_dependent_student['name']}**'in ortalama ipucu kullanımı ({hint_dependent_student['hint_avg']}) yüksek. Problem çözme bağımsızlığını artırıcı bir çalışma planı uygulayın.",
            "action": "Özel Çalışma Planı"
        })
    else:
        # Alternatif Öneri: Grup Çalışması
        recommendations.append({
            "type": "Grup Çalışması",
            "text": "Ortalama zorluk seviyesindeki öğrencileri birbirleriyle eşleştirerek akran desteği grubu oluşturun.",
            "action": "Grup Oluştur"
        })

    return recommendations


# --- ROTALAR ---

@app.route('/')
def index():
    """Ana karşılama sayfası."""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Kullanıcı giriş sayfası."""
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user_data = USERS.get(email)

        if user_data and user_data['password'] == password:  # Gerçekte hash kontrolü yapılmalı
            session.clear()
            session['user_email'] = email
            # Başarılı girişten sonra dashboard'a yönlendir
            return redirect(url_for('teacher_dashboard'))
        else:
            error = 'Yanlış e-posta veya şifre.'

    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Kullanıcı kayıt sayfası."""
    error = None
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']

        if email in USERS:
            error = 'Bu e-posta adresi zaten kayıtlı.'
        else:
            # Sadece demo amaçlı basit kayıt simülasyonu:
            new_id = "TCHR_" + str(random.randint(200, 999))
            USERS[email] = {"password": password, "name": name, "id": new_id}

            # Kayıt sonrası otomatik giriş
            session.clear()
            session['user_email'] = email
            return redirect(url_for('teacher_dashboard'))

    return render_template('register.html', error=error)


@app.route('/logout')
def logout():
    """Kullanıcı oturumunu sonlandırır."""
    session.clear()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required  # Bu decorator, giriş yapılmasını zorunlu kılar.
def teacher_dashboard():
    """Öğretmen Karar Destek Sistemi Dashboard'u."""

    # Veri setine uygun dinamik önerileri üret
    decision_support_recommendations = generate_recommendations(TEACHER_DATA)

    # Template'e gönderilecek bağlam (context)
    context = {
        'user': g.user,  # login_required ile eklenen kullanıcı bilgisi
        'data': TEACHER_DATA,
        'recommendations': decision_support_recommendations,
    }

    # teacher.html template'ini render et
    return render_template('teacher.html', **context)


# ----------------------------------------------------------------------
# ANA ÇALIŞTIRMA BLOĞU (Sadece Flask uygulamasını doğrudan çalıştırmak için)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Bu blok, sadece bu dosya doğrudan çalıştırıldığında (örneğin: python app.py) çalışır.
    # main.py, bu uygulamayı dışarıdan başlatan betiktir.
    print("WARNING: Flask app is running in debug mode. Use 'python main.py' for production-like start.")
    app.run(debug=True, port=5000)