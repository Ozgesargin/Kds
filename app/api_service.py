import os
import json
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- Sabitler ve Yollar ---
# MODEL_DIR, model_training.py tarafından kaydedilen dosyaların yolunu işaret etmeli
MODEL_DIR = os.path.join("models", "kds_v1")
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")
POLICY_PATH = os.path.join(MODEL_DIR, "difficulty_policy.json")

# --- Global Değişkenler ---
app = Flask(__name__)
model = None
scaler = None
feature_names = None
difficulty_policy = None
assets_loaded = False  # Yükleme durumunu izlemek için yeni değişken

# Model eğitimi sırasında kullanılan sayısal özellikler listesi
# Bu liste, scaler'ın sadece bu sütunları dönüştürmesi için gereklidir.
NUMERICAL_FEATURES = [
    'attempt_count', 'ms_first_response', 'hint_count', 'hint_total',
    'hint_independence', 'skill_mastery_score', 'problem_difficulty',
    'normalized_time'
]


# ----------------------------------------------------------------------
# Fonksiyonlar
# ----------------------------------------------------------------------

def load_assets():
    """Kayıtlı modeli, scaler'ı ve diğer bileşenleri yükler."""
    global model, scaler, feature_names, difficulty_policy, assets_loaded

    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
        assets_loaded = False
        return False

    try:
        # Model, Scaler ve Diğer Bileşenleri Yükleme
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(FEATURE_NAMES_PATH, 'r') as f:
            feature_names = json.load(f)
        with open(POLICY_PATH, 'r') as f:
            difficulty_policy = json.load(f)

        print("✅ Tüm model bileşenleri başarıyla yüklendi.")
        assets_loaded = True
        return True
    except Exception as e:
        print(f"HATA: Varlıklar yüklenirken bir sorun oluştu: {e}")
        assets_loaded = False
        return False


def preprocess_and_predict(data):
    """Gelen veriyi işler, ölçekler ve tahmin yapar."""
    global model, scaler, feature_names, assets_loaded

    # Modelin yüklü olduğundan emin olma
    if not assets_loaded or model is None or scaler is None or feature_names is None:
        return None, "Model veya bileşenleri yüklenemedi. Lütfen API durumunu kontrol edin."

    # 1. Gelen JSON verisinden DataFrame oluşturma
    try:
        # Streamlit'ten tek bir sözlük geldiği için index=[0] kullanılıyor.
        df_raw = pd.DataFrame(data, index=[0])
    except Exception as e:
        return None, f"Veri çerçevesi oluşturulurken hata: {e}"

    # 2. Modelin beklediği tüm özelliklerin varlığını kontrol etme
    missing_features = [f for f in feature_names if f not in df_raw.columns]
    if missing_features:
        return None, f"Eksik özellikler: {', '.join(missing_features)}"

    # 3. Özellik sıralamasını modelin eğitim sırasına göre ayarlama
    X_predict = df_raw[feature_names].copy()

    # 4. Ölçeklenecek sayısal sütunları belirleme ve tipini ayarlama
    numeric_cols_to_scale = [col for col in NUMERICAL_FEATURES if col in X_predict.columns]

    # Veri Tipini float64 yapma (scaler'ın beklediği tip)
    X_predict.loc[:, numeric_cols_to_scale] = X_predict.loc[:, numeric_cols_to_scale].astype(np.float64)

    # 5. Ölçekleme
    if scaler and numeric_cols_to_scale:
        X_predict.loc[:, numeric_cols_to_scale] = scaler.transform(X_predict.loc[:, numeric_cols_to_scale])
    else:
        return None, "Ölçekleyici (Scaler) yüklenemedi veya ölçeklenecek sütun yok."

    # 6. Tahmin
    prediction_proba = model.predict_proba(X_predict)[:, 1][0]

    return prediction_proba, None


# ----------------------------------------------------------------------
# API Endpoint'leri
# ----------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    """Gelen özellikler için doğru cevap olasılığını tahmin eder."""

    # Yeni Güvenlik Kontrolü
    if not assets_loaded:
        return jsonify({"error": "Model varlıkları yüklü değil. Sunucu hatası."}), 503

    if not request.json:
        return jsonify({"error": "Lütfen JSON veri gönderin."}), 400

    data = request.json

    # Gelen verinin beklenen bir sözlük olduğunu varsayıyoruz
    prediction_proba, error = preprocess_and_predict(data)

    if error:
        return jsonify({"error": error}), 400

    # Karar politikasına göre zorluk seviyesini belirleme
    easy_threshold = difficulty_policy.get("EASY_THRESHOLD", 0.5)
    medium_threshold = difficulty_policy.get("MEDIUM_THRESHOLD", 0.7)

    if prediction_proba <= easy_threshold:
        difficulty = "Zor (Hard)"
    elif prediction_proba <= medium_threshold:
        difficulty = "Orta (Medium)"
    else:
        difficulty = "Kolay (Easy)"

    response = {
        "P_Correct": round(prediction_proba, 4),
        "Suggested_Difficulty": difficulty,
        # DÜZELTME: Policy dizesini daha net ve standart bir JSON formatına getiriyoruz.
        "Policy": {
            "Easy_Cutoff": easy_threshold,
            "Medium_Cutoff": medium_threshold
        }
    }

    return jsonify(response)


@app.route('/status', methods=['GET'])
def status():
    """API ve model yükleme durumunu kontrol eder."""
    if assets_loaded:
        return jsonify({"status": "OK", "model_version": "kds_v1", "policy": difficulty_policy})
    else:
        return jsonify({"status": "HATA",
                        "message": "Model bileşenleri yüklenemedi. Lütfen konsol çıktılarını kontrol edin."}), 500


# ----------------------------------------------------------------------
# ANA ÇALIŞTIRMA BLOĞU
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if load_assets():
        print("\n🌐 Flask API başlatılıyor...")
        # debug=False ile çalıştırıyoruz (önceki düzeltme)
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
