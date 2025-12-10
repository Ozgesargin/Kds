# app/model_inference.py

import os
import json
import pickle
import pandas as pd
import numpy as np

# Eğitim scriptindeki ile aynı olmalı
NUMERICAL_FEATURES = [
    "attempt_count",
    "ms_first_response",
    "hint_count",
    "hint_total",
    "hint_independence",
    "skill_mastery_score",
    "problem_difficulty",
    "normalized_time",
]

CATEGORICAL_FEATURES = []  # Şu an yok, ama altyapı hazır

# Model dosyalarının yolu (app klasörü içinden)
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "kds_v1"
)

# Ham log dosyası için ZORUNLU kolonlar
REQUIRED_COLS = [
    "user_id",
    "problem_id",
    "skill_id",
    "skill_name",
    "correct",
    "ms_first_response",
]

# Opsiyonel ama varsa kullanılan kolonlar
OPTIONAL_COLS = [
    "attempt_count",
    "hint_count",
    "hint_total",
]


def load_model_components():
    """
    Eğitilmiş modeli, scaler'ı, feature isimlerini ve politika eşiklerini yükler.
    """
    model_path = os.path.join(MODEL_DIR, "logreg_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.json")
    policy_path = os.path.join(MODEL_DIR, "difficulty_policy.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {model_path}. "
            f"Lütfen eğitim scriptini çalıştırıp model dosyalarını kaydettiğinden emin ol."
        )

    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    feature_names = json.load(open(features_path, "r", encoding="utf-8"))

    # Zorluk politikası (kalibrasyon scriptinden gelir)
    if os.path.exists(policy_path):
        difficulty_policy = json.load(open(policy_path, "r", encoding="utf-8"))
    else:
        difficulty_policy = {
            "EASY_THRESHOLD": 0.5,
            "MEDIUM_THRESHOLD": 0.75,
        }

    return model, scaler, feature_names, difficulty_policy


def prepare_inference_matrix(df, scaler, feature_names):
    """
    Eğitimde kullanılan feature setine uygun bir X matrisi hazırlar.
    df'nin zaten create_mastery_features ile üretilmiş kolonlara sahip olduğunu varsayıyoruz.
    """
    X = df.copy()

    # Eğitimde attığımız kolonlar varsa yine atalım
    drop_cols = ["correct", "user_id", "problem_id", "skill_id", "skill_name"]
    for col in drop_cols:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Eksik değerleri doldur ve sayısal tipleri düzelt
    for col in NUMERICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(np.float64)
            X[col] = X[col].fillna(X[col].mean())

    # One-hot kategorik (şu anda yok ama altyapı hazır)
    if CATEGORICAL_FEATURES:
        X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, dummy_na=False)

    # Eksik feature'ları 0 ile doldur, kolon sırasını eğitimdekiyle eşleştir
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_names]

    # Sadece sayısal kolonları scale et
    numeric_cols_to_scale = [c for c in X.columns if c in NUMERICAL_FEATURES]
    if numeric_cols_to_scale:
        X.loc[:, numeric_cols_to_scale] = scaler.transform(X.loc[:, numeric_cols_to_scale])

    return X


# ----------------------------------------------------------------------
# Ham log → Ön işleme → Özellik çıkarımı
# Eğitimde kullandığın pipeline'ın hafif versiyonunu burada tekrar kullanıyoruz.
# ----------------------------------------------------------------------
def preprocess_raw_log_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluation ekranından gelen ham log verisi için:
      - Zorunlu kolonları kontrol eder
      - Eksik opsiyonel kolonları default ile doldurur
      - data_preprocess.clean_and_prepare + model_training.create_mastery_features
        pipeline'ını çalıştırır.
    """
    # Fazla sütun sorun değil, ama şu zorunlular yoksa hata ver:
    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(
            "Yüklenen dosyada eksik zorunlu sütun(lar) var: "
            + ", ".join(missing)
            + ".\n"
            + "Lütfen dosyanın en az şu kolonları içerdiğinden emin olun: "
            + ", ".join(REQUIRED_COLS)
        )

    df = df_raw.copy()

    # Opsiyonel kolonlar yoksa default değer atayalım
    if "attempt_count" not in df.columns:
        df["attempt_count"] = 1
    if "hint_count" not in df.columns:
        df["hint_count"] = 0
    if "hint_total" not in df.columns:
        df["hint_total"] = df["hint_count"]

    # Proje kök dizinini import yoluna ekle (data_preprocess & model_training'i kullanmak için)
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # Eğitimde kullandığın fonksiyonları tekrar kullan
    from data_preprocess import clean_and_prepare  # type: ignore
    from model_training import create_mastery_features  # type: ignore

    df_clean = clean_and_prepare(df)
    df_features = create_mastery_features(df_clean)

    return df_features


# ----------------------------------------------------------------------
# Tahmin Fonksiyonları
# ----------------------------------------------------------------------
def predict_difficulty_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verilen DataFrame için p(correct) ve difficulty label üretir.
    df'nin create_mastery_features ile elde edilmiş kolonlara sahip olduğunu varsayar.
    """
    model, scaler, feature_names, policy = load_model_components()

    X = prepare_inference_matrix(df, scaler, feature_names)
    p_correct = model.predict_proba(X)[:, 1]

    easy_th = float(policy.get("EASY_THRESHOLD", 0.5))
    med_th = float(policy.get("MEDIUM_THRESHOLD", 0.75))

    difficulty_labels = []
    for p in p_correct:
        if p >= med_th:
            difficulty_labels.append("Kolay")
        elif p >= easy_th:
            difficulty_labels.append("Orta")
        else:
            difficulty_labels.append("Zor")

    result_df = df.copy()
    result_df["p_correct"] = p_correct
    result_df["difficulty_level"] = difficulty_labels

    return result_df


def predict_difficulty_for_file(uploaded_file) -> pd.DataFrame:
    """
    Flask'tan gelen dosya objesini (CSV veya Excel) okuyup:
      1) Ham veriye veri ön işleme ve özellik çıkarımı uygular
      2) Model ile p(correct) + difficulty_level hesaplar
      3) Sonuç DataFrame'ini döndürür.

    Fazla sütunlar sorun değil; ZORUNLU sütun isimleri varsa sırası önemsizdir.
    """
    filename = uploaded_file.filename.lower()

    if not filename:
        raise ValueError("Lütfen bir dosya seçin.")

    # Dosyayı pandas ile oku
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        # Excel'de encoding problemi olmaz
        df_raw = pd.read_excel(uploaded_file)
    elif filename.endswith(".csv"):
        # CSV için farklı encoding denemeleri
        tried_encodings = []
        df_raw = None
        for enc in ["utf-8", "utf-8-sig", "cp1254", "latin1"]:
            try:
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file, encoding=enc)
                break
            except UnicodeDecodeError:
                tried_encodings.append(enc)
                continue

        if df_raw is None:
            raise UnicodeDecodeError(
                "utf-8",
                b"",
                0,
                1,
                f"Dosya şu encoding denemeleriyle açılamadı: {', '.join(tried_encodings)}"
            )
    else:
        raise ValueError("Sadece CSV veya Excel dosyaları destekleniyor.")

    if df_raw.empty:
        raise ValueError("Yüklenen dosya boş görünüyor.")

    # 1) Ham log'u eğitimdeki pipeline ile uyumlu hale getir
    df_features = preprocess_raw_log_df(df_raw)

    # 2) Model ile tahmin yap
    result_df = predict_difficulty_for_dataframe(df_features)
    return result_df
