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
#   app/
#     model_inference.py  (bu dosya)
#     models/
#       kds_v1/
#         logreg_model.pkl
#         scaler.pkl
#         feature_names.json
#         difficulty_policy.json
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "kds_v1"
)


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
    df'nin zaten processed_skill_builder benzeri kolonlara sahip olduğunu varsayıyoruz.
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


def predict_difficulty_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verilen DataFrame için p(correct) ve difficulty label üretir.
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
    Flask'tan gelen dosya objesini (CSV veya Excel) okuyup
    difficulty tahminleri içeren DataFrame döndürür.
    """
    filename = uploaded_file.filename.lower()

    if not filename:
        raise ValueError("Lütfen bir dosya seçin.")

    # Dosyayı pandas ile oku
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raise ValueError("Sadece CSV veya Excel dosyaları destekleniyor.")

    if df.empty:
        raise ValueError("Yüklenen dosya boş görünüyor.")

    result_df = predict_difficulty_for_dataframe(df)
    return result_df
