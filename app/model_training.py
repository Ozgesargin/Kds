import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import pickle
import json

# --- Sabitler ve Yollar
INPUT_FILE = os.path.join("../data", "output", "processed_skill_builder.csv")  # Yol düzeltildi
MODEL_OUTPUT_DIR = os.path.join("../models", "kds_v1")

# P(Correct) eşikleri (Karar Politikası)
DIFFICULTY_POLICY = {
    "EASY_THRESHOLD": 0.5,
    "MEDIUM_THRESHOLD": 0.75
}

# Kritik Özellik Kümeleri
NUMERICAL_FEATURES = [
    'attempt_count', 'ms_first_response', 'hint_count', 'hint_total',
    'hint_independence', 'skill_mastery_score', 'problem_difficulty',
    'normalized_time'
]
CATEGORICAL_FEATURES = [
    # Veri setinde varsa buraya eklenecek
]


# ----------------------------------------------------------------------
# Fonksiyonlar
# ----------------------------------------------------------------------
# (create_mastery_features fonksiyonu değişmedi)

def create_mastery_features(df):
    """Kümülatif ustalık skoru, problem zorluğu ve normalize süreyi hesaplar."""
    print("⏳ Beceri bazlı ustalık skorları hesaplanıyor...")

    # ... (kodun bu kısmı değişmedi)

    # 1. Kümülatif Ustalık Skoru (Mastery Score)
    df['skill_mastery_score'] = df.groupby(['user_id', 'skill_id'])['correct'] \
        .transform(lambda x: x.expanding().mean().shift(1))

    global_avg = df['correct'].mean()
    df['skill_mastery_score'] = df['skill_mastery_score'].fillna(global_avg)

    # 2. Problem Zorluk Skoru (Statik)
    problem_avg = df.groupby('problem_id')['correct'].mean().reset_index()
    problem_avg.rename(columns={'correct': 'problem_difficulty'}, inplace=True)
    df = df.merge(problem_avg, on='problem_id', how='left')

    # 3. Normalleştirilmiş Cevap Süresi
    problem_time_avg = df.groupby('problem_id')['ms_first_response'].mean().reset_index()
    problem_time_avg.rename(columns={'ms_first_response': 'avg_problem_time'}, inplace=True)
    df = df.merge(problem_time_avg, on='problem_id', how='left')
    df['normalized_time'] = df['ms_first_response'] / df['avg_problem_time'].replace(0, 1)

    print("✅ Özellik Mühendisliği tamamlandı.")
    return df.drop(columns=['avg_problem_time'])


def prepare_data(df):
    """Veriyi bölme, OHE ve ölçeklendirme işlemlerini yapar."""

    # Eksik değerleri doldurma
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    Y = df['correct']
    X = df.drop(columns=['correct', 'user_id', 'problem_id', 'skill_id', 'skill_name']).copy()

    # 🚨 GÜNCELLEME: Tüm sayısal sütunları burada float'a çeviriyoruz.
    # Bu, alt kümeler oluşturulduğunda (X_train/X_test) veri tipinin kalıcı olmasını sağlar.
    for col in NUMERICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(np.float64)

    # Kategorik Kodlama (One-Hot Encoding)
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, dummy_na=False)

    # user_id bazında bölme
    user_ids = df['user_id'].unique()
    train_users, test_users = train_test_split(user_ids, test_size=0.2, random_state=42)

    X_train = X[df['user_id'].isin(train_users)].copy()
    Y_train = Y[df['user_id'].isin(train_users)].copy()
    X_test = X[df['user_id'].isin(test_users)].copy()
    Y_test = Y[df['user_id'].isin(test_users)].copy()

    # Ölçeklendirme (StandardScaler)
    scaler = StandardScaler()

    # Sadece sayısal sütunları ölçekle (Artık bu sütunlar zaten float64)
    numeric_cols_to_scale = [col for col in X_train.columns if col in NUMERICAL_FEATURES]

    # Ölçekleme ve Atama (Veri tipleri zaten uyumlu olduğu için uyarı kesilecek)
    X_train.loc[:, numeric_cols_to_scale] = scaler.fit_transform(X_train.loc[:, numeric_cols_to_scale])
    X_test.loc[:, numeric_cols_to_scale] = scaler.transform(X_test.loc[:, numeric_cols_to_scale])

    feature_names = X_train.columns.tolist()
    print(f"✅ Eğitim/Test setleri oluşturuldu: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, Y_train, X_test, Y_test, scaler, feature_names


def train_and_save(X_train, Y_train, X_test, Y_test, scaler, feature_names):
    """Modeli eğitir, değerlendirir ve gerekli dosyaları kaydeder."""

    # Model Eğitimi
    model = LogisticRegression(solver='liblinear', C=1.0, penalty='l2', random_state=42)
    model.fit(X_train, Y_train)

    # Değerlendirme
    Y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(Y_test, Y_pred_proba)
    f1 = f1_score(Y_test, model.predict(X_test))

    print(f"\n📈 ROC-AUC Skoru: {roc_auc:.4f}")
    print(f"🎯 F1 Skoru: {f1:.4f}")

    # Kaydetme
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    pickle.dump(model, open(os.path.join(MODEL_OUTPUT_DIR, "logreg_model.pkl"), 'wb'))
    pickle.dump(scaler, open(os.path.join(MODEL_OUTPUT_DIR, "scaler.pkl"), 'wb'))
    json.dump(feature_names, open(os.path.join(MODEL_OUTPUT_DIR, "feature_names.json"), 'w'))
    json.dump(DIFFICULTY_POLICY, open(os.path.join(MODEL_OUTPUT_DIR, "difficulty_policy.json"), 'w'))

    print(f"✅ Model bileşenleri kaydedildi: {MODEL_OUTPUT_DIR}")


# ----------------------------------------------------------------------
# ANA ÇALIŞTIRMA BLOĞU
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Sprint 2: Model Eğitimi Başlıyor ---")

    if not os.path.exists(INPUT_FILE):
        print(f"HATA: İşlenmiş veri bulunamadı: {INPUT_FILE}. Lütfen data_preprocessing.py dosyasını çalıştırın.")
        raise SystemExit(1)

    df = pd.read_csv(INPUT_FILE)

    df_features = create_mastery_features(df)

    X_train, Y_train, X_test, Y_test, scaler, feature_names = prepare_data(df_features)

    train_and_save(X_train, Y_train, X_test, Y_test, scaler, feature_names)

    print("--- Model Eğitim Süreci Tamamlandı! ---")