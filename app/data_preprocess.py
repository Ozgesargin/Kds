import os
import pandas as pd
import numpy as np

# --- 1️⃣ Sabitler ve Yollar (Kök Dizine Göre Ayarlandı) ---
# data/skill_builder_data.csv dosyasının projenin kök dizinine göre yolu
INPUT_PATH = os.path.join("../app/data", "skill_builder_data.csv")
OUTPUT_DIR = os.path.join("../app/data", "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "processed_skill_builder.csv")

# --- 2️⃣ Kritik Sütunlar ---
CRITICAL_COLS = [
    'user_id', 'problem_id', 'skill_id',
    'correct', 'ms_first_response'
]

# ----------------------------------------------------------------------
# Fonksiyonlar
# ----------------------------------------------------------------------

def load_data(path):
    """Veriyi yükler ve ilk kontrolü yapar."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"HATA: Ham veri dosyası bulunamadı: {path}")
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    print(f"✅ Dosya yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


def clean_and_prepare(df):
    """Veriyi temizler, aykırı değerleri yönetir ve özellik ön hazırlığı yapar."""
    df_clean = df.copy()

    # Temizlik Adımı
    subset_cols = ['user_id', 'problem_id', 'ms_first_response', 'correct']
    df_clean = df_clean.drop_duplicates(subset=subset_cols, keep='first')
    df_clean = df_clean.dropna(subset=CRITICAL_COLS)

    # Correct (Doğru/Yanlış) Değerlerinin 0/1 Olmasını Sağlama
    if 'correct' in df_clean.columns:
        df_clean['correct'] = pd.to_numeric(df_clean['correct'], errors='coerce').fillna(0).astype(int)
        df_clean.loc[~df_clean['correct'].isin([0, 1]), 'correct'] = 0

    # hint_count > hint_total düzeltme
    if 'hint_count' in df_clean.columns and 'hint_total' in df_clean.columns:
        mask = df_clean['hint_count'] > df_clean['hint_total']
        df_clean['hint_count'] = np.where(mask, df_clean['hint_total'], df_clean['hint_count'])

    # Zaman Aykırı Değer Yönetimi (Clipping)
    if 'ms_first_response' in df_clean.columns:
        df_clean['ms_first_response'] = pd.to_numeric(df_clean['ms_first_response'], errors='coerce').fillna(0)
        lower_bound = df_clean['ms_first_response'].quantile(0.005)
        upper_bound = df_clean['ms_first_response'].quantile(0.995)
        df_clean['ms_first_response'] = np.clip(df_clean['ms_first_response'], a_min=lower_bound, a_max=upper_bound)
        df_clean.loc[df_clean['ms_first_response'] < 100, 'ms_first_response'] = 100  # Minimum cevap süresi

    # Hint Independence Özelliği
    if 'hint_count' in df_clean.columns and 'hint_total' in df_clean.columns:
        # Hata Giderildi: Hint bağımsızlık skoru
        df_clean['hint_independence'] = 1 - (df_clean['hint_count'] / df_clean['hint_total'].replace(0, 1))

    # ID'leri string yap
    for col in ['user_id', 'problem_id', 'skill_id']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)

    # KRİTİK: Kronolojik Sıralama (Mastery Score hesaplaması için)
    df_clean = df_clean.sort_values(
        ['user_id', 'ms_first_response', 'problem_id'],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return df_clean


# ----------------------------------------------------------------------
# ANA ÇALIŞTIRMA BLOĞU
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Sprint 1: Veri Ön İşleme Başlıyor ---")

    try:
        df = load_data(INPUT_PATH)
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        raise SystemExit(1)

    df_processed = clean_and_prepare(df)

    # Modelleme için gerekli sütunları seçme
    keep_cols = [
        'user_id', 'problem_id', 'skill_id', 'skill_name',
        'correct', 'attempt_count', 'ms_first_response',
        'hint_count', 'hint_total', 'hint_independence'
    ]
    df_final = df_processed[[col for col in keep_cols if col in df_processed.columns]]

    # Kaydetme
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ İşlenmiş veri kaydedildi: {df_final.shape[0]} satır. -> {OUTPUT_PATH}")