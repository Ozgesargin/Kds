import os
import pandas as pd
import numpy as np


# -------------------------------
# 1️⃣ CSV Dosyası Yükleme
# -------------------------------
def load_csv_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    print(f"✅ Dosya yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


# -------------------------------
# 2️⃣ Tekrarlı Verileri Silme
# -------------------------------
def remove_duplicates(df):
    df_clean = df.copy()
    before = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    after = df_clean.shape[0]
    print(f"✅ Tekrarlı veriler temizlendi: {before - after} satır silindi")
    return df_clean


# -------------------------------
# 3️⃣ Eksik Değerleri Temizleme
# -------------------------------
def remove_missing_values(df, critical_columns):
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=critical_columns)
    # correct null ise 0 ata
    if 'correct' in df_clean.columns:
        df_clean['correct'] = df_clean['correct'].fillna(0).astype(int)
    print(f"✅ Kritik sütunlarda eksik değerler temizlendi: {df_clean.shape[0]} satır kaldı")
    return df_clean


# -------------------------------
# 4️⃣ Mantık Hatalarını Düzeltme
# -------------------------------
def fix_logic_errors(df):
    df_clean = df.copy()

    # hint_count > hint_total
    mask = df_clean['hint_count'] > df_clean['hint_total']
    if mask.any():
        df_clean.loc[mask, 'hint_count'] = df_clean.loc[mask, 'hint_total']
        print(f"⚠ {mask.sum()} satırdaki hint_count > hint_total düzeltildi")

    # correct sadece 0 veya 1 olmalı
    if 'correct' in df_clean.columns:
        mask_invalid = ~df_clean['correct'].isin([0, 1])
        if mask_invalid.any():
            df_clean.loc[mask_invalid, 'correct'] = df_clean.loc[mask_invalid, 'correct'].apply(lambda x: 1 if x else 0)
            print(f"⚠ {mask_invalid.sum()} satırdaki correct değeri 0/1 formatına düzeltildi")

    return df_clean


# -------------------------------
# 5️⃣ ms_first_response Nötrleme
# -------------------------------
def normalize_response_time(df, max_seconds=3600):
    df_clean = df.copy()
    if 'ms_first_response' in df_clean.columns:
        df_clean['ms_first_response'] = pd.to_numeric(df_clean['ms_first_response'], errors='coerce').fillna(0)
        df_clean.loc[df_clean['ms_first_response'] > max_seconds, 'ms_first_response'] = max_seconds
    print("✅ ms_first_response nötrleme uygulandı")
    return df_clean


# -------------------------------
# 6️⃣ Hint Independence Özelliği
# -------------------------------
def add_hint_independence(df):
    df_clean = df.copy()
    df_clean['hint_independence'] = 1 - (df_clean['hint_count'] / df_clean['hint_total'].replace(0, 1))
    print("✅ hint_independence sütunu eklendi")
    return df_clean


# -------------------------------
# 7️⃣ Veri Tipi Dönüşümleri
# -------------------------------
def convert_data_types(df):
    df_clean = df.copy()
    str_cols = [
        'user_id', 'problem_id', 'template_id', 'skill_id', 'skill_name',
        'teacher_id', 'student_class_id', 'school_id'
    ]
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str)

    int_cols = ['attempt_count', 'ms_first_response', 'hint_count', 'hint_total', 'correct']
    for col in int_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

    return df_clean


# -------------------------------
# 8️⃣ Sıralı Veri Kontrolü
# -------------------------------
def check_sequential_order(df):
    df_sorted = df.sort_values(['user_id', 'problem_id', 'ms_first_response'])
    issues = 0
    grouped = df_sorted.groupby(['user_id', 'problem_id'])
    for _, group in grouped:
        if not group['ms_first_response'].is_monotonic_increasing:
            issues += 1
    if issues > 0:
        print(f"⚠ {issues} kullanıcı-problem kombinasyonunda ms_first_response sıralama hatası var")
    else:
        print("✅ ms_first_response sıralaması doğru")
    return df_sorted


# -------------------------------
# 9️⃣ Ana İşlem Fonksiyonu
# -------------------------------
def process_data(df):
    critical_columns = [
        'user_id', 'problem_id', 'template_id', 'skill_id', 'skill_name',
        'teacher_id', 'student_class_id', 'school_id', 'hint_count', 'hint_total'
    ]

    df = remove_duplicates(df)
    df = remove_missing_values(df, critical_columns)
    df = fix_logic_errors(df)
    df = normalize_response_time(df)
    df = add_hint_independence(df)
    df = convert_data_types(df)
    df = check_sequential_order(df)

    # Output df’de yalnızca işlediğimiz sütunlar kalsın
    keep_cols = [
        'user_id', 'problem_id', 'template_id', 'skill_id', 'skill_name',
        'teacher_id', 'student_class_id', 'school_id',
        'correct', 'attempt_count', 'ms_first_response',
        'hint_count', 'hint_total', 'hint_independence'
    ]
    df = df[keep_cols]

    return df


# -------------------------------
# 10️⃣ Ana Program
# -------------------------------
if __name__ == "__main__":
    input_path = os.path.join("data", "skill_builder_data_corrected_collapsed.csv")
    output_path = os.path.join("data", "output", "processed_skill_builder.csv")

    print("Assistments KDS veri ön işleme başlıyor...")
    print("Girdi dosyası:", input_path)

    try:
        df = load_csv_file(input_path)
    except FileNotFoundError as e:
        print("Hata:", e)
        raise SystemExit(1)

    df = process_data(df)

    print(f"⏬ Veri temizlendi ve işlendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(df.head())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ İşlenmiş veri kaydedildi: {output_path}")
