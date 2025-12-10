# app/database/db.py

import os
from typing import List, Dict, Any

import pyodbc
import pandas as pd

"""
Bu modül:
  - SQL Server'a bağlanır
  - Egitim.IslemKayitlari + Egitim.Ogrenciler + Egitim.Ogretmenler
    üzerinden JOIN ile etkileşim verisini çeker.

Ayrıca:
  - Evaluation sonuçlarını IslemKayitlari'na yazar
  - DeepMind Math sorularını Egitim.Sorular'a kaydedip
    Egitim.SoruAtamalari ile öğrencilere ödev olarak atar.
"""


# ----------------------------------------------------------------------
# 1. Bağlantı Fonksiyonu
# ----------------------------------------------------------------------
def get_db_connection():
    """
    SQL Server bağlantısı döndürür.

    Ortam değişkenleri:
      KDS_DB_DRIVER (varsayılan: ODBC Driver 17 for SQL Server)
      KDS_DB_SERVER (varsayılan: localhost)
      KDS_DB_NAME   (varsayılan: ASSISTMENT_DB)
      KDS_DB_USER / KDS_DB_PASSWORD (varsa SQL Auth, yoksa Windows Auth)
    """

    driver = os.getenv("KDS_DB_DRIVER", "ODBC Driver 17 for SQL Server")
    server = os.getenv("KDS_DB_SERVER", r"localhost")
    database = os.getenv("KDS_DB_NAME", "ASSISTMENT_DB")

    user = os.getenv("KDS_DB_USER")
    password = os.getenv("KDS_DB_PASSWORD")

    if user and password:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={user};PWD={password};"
        )
    else:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
        )

    return pyodbc.connect(conn_str)


# ----------------------------------------------------------------------
# 2. Etkileşim Verisini Çeken Fonksiyon
# ----------------------------------------------------------------------
def fetch_interaction_data_from_db(
    teacher_id: int | None = None,
    okul_id: int | None = None,
) -> pd.DataFrame:
    """
    Egitim.IslemKayitlari + Ogrenciler + Ogretmenler üzerinden join atar.

    Dönen DataFrame en az şu sütunları içerir:
      - user_id        : ogrenci_id
      - skill_name     : konu/beceri adı
      - correct        : 0/1
      - attempt_count
      - ms_first_response
      - hint_count
      - hint_total

    teacher_id verilirse = sadece o öğretmenin öğrencileri,
    okul_id verilirse   = ayrıca o okula göre de filtreler.
    """

    conn = get_db_connection()

    base_query = """
    SELECT
        ik.ogrenci_id      AS user_id,
        ik.skill_id,
        ik.skill_name,
        ik.correct,
        ik.attempt_count,
        ik.ms_first_response,
        ik.hint_count,
        ik.hint_total
    FROM Egitim.IslemKayitlari AS ik
    INNER JOIN Egitim.Ogrenciler AS o
        ON ik.ogrenci_id = o.ogrenci_id
    INNER JOIN Egitim.Ogretmenler AS og
        ON o.ogretmen_id = og.ogretmen_id
    WHERE 1 = 1
    """

    params: list = []

    if teacher_id is not None:
        base_query += " AND og.ogretmen_id = ?"
        params.append(teacher_id)

    if okul_id is not None:
        base_query += " AND og.okul_id = ?"
        params.append(okul_id)

    df = pd.read_sql_query(base_query, conn, params=params or None)
    conn.close()

    # Temizlik / tip düzeltme
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(float)

    if "hint_count" not in df.columns:
        df["hint_count"] = 0

    return df


# ----------------------------------------------------------------------
# 3. Evaluation Sonuçlarını DB'ye Yazmak (Eski fonksiyonlar)
# ----------------------------------------------------------------------
def insert_evaluation_results(df: pd.DataFrame) -> int:
    """
    Evaluation ekranında öğretmenin yüklediği sonuçları
    Egitim.IslemKayitlari tablosuna ekler.

    Beklenen kolonlar:
      - user_id
      - skill_id
      - skill_name
      - correct
      - attempt_count
      - ms_first_response
      - hint_count
      - hint_total

    Dönüş:
      - Başarıyla eklenen satır sayısı
    """
    if df is None or df.empty:
        return 0

    required_cols = ["user_id", "skill_id", "skill_name", "correct"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Veritabanına yazmak için eksik kolonlar var: " + ", ".join(missing)
        )

    conn = get_db_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO Egitim.IslemKayitlari
            (ogrenci_id, skill_id, skill_name,
             correct, attempt_count, ms_first_response,
             hint_count, hint_total)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    inserted = 0
    for _, row in df.iterrows():
        ogrenci_id = int(row["user_id"])
        skill_id = int(row["skill_id"])
        skill_name = str(row["skill_name"])
        correct = float(row["correct"])

        attempt_count = int(row["attempt_count"]) if "attempt_count" in df.columns else 1
        ms_first_response = float(row["ms_first_response"]) if "ms_first_response" in df.columns else 0.0
        hint_count = int(row["hint_count"]) if "hint_count" in df.columns else 0
        hint_total = int(row["hint_total"]) if "hint_total" in df.columns else hint_count

        cursor.execute(
            insert_sql,
            ogrenci_id,
            skill_id,
            skill_name,
            correct,
            attempt_count,
            ms_first_response,
            hint_count,
            hint_total,
        )
        inserted += 1

    conn.commit()
    conn.close()

    return inserted


def save_evaluation_results_to_db(df: pd.DataFrame) -> None:
    """
    Evaluation ekranında modelden çıkan sonuç DataFrame'ini
    Egitim.IslemKayitlari tablosuna ekler.

    Beklenen kolonlar:
      - user_id
      - skill_id
      - skill_name
      - correct
      - attempt_count
      - ms_first_response
      - hint_count
      - hint_total

    Not: ogrenci_id = user_id varsayımı ile çalışır.
    """
    if df.empty:
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO Egitim.IslemKayitlari
            (ogrenci_id, skill_id, skill_name, correct,
             attempt_count, ms_first_response, hint_count, hint_total)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    for _, row in df.iterrows():
        ogrenci_id = int(row.get("user_id"))
        skill_id = row.get("skill_id")
        skill_name = row.get("skill_name")
        correct = int(row.get("correct", 0))
        attempt_count = int(row.get("attempt_count", 1))
        ms_first_response = float(row.get("ms_first_response", 0))
        hint_count = int(row.get("hint_count", 0))
        hint_total = int(row.get("hint_total", 0))

        cursor.execute(
            insert_sql,
            ogrenci_id,
            skill_id,
            skill_name,
            correct,
            attempt_count,
            ms_first_response,
            hint_count,
            hint_total,
        )

    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# 4. Ödev: Soru kaydetme + atama
# ----------------------------------------------------------------------
def save_homework_questions_for_student(
    teacher_id: int,
    student_id: int,
    questions: List[Dict[str, Any]],
) -> None:
    """
    QuestionService'ten gelen soruları:
      1) Egitim.Sorular tablosuna yazar
      2) Egitim.SoruAtamalari tablosuna atama kaydı açar
    """
    if not questions or teacher_id is None or student_id is None:
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    # Öğretmenin okul_id'sini bul
    okul_id = None
    try:
        cursor.execute(
            "SELECT okul_id FROM Egitim.Ogretmenler WHERE ogretmen_id = ?",
            (teacher_id,),
        )
        row = cursor.fetchone()
        if row:
            okul_id = row.okul_id
    except Exception as e:
        print(f"Uyarı: Ogretmen okul_id bulunamadı: {e}")

    # OUTPUT INSERTED.soru_id ile tek statement'ta identity alıyoruz
    insert_soru_sql = """
        INSERT INTO Egitim.Sorular
            (skill_id, konu_adi, zorluk, metin, olusturan_ogretmen_id, created_at, dogru_cevap)
        OUTPUT INSERTED.soru_id
        VALUES (?, ?, ?, ?, ?, GETDATE(), ?);
    """

    insert_atama_sql = """
        INSERT INTO Egitim.SoruAtamalari
            (soru_id, ogrenci_id, ogretmen_id, okul_id, atama_tarihi, durum)
        VALUES (?, ?, ?, ?, GETDATE(), ?);
    """

    for q in questions:
        question_text = str(q.get("question_text", "")).strip()
        correct_answer = str(q.get("correct_answer", "")).strip()
        difficulty_text = str(q.get("difficulty_text", "Belirtilmemiş"))
        skill_tag = q.get("skill_tag") or "Genel"

        if not question_text:
            continue

        # skill_id NOT NULL ise dummy 0 kullanıyoruz
        skill_id = 0
        konu_adi = skill_tag

        try:
            # Soru kaydı + ID'yi alma
            cursor.execute(
                insert_soru_sql,
                skill_id,
                konu_adi,
                difficulty_text,
                question_text,
                teacher_id,
                correct_answer,
            )
            soru_id_row = cursor.fetchone()
            if not soru_id_row or soru_id_row[0] is None:
                print("Uyarı: INSERTED.soru_id alınamadı.")
                continue

            soru_id = int(soru_id_row[0])

            durum = 0  # 0 = atanmış
            cursor.execute(
                insert_atama_sql,
                soru_id,
                student_id,
                teacher_id,
                okul_id,
                durum,
            )
        except Exception as e:
            print(f"HATA: Ödev sorusu kaydedilemedi: {e}")
            continue

    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# 5. Ödevleri listeleme
# ----------------------------------------------------------------------
def fetch_homework_for_student(student_id: int) -> List[Dict[str, Any]]:
    """
    Egitim.SoruAtamalari + Egitim.Sorular join'i ile
    bir öğrenciye atanmış soruları çeker.
    """
    if student_id is None:
        return []

    conn = get_db_connection()
    query = """
        SELECT
            sa.atama_id,
            sa.atama_tarihi,
            sa.durum,
            s.soru_id,
            s.metin,
            s.zorluk,
            s.created_at,
            s.dogru_cevap
        FROM Egitim.SoruAtamalari AS sa
        INNER JOIN Egitim.Sorular AS s
            ON sa.soru_id = s.soru_id
        WHERE sa.ogrenci_id = ?
        ORDER BY sa.atama_tarihi DESC, sa.atama_id DESC;
    """

    df = pd.read_sql_query(query, conn, params=[student_id])
    conn.close()

    if df.empty:
        return []

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        results.append(
            {
                "assignment_id": int(row["atama_id"]),
                "question_id": int(row["soru_id"]),
                "question_text": str(row["metin"]),
                "correct_answer": str(row.get("dogru_cevap") or ""),
                "difficulty_text": str(row.get("zorluk") or ""),
                "created_at": row.get("atama_tarihi"),
                "status": int(row.get("durum", 0)),
            }
        )

    return results
