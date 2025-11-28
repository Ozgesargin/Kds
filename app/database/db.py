# app/database/db.py

import os
import pyodbc
import pandas as pd

"""
Bu modül:
  - SQL Server'a bağlanır
  - Egitim.IslemKayitlari + Egitim.Ogrenciler + Egitim.Ogretmenler
    üzerinden JOIN ile etkileşim verisini çeker.

fetch_interaction_data_from_db(teacher_id, okul_id):
  -> Dashboard ve Öğrenciler sayfasında kullanacağımız DataFrame'i döndürür.
"""


# ----------------------------------------------------------------------
# 1. Bağlantı Fonksiyonu
# ----------------------------------------------------------------------
def get_db_connection():
    """
    SQL Server bağlantısı döndürür.

    İKİ SEÇENEK:
    1) Windows Authentication (En kolayı, SSMS'te de bu varsa):
       - Aşağıdaki server / database adını kendine göre düzenle
       - Kullanıcı/şifre vermeden çalışır.

    2) SQL kullanıcı adı/şifre ile:
       - Ortam değişkenleri ile veya conn_str'i düzenleyerek kullan.
    """

    driver = os.getenv("KDS_DB_DRIVER", "ODBC Driver 17 for SQL Server")
    server = os.getenv("KDS_DB_SERVER", r"localhost")  # BURAYI GEREKİRSE DEĞİŞTİR
    database = os.getenv("KDS_DB_NAME", "ASSISTMENT_DB")

    # Eğer kullanıcı adı/şifre verildiyse SQL Authentication kullan
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
        # Windows Authentication
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
def fetch_interaction_data_from_db(teacher_id: int | None = None,
                                   okul_id: int | None = None) -> pd.DataFrame:
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

    # Bazı güvenlik/temizlikler
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(float)

    if "hint_count" not in df.columns:
        df["hint_count"] = 0

    return df
