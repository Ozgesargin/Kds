# app/database/data_service.py

import pandas as pd
from .db import fetch_interaction_data_from_db


class DataService:
    """
    Karar Destek Sistemi için ham verileri işleyen ve dashboard / öğrenciler
    sayfasına gönderilecek özet veriyi üreten servis.

    Veri kaynağı:
      -> fetch_interaction_data_from_db(teacher_id, okul_id)
         (SQL Server'dan gerçek veriyi çeker)
    """

    def __init__(self, teacher_id: int | None = None, okul_id: int | None = None):
        """
        teacher_id / okul_id:
          - Giriş yapan öğretmene göre filtrelemek için kullanılır.
          - teacher_id None ise: tüm öğretmenlerin öğrencileri (gerekirse).
        """
        self.teacher_id = teacher_id
        self.okul_id = okul_id
        self.df: pd.DataFrame = fetch_interaction_data_from_db(
            teacher_id=teacher_id,
            okul_id=okul_id,
        )

        if self.df.empty:
            print("WARN: DataService - Etkileşim verisi boş geldi.")
        else:
            print(
                f"INFO: DataService başlatıldı. "
                f"{self.df['user_id'].nunique()} öğrenci, {len(self.df)} etkileşim yüklendi."
            )

    # --------------------------------------------------
    # Yardımcı Hesaplama Fonksiyonları
    # --------------------------------------------------
    def _calculate_student_performance(self) -> list[dict]:
        """
        Her öğrenci (user_id) için:
          - id           : Öğrenci ID (ogrenci_id)
          - name         : 'Öğrenci <id>'  (şimdilik DB'de isim olmadığı için)
          - recent_score : Doğruluk ortalaması (%)
          - hint_avg     : Ortalama ipucu sayısı
        """

        df = self.df
        if df.empty:
            return []

        grouped = df.groupby("user_id").agg(
            recent_score=("correct", "mean"),
            hint_avg=("hint_count", "mean"),
        ).reset_index()

        grouped["recent_score"] = (grouped["recent_score"] * 100).round(1)
        grouped["hint_avg"] = grouped["hint_avg"].round(1)

        students: list[dict] = []
        for _, row in grouped.iterrows():
            uid = int(row["user_id"])
            students.append(
                {
                    "id": uid,
                    "name": f"Öğrenci {uid}",   # İleride isim kolonu gelirse burada map'lersin
                    "recent_score": float(row["recent_score"]),
                    "hint_avg": float(row["hint_avg"]),
                }
            )

        # En riskli öğrenciler en üstte
        students.sort(key=lambda x: x["recent_score"])
        return students

    def _calculate_topic_difficulty(self) -> list[dict]:
        """
        Her konu (skill_name) için:
          - accuracy      : Ortalama doğruluk (%)
          - struggle_level: 100 - accuracy
        """

        df = self.df
        if df.empty:
            return []

        topic_group = df.groupby("skill_name").agg(
            accuracy=("correct", "mean")
        ).reset_index()

        topic_group["accuracy"] = (topic_group["accuracy"] * 100).round(1)
        topic_group["struggle_level"] = (100 - topic_group["accuracy"]).round(1)

        topics: list[dict] = []
        for _, row in topic_group.iterrows():
            topics.append(
                {
                    "name": row["skill_name"],
                    "accuracy": float(row["accuracy"]),
                    "struggle_level": float(row["struggle_level"]),
                }
            )

        topics.sort(key=lambda x: x["struggle_level"], reverse=True)
        return topics

    # --------------------------------------------------
    # Dışarı Açılan Fonksiyonlar
    # --------------------------------------------------
    def get_student_performance(self) -> list[dict]:
        """ /students sayfasında kullanılan liste. """
        return self._calculate_student_performance()

    def predict_and_aggregate_data(self) -> dict:
        """
        Dashboard için gereken özet veri.
        teacher_dashboard view'ında 'data' olarak gönderiyoruz.
        """
        df = self.df

        if df.empty:
            return {
                "class_name": "Veri Yok",
                "teacher_name": "Bilinmeyen Öğretmen",
                "class_overview_accuracy": 0.0,
                "struggling_topics": [],
                "top_struggling_students": [],
                "total_students": 0,
            }

        all_students = self._calculate_student_performance()
        top_struggling_students = all_students[:3]
        struggling_topics = self._calculate_topic_difficulty()

        if all_students:
            class_overview_accuracy = round(
                sum(s["recent_score"] for s in all_students) / len(all_students), 1
            )
        else:
            class_overview_accuracy = 0.0

        total_students = df["user_id"].nunique()

        return {
            "class_name": "Sınıf (DB)",         # İleride gerçek sınıf adı da çekilebilir
            "teacher_name": "DB Öğretmen",      # Template zaten session'daki adı kullanıyor
            "class_overview_accuracy": class_overview_accuracy,
            "struggling_topics": struggling_topics,
            "top_struggling_students": top_struggling_students,
            "total_students": total_students,
        }
