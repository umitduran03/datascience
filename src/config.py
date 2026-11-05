"""Proje genelinde kullanılan sabitler ve dosya yolları.

Tüm modüller yolları ve proje kısıtlarını buradan okur; böylece
eşik değerleri tek bir yerden yönetilir.
"""

import os
from pathlib import Path

# macOS OpenMP kütüphane çakışmasını engelle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Dizin yapısı ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- Veri kaynağı (UCI ML Repository, id=296) -----------------------------
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
RAW_CSV_PATH = RAW_DATA_DIR / "diabetic_data.csv"
IDS_MAPPING_PATH = RAW_DATA_DIR / "IDS_mapping.csv"

# --- Sütun rolleri --------------------------------------------------------
TARGET_RAW = "readmitted"          # Ham hedef: 'NO', '>30', '<30'
TARGET = "readmitted_30d"          # İkili hedef: 1 = '<30', 0 = diğerleri
POSITIVE_LABEL = "<30"
ID_COLUMNS = ["encounter_id", "patient_nbr"]

# Sayısal olarak saklanan ama aslında nominal (kod) olan sütunlar.
NOMINAL_CODE_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

# Veri setinde eksik değerler '?' ile kodlanmıştır.
# DİKKAT: 'None' (A1Cresult, max_glu_serum) eksik veri DEĞİL,
# "test yapılmadı" anlamına gelen geçerli bir kategoridir.
NA_VALUES = ["?"]

# --- Hocanın şartnamesindeki alt sınırlar ---------------------------------
MIN_ROWS = 20_000
MIN_FEATURES = 20
MIN_CATEGORICAL_RATIO = 0.30
MIN_IMBALANCE_RATIO = 3.0          # çoğunluk / azınlık >= 3  (1:3)
MIN_MISSING_RATIO = 0.05           # Faz 2'de sağlanacak (enjeksiyon dahil)

RANDOM_STATE = 42
