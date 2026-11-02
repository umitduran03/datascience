"""Faz 1: Veri setinin proje şartnamesindeki alt sınırları sağladığını
kodla doğrular.

Kontroller (Bölüm 2.1 ve 2.3)
-----------------------------
1. Satır sayısı       >= 20.000
2. Özellik sayısı     >= 20
3. Kategorik oran     >= %30
4. Sınıf dengesizliği >= 1:3  (Sınıf 1: '<30', Sınıf 0: diğerleri)
"""

import pandas as pd

from src import config
from src.data_loader import add_binary_target, load_raw_data


def get_feature_columns(df: pd.DataFrame) -> list:
    """ID ve hedef sütunlarını dışarıda bırakarak özellik listesini döner."""
    excluded = set(config.ID_COLUMNS) | {config.TARGET_RAW, config.TARGET}
    return [col for col in df.columns if col not in excluded]


def get_categorical_columns(df: pd.DataFrame, features: list) -> list:
    """Metin tipindeki ve nominal kod (ID) sütunlarını kategorik sayar."""
    text_cols = [
        col for col in features if not pd.api.types.is_numeric_dtype(df[col])
    ]
    code_cols = [c for c in config.NOMINAL_CODE_COLUMNS if c in features]
    return text_cols + code_cols


def check_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Dört alt sınırı kontrol eder ve sonuç tablosunu döner."""
    df = add_binary_target(df)
    features = get_feature_columns(df)
    categorical = get_categorical_columns(df, features)

    n_rows = len(df)
    n_features = len(features)
    cat_ratio = len(categorical) / n_features

    class_counts = df[config.TARGET].value_counts()
    imbalance = class_counts.max() / class_counts.min()

    rows = [
        ("Satır sayısı", n_rows, f">= {config.MIN_ROWS:,}",
         n_rows >= config.MIN_ROWS),
        ("Özellik sayısı", n_features, f">= {config.MIN_FEATURES}",
         n_features >= config.MIN_FEATURES),
        ("Kategorik oran", f"{cat_ratio:.1%}",
         f">= {config.MIN_CATEGORICAL_RATIO:.0%}",
         cat_ratio >= config.MIN_CATEGORICAL_RATIO),
        ("Dengesizlik (Sınıf 1 : Sınıf 0)", f"1:{imbalance:.2f}",
         f">= 1:{config.MIN_IMBALANCE_RATIO:.0f}",
         imbalance >= config.MIN_IMBALANCE_RATIO),
    ]
    return pd.DataFrame(
        rows, columns=["Kriter", "Değer", "Şart", "Sağlandı"]
    )


def main() -> None:
    """Doğrulamayı çalıştırır ve sonuçları yazdırır."""
    df = load_raw_data()
    report = check_constraints(df)
    print(report.to_string(index=False))

    assert report["Sağlandı"].all(), "Kısıtlardan biri sağlanmadı!"
    print("\nDört kısıtın tamamı sağlandı.")


if __name__ == "__main__":
    main()
