"""Veri setini indirme ve okuma yardımcıları."""

import io
import urllib.request
import zipfile

import pandas as pd

from src import config


def download_dataset(force: bool = False) -> None:
    """UCI'dan Diabetes 130-US Hospitals veri setini indirip açar.

    Parameters
    ----------
    force : bool
        True ise dosya mevcut olsa bile yeniden indirir.
    """
    if config.RAW_CSV_PATH.exists() and not force:
        print(f"Veri zaten mevcut: {config.RAW_CSV_PATH}")
        return

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"İndiriliyor: {config.DATASET_URL}")
    with urllib.request.urlopen(config.DATASET_URL) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(config.RAW_DATA_DIR)
    print(f"Çıkarıldı: {config.RAW_DATA_DIR}")


def load_raw_data() -> pd.DataFrame:
    """Ham CSV'yi doğru eksik değer tanımıyla okur.

    ``keep_default_na=False`` zorunludur: pandas varsayılan olarak
    'None' metnini NaN sayar; bu veri setinde 'None' geçerli bir
    klinik kategoridir (test yapılmadı). Yalnızca '?' eksik kabul edilir.
    """
    if not config.RAW_CSV_PATH.exists():
        download_dataset()

    return pd.read_csv(
        config.RAW_CSV_PATH,
        na_values=config.NA_VALUES,
        keep_default_na=False,
        low_memory=False,
    )


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """'<30' gün içinde yeniden yatışı 1, diğerlerini 0 olarak kodlar."""
    df = df.copy()
    df[config.TARGET] = (
        df[config.TARGET_RAW] == config.POSITIVE_LABEL
    ).astype(int)
    return df


if __name__ == "__main__":
    download_dataset()
