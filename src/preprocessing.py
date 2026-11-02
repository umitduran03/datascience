"""Faz 2: Veri kalitesi, eksik veri enjeksiyonu ve sızıntısız ön işleme.

Şartname karşılığı (Bölüm 2.2, Bölüm 3 - Student 1, Rubrik 1):
- Outliers, duplicates, validity checks & structural anomalies
- En az %5 toplam NaN (doğal + sistematik enjeksiyon)
- Gelişmiş MVI (KNNImputer ve MICE / IterativeImputer) ile baseline
  (SimpleImputer: median/mode) karşılaştırması
- Encoding ve Feature Scaling'in data leakage olmadan uygulanması
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src import config

# --- Sütun grupları -------------------------------------------------------
NUMERIC_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Yüksek kardinaliteli kategorikler -> TargetEncoder + StandardScaler
HIGH_CARDINALITY_COLUMNS = [
    "diag_1", "diag_2", "diag_3", "medical_specialty", "payer_code",
]

# IDS_mapping.csv'de "NULL / Not Available / Not Mapped / Unknown"
# anlamına gelen kodlar: yapısal olarak gizlenmiş eksik değerlerdir.
UNKNOWN_ID_CODES = {
    "admission_type_id": [5, 6, 8],
    "discharge_disposition_id": [18, 25, 26],
    "admission_source_id": [9, 15, 17, 20, 21],
}
INVALID_GENDER = "Unknown/Invalid"

HIGH_MISSING_THRESHOLD = 0.50   # Bu oranın üstünde eksik olan sütun atılır
TARGET_MISSING_RATIO = 0.06     # Şart >= %5; güvenlik payıyla %6


# =========================================================================
# 1. Veri kalitesi (Duplicates, Outliers, Validity & Structural Anomalies)
# =========================================================================
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Kopya, geçerlilik ve yapısal anomali sayılarını raporlar."""
    features = df.drop(columns=config.ID_COLUMNS + [config.TARGET_RAW])
    nunique = features.nunique(dropna=False)
    missing = features.isna().mean()

    rows = [
        ("Tekrarlı encounter_id", int(df["encounter_id"].duplicated().sum())),
        ("Tam kopya satır (ID hariç)",
         int(df.drop(columns=config.ID_COLUMNS).duplicated().sum())),
        ("Geçersiz cinsiyet ('Unknown/Invalid')",
         int((df["gender"] == INVALID_GENDER).sum())),
    ]
    for col, codes in UNKNOWN_ID_CODES.items():
        rows.append((f"Bilinmeyen kod: {col} {codes}",
                     int(df[col].isin(codes).sum())))
    rows.append(("Sabit (tek değerli) sütun",
                 ", ".join(nunique[nunique == 1].index)))
    rows.append((f"%{HIGH_MISSING_THRESHOLD * 100:.0f} üstü eksik sütun",
                 ", ".join(missing[missing > HIGH_MISSING_THRESHOLD].index)))
    return pd.DataFrame(rows, columns=["Kontrol", "Sonuç"])


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Sayısal sütunlarda IQR (1.5 x IQR) yöntemiyle aykırı değer sayımı."""
    rows = []
    for col in NUMERIC_COLUMNS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append((col, low, high, n_out, n_out / len(df)))
    return pd.DataFrame(
        rows, columns=["Sütun", "Alt sınır", "Üst sınır", "Aykırı", "Oran"]
    )


def clean_records(df: pd.DataFrame) -> pd.DataFrame:
    """Satır bazlı kopya temizliği ve şema/geçerlilik dönüşümü uygular.

    Sütun istatistiği kullanmadığı için veri sızıntısı üretmez:
    kopyaları siler, geçersiz/bilinmeyen kodları NaN yapar, nominal ID
    sütunlarını metne çevirir ve ikili hedefi oluşturur.
    """
    df = df.copy()

    # Kopyalar
    df = df.drop_duplicates(subset="encounter_id")
    df = df.loc[~df.drop(columns=config.ID_COLUMNS).duplicated()]

    # Geçerlilik kontrolü: geçersiz / bilinmeyen kodlar -> NaN
    df["gender"] = df["gender"].replace(INVALID_GENDER, np.nan)
    for col, codes in UNKNOWN_ID_CODES.items():
        df[col] = df[col].where(~df[col].isin(codes))
        df[col] = df[col].map(lambda v: np.nan if pd.isna(v) else str(int(v)))

    # İkili hedef: Sınıf 1 = '<30', Sınıf 0 = diğerleri
    df[config.TARGET] = (
        df[config.TARGET_RAW] == config.POSITIVE_LABEL
    ).astype(int)
    return df.drop(columns=config.ID_COLUMNS + [config.TARGET_RAW])


def find_invalid_columns(df_train: pd.DataFrame) -> list:
    """Yalnızca eğitim seti istatistiklerine bakarak sabit (sıfır varyanslı)
    ve aşırı eksik (>%50) sütunları belirler.
    """
    features = df_train.drop(columns=[config.TARGET], errors="ignore")
    nunique = features.nunique(dropna=False)
    constant_cols = list(nunique[nunique <= 1].index)

    missing = features.isna().mean()
    high_missing_cols = list(
        missing[missing > HIGH_MISSING_THRESHOLD].index
    )
    return sorted(set(constant_cols + high_missing_cols))


def clean_data(df: pd.DataFrame,
               train_index: pd.Index = None) -> pd.DataFrame:
    """Satır temizliği yapar ve yalnızca eğitim indeksinden öğrenilen
    sabit / aşırı eksik sütunları çıkarır.
    """
    cleaned = clean_records(df)
    ref = cleaned.loc[train_index] if train_index is not None else cleaned
    drop_cols = find_invalid_columns(ref)
    return cleaned.drop(columns=drop_cols)


# =========================================================================
# 2. Sistematik eksik veri enjeksiyonu (>= %5 NaN şartı)
# =========================================================================
def missing_ratio(df: pd.DataFrame) -> float:
    """Hedef hariç tüm özellik hücrelerindeki toplam NaN oranı."""
    features = df.drop(columns=[config.TARGET], errors="ignore")
    return float(features.isna().sum().sum() / features.size)


def inject_missing(df: pd.DataFrame, target_ratio=TARGET_MISSING_RATIO,
                   columns=None, train_index: pd.Index = None,
                   random_state=config.RANDOM_STATE):
    """Toplam NaN oranı ``target_ratio``'ya ulaşacak şekilde seçili sayısal
    sütunlara tekrarlanabilir (MCAR) eksik veri enjekte eder.

    ``train_index`` verildiğinde enjeksiyon olasılığı yalnızca eğitim
    setindeki eksik veri oranından hesaplanır (sıfır veri sızıntısı).
    """
    columns = columns or NUMERIC_COLUMNS
    df_injected = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=columns)

    ref = df.loc[train_index] if train_index is not None else df
    ref_features = ref.drop(columns=[config.TARGET], errors="ignore")
    n_features = ref_features.shape[1]
    current_nans = ref_features.isna().sum().sum()
    needed = target_ratio * len(ref) * n_features - current_nans
    if needed <= 0:
        return df_injected, mask

    rate = needed / (len(ref) * len(columns))
    rng = np.random.default_rng(random_state)
    for col in columns:
        hit = rng.random(len(df)) < rate
        hit &= df[col].notna().to_numpy()
        mask[col] = hit
        df_injected.loc[hit, col] = np.nan
    return df_injected, mask


# =========================================================================
# 3. Sızıntısız (Leakage-Free) Ön İşleme ve MVI Boru Hattı
# =========================================================================
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Aykırı değerleri yalnızca eğitim setinden öğrenilen alt/üst
    yüzdeliklere kırpar (winsorization). NaN değerler korunur.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(df: pd.DataFrame, imputer: str = "mice") -> Pipeline:
    """Tüm dönüşümleri veri sızıntısı olmadan tek bir Pipeline'da toplar.

    Parameters
    ----------
    df : pd.DataFrame
        Özellik sütunlarını belirlemek için şablon veri çerçevesi.
    imputer : {'simple', 'knn', 'mice'}
        - 'simple': Baseline (sayısal median, kategorik most_frequent/mode)
        - 'knn'   : Gelişmiş MVI 1 — KNNImputer (n_neighbors=25)
        - 'mice'  : Gelişmiş MVI 2 — IterativeImputer / MICE (kodlanmış
                    tüm klinik özelliklerle iteratif regresyon)
    """
    if imputer not in {"simple", "knn", "mice"}:
        raise ValueError("imputer 'simple', 'knn' veya 'mice' olmalıdır.")

    features = [c for c in df.columns if c != config.TARGET]
    high_card = [c for c in HIGH_CARDINALITY_COLUMNS if c in features]
    low_card = [
        c for c in features
        if c not in NUMERIC_COLUMNS and c not in high_card
    ]

    num_steps = [
        ("clip", QuantileClipper()),
        ("scale", StandardScaler()),
    ]
    if imputer == "simple":
        num_steps.append(("impute", SimpleImputer(strategy="median")))
        cat_kwargs = {"strategy": "most_frequent"}
    elif imputer == "knn":
        num_steps.append(("impute", KNNImputer(n_neighbors=25)))
        cat_kwargs = {"strategy": "constant", "fill_value": "Missing"}
    else:
        # Kategorik eksikler 'Missing' göstergesiyle kodlanır; sayısal NaN'ler
        # tüm kodlanmış özellikler kullanılarak MICE adımında doldurulur.
        cat_kwargs = {"strategy": "constant", "fill_value": "Missing"}

    cv_gen = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=config.RANDOM_STATE
    )
    high_card_pipe = Pipeline([
        ("impute", SimpleImputer(**cat_kwargs)),
        ("encode", TargetEncoder(target_type="binary", cv=cv_gen)),
        ("scale", StandardScaler()),
    ])
    low_card_pipe = Pipeline([
        ("impute", SimpleImputer(**cat_kwargs)),
        ("encode", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False)),
    ])

    ct = ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_COLUMNS),
        ("high_card", high_card_pipe, high_card),
        ("low_card", low_card_pipe, low_card),
    ], verbose_feature_names_out=False)

    steps = [("ct", ct)]
    if imputer == "mice":
        steps.append((
            "mice",
            IterativeImputer(
                max_iter=10,
                skip_complete=True,
                random_state=config.RANDOM_STATE,
            ),
        ))
    return Pipeline(steps)


# =========================================================================
# 4. MVI Karşılaştırması (Baseline vs. MICE)
# =========================================================================
def evaluate_imputation(pipeline: Pipeline, X_true: pd.DataFrame,
                        X_missing: pd.DataFrame,
                        mask: pd.DataFrame) -> pd.DataFrame:
    """Enjekte edilen test hücrelerinde gerçek ve doldurulmuş değerleri
    (standartlaştırılmış ölçekte) RMSE ve MAE ile karşılaştırır.

    ``pipeline`` yalnızca X_train üzerinde fit edilmiş olmalıdır.
    """
    transformed = pipeline.transform(X_missing)
    imputed_num = transformed[:, :len(NUMERIC_COLUMNS)]

    num_pipe = pipeline.named_steps["ct"].named_transformers_["num"]
    scaler_only = Pipeline(num_pipe.steps[:2])
    true_scaled = scaler_only.transform(X_true[NUMERIC_COLUMNS])

    rows = []
    all_errors = []
    for j, col in enumerate(NUMERIC_COLUMNS):
        m = mask[col].to_numpy()
        err = imputed_num[m, j] - true_scaled[m, j]
        all_errors.append(err)
        rows.append((
            col,
            int(m.sum()),
            float(np.sqrt(np.mean(err ** 2))),
            float(np.mean(np.abs(err))),
        ))

    flat_err = np.concatenate(all_errors)
    rows.append((
        "TOPLAM",
        int(mask.to_numpy().sum()),
        float(np.sqrt(np.mean(flat_err ** 2))),
        float(np.mean(np.abs(flat_err))),
    ))
    return pd.DataFrame(rows, columns=["Sütun", "Enjekte NaN", "RMSE", "MAE"])


def transform_to_dataframe(pipeline: Pipeline,
                           X: pd.DataFrame) -> pd.DataFrame:
    """Fit edilmiş Pipeline çıktısını sütun isimli DataFrame'e dönüştürür."""
    arr = pipeline.transform(X)
    cols = list(pipeline.get_feature_names_out())
    return pd.DataFrame(arr, index=X.index, columns=cols)
