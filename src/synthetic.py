"""Faz 3: İki farklı teknikle (SMOTE ve CTGAN) sentetik veri üretimi,
istatistiksel kalite ölçümü (KS Testi, Wasserstein Mesafesi, Korelasyon Farkı)
ve makine öğrenmesi fayda testleri (TSTR vs. TRTR).

Şartname karşılığı (Bölüm 2.4, Bölüm 3 - Student 1, Rubrik 2 ve 4):
- İki farklı sentetik veri tekniği: imblearn (SMOTE) ve sdv (CTGAN)
- İstatistiksel sadakat: Kolmogorov-Smirnov Testi, Wasserstein Mesafesi
  ve |Corr_Real - Corr_Synthetic| korelasyon farkı özeti
- ML fayda testi: TSTR vs. TRTR (ve 1:1 dengelenmiş eğitim setleri)
- Metrik kuralı: Accuracy KULLANILMAZ; yalnızca Macro F1, Weighted F1,
  ROC-AUC ve PR-AUC raporlanır.
"""

import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp, wasserstein_distance
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src import config
from src import preprocessing as pp

CONTINUOUS_COLUMNS = pp.NUMERIC_COLUMNS + pp.HIGH_CARDINALITY_COLUMNS
TABLE_NAME = "patients"


# =========================================================================
# 1. Teknik 1 — SMOTE (imbalanced-learn)
# =========================================================================
def generate_smote_datasets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_tstr_per_class: int = 9000,
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """SMOTE ile hem 1:1 sınıf dengeleme hem de %100 saf sentetik TSTR
    veri seti üretir.

    TSTR setinde hiçbir gerçek hasta satırı yer almaz; hem Sınıf 0 hem
    Sınıf 1 örnekleri yalnızca SMOTE interpolasyonu ile üretilen yeni
    sentetik satırlardan oluşur.
    """
    # A) Eğitim setini 1:1 dengeleme (Sınıf 1 -> Sınıf 0 sayısına tamamlanır)
    smote_bal = SMOTE(random_state=random_state)
    X_bal, y_bal = smote_bal.fit_resample(X_train, y_train)
    X_syn_pos = X_bal.iloc[len(X_train):].reset_index(drop=True)

    # B) TSTR için saf sentetik Sınıf 0 üretimi (gerçek satır kopyalamadan)
    X_neg_real = X_train[y_train == 0]
    X_pos_real = X_train[y_train == 1]
    n_seed_neg = min(len(X_pos_real) // 2, len(X_neg_real))
    X_neg_seed = X_neg_real.sample(n=n_seed_neg, random_state=random_state)

    X_temp = pd.concat([X_neg_seed, X_pos_real], ignore_index=True)
    y_temp = pd.Series(
        [0] * len(X_neg_seed) + [1] * len(X_pos_real),
        name=config.TARGET,
    )
    smote_neg = SMOTE(
        sampling_strategy={0: len(X_neg_seed) + n_tstr_per_class},
        random_state=random_state,
    )
    X_temp_res, _ = smote_neg.fit_resample(X_temp, y_temp)
    X_syn_neg_tstr = X_temp_res.iloc[len(X_temp):].reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    pos_idx = rng.choice(len(X_syn_pos), size=n_tstr_per_class, replace=False)
    X_syn_pos_tstr = X_syn_pos.iloc[pos_idx].reset_index(drop=True)

    X_tstr = pd.concat([X_syn_neg_tstr, X_syn_pos_tstr], ignore_index=True)
    y_tstr = pd.Series(
        [0] * len(X_syn_neg_tstr) + [1] * len(X_syn_pos_tstr),
        name=config.TARGET,
    )
    return {
        "X_balanced": X_bal,
        "y_balanced": y_bal,
        "X_syn_pos": X_syn_pos,
        "X_tstr": X_tstr,
        "y_tstr": y_tstr,
    }


# =========================================================================
# 2. Teknik 2 — CTGAN (SDV: Synthetic Data Vault)
# =========================================================================
def _build_compact_train(
    X_train: pd.DataFrame,
    df_missing: pd.DataFrame,
) -> tuple:
    """154 sütunlu matrisi CTGAN için yapısal olarak geçerli 42 sütunlu
    (13 sürekli + 29 kategorik) tabloya dönüştürür.

    Böylece CTGAN her kategorik değişken için karşılıklı dışlayıcı
    (mutually exclusive) tek bir kategori üretir; örnekleme sonrasında
    tekrar birebir 154 sütunlu uzaya açılır.
    """
    low_card = [
        c for c in df_missing.columns
        if c != config.TARGET and c not in CONTINUOUS_COLUMNS
    ]
    cat_train = (
        df_missing.loc[X_train.index, low_card]
        .fillna("Missing")
        .astype(str)
    )
    compact = pd.concat([X_train[CONTINUOUS_COLUMNS], cat_train], axis=1)
    return compact, low_card


def _expand_to_encoded_space(
    df_compact: pd.DataFrame,
    low_card: list,
    target_columns: pd.Index,
) -> pd.DataFrame:
    """42 sütunlu CTGAN çıktısını 154 sütunlu model uzayına dönüştürür."""
    ohe_cols = [c for c in target_columns if c not in CONTINUOUS_COLUMNS]
    dummies = pd.get_dummies(df_compact[low_card], dtype=float)
    dummies = dummies.reindex(columns=ohe_cols, fill_value=0.0)
    combined = pd.concat(
        [df_compact[CONTINUOUS_COLUMNS].astype(float), dummies], axis=1
    )
    return combined[target_columns]


def _build_sdv_metadata(low_card: list) -> Metadata:
    """SDV 2.x Metadata nesnesini oluşturur."""
    cols = {c: {"sdtype": "numerical"} for c in CONTINUOUS_COLUMNS}
    for c in low_card:
        cols[c] = {"sdtype": "categorical"}
    return Metadata.load_from_dict({
        "tables": {TABLE_NAME: {"columns": cols}}
    })


def generate_ctgan_datasets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df_missing: pd.DataFrame,
    fit_rows_per_class: int = 6000,
    epochs: int = 50,
    n_tstr_per_class: int = 9000,
    random_state: int = config.RANDOM_STATE,
) -> dict:
    """SDV CTGANSynthesizer ile hem 1:1 sınıf dengeleme hem de %100 saf
    sentetik TSTR veri seti üretir.

    Yalnızca eğitim seti (X_train) kullanılır; test setine hiç dokunulmaz.
    """
    compact_train, low_card = _build_compact_train(X_train, df_missing)
    metadata = _build_sdv_metadata(low_card)

    pos_real = compact_train[y_train == 1]
    neg_real = compact_train[y_train == 0]
    pos_fit = pos_real.sample(
        n=min(fit_rows_per_class, len(pos_real)), random_state=random_state
    )
    neg_fit = neg_real.sample(
        n=min(fit_rows_per_class, len(neg_real)), random_state=random_state
    )

    ctgan_kwargs = {
        "epochs": epochs,
        "batch_size": 500,
        "generator_dim": (128, 128),
        "discriminator_dim": (128, 128),
        "enable_gpu": False,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen_pos = CTGANSynthesizer(metadata, **ctgan_kwargs)
        gen_pos.fit({TABLE_NAME: pos_fit})

        gen_neg = CTGANSynthesizer(metadata, **ctgan_kwargs)
        gen_neg.fit({TABLE_NAME: neg_fit})

        n_need_pos = int((y_train == 0).sum() - (y_train == 1).sum())
        raw_pos_bal = gen_pos.sample(num_rows=n_need_pos)[TABLE_NAME]
        raw_pos_tstr = gen_pos.sample(num_rows=n_tstr_per_class)[TABLE_NAME]
        raw_neg_tstr = gen_neg.sample(num_rows=n_tstr_per_class)[TABLE_NAME]

    X_syn_pos = _expand_to_encoded_space(
        raw_pos_bal, low_card, X_train.columns
    ).reset_index(drop=True)
    X_pos_tstr = _expand_to_encoded_space(
        raw_pos_tstr, low_card, X_train.columns
    ).reset_index(drop=True)
    X_neg_tstr = _expand_to_encoded_space(
        raw_neg_tstr, low_card, X_train.columns
    ).reset_index(drop=True)

    X_bal = pd.concat([X_train.reset_index(drop=True), X_syn_pos],
                      ignore_index=True)
    y_bal = pd.Series(
        np.concatenate([
            y_train.to_numpy(),
            np.ones(len(X_syn_pos), dtype=int),
        ]),
        name=config.TARGET,
    )

    X_tstr = pd.concat([X_neg_tstr, X_pos_tstr], ignore_index=True)
    y_tstr = pd.Series(
        [0] * len(X_neg_tstr) + [1] * len(X_pos_tstr),
        name=config.TARGET,
    )
    return {
        "X_balanced": X_bal,
        "y_balanced": y_bal,
        "X_syn_pos": X_syn_pos,
        "X_tstr": X_tstr,
        "y_tstr": y_tstr,
    }


# =========================================================================
# 3. İstatistiksel Kalite Testleri (KS Testi, Wasserstein, Korelasyon Farkı)
# =========================================================================
def evaluate_statistical_fidelity(
    X_real: pd.DataFrame,
    X_synthetic: pd.DataFrame,
    columns: list = None,
) -> pd.DataFrame:
    """Sütun bazında Kolmogorov-Smirnov (KS) istatistiği, KS Tamamlayıcı
    skoru (1 - KS) ve Wasserstein-1 mesafesini hesaplar.
    """
    columns = columns or CONTINUOUS_COLUMNS
    rows = []
    for col in columns:
        r_vals = X_real[col].to_numpy()
        s_vals = X_synthetic[col].to_numpy()
        ks_res = ks_2samp(r_vals, s_vals)
        w_dist = wasserstein_distance(r_vals, s_vals)
        rows.append((
            col,
            float(ks_res.statistic),
            float(1.0 - ks_res.statistic),
            float(ks_res.pvalue),
            float(w_dist),
        ))

    df_res = pd.DataFrame(
        rows,
        columns=[
            "Özellik",
            "KS_İstatistiği",
            "KS_Benzerlik_(1-KS)",
            "KS_p_değeri",
            "Wasserstein_Mesafesi",
        ],
    )
    mean_row = pd.DataFrame([(
        "ORTALAMA",
        df_res["KS_İstatistiği"].mean(),
        df_res["KS_Benzerlik_(1-KS)"].mean(),
        np.nan,
        df_res["Wasserstein_Mesafesi"].mean(),
    )], columns=df_res.columns)
    return pd.concat([df_res, mean_row], ignore_index=True)


def correlation_difference_summary(
    X_real: pd.DataFrame,
    X_synthetic: pd.DataFrame,
    columns: list = None,
) -> dict:
    """|Corr_Real - Corr_Synthetic| mutlak fark matrisinin ortalama ve
    maksimum değerlerini döner.
    """
    columns = columns or CONTINUOUS_COLUMNS
    corr_real = X_real[columns].corr()
    corr_syn = X_synthetic[columns].corr()
    delta = (corr_real - corr_syn).abs()
    mask = ~np.eye(len(columns), dtype=bool)
    off_diag = delta.to_numpy()[mask]
    return {
        "Ortalama_|ΔCorr|": float(np.mean(off_diag)),
        "Medyan_|ΔCorr|": float(np.median(off_diag)),
        "Maksimum_|ΔCorr|": float(np.max(off_diag)),
    }


# =========================================================================
# 4. Makine Öğrenmesi Fayda Testleri (TSTR vs. TRTR — Accuracy Yasak!)
# =========================================================================
def evaluate_ml_utility(
    datasets: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """TRTR, TSTR (SMOTE / CTGAN) ve 1:1 Dengelenmiş (Real + Synthetic)
    senaryoları gerçek test seti (X_test) üzerinde karşılaştırır.

    Şartname gereği Accuracy kullanılmaz; yalnızca Macro F1, Weighted F1,
    ROC-AUC ve PR-AUC raporlanır.
    """
    learners = {
        "LogisticRegression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
    }

    rows = []
    for scenario_name, (X_tr, y_tr) in datasets.items():
        for model_name, base_clf in learners.items():
            clf = base_clf.__class__(**base_clf.get_params())
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_test)[:, 1]
            pred = clf.predict(X_test)
            rows.append((
                scenario_name,
                model_name,
                len(X_tr),
                float(np.mean(y_tr)),
                float(f1_score(y_test, pred, average="macro")),
                float(f1_score(y_test, pred, average="weighted")),
                float(roc_auc_score(y_test, prob)),
                float(average_precision_score(y_test, prob)),
            ))

    return pd.DataFrame(
        rows,
        columns=[
            "Senaryo",
            "Model",
            "Eğitim_Satır",
            "Pozitif_Oran",
            "Macro_F1",
            "Weighted_F1",
            "ROC_AUC",
            "PR_AUC",
        ],
    )
