"""Faz 5: Derin Tablo Mimarisi TabNet (pytorch-tabnet), Dikkat Maskeleri,
Kayıp Eğrisi Optimizasyonu ve Karşılaştırmalı Analiz.

Şartname karşılığı (Bölüm 2.5, Bölüm 3 - Student 3, Rubrik 3, 4 ve 5):
- Kural 1 (Metrik Yasağı): Accuracy KESİNLİKLE kullanılmaz; modeller
  yalnızca Macro F1-Score, Weighted F1-Score, ROC-AUC ve PR-AUC
  üzerinden değerlendirilir.
- Kural 2 (Model 3 — Deep Tabular TabNet): pytorch-tabnet kütüphanesi
  (TabNetClassifier) kullanılarak derin öğrenme mimarisi kurulur. Nd, Na
  adımları ve mask_type (sparsemax / entmax) optimize edilir; eğitim ve
  doğrulama kayıp eğrileri (loss curves) takip edilir.
- Kural 3 (Attention & Explain): tabnet.explain() mekanizmasıyla seyrek
  dikkat maskeleri (sparse attention masks) ve küresel özellik önemleri
  (feature_importances_) çıkarılır.
- Kural 4 (Kod Kalitesi): PEP-8 ve Flake8 uyumlu, modüler ve canlı jüri kod
  savunmasına hazır temiz fonksiyon yapısı.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import torch

from src import config

TABNET_OUTPUTS_PATH = config.PROCESSED_DATA_DIR / "tabnet_outputs.pkl"
METRIC_COLUMNS = ["Macro_F1", "Weighted_F1", "ROC_AUC", "PR_AUC"]


# =========================================================================
# 1. Metrik Değerlendirme Yardımcıları (Accuracy Kesinlikle Yasak!)
# =========================================================================
def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Şartnamede zorunlu kılınan 4 dengesiz veri metriğini hesaplar.

    Accuracy (Doğruluk) metriği sınıf dengesizliğinde aşırı yanıltıcı
    olduğu için şartname gereği kesinlikle hesaplanmaz ve raporlanmaz.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    return {
        "Macro_F1": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "Weighted_F1": float(
            f1_score(
                y_true_arr, y_pred_arr, average="weighted", zero_division=0
            )
        ),
        "ROC_AUC": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "PR_AUC": float(average_precision_score(y_true_arr, y_prob_arr)),
    }


def normalized_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Satır bazlı (True Class) normalize edilmiş karmaşıklık matrisi."""
    return confusion_matrix(y_true, y_pred, normalize="true")


# =========================================================================
# 2. TabNetClassifier Oluşturma ve Eğitme
# =========================================================================
def build_tabnet_classifier(
    n_d: int = 16,
    n_a: int = 16,
    n_steps: int = 3,
    gamma: float = 1.3,
    mask_type: str = "sparsemax",
    lambda_sparse: float = 1e-3,
    learning_rate: float = 2e-2,
    random_state: int = config.RANDOM_STATE,
    verbose: int = 0,
) -> TabNetClassifier:
    """Hocanın şartnamesine uygun parametrelerle TabNetClassifier örneği kurar.

    Parameters
    ----------
    n_d : int
        Karar tahmin katmanının (decision prediction layer) boyutu.
    n_a : int
        Her adımdaki dikkat gömme katmanının (attention embedding) boyutu.
    n_steps : int
        Mimari içindeki ardışık karar ve maskeleme adımı sayısı.
    gamma : float
        Önceki adımlarda seçilen özelliklerin tekrar kullanımını cezalandıran
        ve seyrekliği artıran katsayı.
    mask_type : {'sparsemax', 'entmax'}
        Dikkat haritası aktivasyon fonksiyonu.
    lambda_sparse : float
        Seyreklik düzenlileştirme (sparsity regularization) ağırlığı.
    learning_rate : float
        Adam optimizer için başlangıç öğrenme oranı.
    random_state : int
        Tekrarlanabilirlik tohumu.
    verbose : int
        Konsol çıktı ayrıntısı.
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    return TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        n_independent=2,
        n_shared=2,
        mask_type=mask_type,
        lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=learning_rate),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.9),
        seed=random_state,
        verbose=verbose,
    )


def fit_tabnet(
    model: TabNetClassifier,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray = None,
    y_val: pd.Series | np.ndarray = None,
    max_epochs: int = 25,
    patience: int = 7,
    batch_size: int = 2048,
    virtual_batch_size: int = 256,
) -> TabNetClassifier:
    """TabNet modelini dengesiz sınıf ağırlıkları (weights=1) ve erken durdurma
    (early stopping) ile eğitir.
    """
    X_tr_arr = np.asarray(X_train, dtype=np.float32)
    y_tr_arr = np.asarray(y_train, dtype=int)

    if X_val is not None and y_val is not None:
        eval_set = [
            (np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=int))
        ]
        eval_name = ["val"]
    else:
        eval_set = [(X_tr_arr, y_tr_arr)]
        eval_name = ["train"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train=X_tr_arr,
            y_train=y_tr_arr,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=["auc"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            weights=1,
            drop_last=False,
        )
    return model


def predict_tabnet(
    model: TabNetClassifier,
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """TabNet'ten sınıf tahmini (y_pred) ve pozitif olasılık (y_prob) döner."""
    X_arr = np.asarray(X, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = np.asarray(model.predict(X_arr), dtype=int)
        y_prob = np.asarray(model.predict_proba(X_arr)[:, 1], dtype=float)
    return y_pred, y_prob


# =========================================================================
# 3. Mimari İnce Ayar (Nd/Na ve Mask Type Karşılaştırması)
# =========================================================================
def evaluate_tabnet_architectures(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    configs: list[dict] = None,
    val_size: float = 0.20,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, TabNetClassifier, dict]:
    """Farklı n_d/n_a ve mask_type (sparsemax vs entmax) ayarlarını
    yalnızca eğitim seti içinde veri sızıntısız doğrulamayla karşılaştırır.
    """
    if configs is None:
        configs = [
            {
                "n_d": 16, "n_a": 16, "mask_type": "sparsemax",
                "label": "Nd=16, Na=16 (Sparsemax)",
            },
            {
                "n_d": 16, "n_a": 16, "mask_type": "entmax",
                "label": "Nd=16, Na=16 (Entmax)",
            },
            {
                "n_d": 8, "n_a": 8, "mask_type": "sparsemax",
                "label": "Nd=8, Na=8 (Sparsemax)",
            },
        ]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state,
    )

    results = []
    best_score = -1.0
    best_model = None
    best_cfg = None

    for cfg in configs:
        print(f"  -> Eğitiliyor: {cfg['label']}...", flush=True)
        t_cfg = time.time()
        model = build_tabnet_classifier(
            n_d=cfg["n_d"],
            n_a=cfg["n_a"],
            mask_type=cfg["mask_type"],
            random_state=random_state,
            verbose=0,
        )
        fit_tabnet(
            model,
            X_tr,
            y_tr,
            X_val=X_val,
            y_val=y_val,
            max_epochs=15,
            patience=5,
            batch_size=2048,
            virtual_batch_size=256,
        )
        y_val_pred, y_val_prob = predict_tabnet(model, X_val)
        metrics = evaluate_predictions(y_val, y_val_pred, y_val_prob)
        print(
            f"  -> Tamamlandı: {cfg['label']} ({time.time() - t_cfg:.1f} s) "
            f"| Val ROC-AUC: {metrics['ROC_AUC']:.4f} | "
            f"PR-AUC: {metrics['PR_AUC']:.4f}",
            flush=True,
        )

        # Bileşik seçim skoru (Macro F1 + PR-AUC + ROC-AUC)
        composite_score = (
            0.50 * metrics["Macro_F1"]
            + 0.25 * metrics["PR_AUC"]
            + 0.25 * metrics["ROC_AUC"]
        )
        results.append({
            "Yapılandırma": cfg["label"],
            "n_d": cfg["n_d"],
            "n_a": cfg["n_a"],
            "mask_type": cfg["mask_type"],
            "Bileşik_Skor": composite_score,
            **metrics,
        })

        if composite_score > best_score:
            best_score = composite_score
            best_model = model
            best_cfg = cfg

    res_df = pd.DataFrame(results).sort_values(
        by="Bileşik_Skor", ascending=False
    ).reset_index(drop=True)

    return res_df, best_model, best_cfg


# =========================================================================
# 4. Attention Maskeleri ve Özellik Önemleri (Explain Mekanizması)
# =========================================================================
def extract_tabnet_explain(
    model: TabNetClassifier,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str],
) -> dict:
    """tabnet.explain() mekanizmasını işleterek dikkat matrislerini,
    adım bazlı maskeleri ve küresel özellik önemlerini döner.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explain_matrix, masks = model.explain(X_arr)

    # masks sözlüğü: {0: step_0_mask, 1: step_1_mask, ...}
    step_weights = {}
    for step_idx, mask in masks.items():
        step_weights[f"Step_{step_idx + 1}"] = mask.mean(axis=0)

    step_df = pd.DataFrame(step_weights, index=feature_names)

    # Küresel özellik önemleri (TabNet built-in feature_importances_)
    global_imp = model.feature_importances_
    total = float(global_imp.sum()) or 1.0
    imp_df = pd.DataFrame({
        "Özellik": feature_names,
        "Önem_Skoru": global_imp,
        "Önem_Oranı": global_imp / total,
        "Önem_%": (global_imp / total) * 100.0,
    }).sort_values(by="Önem_Skoru", ascending=False).reset_index(drop=True)

    return {
        "explain_matrix": explain_matrix,
        "masks": masks,
        "step_masks_df": step_df,
        "feature_importances": imp_df,
    }


def aggregate_tabnet_importances(
    feat_imp_df: pd.DataFrame,
    raw_columns: list[str],
) -> pd.DataFrame:
    """One-hot ve target-encoded sütun önemlerini orijinal klinik değişken
    isimleri altında gruplayarak toplam önemlerini hesaplar.
    """
    grouped_rows = []
    for raw_col in raw_columns:
        sub = feat_imp_df[
            feat_imp_df["Özellik"].str.startswith(f"{raw_col}_")
            | (feat_imp_df["Özellik"] == raw_col)
        ]
        raw_score = float(sub["Önem_Skoru"].sum())
        grouped_rows.append({"Özellik": raw_col, "Önem_Skoru": raw_score})

    agg_df = pd.DataFrame(grouped_rows)
    total = float(agg_df["Önem_Skoru"].sum()) or 1.0
    agg_df["Önem_Oranı"] = agg_df["Önem_Skoru"] / total
    agg_df["Önem_%"] = agg_df["Önem_Oranı"] * 100.0
    return agg_df.sort_values(
        by="Önem_Skoru", ascending=False
    ).reset_index(drop=True)


# =========================================================================
# 5. Çıktı Kayıt ve Yükleme
# =========================================================================
def save_tabnet_outputs(
    outputs: dict,
    path: Path = None,
) -> Path:
    """TabNet modelini, tahminlerini ve görselleştirme çıktılarını kaydeder."""
    target_path = Path(path) if path is not None else TABNET_OUTPUTS_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(outputs, target_path)
    return target_path


def load_tabnet_outputs(path: Path = None) -> dict:
    """Diskten önceden kaydedilmiş TabNet çıktılarını yükler."""
    target_path = Path(path) if path is not None else TABNET_OUTPUTS_PATH
    return pd.read_pickle(target_path)
