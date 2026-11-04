"""Faz 4: Referans Perceptron, Ağaç Tabanlı GBDT (LightGBM / XGBoost) ve
Optuna ile Sızıntısız (Leakage-Free) Hiperparametre Optimizasyonu.

Şartname karşılığı (Bölüm 2.5, Bölüm 3 - Student 2, Rubrik 3 ve 4):
- Kural 1 (Metrik Yasağı): Accuracy KESİNLİKLE kullanılmaz; yalnızca
  Macro F1-Score, Weighted F1-Score, ROC-AUC ve PR-AUC raporlanır.
- Kural 2 (Model 1 — Baseline Perceptron): Farklı öğrenme oranları (eta0)
  ve sınıf ağırlıkları (class_weight) Stratified K-Fold CV ile yalnızca
  X_train üzerinde taranır; decision_function çıktıları sigmoid (expit)
  ile [0, 1] olasılık/skor aralığına dönüştürülür.
- Kural 3 (Model 2 — Tree-Based GBDT & Optuna): LightGBM ve XGBoost
  modelleri karşılaştırılır; Optuna + StratifiedKFold ile yalnızca
  X_train üzerinde hiperparametre optimizasyonu yapılır ve en iyi modelin
  özellik önemleri (feature_importances_) çıkarılır.
- Kural 4 (Faz 5 Entegrasyonu): Faz 5'teki zorunlu grafikler (Grafik 4, 5, 6)
  için y_pred, y_prob ve özellik önemleri data/processed/faz4_outputs.pkl
  dosyasına kaydedilir.
"""

from pathlib import Path
import warnings

from lightgbm import LGBMClassifier
import numpy as np
import optuna
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src import config
from src import preprocessing as pp

optuna.logging.set_verbosity(optuna.logging.WARNING)

FAZ4_OUTPUTS_PATH = config.PROCESSED_DATA_DIR / "faz4_outputs.pkl"
METRIC_COLUMNS = ["Macro_F1", "Weighted_F1", "ROC_AUC", "PR_AUC"]

DEFAULT_ETA0_GRID = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
DEFAULT_CLASS_WEIGHTS = ["balanced", None]


# =========================================================================
# 1. Ortak Değerlendirme Yardımcıları (Accuracy Kesinlikle Yasak!)
# =========================================================================
def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Şartnamede zorunlu kılınan 4 dengesiz veri metriğini hesaplar.

    Accuracy (Doğruluk) metriği dengesiz veride yanıltıcı olduğu için
    şartname gereği kesinlikle hesaplanmaz ve raporlanmaz.
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
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Faz 5'teki 6. zorunlu grafik (Confusion Matrix Grid) için satır bazlı
    normalize edilmiş karmaşıklık matrisini döner.
    """
    return confusion_matrix(y_true, y_pred, normalize="true")


# =========================================================================
# 2. Model 1 — Referans (Baseline) Perceptron ve Öğrenme Oranı Taraması
# =========================================================================
def perceptron_probabilities(
    model: Perceptron,
    X: pd.DataFrame,
) -> np.ndarray:
    """Perceptron decision_function skorlarını lojistik sigmoid (expit) ile
    monotonik olarak (0, 1) olasılık/skor aralığına dönüştürür.

    Bu dönüşüm sıralamayı birebir koruduğu için ROC-AUC ve PR-AUC
    değerlerini değiştirmez ve karar sınırını (z=0 <-> p=0.5) korur.
    """
    raw_scores = model.decision_function(np.asarray(X, dtype=float))
    return np.asarray(expit(raw_scores), dtype=float)


def build_perceptron(
    eta0: float = 0.01,
    class_weight: str = "balanced",
    max_iter: int = 1000,
    tol: float = 1e-3,
    random_state: int = config.RANDOM_STATE,
) -> Perceptron:
    """Verilen öğrenme oranı (eta0) ve sınıf ağırlığı ile Perceptron kurar."""
    return Perceptron(
        eta0=eta0,
        class_weight=class_weight,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )


def tune_perceptron(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    eta0_grid: list = None,
    class_weights: list = None,
    n_splits: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> tuple:
    """Farklı öğrenme oranlarını (eta0) ve sınıf ağırlıklarını (class_weight)
    yalnızca X_train üzerinde Stratified K-Fold CV ile karşılaştırır.

    Veri sızıntısını önlemek için X_test bu aşamada kesinlikle kullanılmaz.
    En yüksek CV Macro_F1 + PR_AUC dengesini sağlayan model tüm X_train
    üzerinde yeniden eğitilerek döndürülür.
    """
    eta0_grid = eta0_grid or DEFAULT_ETA0_GRID
    class_weights = class_weights or DEFAULT_CLASS_WEIGHTS

    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=int)
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    rows = []
    for cw in class_weights:
        cw_label = "balanced" if cw == "balanced" else "None (Ağırlıksız)"
        for eta0 in eta0_grid:
            fold_metrics = []
            for tr_idx, val_idx in skf.split(X_arr, y_arr):
                clf = build_perceptron(
                    eta0=eta0,
                    class_weight=cw,
                    random_state=random_state,
                )
                clf.fit(X_arr[tr_idx], y_arr[tr_idx])
                y_val_pred = clf.predict(X_arr[val_idx])
                y_val_prob = perceptron_probabilities(clf, X_arr[val_idx])
                fold_metrics.append(
                    evaluate_predictions(
                        y_arr[val_idx], y_val_pred, y_val_prob
                    )
                )

            fold_df = pd.DataFrame(fold_metrics)
            rows.append({
                "eta0": float(eta0),
                "class_weight": cw_label,
                "raw_class_weight": cw,
                "CV_Macro_F1": float(fold_df["Macro_F1"].mean()),
                "CV_Macro_F1_Std": float(fold_df["Macro_F1"].std()),
                "CV_Weighted_F1": float(fold_df["Weighted_F1"].mean()),
                "CV_ROC_AUC": float(fold_df["ROC_AUC"].mean()),
                "CV_PR_AUC": float(fold_df["PR_AUC"].mean()),
            })

    tuning_df = pd.DataFrame(rows)
    tuning_df["Seçim_Skoru"] = (
        tuning_df["CV_Macro_F1"]
        + tuning_df["CV_ROC_AUC"]
        + tuning_df["CV_PR_AUC"]
    ) / 3.0
    tuning_df = tuning_df.sort_values(
        by=["Seçim_Skoru", "CV_PR_AUC", "CV_Macro_F1"],
        ascending=False,
    ).reset_index(drop=True)

    best_row = tuning_df.iloc[0]
    best_params = {
        "eta0": float(best_row["eta0"]),
        "class_weight": best_row["raw_class_weight"],
    }

    best_model = build_perceptron(
        eta0=best_params["eta0"],
        class_weight=best_params["class_weight"],
        random_state=random_state,
    )
    best_model.fit(X_arr, y_arr)

    display_df = tuning_df.drop(columns=["raw_class_weight"])
    return best_model, display_df, best_params


# =========================================================================
# 3. Model 2 — Ağaç Tabanlı GBDT (LightGBM / XGBoost) ve Optuna
# =========================================================================
def build_baseline_lightgbm(
    scale_pos_weight: float = 1.0,
    random_state: int = config.RANDOM_STATE,
) -> LGBMClassifier:
    """Varsayılan hiperparametrelere sahip referans LightGBM modeli kurar."""
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_baseline_xgboost(
    scale_pos_weight: float = 1.0,
    random_state: int = config.RANDOM_STATE,
) -> XGBClassifier:
    """Varsayılan hiperparametrelere sahip referans XGBoost modeli kurar."""
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def compare_gbdt_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """LightGBM ve XGBoost adaylarını yalnızca X_train üzerinde Stratified
    K-Fold CV ile karşılaştırarak Optuna için temel GBDT mimarisini doğrular.
    """
    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=int)
    pos_ratio = float((y_arr == 0).sum() / max((y_arr == 1).sum(), 1))

    candidates = {
        "LightGBM (Varsayılan - Dengeli)": lambda: build_baseline_lightgbm(
            scale_pos_weight=pos_ratio, random_state=random_state
        ),
        "XGBoost (Varsayılan - Dengeli)": lambda: build_baseline_xgboost(
            scale_pos_weight=pos_ratio, random_state=random_state
        ),
    }

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    rows = []
    for name, builder in candidates.items():
        fold_metrics = []
        for tr_idx, val_idx in skf.split(X_arr, y_arr):
            clf = builder()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_arr[tr_idx], y_arr[tr_idx])
                y_val_pred = clf.predict(X_arr[val_idx])
                y_val_prob = clf.predict_proba(X_arr[val_idx])[:, 1]
            fold_metrics.append(
                evaluate_predictions(y_arr[val_idx], y_val_pred, y_val_prob)
            )

        fold_df = pd.DataFrame(fold_metrics)
        rows.append({
            "Model_Adayı": name,
            "CV_Macro_F1": float(fold_df["Macro_F1"].mean()),
            "CV_Weighted_F1": float(fold_df["Weighted_F1"].mean()),
            "CV_ROC_AUC": float(fold_df["ROC_AUC"].mean()),
            "CV_PR_AUC": float(fold_df["PR_AUC"].mean()),
        })
    return pd.DataFrame(rows)


def optimize_gbdt_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    n_splits: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> tuple:
    """LightGBM hiperparametrelerini Optuna (TPE Sampler) ve StratifiedKFold
    kullanarak yalnızca X_train üzerinde optimize eder (sıfır veri sızıntısı).

    Parameters
    ----------
    X_train : pd.DataFrame
        İşlenmiş eğitim özellikleri (81.412 x 154).
    y_train : pd.Series
        İkili eğitim hedefi (readmitted_30d).
    n_trials : int
        Optuna deneme sayısı.
    n_splits : int
        Stratified K-Fold katlama sayısı.

    Returns
    -------
    best_model : LGBMClassifier
        En iyi hiperparametrelerle tüm X_train üzerinde eğitilmiş model.
    best_params : dict
        Optuna tarafından bulunan en iyi hiperparametre sözlüğü.
    trials_df : pd.DataFrame
        Tüm denemelerin parametre ve CV metrik geçmişi.
    study : optuna.Study
        Optuna çalışma nesnesi.
    """
    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=int)
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 150, 400, step=50
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.12, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 30, 200
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-3, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10.0, log=True
            ),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 4.5, 7.5
            ),
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_metrics = []
        for tr_idx, val_idx in skf.split(X_arr, y_arr):
            model = LGBMClassifier(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_arr[tr_idx], y_arr[tr_idx])
                y_val_pred = model.predict(X_arr[val_idx])
                y_val_prob = model.predict_proba(X_arr[val_idx])[:, 1]
            fold_metrics.append(
                evaluate_predictions(y_arr[val_idx], y_val_pred, y_val_prob)
            )

        fold_df = pd.DataFrame(fold_metrics)
        mean_macro_f1 = float(fold_df["Macro_F1"].mean())
        mean_weighted_f1 = float(fold_df["Weighted_F1"].mean())
        mean_roc_auc = float(fold_df["ROC_AUC"].mean())
        mean_pr_auc = float(fold_df["PR_AUC"].mean())

        trial.set_user_attr("CV_Macro_F1", mean_macro_f1)
        trial.set_user_attr("CV_Weighted_F1", mean_weighted_f1)
        trial.set_user_attr("CV_ROC_AUC", mean_roc_auc)
        trial.set_user_attr("CV_PR_AUC", mean_pr_auc)

        # Hem azınlık sınıfı F1 başarısını (Macro F1) hem de olasılık sıralama
        # kalitesini (PR-AUC ve ROC-AUC) birlikte maksimize eden bileşik hedef
        return 0.50 * mean_macro_f1 + 0.25 * mean_pr_auc + 0.25 * mean_roc_auc

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {
        **study.best_params,
        "subsample_freq": 1,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    best_model = LGBMClassifier(**best_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X_arr, y_arr)

    trial_rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        trial_rows.append({
            "Trial": int(t.number),
            "Hedef_Skor": float(t.value),
            "CV_Macro_F1": float(t.user_attrs["CV_Macro_F1"]),
            "CV_Weighted_F1": float(t.user_attrs["CV_Weighted_F1"]),
            "CV_ROC_AUC": float(t.user_attrs["CV_ROC_AUC"]),
            "CV_PR_AUC": float(t.user_attrs["CV_PR_AUC"]),
            **t.params,
        })
    trials_df = pd.DataFrame(trial_rows).sort_values(
        by="Hedef_Skor", ascending=False
    ).reset_index(drop=True)

    return best_model, best_params, trials_df, study


def predict_gbdt(
    model,
    X: pd.DataFrame,
) -> tuple:
    """Eğitilmiş GBDT modelinden sınıf tahmini (y_pred) ve pozitif sınıf
    olasılığı (y_prob) üretir.
    """
    X_arr = np.asarray(X, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = np.asarray(model.predict(X_arr), dtype=int)
        y_prob = np.asarray(model.predict_proba(X_arr)[:, 1], dtype=float)
    return y_pred, y_prob


# =========================================================================
# 4. GBDT Özellik Önemleri (Faz 5 Grafik 4: TabNet vs. GBDT İçin)
# =========================================================================
def extract_feature_importances(
    model,
    feature_names: list,
) -> pd.DataFrame:
    """Eğitilmiş ağaç modelinin feature_importances_ değerlerini ham,
    normalize (toplamı 1) ve yüzde (%) olarak sıralı DataFrame'e dönüştürür.
    """
    raw_imp = np.asarray(model.feature_importances_, dtype=float)
    total = float(raw_imp.sum()) or 1.0
    norm_imp = raw_imp / total

    df_imp = pd.DataFrame({
        "Özellik": list(feature_names),
        "Önem_Skoru": raw_imp,
        "Önem_Oranı": norm_imp,
        "Önem_%": norm_imp * 100.0,
    })
    return df_imp.sort_values(
        by="Önem_Skoru", ascending=False
    ).reset_index(drop=True)


def aggregate_feature_importances(
    encoded_imp_df: pd.DataFrame,
    raw_columns: list = None,
) -> pd.DataFrame:
    """154 kodlanmış (One-Hot + Sürekli) özelliğin önem skorlarını 42 ana
    klinik özelliğe gruplayarak toplar.

    Faz 5'te TabNet maskeleri ister 154 sütun ister 42 ana özellik bazında
    çizilsin, her iki seviye de hazır olarak sunulur.
    """
    if raw_columns is None:
        raw_columns = list(pp.NUMERIC_COLUMNS) + list(
            pp.HIGH_CARDINALITY_COLUMNS
        )

    known_continuous = set(pp.NUMERIC_COLUMNS) | set(
        pp.HIGH_CARDINALITY_COLUMNS
    )
    sorted_raw = sorted(raw_columns, key=len, reverse=True)

    def map_to_parent(col_name: str) -> str:
        if col_name in known_continuous:
            return col_name
        for parent in sorted_raw:
            if col_name == parent or col_name.startswith(f"{parent}_"):
                return parent
        return col_name.rsplit("_", 1)[0]

    grouped = encoded_imp_df.copy()
    grouped["Ana_Özellik"] = grouped["Özellik"].map(map_to_parent)
    agg_df = (
        grouped.groupby("Ana_Özellik", as_index=False)["Önem_Skoru"]
        .sum()
        .rename(columns={"Ana_Özellik": "Özellik"})
    )
    total = float(agg_df["Önem_Skoru"].sum()) or 1.0
    agg_df["Önem_Oranı"] = agg_df["Önem_Skoru"] / total
    agg_df["Önem_%"] = agg_df["Önem_Oranı"] * 100.0
    return agg_df.sort_values(
        by="Önem_Skoru", ascending=False
    ).reset_index(drop=True)


# =========================================================================
# 5. Faz 4 Karşılaştırma Tablosu ve Faz 5 Çıktı Kayıt / Yükleme
# =========================================================================
def evaluate_models_on_test(
    models_dict: dict,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Birden fazla modelin X_test tahminlerini tek bir karşılaştırma
    tablosunda (Macro_F1, Weighted_F1, ROC_AUC, PR_AUC) birleştirir.
    """
    rows = []
    for model_name, payload in models_dict.items():
        metrics = evaluate_predictions(
            y_test, payload["y_pred"], payload["y_prob"]
        )
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows)


def save_faz4_outputs(
    outputs: dict,
    path: Path = None,
) -> Path:
    """Faz 5'te Yunus Emre'nin çizeceği 4., 5. ve 6. zorunlu grafikler için
    Perceptron ve optimize GBDT çıktılarını diske kaydeder.
    """
    target_path = Path(path) if path is not None else FAZ4_OUTPUTS_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(outputs, target_path)
    return target_path


def load_faz4_outputs(path: Path = None) -> dict:
    """Faz 5 modüllerinin ve notebook'unun Faz 4 tahminlerini, olasılıklarını
    ve GBDT özellik önemlerini doğrudan okuyabilmesini sağlar.
    """
    target_path = Path(path) if path is not None else FAZ4_OUTPUTS_PATH
    return pd.read_pickle(target_path)


def get_faz5_plot_payload(path: Path = None) -> dict:
    """Faz 5'teki 4., 5. ve 6. zorunlu grafik fonksiyonlarının doğrudan
    kullanabileceği özet sözlüğü döner.
    """
    data = load_faz4_outputs(path=path)
    return {
        "y_test": data["y_test"],
        "models": {
            "Perceptron (Baseline)": {
                "y_pred": data["perceptron"]["y_pred"],
                "y_prob": data["perceptron"]["y_prob"],
                "metrics": data["perceptron"]["metrics"],
                "confusion_matrix_norm": data["perceptron"][
                    "confusion_matrix_norm"
                ],
            },
            "LightGBM (Optuna GBDT)": {
                "y_pred": data["gbdt"]["y_pred"],
                "y_prob": data["gbdt"]["y_prob"],
                "metrics": data["gbdt"]["metrics"],
                "confusion_matrix_norm": data["gbdt"][
                    "confusion_matrix_norm"
                ],
            },
        },
        "gbdt_feature_importances_154": data["gbdt"]["feature_importances"],
        "gbdt_feature_importances_42": data["gbdt"][
            "feature_importances_grouped"
        ],
    }
