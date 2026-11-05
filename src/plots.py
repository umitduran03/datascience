"""Faz 5: Proje Şartnamesindeki 6 Zorunlu Teşhis ve Karşılaştırma Grafiği.

Şartname karşılığı (Bölüm 2.4, Bölüm 2.5, Bölüm 3 - Student 3, Rubrik 4):
1. 01_missingness_heatmap_dendrogram.png: missingno ile eksik veri örüntüsü
   ısı haritası ve hiyerarşik kümeleme dendrogramı.
2. 02_real_vs_synthetic_distributions.png: En önemli özellikler için Gerçek,
   SMOTE ve CTGAN üst üste binmiş KDE/Histogram dağılımları.
3. 03_correlation_difference_matrix.png: Gerçek ve sentetik veri arasındaki
   yapısal korumayı kanıtlayan |Corr_Real - Corr_Synthetic| fark matrisi.
4. 04_tabnet_masks_vs_gbdt_importance.png: TabNet attention maskeleri ile
   GBDT özellik önemlerinin karşılaştırmalı görseli.
5. 05_combined_roc_and_pr_curves.png: 3 modelin (Perceptron, GBDT, TabNet)
   birleşik ROC-AUC ve Precision-Recall (PR-AUC) eğrileri.
6. 06_normalized_confusion_matrix_grid.png: 3 modelin yan yana normalize
   edilmiş Karmaşıklık Matrisleri Grid'i.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config

FIGURES_DIR = config.FIGURES_DIR
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def _ensure_figures_dir() -> Path:
    """Grafiklerin kaydedileceği klasörü oluşturur ve döner."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


# =========================================================================
# 1. Grafik 1: Eksik Veri Isı Haritası ve Dendrogramı
# =========================================================================
def plot_missingness_patterns(
    df_missing: pd.DataFrame,
    save_path: Path = None,
) -> plt.Figure:
    """missingno kütüphanesi ile eksik veri korelasyon ısı haritası ve
    hiyerarşik dendrogramı tek figürde birleştirir.
    """
    _ensure_figures_dir()
    out_path = save_path or (
        FIGURES_DIR / "01_missingness_heatmap_dendrogram.png"
    )

    # Yalnızca eksik değer içeren sütunları seç
    missing_cols = df_missing.columns[df_missing.isna().sum() > 0]
    df_sub = df_missing[missing_cols]

    fig = plt.figure(figsize=(16, 7), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1)
    msno.heatmap(
        df_sub,
        ax=ax1,
        fontsize=9,
        cmap="coolwarm",
        cbar=True,
    )
    ax1.set_title(
        "(A) Eksiklik Korelasyon Isı Haritası (Nullity Correlation)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    ax2 = fig.add_subplot(1, 2, 2)
    msno.dendrogram(
        df_sub,
        ax=ax2,
        fontsize=9,
        orientation="top",
    )
    ax2.set_title(
        "(B) Eksiklik Örüntüsü Hiyerarşik Kümeleme Dendrogramı",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    plt.suptitle(
        "Zorunlu Grafik 1: Eksik Veri Örüntüleri ve Birlikte Eksiklik Analizi",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


# =========================================================================
# 2. Grafik 2: Gerçek vs. Sentetik (SMOTE & CTGAN) Özellik Dağılımları
# =========================================================================
def plot_real_vs_synthetic_distributions(
    X_real_pos: pd.DataFrame,
    X_syn_smote: pd.DataFrame,
    X_syn_ctgan: pd.DataFrame,
    features: list[str] = None,
    save_path: Path = None,
) -> plt.Figure:
    """En önemli sürekli özellikler için Gerçek, SMOTE ve CTGAN pozitif
    örneklerinin dağılımlarını (Overlaid KDE) karşılaştırır.
    """
    _ensure_figures_dir()
    out_path = save_path or (
        FIGURES_DIR / "02_real_vs_synthetic_distributions.png"
    )

    if features is None:
        features = [
            "num_lab_procedures",
            "num_medications",
            "time_in_hospital",
            "number_diagnoses",
            "num_procedures",
            "number_inpatient",
        ]
    avail_features = [f for f in features if f in X_real_pos.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()

    for idx, col in enumerate(avail_features):
        ax = axes[idx]
        sns.kdeplot(
            X_real_pos[col],
            ax=ax,
            color="#1f77b4",
            label="Gerçek Pozitif",
            linewidth=2.2,
            fill=True,
            alpha=0.15,
        )
        sns.kdeplot(
            X_syn_smote[col],
            ax=ax,
            color="#2ca02c",
            label="SMOTE Sentetik",
            linewidth=1.8,
            linestyle="--",
        )
        sns.kdeplot(
            X_syn_ctgan[col],
            ax=ax,
            color="#d62728",
            label="CTGAN Sentetik",
            linewidth=1.8,
            linestyle=":",
        )
        ax.set_title(f"Özellik: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Standartlaştırılmış Değer", fontsize=9)
        ax.set_ylabel("Yoğunluk (Density)", fontsize=9)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for idx in range(len(avail_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(
        "Zorunlu Grafik 2: Gerçek vs. Sentetik (SMOTE ve CTGAN) "
        "Dağılım Karşılaştırması",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


# =========================================================================
# 3. Grafik 3: Korelasyon Fark Matrisi (|Corr_Real - Corr_Synthetic|)
# =========================================================================
def plot_correlation_difference_matrix(
    X_real_pos: pd.DataFrame,
    X_syn_smote: pd.DataFrame,
    X_syn_ctgan: pd.DataFrame,
    features: list[str] = None,
    save_path: Path = None,
) -> plt.Figure:
    """Gerçek ve sentetik veri arasındaki yapısal korelasyon korumasını
    |Corr_Real - Corr_Synthetic| mutlak fark matrisiyle görselleştirir.
    """
    _ensure_figures_dir()
    out_path = save_path or (
        FIGURES_DIR / "03_correlation_difference_matrix.png"
    )

    if features is None:
        features = [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
        ]
    cols = [c for c in features if c in X_real_pos.columns]

    corr_real = X_real_pos[cols].corr()
    corr_smote = X_syn_smote[cols].corr()
    corr_ctgan = X_syn_ctgan[cols].corr()

    diff_smote = (corr_real - corr_smote).abs()
    diff_ctgan = (corr_real - corr_ctgan).abs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    triu_idx = np.triu_indices_from(diff_smote, k=1)
    mae_smote = float(diff_smote.values[triu_idx].mean())
    mae_ctgan = float(diff_ctgan.values[triu_idx].mean())

    sns.heatmap(
        diff_smote,
        ax=ax1,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.35,
        cbar_kws={"label": "|Δ Korelasyon|"},
    )
    ax1.set_title(
        f"(A) |Corr_Gerçek - Corr_SMOTE|\n"
        f"Ortalama Mutlak Hata (MAE): {mae_smote:.3f}",
        fontsize=11,
        fontweight="bold",
    )

    sns.heatmap(
        diff_ctgan,
        ax=ax2,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.35,
        cbar_kws={"label": "|Δ Korelasyon|"},
    )
    ax2.set_title(
        f"(B) |Corr_Gerçek - Corr_CTGAN|\n"
        f"Ortalama Mutlak Hata (MAE): {mae_ctgan:.3f}",
        fontsize=11,
        fontweight="bold",
    )

    plt.suptitle(
        "Zorunlu Grafik 3: Yapısal Korelasyon Koruma Fark Matrisi "
        "(|Corr_Real - Corr_Synthetic|)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


# =========================================================================
# 4. Grafik 4: TabNet Attention Maskeleri vs. GBDT Özellik Önemi
# =========================================================================
def plot_tabnet_masks_vs_gbdt_importance(
    tabnet_step_masks_df: pd.DataFrame,
    gbdt_feat_imp_df: pd.DataFrame,
    top_n: int = 12,
    save_path: Path = None,
) -> plt.Figure:
    """TabNet adımları boyunca uygulanan seyrek dikkat maskeleri (sparse
    attention maps) ile GBDT özellik önemlerini yan yana karşılaştırır.
    """
    _ensure_figures_dir()
    out_path = save_path or (
        FIGURES_DIR / "04_tabnet_masks_vs_gbdt_importance.png"
    )

    # GBDT'nin en önemli top_n özelliğini referans al
    top_features = list(gbdt_feat_imp_df["Özellik"].head(top_n))

    # TabNet adım maskelerini bu özellikler için sırala
    avail_top = [f for f in top_features if f in tabnet_step_masks_df.index]
    tabnet_sub = tabnet_step_masks_df.loc[avail_top]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 7), dpi=300, gridspec_kw={"width_ratios": [1.1, 1]}
    )

    # Sol Panel: TabNet Adım Bazlı Attention Isı Haritası
    sns.heatmap(
        tabnet_sub,
        ax=ax1,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar_kws={"label": "Ortalama Dikkat Ağırlığı (Attention Weight)"},
    )
    ax1.set_title(
        "(A) TabNet Ardışık Karar Adımları Dikkat Maskeleri (Sparse Masks)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Karar Adımları (Decision Steps)", fontsize=10)
    ax1.set_ylabel("Klinik Özellikler", fontsize=10)

    # Sağ Panel: GBDT Özellik Önemleri Yatay Çubuk Grafiği
    gbdt_sub = gbdt_feat_imp_df[
        gbdt_feat_imp_df["Özellik"].isin(avail_top)
    ].sort_values(by="Önem_%", ascending=True)

    y_pos = np.arange(len(gbdt_sub))
    bars = ax2.barh(
        y_pos,
        gbdt_sub["Önem_%"],
        color="#2b5c8f",
        edgecolor="#183654",
        height=0.65,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(gbdt_sub["Özellik"], fontsize=9)
    ax2.set_xlabel("GBDT Nispi Özellik Önemi (%)", fontsize=10)
    ax2.set_title(
        "(B) LightGBM (Optuna GBDT) Küresel Özellik Önemleri",
        fontsize=11,
        fontweight="bold",
    )

    for bar in bars:
        w = bar.get_width()
        ax2.annotate(
            f"{w:.1f}%",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )
    ax2.set_xlim(0, max(gbdt_sub["Önem_%"]) * 1.15)

    plt.suptitle(
        "Zorunlu Grafik 4: TabNet Seyrek Dikkat Maskeleri vs. GBDT "
        "Özellik Önemleri",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


# =========================================================================
# 5. Grafik 5: Birleşik ROC-AUC ve Precision-Recall (PR-AUC) Eğrileri
# =========================================================================
def plot_combined_roc_and_pr_curves(
    y_test: pd.Series | np.ndarray,
    models_dict: dict,
    save_path: Path = None,
) -> plt.Figure:
    """Üç modelin (Perceptron, GBDT, TabNet) performansını tek figürde
    yan yana birleşik ROC ve Precision-Recall eğrileriyle gösterir.
    """
    _ensure_figures_dir()
    out_path = save_path or (
        FIGURES_DIR / "05_combined_roc_and_pr_curves.png"
    )

    y_true_arr = np.asarray(y_test, dtype=int)
    prevalence = float(y_true_arr.mean())

    palette = {
        "Perceptron": "#d95f02",
        "GBDT": "#1b9e77",
        "TabNet": "#7570b3",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # (A) ROC Eğrileri
    for name, payload in models_dict.items():
        y_prob = payload["y_prob"]
        fpr, tpr, _ = roc_curve(y_true_arr, y_prob)
        auc_score = payload["metrics"]["ROC_AUC"]

        color = "#333333"
        for key in palette:
            if key in name:
                color = palette[key]
                break

        ax1.plot(
            fpr,
            tpr,
            label=f"{name} (ROC-AUC = {auc_score:.4f})",
            color=color,
            linewidth=2.2,
        )

    ax1.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="#888888",
        linewidth=1.2,
        label="Rastgele Tahmin (AUC = 0.5000)",
    )
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=10)
    ax1.set_ylabel("Doğru Pozitif Oranı (TPR / Recall)", fontsize=10)
    ax1.set_title(
        "(A) Birleşik ROC Eğrileri (Receiver Operating Characteristic)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # (B) Precision-Recall Eğrileri
    for name, payload in models_dict.items():
        y_prob = payload["y_prob"]
        precision, recall, _ = precision_recall_curve(y_true_arr, y_prob)
        pr_auc = payload["metrics"]["PR_AUC"]

        color = "#333333"
        for key in palette:
            if key in name:
                color = palette[key]
                break

        ax2.plot(
            recall,
            precision,
            label=f"{name} (PR-AUC = {pr_auc:.4f})",
            color=color,
            linewidth=2.2,
        )

    ax2.axhline(
        prevalence,
        linestyle="--",
        color="#888888",
        linewidth=1.2,
        label=f"Taban Prevalans Oranı ({prevalence:.1%})",
    )
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel("Duyarlılık / Recall (Pozitifleri Yakalama)", fontsize=10)
    ax2.set_ylabel(
        "Kesinlik / Precision (Pozitif Tahmin Doğruluğu)", fontsize=10
    )
    ax2.set_title(
        "(B) Birleşik Precision-Recall Eğrileri (Dengesiz Veri Odaklı)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.suptitle(
        "Zorunlu Grafik 5: Çoklu Model Birleşik ROC-AUC ve Precision-Recall "
        "Eğrileri",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


# =========================================================================
# 6. Grafik 6: 3'lü Normalize Karmaşıklık Matrisleri Grid'i
# =========================================================================
def plot_normalized_confusion_matrix_grid(
    models_dict: dict,
    save_path: Path = None,
) -> plt.Figure:
    """Üç modelin (Perceptron, GBDT, TabNet) satır bazlı normalize edilmiş
    karmaşıklık matrislerini yan yana (1x3 grid) görselleştirir.
    """
    _ensure_figures_dir()
    out_path = save_path or (
        FIGURES_DIR / "06_normalized_confusion_matrix_grid.png"
    )

    n_models = len(models_dict)
    fig, axes = plt.subplots(
        1, n_models, figsize=(5.5 * n_models, 4.8), dpi=300
    )
    if n_models == 1:
        axes = [axes]

    labels = ["Yatış Yok / >30g (0)", "<30g Yeniden Yatış (1)"]

    for idx, (name, payload) in enumerate(models_dict.items()):
        ax = axes[idx]
        cm_norm = payload["confusion_matrix_norm"]

        cbar_arg = (idx == n_models - 1)
        cbar_kw = (
            {"label": "Normalize Oran"} if idx == n_models - 1 else None
        )

        sns.heatmap(
            cm_norm,
            ax=ax,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            cbar=cbar_arg,
            cbar_kws=cbar_kw,
        )
        macro_f1 = payload["metrics"]["Macro_F1"]
        pr_auc = payload["metrics"]["PR_AUC"]

        ax.set_title(
            f"{name}\nMacro F1: {macro_f1:.4f} | PR-AUC: {pr_auc:.4f}",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Gerçek Sınıf", fontsize=9)
        else:
            ax.set_ylabel("")

    plt.suptitle(
        "Zorunlu Grafik 6: Normalize Edilmiş Karmaşıklık Matrisleri "
        "(Confusion Matrix Grid)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


# =========================================================================
# 7. Ek Grafik: TabNet Eğitim ve Doğrulama Kayıp Eğrileri
# =========================================================================
def plot_tabnet_loss_curves(
    tabnet_history: dict,
    save_path: Path = None,
) -> plt.Figure:
    """TabNet modelinin epoch'lar boyunca eğitim ve doğrulama kayıp eğrilerini
    (loss curves) görselleştirir.
    """
    _ensure_figures_dir()
    out_path = save_path or (
        FIGURES_DIR / "07_tabnet_training_loss_curves.png"
    )

    hist = (
        tabnet_history.history
        if hasattr(tabnet_history, "history")
        else tabnet_history
    )

    epochs = range(1, len(hist["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    ax1.plot(
        epochs,
        hist["loss"],
        label="Eğitim Kaybı (Train Loss)",
        color="#1f77b4",
        linewidth=2.0,
    )
    val_loss_key = None
    for k in hist:
        if "loss" in k and k != "loss":
            val_loss_key = k
            break
    if val_loss_key:
        ax1.plot(
            epochs,
            hist[val_loss_key],
            label="Doğrulama Kaybı (Val Loss)",
            color="#ff7f0e",
            linewidth=2.0,
            linestyle="--",
        )
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Kayıp (Loss)", fontsize=10)
    ax1.set_title(
        "(A) Epoch Bazlı Kayıp Eğrileri (Loss Optimization)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=9)

    val_auc_key = None
    for k in hist:
        if "auc" in k.lower():
            val_auc_key = k
            break

    if val_auc_key:
        ax2.plot(
            epochs,
            hist[val_auc_key],
            label=f"Doğrulama AUC ({val_auc_key})",
            color="#2ca02c",
            linewidth=2.0,
        )
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("ROC-AUC Skoru", fontsize=10)
        ax2.set_title(
            "(B) Epoch Bazlı Doğrulama Performansı (Validation AUC)",
            fontsize=11,
            fontweight="bold",
        )
        ax2.legend(loc="lower right", fontsize=9)
    else:
        ax2.set_visible(False)

    plt.suptitle(
        "TabNet Derin Öğrenme Eğitimi ve Kayıp Eğrisi Optimizasyonu",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
