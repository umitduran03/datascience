# Diabetes 130-US Hospitals — 30 Günlük Yeniden Yatış Tahmini

Dengesiz ve yüksek boyutlu tablo verisi üzerinde uçtan uca makine öğrenmesi
boru hattı: veri kalitesi, gelişmiş eksik veri doldurma (MVI), sentetik veri
üretimi ve Perceptron / GBDT / TabNet modellerinin karşılaştırması.

## Veri Seti

- **Kaynak:** [UCI ML Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296)
- **Hedef:** `readmitted == '<30'` → 1, diğerleri → 0 (ikili, dengesiz)
- Ham veri repoya eklenmez; aşağıdaki komutla indirilir.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m src.data_loader       # veriyi data/raw/ altına indirir
python -m src.validate_constraints
```

## Klasör Yapısı

```
data/raw/            Ham veri (git'e eklenmez)
data/processed/      İşlenmiş veri (git'e eklenmez)
notebooks/           Faz bazlı Jupyter notebook'ları
reports/figures/     Zorunlu 6 grafik
src/                 Yeniden kullanılabilir Python modülleri
```

## Ekip

| Üye | Rol |
|---|---|
| Ümit | Veri Mimarı & Kalite Uzmanı — Sözlü Sunum |
| Ertuğrul | Klasik & Ağaç Modelleri Uzmanı — Kod Savunması |
| Yunus Emre | TabNet & Analitik Uzmanı — Kod Savunması |

## İlerleme

- [x] Faz 1 — Veri seti seçimi ve kısıt doğrulaması (`notebooks/01_faz1_veri_dogrulama.ipynb`)
- [x] Faz 2 — Veri kalitesi, MVI, encoding & scaling (`notebooks/02_faz2_on_isleme_ve_mvi.ipynb`)
- [x] Faz 3 — Sentetik veri (SMOTE + CTGAN) ve kalite testleri (`notebooks/03_faz3_sentetik_veri_ve_kalite.ipynb`)
- [x] Faz 4 — Perceptron & GBDT (`notebooks/04_faz4_perceptron_ve_gbdt.ipynb`)
- [x] Faz 5 — TabNet & zorunlu grafikler (`notebooks/05_faz5_tabnet_ve_zorunlu_grafikler.ipynb`)
- [ ] Faz 6 — Final metrikler, sunum, teslim
