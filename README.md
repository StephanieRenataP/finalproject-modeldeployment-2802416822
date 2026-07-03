# Credit Score — Training Pipeline + Deployment (v3)

Sinkron dengan `credit_score_model_3.ipynb`. Perubahan dibanding versi sebelumnya:
`RobustScaler` (bukan `StandardScaler`), model `XGBoost` ditambahkan, ruang
hyperparameter diperluas (`n_iter=10`), tiga fitur interaksi domain-spesifik
(`Debt_to_Income`, `Credit_Utilization_Intensity`, `Free_Cash_Flow`), eksperimen
`class_weight` vs `SMOTE`, dan threshold tuning kelas minoritas `Good`.

## File

| File | Fungsi |
|---|---|
| `training_pipeline.py` | Modul utama (OOP): cleaning, feature engineering, 6-model comparison, tuning, SMOTE experiment, evaluasi, threshold tuning, MLflow tracking, penyimpanan `.pkl`. **Jangan dijalankan langsung.** |
| `run_training.py` | Entry point CLI — jalankan file ini. |
| `inference.py` | Modul inferencing (OOP): `CreditScoreInferenceService`, mendukung toggle threshold kelas minoritas. |
| `app.py` | Web app Streamlit (form manual + upload CSV batch + toggle threshold). |
| `test_inference.py` | 3 test case (satu per kelas) + test toggle threshold, memakai baris data asli. |
| `requirements.txt` | Dependensi (termasuk `xgboost`, `imbalanced-learn`, `scipy`). |

## Menjalankan

```bash
pip install -r requirements.txt

# Training (hasil: models/credit_score_model.pkl + tracking di sqlite:///mlflow.db)
python run_training.py --data-path data_D.csv

# Lihat eksperimen
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Test deployment
python test_inference.py

# Jalankan web app
streamlit run app.py
```

## Alur pipeline (`CreditScoreTrainingPipeline.run()`)

1. Load data + laporan kualitas data (missingness, skewness) → dicatat ke MLflow.
2. Split latih/uji berbasis nasabah (`CustomerAwareSplitter`).
3. Bandingkan 6 algoritma via `StratifiedGroupKFold` 3-fold (`ModelExperimentRunner`):
   Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
4. `RandomizedSearchCV` (10 iterasi) pada model terbaik (`HyperparameterTuner`).
5. Eksperimen `class_weight="balanced"` vs SMOTE, fokus F1-score kelas `Good`
   (`ImbalanceExperimentRunner`, pakai `imblearn.pipeline.Pipeline` agar SMOTE
   tidak bocor ke data validasi).
6. Fit model final dengan strategi imbalance terpilih pada seluruh data latih.
7. Evaluasi pada data uji holdout (`ModelEvaluator`).
8. Threshold tuning kelas `Good` via precision-recall curve (`MinorityClassThresholdTuner`).
9. Log model + registry ke MLflow (`serialization_format="cloudpickle"`, karena
   pipeline memuat class custom `CreditDataCleaner` & `XGBWithLabelEncoding`).
10. Simpan artefak `.pkl` (`ArtifactManager`) — sekarang termasuk `good_class_threshold`
    dan `minority_class` sebagai metadata tambahan untuk inferencing.

## Hasil training pada `data_D.csv` penuh (25.000 baris, 8.959 nasabah latih)

- Perbandingan model (F1-macro CV): **Random Forest 0.6595** > Gradient Boosting
  0.6530 > XGBoost 0.6515 > KNN 0.6262 > Decision Tree 0.6181 > Logistic
  Regression 0.3970 (LR anjlok karena `Monthly_Balance` sangat condong/corrupted,
  bahkan setelah `RobustScaler`).
- Hyperparameter terbaik (Random Forest): `n_estimators=200, max_depth=16,
  min_samples_split=15, min_samples_leaf=2, max_features=0.5` (F1-macro CV 0.6636).
- Eksperimen imbalance: baseline (`class_weight`) F1-Good=0.6041 vs SMOTE
  F1-Good=0.5997 → **baseline dipilih** (SMOTE tidak memberi perbaikan).
- Evaluasi data uji: **Accuracy 0.6730, F1-macro 0.6638**, F1 per kelas: Poor 0.67,
  Standard 0.70, Good 0.62.
- Threshold tuning kelas Good: ambang batas optimal **0.355** (F1=0.6246,
  Precision=0.508, Recall=0.811) — trade-off eksplisit, precision turun demi
  recall jauh lebih tinggi pada kelas minoritas.

## Test deployment (3 test case, data asli dari `data_D.csv`)

Semua LULUS dengan keyakinan tinggi:
```
Good     -> prediksi Good     (93.97%)
Standard -> prediksi Standard (85.82%)
Poor     -> prediksi Poor     (88.71%)
```
