# Fraud Detection Pipeline (Project 2 - DecodeLabs)

Leak-free Supervised Learning pipeline for fraud detection using SMOTE,
Logistic Regression, Random Forest, GridSearchCV, and strict
Precision/Recall/ROC-AUC evaluation.

## Files in this folder
- `fraud_detection_pipeline.py` -> main script (run this)
- `creditcard.csv` -> dataset (synthetic, ~0.17% fraud rate, same shape as
  the real Kaggle "Credit Card Fraud Detection" dataset). Replace this
  file with the real Kaggle `creditcard.csv` any time for real results.
- `requirements.txt` -> python packages needed
- `model_evaluation.png` -> sample output plot (confusion matrices, ROC
  curves, metric comparison) from a prior run

## How to run on WSL

1. Open your WSL terminal and go to the folder where you put this:
   ```bash
   cd /path/to/fraud_detection_project
   ```

2. (Recommended) create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:
   ```bash
   python fraud_detection_pipeline.py creditcard.csv
   ```

   If you ever want to skip the CSV and let it auto-generate a fresh
   synthetic dataset instead, just run it with no argument:
   ```bash
   python fraud_detection_pipeline.py
   ```

5. Output:
   - Console: best hyperparameters, Precision/Recall/F1/ROC-AUC for both
     models, full classification report, final summary table.
   - File: `model_evaluation.png` (confusion matrices + ROC curves +
     metric bar chart) gets saved in the same folder.

## Using the REAL Kaggle dataset (better for your submission)
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv` (284,807 rows)
3. Replace the `creditcard.csv` in this folder with the downloaded one
4. Re-run step 4 above — same command, real data, real results.

## What this pipeline does (mapped to the brief)
- Stratified train/test split BEFORE any resampling/scaling (no leakage)
- `imblearn.pipeline.Pipeline` (not sklearn's) so SMOTE only ever touches
  the training fold, even during cross-validation
- Logistic Regression pipeline: StandardScaler -> SMOTE -> Classifier
- Random Forest pipeline: SMOTE -> Classifier (no scaler needed, tree
  splits are scale-invariant)
- `GridSearchCV` (StratifiedKFold) tunes SMOTE's `k_neighbors` together
  with each model's hyperparameters, scored on ROC-AUC
- Final evaluation uses Precision, Recall, F1, ROC-AUC and Confusion
  Matrix only — "Accuracy" is deliberately never used as a judgment
  metric, since with ~99.8% legitimate transactions a model that always
  predicts "Legitimate" would score ~99.8% accuracy while catching zero
  fraud.
