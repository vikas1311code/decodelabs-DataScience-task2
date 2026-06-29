"""
=========================================================================
 PROJECT 2 : SUPERVISED LEARNING - FRAUD DETECTION PIPELINE
 DecodeLabs - Industrial Training Kit (Batch 2026)
=========================================================================

GOAL
----
Build a leak-free, production-grade classification pipeline that detects
fraudulent transactions in a highly imbalanced dataset, using:
    - SMOTE (imblearn) for class imbalance, applied ONLY inside the
      training fold (never before train/test split -> no data leakage)
    - Logistic Regression  (needs scaling)
    - Random Forest        (scale-invariant, no scaler needed)
    - GridSearchCV for hyperparameter tuning (tuned holistically, with
      SMOTE re-applied fold-by-fold automatically by imblearn.Pipeline)
    - Evaluation strictly via Precision, Recall, F1, ROC-AUC and the
      Confusion Matrix -> "Accuracy" is intentionally ignored, because
      with a ~99.8% / 0.2% split it is a meaningless, misleading metric.

HOW TO USE WITH YOUR OWN DATA (recommended)
--------------------------------------------
This script is built for the classic Kaggle "Credit Card Fraud Detection"
dataset (284,807 rows, columns Time, V1...V28, Amount, Class).
If you have creditcard.csv, just put it next to this script and run:

    python fraud_detection_pipeline.py creditcard.csv

If no path is given, the script auto-generates a SYNTHETIC dataset with
the same extreme imbalance (~0.17% fraud) so the pipeline still runs
end-to-end and you can see all the metrics/plots working correctly.
=========================================================================
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# -------------------------------------------------------------------------
# 1. LOAD DATA  (real file if provided, else synthetic fallback)
# -------------------------------------------------------------------------
def load_data(path: str | None):
    if path:
        print(f"Loading real dataset from: {path}")
        df = pd.read_csv(path)
        if "Class" not in df.columns:
            raise ValueError("Expected a target column named 'Class' (0=legit, 1=fraud).")
        return df

    print("No dataset path given -> generating a SYNTHETIC imbalanced dataset "
          "(mirrors the real creditcard.csv structure: ~0.17% fraud).")
    from sklearn.datasets import make_classification

    n_samples = 20_000
    fraud_ratio = 0.0017  # mirrors the real-world 0.17% fraud rate

    X, y = make_classification(
        n_samples=n_samples,
        n_features=28,           # mimic V1...V28
        n_informative=12,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.001,
        class_sep=1.2,
        random_state=RANDOM_STATE,
    )

    df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(X.shape[1])])
    df["Time"] = np.sort(np.random.uniform(0, 172800, size=n_samples))  # 2 days, in seconds
    df["Amount"] = np.round(np.abs(np.random.lognormal(mean=3.0, sigma=1.5, size=n_samples)), 2)
    df["Class"] = y
    return df


# -------------------------------------------------------------------------
# 2. SPLIT FIRST -> THEN BUILD PIPELINES (Zero-Leakage Protocol)
# -------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified split: preserves the real-world 99.8/0.2 imbalance in BOTH
    # the train set and the held-out test set. SMOTE/Scaling happen AFTER
    # this split, and only ever touch the training fold.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain shape: {X_train.shape}, Fraud rate: {y_train.mean()*100:.3f}%")
    print(f"Test  shape: {X_test.shape}, Fraud rate: {y_test.mean()*100:.3f}%")
    return X_train, X_test, y_train, y_test


# -------------------------------------------------------------------------
# 3. BUILD THE TWO LEAK-FREE PIPELINES
# -------------------------------------------------------------------------
def build_pipelines():
    # Logistic Regression NEEDS scaling (sensitive to feature magnitude),
    # so: Scaler -> SMOTE -> Classifier
    lr_pipeline = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    # Random Forest is scale-invariant (splits are ordinal partitions),
    # so no scaler needed: SMOTE -> Classifier
    rf_pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    return lr_pipeline, rf_pipeline


# -------------------------------------------------------------------------
# 4. HYPERPARAMETER TUNING VIA GridSearchCV
#    -> SMOTE is re-applied INSIDE every fold for every parameter
#       combination, so the validation fold is never resampled/leaked.
# -------------------------------------------------------------------------
def tune_model(pipeline, param_grid, X_train, y_train, name):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",   # tune on ROC-AUC, NOT accuracy
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    print(f"\nTuning {name} ...")
    grid.fit(X_train, y_train)
    print(f"Best params for {name}: {grid.best_params_}")
    print(f"Best CV ROC-AUC for {name}: {grid.best_score_:.4f}")
    return grid.best_estimator_


# -------------------------------------------------------------------------
# 5. EVALUATION: Precision / Recall / F1 / ROC-AUC / Confusion Matrix
#    "Accuracy" is deliberately NOT used as the headline metric.
# -------------------------------------------------------------------------
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n===== {name} : Test Set Performance =====")
    print(f"Precision : {precision:.4f}  (when we flag fraud, how often are we right?)")
    print(f"Recall    : {recall:.4f}  (of all real fraud, how much did we catch?)")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    return {
        "name": name, "precision": precision, "recall": recall,
        "f1": f1, "roc_auc": roc_auc, "cm": cm,
        "y_proba": y_proba,
    }


# -------------------------------------------------------------------------
# 6. PLOTS: Confusion Matrices + ROC Curves for both models
# -------------------------------------------------------------------------
def plot_results(results, y_test, out_path="model_evaluation.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for i, res in enumerate(results):
        sns.heatmap(
            res["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
            ax=axes[0, i],
        )
        axes[0, i].set_title(f"{res['name']} - Confusion Matrix")
        axes[0, i].set_xlabel("Predicted")
        axes[0, i].set_ylabel("Actual")

    ax_roc = axes[1, 0]
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax_roc.plot(fpr, tpr, label=f"{res['name']} (AUC={res['roc_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_title("ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()

    ax_bar = axes[1, 1]
    metrics_df = pd.DataFrame(results)[["name", "precision", "recall", "f1", "roc_auc"]]
    metrics_df = metrics_df.set_index("name")
    metrics_df.plot(kind="bar", ax=ax_bar)
    ax_bar.set_title("Metric Comparison (Accuracy intentionally excluded)")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved evaluation plots to: {out_path}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_data(data_path)

    print("\nClass distribution (raw data):")
    print(df["Class"].value_counts(normalize=True).rename("proportion"))

    X_train, X_test, y_train, y_test = split_data(df)
    lr_pipeline, rf_pipeline = build_pipelines()

    # --- Logistic Regression hyperparameter grid ---
    lr_param_grid = {
        "smote__k_neighbors": [3, 5],
        "classifier__C": [0.01, 0.1, 1.0],
    }
    best_lr = tune_model(lr_pipeline, lr_param_grid, X_train, y_train, "Logistic Regression")

    # --- Random Forest hyperparameter grid ---
    rf_param_grid = {
        "smote__k_neighbors": [3, 5],
        "classifier__n_estimators": [100],
        "classifier__max_depth": [10, None],
    }
    best_rf = tune_model(rf_pipeline, rf_param_grid, X_train, y_train, "Random Forest")

    # --- Final evaluation on the untouched test set ---
    results = [
        evaluate(best_lr, X_test, y_test, "Logistic Regression"),
        evaluate(best_rf, X_test, y_test, "Random Forest"),
    ]

    plot_results(results, y_test)

    print("\n========================================================")
    print(" SUMMARY")
    print("========================================================")
    for res in results:
        print(f"{res['name']:20s} | Precision: {res['precision']:.3f} | "
              f"Recall: {res['recall']:.3f} | F1: {res['f1']:.3f} | "
              f"ROC-AUC: {res['roc_auc']:.3f}")
    print("\nNote: 'Accuracy' was intentionally never used to judge these "
          "models, since with ~99.8% legitimate transactions a model that "
          "predicts everything as 'Legitimate' would score ~99.8% accuracy "
          "while catching ZERO fraud.")


if __name__ == "__main__":
    main()
