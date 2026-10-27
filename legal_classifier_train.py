"""
Legal Case Outcome Predictor
============================
Dataset  : fj.csv  (+ coi.csv to expand cited Articles)
Target   : decision  (0 = Against Petitioner | 1 = In Favor)
Model    : TF-IDF  +  calibrated LinearSVC / RandomForest / XGBoost soft-voting ensemble
Output   : models/legal_classifier.pkl  (a dataset_loader.LegalClassifier)

Only features that are also available for a user's free-text scenario are used, so the
reported test accuracy matches what the app sees at prediction time.

Run:
    python legal_classifier_train.py
Load:
    import joblib
    model = joblib.load("models/legal_classifier.pkl")
    model.predict_proba(["case text here"])
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from dataset_loader import LABELS, LegalClassifier, build_case_text, load_dataset

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not found — pip install xgboost to enable it.")

# ──────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────────────────────
MODEL_DIR   = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH  = MODEL_DIR / "legal_classifier.pkl"
METRICS_LOG = MODEL_DIR / "metrics_log.csv"
CM_PNG      = MODEL_DIR / "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
WEIGHTS      = {"svc": 1.5, "rf": 1.0, "xgb": 1.5}


# ──────────────────────────────────────────────────────────────
# MODEL TRAINERS
# ──────────────────────────────────────────────────────────────
def make_vectorizer():
    return TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), stop_words="english",
        sublinear_tf=True, min_df=2, max_df=0.95, dtype=np.float32,
    )


def train_svc(Xtr, ytr):
    svc = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
        cv=3,
    )
    return svc.fit(Xtr, ytr)


def train_rf(Xtr, ytr):
    rf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    return rf.fit(Xtr, ytr)


def train_xgb(Xtr, ytr):
    """XGBoost with early stopping on an internal validation split (no test leakage)."""
    tr, val = train_test_split(np.arange(Xtr.shape[0]), test_size=0.15,
                               random_state=RANDOM_STATE, stratify=ytr)
    xgb = XGBClassifier(
        objective="binary:logistic", tree_method="hist",
        n_estimators=700, learning_rate=0.08, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
        eval_metric="logloss", early_stopping_rounds=50,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb.fit(Xtr[tr], ytr[tr], sample_weight=compute_sample_weight("balanced", ytr[tr]),
            eval_set=[(Xtr[val], ytr[val])], verbose=False)
    return xgb


def ensemble_proba(models: dict, X) -> np.ndarray:
    total = sum(WEIGHTS[n] for n in models)
    return sum(m.predict_proba(X) * WEIGHTS[n] for n, m in models.items()) / total


# ──────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ──────────────────────────────────────────────────────────────
def plot_cm(cm_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_df.values, cmap="Blues")
    fig.colorbar(im)
    ticks = np.arange(len(cm_df.index))
    ax.set_xticks(ticks); ax.set_xticklabels(cm_df.columns, rotation=30, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(cm_df.index)
    thresh = cm_df.values.max() / 2.0
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, int(cm_df.values[i, j]), ha="center", va="center",
                    color="white" if cm_df.values[i, j] > thresh else "black")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Confusion matrix saved -> {path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def train():
    fj = load_dataset("fj.csv")
    if fj.empty:
        raise SystemExit("dataset/fj.csv not found.")
    print(f"[DATA] {len(fj)} cases | class 0: {(fj['decision'] == 0).sum()} | "
          f"class 1: {(fj['decision'] == 1).sum()}")

    # Raw text only; LegalClassifier applies build_case_text again at predict time
    X_raw = (fj["case title"] + " " + fj["issues"]).to_numpy()
    y_all = fj["decision"].to_numpy()

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all,
    )
    print(f"[SPLIT] train={len(y_tr)} | test={len(y_te)}")

    print("[FEAT] Vectorizing...")
    vec = make_vectorizer()
    Xtr = vec.fit_transform([build_case_text(t) for t in X_tr_raw])
    Xte = vec.transform([build_case_text(t) for t in X_te_raw])

    # Cross-validation (on the training split only)
    print(f"\n[CV] {CV_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    svc_cv, rf_cv = [], []
    for fold, (ti, vi) in enumerate(cv.split(Xtr, y_tr), 1):
        svc_cv.append(accuracy_score(y_tr[vi], train_svc(Xtr[ti], y_tr[ti]).predict(Xtr[vi])))
        rf_cv.append(accuracy_score(y_tr[vi], train_rf(Xtr[ti], y_tr[ti]).predict(Xtr[vi])))
        print(f"  Fold {fold}: SVC={svc_cv[-1]:.4f}  RF={rf_cv[-1]:.4f}")
    print(f"\n  SVC CV: {np.mean(svc_cv):.4f} +/- {np.std(svc_cv):.4f}")
    print(f"  RF  CV: {np.mean(rf_cv):.4f} +/- {np.std(rf_cv):.4f}")

    # Final models
    models = {"svc": train_svc(Xtr, y_tr), "rf": train_rf(Xtr, y_tr)}
    if XGB_AVAILABLE:
        models["xgb"] = train_xgb(Xtr, y_tr)
    for name, m in models.items():
        print(f"  {name.upper():<3} test acc: {accuracy_score(y_te, m.predict(Xte)):.4f}")

    # Ensemble metrics
    p_ens  = ensemble_proba(models, Xte)
    y_pred = np.argmax(p_ens, axis=1)

    acc         = accuracy_score(y_te, y_pred)
    f1_macro    = f1_score(y_te, y_pred, average="macro")
    f1_weighted = f1_score(y_te, y_pred, average="weighted")
    auc         = roc_auc_score(y_te, p_ens[:, 1])

    print(f"\n{'=' * 50}")
    print(f"  Test Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  ROC-AUC        : {auc:.4f}")
    print(f"  F1 Macro       : {f1_macro:.4f}")
    print(f"  F1 Weighted    : {f1_weighted:.4f}")
    print(f"{'=' * 50}")

    tgt_names = [LABELS[c] for c in sorted(LABELS)]
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=tgt_names, digits=4))

    cm_df = pd.DataFrame(confusion_matrix(y_te, y_pred, labels=sorted(LABELS)),
                         index=tgt_names, columns=tgt_names)
    print("Confusion Matrix:")
    print(cm_df, "\n")
    plot_cm(cm_df, CM_PNG)

    # Save
    clf = LegalClassifier(vec=vec, models=models,
                          weights={n: WEIGHTS[n] for n in models}, classes_=sorted(LABELS))
    joblib.dump(clf, MODEL_PATH, compress=3)
    print(f"Model saved -> {MODEL_PATH}")

    # Log
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "+".join(models), "accuracy": round(acc, 6), "roc_auc": round(auc, 6),
        "f1_macro": round(f1_macro, 6), "f1_weighted": round(f1_weighted, 6),
    }
    prev = pd.read_csv(METRICS_LOG) if METRICS_LOG.exists() else pd.DataFrame()
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(METRICS_LOG, index=False)
    print(f"Metrics logged -> {METRICS_LOG}")


if __name__ == "__main__":
    train()
