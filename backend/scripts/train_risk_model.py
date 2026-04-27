"""
FINORA — ML Risk Model Trainer
================================
Trains an Isolation Forest + XGBoost ensemble on synthetic
transaction data to detect anomalous financial transactions.

Features engineered:
  - amount_zscore        : deviation from user's 90d average
  - amount_log           : log-scaled amount
  - is_known_mcc         : 1 if merchant category in user's top MCCs
  - is_home_city         : 1 if transaction city == user's home city
  - is_foreign           : 1 if merchant_country != India
  - high_risk_mcc        : 1 if MCC is Wire Transfer / Gambling / FX
  - is_high_risk_user    : 1 if user risk_tier == High

Output:
  backend/models/saved/risk_model.pkl   ← trained ensemble
  backend/models/saved/feature_cols.pkl ← feature column list
  backend/models/saved/model_report.json ← evaluation metrics

Run:
  python scripts/train_risk_model.py
"""

import os
import sys
import json
import math
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data" / "raw"
SAVE_DIR   = BASE_DIR / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── High-risk MCCs (Wire Transfer, Gambling, Currency Exchange)
HIGH_RISK_MCCS = {"4829", "7995", "6051"}

# ── Anomaly score thresholds (from FRAUD-001 internal policy)
THRESHOLD_BLOCK  = 0.65
THRESHOLD_HOLD   = 0.50
THRESHOLD_REVIEW = 0.38


# ─────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    txn_path     = DATA_DIR / "transactions.csv"
    profile_path = DATA_DIR / "user_profiles.csv"

    if not txn_path.exists():
        raise FileNotFoundError(f"transactions.csv not found at {txn_path}")
    if not profile_path.exists():
        raise FileNotFoundError(f"user_profiles.csv not found at {profile_path}")

    txns     = pd.read_csv(txn_path)
    profiles = pd.read_csv(profile_path)

    print(f"  ✅ Loaded {len(txns):,} transactions")
    print(f"  ✅ Loaded {len(profiles):,} user profiles")
    print(f"  ✅ Anomaly rate: {txns['is_anomaly'].mean()*100:.1f}%")

    return txns, profiles


# ─────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────

def engineer_features(
    txns: pd.DataFrame,
    profiles: pd.DataFrame
) -> pd.DataFrame:
    """
    Join transactions with user profiles and engineer
    behaviour-relative features.

    Key insight: A ₹50,000 transaction is only suspicious
    relative to THAT user's normal behaviour — not globally.
    """
    # Merge on user_id
    df = txns.merge(profiles, on="user_id", how="left")

    # ── Amount features ───────────────────────────────────
    df["avg_txn_amount_90d"]    = df["avg_txn_amount_90d"].fillna(df["amount"].median())
    df["stddev_txn_amount_90d"] = df["stddev_txn_amount_90d"].fillna(1.0)
    df["stddev_txn_amount_90d"] = df["stddev_txn_amount_90d"].replace(0, 1.0)

    # Z-score: how many standard deviations from user's average
    df["amount_zscore"] = (
        (df["amount"] - df["avg_txn_amount_90d"]) /
        df["stddev_txn_amount_90d"]
    ).clip(-10, 10)

    # Log-scaled amount (handles right-skewed INR distribution)
    df["amount_log"] = np.log1p(df["amount"])

    # Amount ratio to user average
    df["amount_ratio"] = (
        df["amount"] / df["avg_txn_amount_90d"].replace(0, 1)
    ).clip(0, 20)

    # ── Merchant Category features ────────────────────────
    def is_known_mcc(row):
        if pd.isna(row.get("top_merchant_categories")):
            return 0
        top_mccs = str(row["top_merchant_categories"]).split("|")
        return 1 if str(row["merchant_category_code"]) in top_mccs else 0

    df["is_known_mcc"]    = df.apply(is_known_mcc, axis=1)
    df["is_unknown_mcc"]  = 1 - df["is_known_mcc"]
    df["high_risk_mcc"]   = df["merchant_category_code"].astype(str).isin(
        HIGH_RISK_MCCS
    ).astype(int)

    # ── Geography features ────────────────────────────────
    df["home_city"] = df["home_city"].fillna("Unknown")
    df["merchant_city"] = df["merchant_city"].fillna("Unknown")

    df["is_home_city"] = (
        df["merchant_city"].str.lower() ==
        df["home_city"].str.lower()
    ).astype(int)

    df["is_foreign"] = (
        df["merchant_country"].str.lower() != "india"
    ).astype(int)

    df["ip_mismatch"] = (
        df["ip_country"].str.lower() != "india"
    ).astype(int)

    # ── User risk features ────────────────────────────────
    df["is_high_risk_user"] = (
        df["risk_tier"].str.lower() == "high"
    ).astype(int)

    df["is_medium_risk_user"] = (
        df["risk_tier"].str.lower() == "medium"
    ).astype(int)

    # ── Channel encoding ──────────────────────────────────
    channel_risk = {
        "upi": 0, "card": 1, "netbanking": 1,
        "wallet": 0, "wire": 3
    }
    df["channel_risk"] = df["channel"].str.lower().map(
        channel_risk
    ).fillna(1)

    # ── Composite risk score features ─────────────────────
    # Multiple risk signals firing together
    df["risk_signal_count"] = (
        df["is_unknown_mcc"] +
        df["high_risk_mcc"] +
        df["is_foreign"] +
        df["ip_mismatch"] +
        (df["amount_zscore"] > 3).astype(int) +
        df["is_high_risk_user"]
    )

    return df


# ─────────────────────────────────────────────────────────
# STEP 3 — PREPARE FEATURE MATRIX
# ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "amount_zscore",
    "amount_log",
    "amount_ratio",
    "is_known_mcc",
    "is_unknown_mcc",
    "high_risk_mcc",
    "is_home_city",
    "is_foreign",
    "ip_mismatch",
    "is_high_risk_user",
    "is_medium_risk_user",
    "channel_risk",
    "risk_signal_count",
]


def prepare_matrices(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0).values
    y = df["is_anomaly"].astype(int).values
    return X, y


# ─────────────────────────────────────────────────────────
# STEP 4 — TRAIN MODELS
# ─────────────────────────────────────────────────────────

def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """
    Isolation Forest — unsupervised anomaly detection.
    Learns what 'normal' looks like from the full dataset.
    contamination = expected anomaly rate (~5%)
    """
    print("\n  Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)
    print("  ✅ Isolation Forest trained")
    return iso


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> xgb.XGBClassifier:
    """
    XGBoost supervised classifier.
    Uses is_anomaly labels from synthetic data.
    scale_pos_weight handles class imbalance.
    """
    print("\n  Training XGBoost classifier...")

    # Handle class imbalance — ~95% normal, ~5% anomaly
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / max(pos_count, 1)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print(f"  ✅ XGBoost trained | best iteration: {xgb_model.best_iteration}")
    return xgb_model


# ─────────────────────────────────────────────────────────
# STEP 5 — ENSEMBLE + EVALUATE
# ─────────────────────────────────────────────────────────

def ensemble_score(
    iso: IsolationForest,
    xgb_model: xgb.XGBClassifier,
    X: np.ndarray,
    iso_weight: float = 0.35,
    xgb_weight: float = 0.65
) -> np.ndarray:
    """
    Combine Isolation Forest and XGBoost scores.

    IF score: raw decision function normalized to [0,1]
    XGB score: probability of anomaly class
    Final: weighted average
    """
    # Isolation Forest — lower = more anomalous, normalize to [0,1]
    iso_raw    = iso.decision_function(X)
    iso_min    = iso_raw.min()
    iso_max    = iso_raw.max()
    iso_scores = 1 - (iso_raw - iso_min) / max(iso_max - iso_min, 1e-8)

    # XGBoost — probability of class 1 (anomaly)
    xgb_scores = xgb_model.predict_proba(X)[:, 1]

    # Weighted ensemble
    final = (iso_weight * iso_scores) + (xgb_weight * xgb_scores)
    return final.clip(0, 1)


def evaluate(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float = THRESHOLD_BLOCK
) -> dict:
    """Evaluate ensemble at a given decision threshold."""
    y_pred = (scores >= threshold).astype(int)

    report = {
        "threshold":        threshold,
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":          round(roc_auc_score(y_true, scores), 4),
        "total_flagged":    int(y_pred.sum()),
        "true_positives":   int(((y_pred == 1) & (y_true == 1)).sum()),
        "false_positives":  int(((y_pred == 1) & (y_true == 0)).sum()),
        "false_negatives":  int(((y_pred == 0) & (y_true == 1)).sum()),
    }
    return report


def action_from_score(score: float) -> str:
    """Map ensemble score to action using FRAUD-001 thresholds."""
    if score >= THRESHOLD_BLOCK:
        return "BLOCK"
    elif score >= THRESHOLD_HOLD:
        return "HOLD"
    elif score >= THRESHOLD_REVIEW:
        return "REVIEW"
    return "CLEAR"


# ─────────────────────────────────────────────────────────
# STEP 6 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────

def save_artifacts(
    iso: IsolationForest,
    xgb_model: xgb.XGBClassifier,
    report: dict
):
    # Save ensemble models together
    model_bundle = {
        "isolation_forest": iso,
        "xgboost":          xgb_model,
        "feature_cols":     FEATURE_COLS,
        "thresholds": {
            "block":  THRESHOLD_BLOCK,
            "hold":   THRESHOLD_HOLD,
            "review": THRESHOLD_REVIEW
        },
        "iso_weight": 0.35,
        "xgb_weight": 0.65
    }

    model_path = SAVE_DIR / "risk_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\n  ✅ Model saved → {model_path}")

    # Save feature columns separately for quick reference
    feat_path = SAVE_DIR / "feature_cols.pkl"
    with open(feat_path, "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    # Save evaluation report as JSON
    report_path = SAVE_DIR / "model_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Report saved → {report_path}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  FINORA — Transaction Risk Model Trainer")
    print("="*55)

    # ── Load ──────────────────────────────────────────────
    print("\n📂 Loading data...")
    txns, profiles = load_data()

    # ── Feature Engineering ───────────────────────────────
    print("\n⚙️  Engineering features...")
    df = engineer_features(txns, profiles)
    X, y = prepare_matrices(df)
    print(f"  ✅ Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  ✅ Features: {FEATURE_COLS}")

    # ── Train / Val Split ─────────────────────────────────
    print("\n✂️  Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,}")
    print(f"  Anomalies in train: {y_train.sum()} | val: {y_val.sum()}")

    # ── Train Models ──────────────────────────────────────
    print("\n🤖 Training models...")
    iso       = train_isolation_forest(X_train)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    # ── Evaluate ──────────────────────────────────────────
    print("\n📊 Evaluating ensemble on validation set...")
    val_scores = ensemble_score(iso, xgb_model, X_val)
    print(f"\n  Score distribution (validation set):")
    print(f"  Min:    {val_scores.min():.4f}")
    print(f"  Max:    {val_scores.max():.4f}")
    print(f"  Mean:   {val_scores.mean():.4f}")
    print(f"  Median: {float(np.median(val_scores)):.4f}")
    print(f"  >0.85:  {(val_scores > 0.85).sum()} samples")
    print(f"  >0.65:  {(val_scores > 0.65).sum()} samples")
    print(f"  >0.45:  {(val_scores > 0.45).sum()} samples")
    print(f"  >0.30:  {(val_scores > 0.30).sum()} samples")

    report = evaluate(val_scores, y_val, threshold=0.65)

    print(f"\n  {'Metric':<20} {'Value'}")
    print(f"  {'─'*35}")
    print(f"  {'ROC-AUC':<20} {report['roc_auc']}")
    print(f"  {'Precision':<20} {report['precision']}")
    print(f"  {'Recall':<20} {report['recall']}")
    print(f"  {'F1 Score':<20} {report['f1_score']}")
    print(f"  {'Flagged':<20} {report['total_flagged']}")
    print(f"  {'True Positives':<20} {report['true_positives']}")
    print(f"  {'False Positives':<20} {report['false_positives']}")
    print(f"  {'False Negatives':<20} {report['false_negatives']}")

    # Action distribution
    print("\n  Action distribution on validation set:")
    actions = [action_from_score(s) for s in val_scores]
    for action in ["BLOCK", "HOLD", "REVIEW", "CLEAR"]:
        count = actions.count(action)
        pct   = count / len(actions) * 100
        bar   = "█" * int(pct / 2)
        print(f"  {action:<8} {bar:<25} {count:>4} ({pct:.1f}%)")

    # ── Save ──────────────────────────────────────────────
    print("\n💾 Saving artifacts...")
    save_artifacts(iso, xgb_model, report)

    print("\n" + "="*55)
    print("  ✅ Training complete!")
    print(f"  Model: backend/models/saved/risk_model.pkl")
    print(f"  ROC-AUC: {report['roc_auc']} | F1: {report['f1_score']}")
    print("\n  ⭐ Interview talking point:")
    print(f"  'My ensemble (IF + XGBoost) achieves {report['roc_auc']}")
    print(f"   ROC-AUC on 10,000 synthetic fintech transactions'")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()