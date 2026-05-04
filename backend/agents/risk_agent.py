"""
FINORA — Agent 2: Transaction Risk Agent
==========================================
Analyzes individual transactions for anomalies using:
  1. Feature engineering against user profile baseline
  2. ML Ensemble (Isolation Forest + XGBoost) scoring
  3. Groq LLM for human-readable risk explanation
  4. Policy cross-referencing (FRAUD-001, AML-001, AML-002)
"""

import os
import sys
import math
import json
import pickle
import uuid
import re
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "saved" / "risk_model.pkl"
DATA_DIR   = BASE_DIR / "data" / "raw"

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL     = "llama-3.3-70b-versatile"
HIGH_RISK_MCCS = {"4829", "7995", "6051"}

POLICY_REFS = {
    "BLOCK": [
        {"policy_id": "FRAUD-001", "policy_name": "Fraud Prevention & Response Policy",
         "rule": "Block if ML score exceeds threshold per FRAUD-001"},
        {"policy_id": "AML-002", "policy_name": "Suspicious Transaction Monitoring Policy",
         "rule": "High ML scores require immediate action"}
    ],
    "HOLD": [
        {"policy_id": "AML-002", "policy_name": "Suspicious Transaction Monitoring Policy",
         "rule": "Scores above hold threshold require human analyst review"},
        {"policy_id": "AML-001", "policy_name": "Anti-Money Laundering & CFT Policy",
         "rule": "Enhanced Due Diligence for suspicious patterns"}
    ],
    "REVIEW": [
        {"policy_id": "AML-002", "policy_name": "Suspicious Transaction Monitoring Policy",
         "rule": "Velocity checks and pattern monitoring required"},
        {"policy_id": "FRAUD-001", "policy_name": "Fraud Prevention & Response Policy",
         "rule": "100% real-time fraud detection coverage"}
    ],
    "CLEAR": []
}

FOREIGN_POLICY = {
    "policy_id": "AML-001",
    "policy_name": "Anti-Money Laundering & CFT Policy",
    "rule": "Enhanced Due Diligence required for cross-border transactions"
}

_model_bundle = None
_llm = None


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found. Run train_risk_model.py first.")
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)
        logger.info("[Agent2] Risk model loaded")
    return _model_bundle


def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL,
                        temperature=0.1, max_tokens=1024)
    return _llm


async def fetch_transaction(transaction_id, db):
    result = await db.execute(
        text("SELECT * FROM transactions WHERE transaction_id = :tid"),
        {"tid": transaction_id}
    )
    row = result.mappings().first()
    return dict(row) if row else None


async def fetch_user_profile(user_id, db):
    result = await db.execute(
        text("SELECT * FROM user_profiles WHERE user_id = :uid"),
        {"uid": user_id}
    )
    row = result.mappings().first()
    return dict(row) if row else None


def engineer_features(txn, profile):
    amount      = float(txn.get("amount", 0))
    user_avg    = float(profile.get("avg_txn_amount_90d", amount))
    user_std    = float(profile.get("stddev_txn_amount_90d", 1)) or 1.0
    home_city   = str(profile.get("home_city", "")).lower()
    txn_city    = str(txn.get("merchant_city", "")).lower()
    txn_country = str(txn.get("merchant_country", "India")).lower()
    ip_country  = str(txn.get("ip_country", "India")).lower()
    mcc         = str(txn.get("merchant_category_code", ""))
    channel     = str(txn.get("channel", "")).lower()
    risk_tier   = str(profile.get("risk_tier", "Low")).lower()
    top_mccs    = [m.strip() for m in str(profile.get("top_merchant_categories", "")).split("|") if m.strip()]

    channel_risk_map = {"upi": 0, "card": 1, "netbanking": 1, "wallet": 0, "wire": 3}

    amount_zscore     = max(-10, min(10, (amount - user_avg) / user_std))
    amount_log        = math.log1p(amount)
    amount_ratio      = min(amount / max(user_avg, 1), 20)
    is_known_mcc      = 1 if mcc in top_mccs else 0
    is_unknown_mcc    = 1 - is_known_mcc
    high_risk_mcc     = 1 if mcc in HIGH_RISK_MCCS else 0
    is_home_city      = 1 if txn_city == home_city else 0
    is_foreign        = 1 if txn_country != "india" else 0
    ip_mismatch       = 1 if ip_country != "india" else 0
    is_high_risk      = 1 if risk_tier == "high" else 0
    is_medium_risk    = 1 if risk_tier == "medium" else 0
    ch_risk           = channel_risk_map.get(channel, 1)
    risk_signal_count = (is_unknown_mcc + high_risk_mcc + is_foreign +
                         ip_mismatch + (1 if amount_zscore > 3 else 0) + is_high_risk)

    return {
        "vector": [amount_zscore, amount_log, amount_ratio, is_known_mcc,
                   is_unknown_mcc, high_risk_mcc, is_home_city, is_foreign,
                   ip_mismatch, is_high_risk, is_medium_risk, ch_risk, risk_signal_count],
        "raw": {
            "amount_zscore": round(amount_zscore, 3),
            "amount_ratio": round(amount_ratio, 2),
            "is_known_mcc": is_known_mcc,
            "is_home_city": is_home_city,
            "is_foreign": is_foreign,
            "ip_mismatch": ip_mismatch,
            "high_risk_mcc": high_risk_mcc,
            "risk_signal_count": risk_signal_count,
            "user_avg": round(user_avg, 2),
            "user_std": round(user_std, 2),
            "top_mccs": top_mccs
        }
    }


def run_ml_inference(feature_vector):
    bundle     = get_model_bundle()
    iso        = bundle["isolation_forest"]
    xgb_m      = bundle["xgboost"]
    thresholds = bundle["thresholds"]
    iso_w      = bundle.get("iso_weight", 0.35)
    xgb_w      = bundle.get("xgb_weight", 0.65)

    X         = np.array([feature_vector])
    iso_raw   = iso.decision_function(X)[0]
    iso_score = float(np.clip(1 - (iso_raw + 0.5), 0, 1))
    xgb_score = float(xgb_m.predict_proba(X)[0][1])
    final     = float(np.clip((iso_w * iso_score) + (xgb_w * xgb_score), 0, 1))

    if final >= thresholds["block"]:   action = "BLOCK"
    elif final >= thresholds["hold"]:  action = "HOLD"
    elif final >= thresholds["review"]:action = "REVIEW"
    else:                              action = "CLEAR"

    return {"risk_score": round(final, 4), "iso_score": round(iso_score, 4),
            "xgb_score": round(xgb_score, 4), "action": action}


def detect_anomaly_types(features_raw, txn):
    types = []
    if features_raw["amount_zscore"] > 3:    types.append("amount_outlier")
    if not features_raw["is_known_mcc"]:     types.append("mcc_mismatch")
    if not features_raw["is_home_city"]:     types.append("geo_mismatch")
    if features_raw["is_foreign"]:           types.append("foreign_country")
    if features_raw["ip_mismatch"]:          types.append("ip_country_mismatch")
    if features_raw["high_risk_mcc"]:        types.append("high_risk_merchant_category")
    return types if types else ["statistical_anomaly"]


SYSTEM_PROMPT = """You are FINORA's Transaction Risk Agent for Indian fintech compliance.
Explain transaction anomalies specifically, citing exact amounts, cities, and merchant categories.
Reference internal policy IDs where relevant. Respond with valid JSON only."""


def get_llm_explanation(txn, profile, features_raw, ml_result, anomaly_types):
    from langchain.schema import HumanMessage, SystemMessage

    if ml_result["action"] == "CLEAR":
        return {"explanation": "Transaction matches user's normal behaviour patterns.",
                "evidence": ["Amount within normal range", "Known merchant category",
                             "Expected geography"],
                "recommended_action": "No action required.", "confidence": 0.05}

    prompt = f"""TRANSACTION: {txn.get('transaction_id')} | ₹{txn.get('amount'):,.2f}
Merchant: {txn.get('merchant_name')} | MCC: {txn.get('merchant_category_code')}
City: {txn.get('merchant_city')} | Country: {txn.get('merchant_country')}
Channel: {txn.get('channel')} | IP Country: {txn.get('ip_country')}

USER BASELINE:
Avg: ₹{features_raw['user_avg']:,.2f} | Std: ₹{features_raw['user_std']:,.2f}
Home City: {profile.get('home_city')} | Risk Tier: {profile.get('risk_tier')}
Known MCCs: {', '.join(features_raw['top_mccs'][:5])}

ML RESULT:
Risk Score: {ml_result['risk_score']} | Action: {ml_result['action']}
Anomaly Types: {', '.join(anomaly_types)}
Amount Z-Score: {features_raw['amount_zscore']} | Ratio: {features_raw['amount_ratio']}x avg
Risk Signals: {features_raw['risk_signal_count']}/6 fired

Respond ONLY with JSON:
{{
  "explanation": "2-3 sentences explaining WHY this is suspicious with specific numbers",
  "evidence": ["evidence point 1 with exact values", "evidence point 2", "evidence point 3"],
  "recommended_action": "specific action for compliance team",
  "confidence": 0.0
}}"""

    try:
        llm      = get_llm()
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                               HumanMessage(content=prompt)])

        # Guard against empty response
        if not response or not response.content or not response.content.strip():
            raise ValueError("Empty response from Groq")

        raw = re.sub(r"^```json\s*", "", response.content.strip())
        raw = re.sub(r"\s*```$", "", raw).strip()

        # Guard against non-JSON response
        if not raw.startswith("{"):
            raise ValueError(f"Non-JSON response: {raw[:100]}")

        return json.loads(raw)

    except Exception as e:
        logger.error(f"[Agent2] LLM error: {e}")
        return {
            "explanation": (
                f"Transaction ₹{txn.get('amount'):,.2f} at "
                f"{txn.get('merchant_name', 'unknown merchant')} "
                f"flagged for {', '.join(anomaly_types)}. "
                f"Risk score {ml_result['risk_score']} triggers "
                f"{ml_result['action']} action per FRAUD-001 policy."
            ),
            "evidence": [
                f"Risk score: {ml_result['risk_score']} "
                f"(threshold: {ml_result['action']})",
                f"Amount ratio: {features_raw['amount_ratio']}x "
                f"user's 90-day average of ₹{features_raw['user_avg']:,.2f}",
                f"Anomaly signals detected: "
                f"{', '.join(anomaly_types)}"
            ],
            "recommended_action": (
                f"Manual review required. Compliance team to verify "
                f"transaction per AML-002 monitoring policy."
            ),
            "confidence": ml_result["risk_score"]
        }


def get_policy_refs(action, anomaly_types):
    refs = list(POLICY_REFS.get(action, []))
    if "foreign_country" in anomaly_types or "ip_country_mismatch" in anomaly_types:
        if FOREIGN_POLICY not in refs:
            refs.append(FOREIGN_POLICY)
    return refs


def build_audit_entry(txn, result):
    return {
        "log_id": str(uuid.uuid4()),
        "agent_name": "transaction_risk_agent",
        "trigger_type": "transaction_analysis",
        "input_summary": f"TXN {txn.get('transaction_id')} | ₹{txn.get('amount')} | {txn.get('merchant_name', '')[:50]}",
        "output_summary": f"Action: {result['action']} | Score: {result['risk_score']} | Types: {', '.join(result['anomaly_types'])}",
        "full_reasoning": result.get("explanation", "")[:1000],
        "confidence_score": result.get("confidence", 0.0),
        "action_taken": result["action"],
        "timestamp": datetime.utcnow().isoformat(),
        "related_entity_id": txn.get("transaction_id"),
        "related_entity_type": "transaction",
        "query_text": None
    }


async def analyze_transaction(transaction_id, db):
    logger.info(f"\n[Agent2] === Analyzing: {transaction_id} ===")

    txn = await fetch_transaction(transaction_id, db)
    if not txn:
        return {"error": f"Transaction {transaction_id} not found",
                "transaction_id": transaction_id, "action": "ERROR"}

    profile = await fetch_user_profile(txn["user_id"], db)
    if not profile:
        profile = {"user_id": txn["user_id"], "user_name": "Unknown",
                   "home_city": txn.get("merchant_city", "Unknown"),
                   "risk_tier": "Medium", "avg_txn_amount_90d": txn["amount"],
                   "stddev_txn_amount_90d": 1.0,
                   "top_merchant_categories": txn.get("merchant_category_code", "")}

    features      = engineer_features(txn, profile)
    ml_result     = run_ml_inference(features["vector"])
    anomaly_types = detect_anomaly_types(features["raw"], txn)
    llm_output    = get_llm_explanation(txn, profile, features["raw"],
                                        ml_result, anomaly_types)

    result = {
        "transaction_id":     transaction_id,
        "user_id":            txn["user_id"],
        "user_name":          profile.get("user_name", "Unknown"),
        "action":             ml_result["action"],
        "risk_score":         ml_result["risk_score"],
        "iso_score":          ml_result["iso_score"],
        "xgb_score":          ml_result["xgb_score"],
        "anomaly_types":      anomaly_types,
        "explanation":        llm_output.get("explanation", ""),
        "evidence":           llm_output.get("evidence", []),
        "recommended_action": llm_output.get("recommended_action", ""),
        "policy_refs":        get_policy_refs(ml_result["action"], anomaly_types),
        "user_context": {
            "home_city":          profile.get("home_city"),
            "risk_tier":          profile.get("risk_tier"),
            "avg_txn_amount_90d": features["raw"]["user_avg"],
            "amount_zscore":      features["raw"]["amount_zscore"],
            "amount_ratio":       features["raw"]["amount_ratio"]
        },
        "transaction": {
            "amount":           txn.get("amount"),
            "merchant_name":    txn.get("merchant_name"),
            "merchant_city":    txn.get("merchant_city"),
            "merchant_country": txn.get("merchant_country"),
            "channel":          txn.get("channel"),
            "timestamp":        str(txn.get("timestamp", ""))
        },
        "confidence":   llm_output.get("confidence", ml_result["risk_score"]),
        "analyzed_at":  datetime.utcnow().isoformat()
    }

    logger.success(f"[Agent2] Done | action={result['action']} | score={result['risk_score']}")
    return result


async def analyze_batch(transaction_ids, db):
    results = []
    for tid in transaction_ids:
        try:
            r = await analyze_transaction(tid, db)
            results.append(r)
        except Exception as e:
            logger.error(f"[Agent2] Batch error on {tid}: {e}")
            results.append({"transaction_id": tid, "action": "ERROR", "error": str(e)})
    return results


if __name__ == "__main__":
    print("\n[Agent2] Offline ML test (no DB, no LLM)")
    print("─" * 50)

    txn_df  = pd.read_csv(DATA_DIR / "transactions.csv")
    prof_df = pd.read_csv(DATA_DIR / "user_profiles.csv")

    anomalies = txn_df[txn_df["is_anomaly"] == True].head(3)
    normals   = txn_df[txn_df["is_anomaly"] == False].head(2)

    for _, row in pd.concat([anomalies, normals]).iterrows():
        txn     = row.to_dict()
        matches = prof_df[prof_df["user_id"] == txn["user_id"]]
        if matches.empty:
            continue
        profile       = matches.iloc[0].to_dict()
        features      = engineer_features(txn, profile)
        ml_result     = run_ml_inference(features["vector"])
        anomaly_types = detect_anomaly_types(features["raw"], txn)

        expected = "ANOMALY" if txn["is_anomaly"] else "NORMAL"
        correct  = "✅" if (ml_result["action"] != "CLEAR") == bool(txn["is_anomaly"]) else "⚠️"

        print(f"\n{correct} {txn['transaction_id']} | Expected: {expected}")
        print(f"   ML Action: {ml_result['action']} | Score: {ml_result['risk_score']}")
        print(f"   Detected:  {', '.join(anomaly_types)}")
        print(f"   Z-Score:   {features['raw']['amount_zscore']}")