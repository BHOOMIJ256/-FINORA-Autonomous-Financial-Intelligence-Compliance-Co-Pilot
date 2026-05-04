"""
FINORA — Transaction Risk API Routes
======================================
Endpoints:
  POST /api/transactions/analyze       — analyze single transaction
  POST /api/transactions/batch         — analyze multiple transactions
  GET  /api/transactions/flagged       — get all flagged transactions
  GET  /api/transactions/feed          — live risk feed (last 50 analyzed)
  GET  /api/transactions/health        — agent health check
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from agents.risk_agent import (
    analyze_transaction, analyze_batch, build_audit_entry
)

router = APIRouter()


# ── Request / Response schemas ────────────────────────────

class AnalyzeRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN000042")


class BatchRequest(BaseModel):
    transaction_ids: list[str] = Field(..., max_items=50)


class PolicyRef(BaseModel):
    policy_id:   str
    policy_name: str
    rule:        str


class TransactionRiskResponse(BaseModel):
    transaction_id:     str
    user_id:            str
    user_name:          str
    action:             str
    risk_score:         float
    anomaly_types:      list[str]
    explanation:        str
    evidence:           list[str]
    recommended_action: str
    policy_refs:        list[dict]
    user_context:       dict
    transaction:        dict
    confidence:         float
    analyzed_at:        str
    error:              Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────

@router.post("/analyze", response_model=TransactionRiskResponse)
async def analyze_single_transaction(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a single transaction for risk.
    Returns full structured risk report with ML score,
    anomaly types, LLM explanation, and policy references.
    """
    logger.info(f"[API] /transactions/analyze → {request.transaction_id}")

    try:
        result = await analyze_transaction(request.transaction_id, db)

        if result.get("action") == "ERROR":
            raise HTTPException(status_code=404, detail=result.get("error"))

        # Log audit entry in background
        txn_data = result.get("transaction", {})
        txn_data["transaction_id"] = result["transaction_id"]
        txn_data["user_id"]        = result["user_id"]
        audit_entry = build_audit_entry(txn_data, result)
        background_tasks.add_task(_persist_audit_log, audit_entry, db)

        return TransactionRiskResponse(**{
            k: v for k, v in result.items()
            if k in TransactionRiskResponse.model_fields
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def analyze_batch_transactions(
    request: BatchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze up to 50 transactions in one call.
    Used by dashboard live risk feed.
    """
    logger.info(f"[API] /transactions/batch → {len(request.transaction_ids)} txns")
    results = await analyze_batch(request.transaction_ids, db)
    flagged = [r for r in results if r.get("action") in ("BLOCK", "HOLD")]
    return {
        "total":    len(results),
        "flagged":  len(flagged),
        "results":  results
    }


@router.get("/flagged")
async def get_flagged_transactions(
    action: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """
    Return transactions flagged as BLOCK or HOLD.
    Pulls from audit_logs table populated by Agent 4.
    """
    try:
        if action:
            query = text("""
                SELECT * FROM audit_logs
                WHERE agent_name = 'transaction_risk_agent'
                AND action_taken = :action
                ORDER BY timestamp DESC LIMIT :lim
            """)
            result = await db.execute(query, {"action": action.upper(), "lim": limit})
        else:
            query = text("""
                SELECT * FROM audit_logs
                WHERE agent_name = 'transaction_risk_agent'
                AND action_taken IN ('BLOCK', 'HOLD')
                ORDER BY timestamp DESC LIMIT :lim
            """)
            result = await db.execute(query, {"lim": limit})

        rows = [dict(r) for r in result.mappings().all()]
        return {"count": len(rows), "flagged": rows}

    except Exception as e:
        logger.error(f"[API] flagged error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample-anomalies")
async def get_sample_anomalies(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Fetch sample anomalous transaction IDs from the database.
    Useful for testing the analyze endpoint without knowing IDs.
    """
    try:
        result = await db.execute(
            text("""
                SELECT transaction_id, user_id, amount,
                       merchant_name, anomaly_type
                FROM transactions
                WHERE is_anomaly = true
                ORDER BY RANDOM()
                LIMIT :lim
            """),
            {"lim": limit}
        )
        rows = [dict(r) for r in result.mappings().all()]
        return {"count": len(rows), "samples": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def transaction_agent_health(db: AsyncSession = Depends(get_db)):
    """Check Agent 2 health — model, DB, LLM."""
    import os
    from pathlib import Path

    status = {"agent": "transaction_risk_agent", "status": "ok", "checks": {}}

    # Check ML model
    model_path = Path("models/saved/risk_model.pkl")
    status["checks"]["ml_model"] = {
        "status": "ok" if model_path.exists() else "missing",
        "path": str(model_path)
    }

    # Check DB connection
    try:
        result = await db.execute(text("SELECT COUNT(*) FROM transactions"))
        count  = result.scalar()
        status["checks"]["database"] = {"status": "ok", "transactions": count}
    except Exception as e:
        status["checks"]["database"] = {"status": "error", "detail": str(e)}
        status["status"] = "degraded"

    # Check Groq key
    status["checks"]["groq_api_key"] = {
        "status": "ok" if os.getenv("GROQ_API_KEY") else "missing"
    }

    return status


# ── Background task ───────────────────────────────────────

from agents.audit_agent import persist_log

async def _persist_audit_log(entry: dict, db: AsyncSession):
    await persist_log(entry, db)