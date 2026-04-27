"""
FINORA — Compliance API Routes
================================
File location: backend/api/routes/compliance.py

Endpoints:
  POST /api/compliance/analyze     — run full compliance analysis
  GET  /api/compliance/policies    — list all internal policies
  GET  /api/compliance/health      — agent health check
"""

import asyncio
import os
from functools import partial
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from agents.regulatory_agent import (
    analyze_regulation,
    load_internal_policies,
    build_audit_entry,
    get_qdrant_client,
)

# ── Router with explicit prefix ───────────────────────────
# Fix 1: prefix set HERE not in main.py — prevents /analyze vs
# /api/compliance/analyze mismatch between docs and reality
router = APIRouter(prefix="/api/compliance", tags=["Compliance — Agent 1"])

# ── Risk level normaliser ─────────────────────────────────
# Fix 2: LLM sometimes returns "HIGH", "high risk", "Moderate" etc.
# This guards the frontend from unexpected values
VALID_RISK_LEVELS = {"High", "Medium", "Low", "Unknown"}

def normalise_risk_level(raw: str) -> str:
    capitalised = raw.strip().capitalize()
    return capitalised if capitalised in VALID_RISK_LEVELS else "Unknown"


# ── Request / Response schemas ────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Natural language query about a regulation",
        example="Latest RBI guidelines on KYC for payment aggregators"
    )
    # Fix 3: top_k is accepted AND passed through to the agent
    # Previously it was accepted but silently discarded
    top_k: Optional[int] = Field(
        default=6,
        ge=1,
        le=15,
        description="Number of regulatory chunks to retrieve from Qdrant"
    )


class AffectedPolicy(BaseModel):
    policy_id:   str
    policy_name: str
    reason:      str


class AnalyzeResponse(BaseModel):
    query:             str
    summary:           str
    affected_policies: list[AffectedPolicy]
    gap_report:        str
    risk_level:        str
    retrieved_sources: list[dict]
    confidence:        float
    error:             Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_compliance(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    Run a full regulatory compliance analysis.

    Retrieves relevant RBI/SEBI regulatory chunks from Qdrant,
    cross-references against internal policies, and returns a
    structured compliance gap report.
    """
    logger.info(f"[API] POST /api/compliance/analyze — query: {request.query[:60]}")

    try:
        # Fix 4: analyze_regulation is synchronous (uses SentenceTransformer
        # + QdrantClient, both blocking). Calling it directly inside async
        # blocks the entire FastAPI event loop for 3-8 seconds.
        # run_in_executor moves it to a thread pool — other requests
        # continue processing while this one runs.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(analyze_regulation, request.query, request.top_k)
        )

        # Build audit entry and persist in background (non-blocking)
        audit_entry = build_audit_entry(request.query, result)
        background_tasks.add_task(_persist_audit_log, audit_entry)

        # Defensive construction — skip any malformed policy dicts
        # that are missing required fields
        valid_policies = [
            AffectedPolicy(**p)
            for p in result.get("affected_policies", [])
            if isinstance(p, dict)
            and all(k in p for k in ["policy_id", "policy_name", "reason"])
        ]

        return AnalyzeResponse(
            query=request.query,
            summary=result.get("summary", ""),
            affected_policies=valid_policies,
            gap_report=result.get("gap_report", ""),
            # Fix 5: normalise before returning — guards against
            # "HIGH", "high risk", "Moderate", "N/A" from the LLM
            risk_level=normalise_risk_level(result.get("risk_level", "Unknown")),
            retrieved_sources=result.get("retrieved_sources", []),
            confidence=float(result.get("confidence", 0.0)),
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"[API] /compliance/analyze failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/policies")
async def get_internal_policies():
    """
    Return all internal company policies from internal_policies.csv.
    Used by the frontend Compliance Center to display the policy list.
    """
    policies = load_internal_policies()
    return {
        "count":    len(policies),
        "policies": policies
    }


@router.get("/health")
async def compliance_health():
    """
    Health check for Agent 1.
    Verifies Qdrant connection, policy file presence, and Groq API key.
    Safe to call repeatedly — used by the dashboard status indicator.
    """
    status = {
        "agent":  "regulatory_watch",
        "status": "ok",
        "checks": {}
    }

    # Check Qdrant
    try:
        # Fix 6: reuse the singleton from regulatory_agent instead of
        # creating a new QdrantClient on every health check call
        client = get_qdrant_client()
        info   = client.get_collection("regulatory_docs")
        status["checks"]["qdrant"] = {
            "status": "ok",
            "chunks": info.points_count
        }
    except Exception as e:
        status["checks"]["qdrant"] = {
            "status": "error",
            "detail": str(e)
        }
        status["status"] = "degraded"

    # Check internal policies file
    policies = load_internal_policies()
    status["checks"]["internal_policies"] = {
        "status": "ok" if policies else "missing",
        "count":  len(policies)
    }

    # Check Groq API key presence
    groq_key = os.getenv("GROQ_API_KEY", "")
    status["checks"]["groq_api_key"] = {
        "status": "ok" if groq_key else "missing"
    }

    # Set overall status to degraded if any check failed
    if any(
        v.get("status") not in ("ok",)
        for v in status["checks"].values()
        if isinstance(v, dict)
    ):
        status["status"] = "degraded"

    return status


# ── Background task ───────────────────────────────────────

async def _persist_audit_log(entry: dict):
    """
    Persist audit log entry to PostgreSQL.

    Currently logs to console only.
    When Agent 4 (Audit Trail Agent) is built, replace the logger
    call with an actual DB insert. The signature will become:
        async def _persist_audit_log(entry: dict, db: AsyncSession)
    and the caller will pass the db session explicitly:
        background_tasks.add_task(_persist_audit_log, audit_entry, db)
    """
    logger.debug(
        f"[AuditLog] id={entry.get('log_id')} | "
        f"agent={entry.get('agent_name')} | "
        f"action={entry.get('action_taken')} | "
        f"confidence={entry.get('confidence_score')} | "
        f"ts={entry.get('timestamp')}"
    )