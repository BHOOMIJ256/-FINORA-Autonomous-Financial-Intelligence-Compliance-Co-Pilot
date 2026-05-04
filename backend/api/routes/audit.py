"""
FINORA — Audit Trail API Routes
=================================
Endpoints:
  POST /api/audit/log              — manually persist an audit entry
  GET  /api/audit/logs             — retrieve logs with filters
  GET  /api/audit/logs/{log_id}    — get single log entry
  GET  /api/audit/summary          — dashboard stats
  POST /api/audit/query            — natural language query interface
  GET  /api/audit/health           — agent health check
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from agents.audit_agent import (
    persist_log,
    query_logs,
    get_log_by_id,
    get_summary,
    answer_audit_query
)

router = APIRouter()


# ── Request / Response Schemas ────────────────────────────

class AuditLogEntry(BaseModel):
    log_id:              Optional[str] = None
    agent_name:          str
    trigger_type:        str
    input_summary:       Optional[str] = None
    output_summary:      Optional[str] = None
    full_reasoning:      Optional[str] = None
    confidence_score:    Optional[float] = 0.0
    action_taken:        Optional[str] = None
    timestamp:           Optional[str] = None
    related_entity_id:   Optional[str] = None
    related_entity_type: Optional[str] = None
    query_text:          Optional[str] = None


class NLQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=300,
        example="What transactions did FINORA block today?"
    )


# ── Endpoints ─────────────────────────────────────────────

@router.post("/log", status_code=201)
async def create_audit_log(
    entry: AuditLogEntry,
    db: AsyncSession = Depends(get_db)
):
    """
    Manually persist a single audit log entry.
    Primarily called internally by other agents as background tasks.
    """
    success = await persist_log(entry.model_dump(), db)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to persist audit log"
        )
    return {"status": "ok", "log_id": entry.log_id}


@router.get("/logs")
async def get_audit_logs(
    agent_name:          Optional[str] = Query(None, description="Filter by agent name"),
    action_taken:        Optional[str] = Query(None, description="BLOCK | HOLD | REVIEW | CLEAR | REPORT_GENERATED"),
    related_entity_type: Optional[str] = Query(None, description="transaction | regulation | market_scan"),
    related_entity_id:   Optional[str] = Query(None, description="Specific entity ID"),
    since_hours:         Optional[int] = Query(None, description="Only logs from last N hours"),
    limit:               int           = Query(50, ge=1, le=200),
    offset:              int           = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve audit logs with optional filters.
    All parameters are optional — omit to get all logs.

    Examples:
      /api/audit/logs?agent_name=transaction_risk_agent&action_taken=BLOCK
      /api/audit/logs?since_hours=24&limit=20
      /api/audit/logs?related_entity_type=transaction
    """
    try:
        logs = await query_logs(
            db,
            agent_name=agent_name,
            action_taken=action_taken,
            related_entity_type=related_entity_type,
            related_entity_id=related_entity_id,
            since_hours=since_hours,
            limit=limit,
            offset=offset
        )
        return {
            "count":   len(logs),
            "offset":  offset,
            "limit":   limit,
            "filters": {
                "agent_name":          agent_name,
                "action_taken":        action_taken,
                "related_entity_type": related_entity_type,
                "since_hours":         since_hours
            },
            "logs": logs
        }
    except Exception as e:
        logger.error(f"[API] /audit/logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/{log_id}")
async def get_single_log(
    log_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve a single audit log entry by its UUID.
    Returns the full entry including complete reasoning chain.
    """
    log = await get_log_by_id(log_id, db)
    if not log:
        raise HTTPException(
            status_code=404,
            detail=f"Audit log {log_id} not found"
        )
    return log


@router.get("/summary")
async def get_audit_summary(db: AsyncSession = Depends(get_db)):
    """
    Dashboard summary statistics.

    Returns:
      - total_logs: total decisions logged
      - last_24h: activity in last 24 hours
      - high_confidence: decisions with confidence >= 0.8
      - by_agent: count per agent
      - by_action: count per action type
      - recent_activity: last 5 log entries
    """
    try:
        summary = await get_summary(db)
        return summary
    except Exception as e:
        logger.error(f"[API] /audit/summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def natural_language_audit_query(
    request: NLQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Natural language audit query interface.

    Ask anything about FINORA's decision history in plain English.

    Example questions:
      - "What transactions did FINORA block today?"
      - "Show me all compliance reports from this week"
      - "Which agent made the most decisions recently?"
      - "What was the reasoning behind the last BLOCK decision?"
      - "How many transactions were flagged for review?"
    """
    logger.info(f"[API] /audit/query → {request.question[:60]}")
    try:
        result = await answer_audit_query(request.question, db)
        return result
    except Exception as e:
        logger.error(f"[API] /audit/query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def audit_health(db: AsyncSession = Depends(get_db)):
    """Check Agent 4 health — DB connection and log count."""
    try:
        from sqlalchemy import text
        result = await db.execute(text("SELECT COUNT(*) FROM audit_logs"))
        count  = result.scalar() or 0
        return {
            "agent":      "audit_trail_agent",
            "status":     "ok",
            "total_logs": count
        }
    except Exception as e:
        return {
            "agent":  "audit_trail_agent",
            "status": "error",
            "detail": str(e)
        }