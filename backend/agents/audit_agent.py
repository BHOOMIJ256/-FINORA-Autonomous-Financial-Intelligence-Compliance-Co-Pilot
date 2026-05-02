"""
FINORA — Agent 4: Audit Trail Agent
======================================
Handles persistence and querying of all agent decisions.

Three core functions:
  1. persist_log(entry, db)         — write audit entry to PostgreSQL
  2. query_logs(filters, db)        — retrieve logs with flexible filters
  3. answer_audit_query(question, db) — NL query → structured answer via LLM

Every agent (1, 2, 3) calls persist_log() as a background task
after every decision. Agent 4 owns the read path entirely.
"""

import os
import re
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.3-70b-versatile"

# ── Singleton ─────────────────────────────────────────────
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
    return _llm


# ─────────────────────────────────────────────────────────
# COMPONENT 1 — PERSISTENCE (WRITE PATH)
# ─────────────────────────────────────────────────────────

async def persist_log(entry: dict, db: AsyncSession) -> bool:
    """
    Persist a single audit log entry to PostgreSQL.

    Called as a background task by all three agents.
    Replaces all placeholder _persist_audit_log functions.

    Args:
        entry: audit entry dict from build_audit_entry()
        db:    AsyncSession from FastAPI dependency

    Returns:
        True on success, False on failure
    """
    try:
        # Ensure log_id exists
        if not entry.get("log_id"):
            entry["log_id"] = str(uuid.uuid4())

        # Ensure timestamp exists
        if not entry.get("timestamp"):
            entry["timestamp"] = datetime.utcnow().isoformat()

        # Truncate long fields to fit column limits
        entry["input_summary"]   = str(entry.get("input_summary",  "") or "")[:500]
        entry["output_summary"]  = str(entry.get("output_summary", "") or "")[:500]
        entry["full_reasoning"]  = str(entry.get("full_reasoning", "") or "")[:2000]
        entry["action_taken"]    = str(entry.get("action_taken",   "") or "")[:50]
        entry["agent_name"]      = str(entry.get("agent_name",     "") or "")[:50]
        entry["trigger_type"]    = str(entry.get("trigger_type",   "") or "")[:50]
        entry["query_text"]      = str(entry.get("query_text",     "") or "")[:500] or None

        await db.execute(
            text("""
                INSERT INTO audit_logs (
                    log_id, agent_name, trigger_type,
                    input_summary, output_summary, full_reasoning,
                    confidence_score, action_taken, timestamp,
                    related_entity_id, related_entity_type, query_text
                ) VALUES (
                    :log_id, :agent_name, :trigger_type,
                    :input_summary, :output_summary, :full_reasoning,
                    :confidence_score, :action_taken, :timestamp,
                    :related_entity_id, :related_entity_type, :query_text
                )
                ON CONFLICT (log_id) DO NOTHING
            """),
            {
                "log_id":              entry["log_id"],
                "agent_name":          entry["agent_name"],
                "trigger_type":        entry["trigger_type"],
                "input_summary":       entry["input_summary"],
                "output_summary":      entry["output_summary"],
                "full_reasoning":      entry["full_reasoning"],
                "confidence_score":    float(entry.get("confidence_score", 0.0) or 0.0),
                "action_taken":        entry["action_taken"],
                "timestamp":           entry["timestamp"],
                "related_entity_id":   entry.get("related_entity_id"),
                "related_entity_type": entry.get("related_entity_type"),
                "query_text":          entry.get("query_text")
            }
        )
        await db.commit()
        logger.debug(
            f"[Agent4] Persisted: {entry['agent_name']} | "
            f"{entry['action_taken']} | {entry['log_id'][:8]}..."
        )
        return True

    except Exception as e:
        logger.error(f"[Agent4] Persist failed: {e}")
        await db.rollback()
        return False


async def persist_log_batch(entries: list[dict], db: AsyncSession) -> int:
    """Persist multiple audit entries at once. Returns count of successful inserts."""
    success = 0
    for entry in entries:
        if await persist_log(entry, db):
            success += 1
    return success


# ─────────────────────────────────────────────────────────
# COMPONENT 2 — QUERYING (READ PATH)
# ─────────────────────────────────────────────────────────

async def query_logs(
    db: AsyncSession,
    agent_name:          Optional[str] = None,
    action_taken:        Optional[str] = None,
    related_entity_type: Optional[str] = None,
    related_entity_id:   Optional[str] = None,
    since_hours:         Optional[int] = None,
    limit:               int = 50,
    offset:              int = 0
) -> list[dict]:
    """
    Retrieve audit logs with flexible filters.

    All filters are optional — pass only what you need.
    Results ordered by timestamp DESC (newest first).
    """
    conditions = ["1=1"]
    params     = {"limit": limit, "offset": offset}

    if agent_name:
        conditions.append("agent_name = :agent_name")
        params["agent_name"] = agent_name

    if action_taken:
        conditions.append("action_taken = :action_taken")
        params["action_taken"] = action_taken.upper()

    if related_entity_type:
        conditions.append("related_entity_type = :related_entity_type")
        params["related_entity_type"] = related_entity_type

    if related_entity_id:
        conditions.append("related_entity_id = :related_entity_id")
        params["related_entity_id"] = related_entity_id

    if since_hours:
        cutoff = (
            datetime.utcnow() - timedelta(hours=since_hours)
        ).isoformat()
        conditions.append("timestamp >= :cutoff")
        params["cutoff"] = cutoff

    where_clause = " AND ".join(conditions)

    result = await db.execute(
        text(f"""
            SELECT
                log_id, agent_name, trigger_type,
                input_summary, output_summary, full_reasoning,
                confidence_score, action_taken, timestamp,
                related_entity_id, related_entity_type, query_text
            FROM audit_logs
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT :limit OFFSET :offset
        """),
        params
    )

    rows = [dict(r) for r in result.mappings().all()]
    logger.debug(f"[Agent4] query_logs returned {len(rows)} rows")
    return rows


async def get_log_by_id(log_id: str, db: AsyncSession) -> Optional[dict]:
    """Retrieve a single audit log entry by its ID."""
    result = await db.execute(
        text("SELECT * FROM audit_logs WHERE log_id = :log_id"),
        {"log_id": log_id}
    )
    row = result.mappings().first()
    return dict(row) if row else None


async def get_summary(db: AsyncSession) -> dict:
    """
    Dashboard summary — counts by agent, action, and recent activity.
    Returns aggregated statistics for the FINORA dashboard.
    """
    try:
        # Total logs
        total_result = await db.execute(
            text("SELECT COUNT(*) FROM audit_logs")
        )
        total = total_result.scalar() or 0

        # Counts by agent
        agent_result = await db.execute(text("""
            SELECT agent_name, COUNT(*) as count
            FROM audit_logs
            GROUP BY agent_name
            ORDER BY count DESC
        """))
        by_agent = {r["agent_name"]: r["count"]
                    for r in agent_result.mappings().all()}

        # Counts by action
        action_result = await db.execute(text("""
            SELECT action_taken, COUNT(*) as count
            FROM audit_logs
            GROUP BY action_taken
            ORDER BY count DESC
        """))
        by_action = {r["action_taken"]: r["count"]
                     for r in action_result.mappings().all()}

        # Last 24 hours activity
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        recent_result = await db.execute(
            text("SELECT COUNT(*) FROM audit_logs WHERE timestamp >= :cutoff"),
            {"cutoff": cutoff}
        )
        last_24h = recent_result.scalar() or 0

        # Most recent 5 entries
        recent_logs = await query_logs(db, limit=5)

        # High confidence decisions (confidence >= 0.8)
        high_conf_result = await db.execute(
            text("""
                SELECT COUNT(*) FROM audit_logs
                WHERE confidence_score >= 0.8
            """)
        )
        high_confidence = high_conf_result.scalar() or 0

        return {
            "total_logs":       total,
            "last_24h":         last_24h,
            "high_confidence":  high_confidence,
            "by_agent":         by_agent,
            "by_action":        by_action,
            "recent_activity":  recent_logs
        }

    except Exception as e:
        logger.error(f"[Agent4] Summary error: {e}")
        return {
            "total_logs": 0, "last_24h": 0,
            "high_confidence": 0,
            "by_agent": {}, "by_action": {},
            "recent_activity": [],
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────
# COMPONENT 3 — NL QUERY INTERFACE
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FINORA's Audit Intelligence Agent.
You analyze audit logs from a financial AI system and answer
questions about decisions made by the system.

Be specific: cite log IDs, timestamps, amounts, and reasoning
from the provided logs. If the logs don't contain enough
information to answer fully, say so clearly.

Always respond with valid JSON only. No markdown, no preamble.
"""


def extract_filters_from_question(question: str) -> dict:
    """
    Parse a natural language question into structured filters.
    Simple rule-based extraction — covers common query patterns.
    """
    q       = question.lower()
    filters = {}

    # Agent filter
    if "transaction" in q or "block" in q or "fraud" in q:
        filters["agent_name"] = "transaction_risk_agent"
    elif "compliance" in q or "regulation" in q or "policy" in q or "rbi" in q:
        filters["agent_name"] = "regulatory_watch_agent"
    elif "market" in q or "sentiment" in q or "client" in q or "news" in q:
        filters["agent_name"] = "market_sentiment_agent"

    # Action filter
    if "blocked" in q or "block" in q:
        filters["action_taken"] = "BLOCK"
    elif "hold" in q or "held" in q:
        filters["action_taken"] = "HOLD"
    elif "review" in q:
        filters["action_taken"] = "REVIEW"
    elif "cleared" in q or "clear" in q:
        filters["action_taken"] = "CLEAR"

    # Time filter
    if "today" in q or "24 hour" in q or "last 24" in q:
        filters["since_hours"] = 24
    elif "this week" in q or "last week" in q or "7 day" in q:
        filters["since_hours"] = 168
    elif "last hour" in q or "past hour" in q:
        filters["since_hours"] = 1

    # Entity type
    if "transaction" in q:
        filters["related_entity_type"] = "transaction"
    elif "regulation" in q or "circular" in q:
        filters["related_entity_type"] = "regulation"
    elif "market" in q or "scan" in q:
        filters["related_entity_type"] = "market_scan"

    return filters


async def answer_audit_query(
    question: str,
    db: AsyncSession
) -> dict[str, Any]:
    """
    Natural language audit query interface.

    Takes a plain English question, retrieves relevant logs,
    and uses Groq LLM to synthesize a human-readable answer.

    Examples:
        "What transactions did FINORA block today?"
        "Show me all compliance reports from this week"
        "Which agent made the most decisions recently?"
        "What was the reasoning behind the last BLOCK decision?"
    """
    logger.info(f"[Agent4] NL Query: {question[:80]}")

    # Step 1 — Extract filters from question
    filters = extract_filters_from_question(question)
    logger.debug(f"[Agent4] Extracted filters: {filters}")

    # Step 2 — Retrieve relevant logs
    logs = await query_logs(
        db,
        agent_name=filters.get("agent_name"),
        action_taken=filters.get("action_taken"),
        related_entity_type=filters.get("related_entity_type"),
        since_hours=filters.get("since_hours"),
        limit=20
    )

    if not logs:
        return {
            "question":  question,
            "answer":    "No audit logs found matching your query. The system may not have processed any relevant decisions yet.",
            "log_count": 0,
            "logs":      [],
            "filters_applied": filters
        }

    # Step 3 — Format logs for LLM
    log_summary = ""
    for i, log in enumerate(logs[:10], 1):
        log_summary += (
            f"\n[{i}] ID: {log['log_id'][:12]}... | "
            f"Agent: {log['agent_name']} | "
            f"Action: {log['action_taken']} | "
            f"Time: {log['timestamp'][:19]}\n"
            f"     Input:  {log['input_summary'][:100]}\n"
            f"     Output: {log['output_summary'][:100]}\n"
            f"     Confidence: {log['confidence_score']}\n"
        )

    # Step 4 — LLM synthesis
    prompt = f"""QUESTION: {question}

RETRIEVED AUDIT LOGS ({len(logs)} total, showing top 10):
{log_summary}

Respond ONLY with this JSON:
{{
  "answer": "Direct 2-4 sentence answer to the question using specific details from the logs",
  "key_findings": [
    "specific finding 1 with details",
    "specific finding 2",
    "specific finding 3"
  ],
  "recommendation": "actionable recommendation based on the audit data",
  "confidence": 0.0
}}

Set confidence 0.0-1.0 based on how well the logs answer the question."""

    try:
        from langchain.schema import HumanMessage, SystemMessage
        llm      = get_llm()
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        raw = re.sub(r"^```json\s*", "", response.content.strip())
        raw = re.sub(r"\s*```$",     "", raw).strip()
        llm_result = json.loads(raw)

    except Exception as e:
        logger.error(f"[Agent4] NL query LLM error: {e}")
        llm_result = {
            "answer":          f"Found {len(logs)} matching log entries.",
            "key_findings":    [log["output_summary"] for log in logs[:3]],
            "recommendation":  "Review the retrieved logs for details.",
            "confidence":      0.5
        }

    return {
        "question":        question,
        "answer":          llm_result.get("answer", ""),
        "key_findings":    llm_result.get("key_findings", []),
        "recommendation":  llm_result.get("recommendation", ""),
        "confidence":      llm_result.get("confidence", 0.5),
        "log_count":       len(logs),
        "logs":            logs[:10],
        "filters_applied": filters
    }