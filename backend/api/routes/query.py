from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger
from agents.sentiment_agent import run_sentiment_scan, build_audit_entry

router = APIRouter()

@router.get("/sentiment/scan")
async def trigger_sentiment_scan(background_tasks: BackgroundTasks):
    """
    Trigger a full market sentiment scan.
    Fetches news, runs FinBERT, maps to clients, generates reports.
    """
    logger.info("[API] /query/sentiment/scan triggered")
    try:
        result = run_sentiment_scan()
        audit  = build_audit_entry(result)
        background_tasks.add_task(_log_audit, audit)
        return result
    except Exception as e:
        logger.error(f"[API] Sentiment scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/clients/{client_id}")
async def get_client_sentiment(client_id: str):
    """Get the latest sentiment context for a specific client."""
    try:
        result  = run_sentiment_scan()
        all_clients = (
            result.get("high_impact_clients", []) +
            result.get("medium_impact_clients", [])
        )
        match = next(
            (c for c in all_clients if c["client_id"] == client_id),
            None
        )
        if not match:
            return {"client_id": client_id, "impact_level": "None",
                    "message": "No significant market impact detected"}
        return match
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def query_health():
    return {"agent": "market_sentiment_agent", "status": "ok"}


async def _log_audit(entry: dict):
    logger.debug(f"[AuditLog] {entry['agent_name']} | {entry['action_taken']}")