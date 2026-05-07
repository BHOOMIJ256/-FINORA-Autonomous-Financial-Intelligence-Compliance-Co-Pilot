from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger
from agents.orchestrator import run_orchestrator

router = APIRouter()

class OrchestratorRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        example="Analyze transaction TXN000050"
    )

@router.post("/ask")
async def orchestrate(request: OrchestratorRequest):
    """
    FINORA's unified intelligence endpoint.
    Routes your query to the right agents automatically.
    
    Examples:
      - "What does RBI say about KYC?"
      - "Analyze transaction TXN000050"  
      - "Give me a full intelligence briefing"
      - "What did FINORA decide today?"
    """
    logger.info(f"[API] /orchestrate/ask → {request.query[:60]}")
    try:
        result = run_orchestrator(request.query)
        return result
    except Exception as e:
        logger.error(f"[API] Orchestrator error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def orchestrator_health():
    return {"agent": "orchestrator", "status": "ok"}