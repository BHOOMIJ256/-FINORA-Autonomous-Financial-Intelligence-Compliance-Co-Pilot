from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from core.config import settings
from core.database import init_db
from core.vector_store import init_collections
from api.routes import transactions, compliance, query, audit, orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    init_collections()
    logger.info("Database & Vector Store initialized")
    yield
    # ── Shutdown ─────────────────────────────────────────
    logger.info("Shutting down FINORA")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Autonomous Financial Intelligence & Compliance Co-Pilot",
    lifespan=lifespan,
)

# CORS — allows React frontend to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────
app.include_router(transactions.router, prefix="/api/transactions", tags=["Transactions"])
app.include_router(compliance.router, tags=["Compliance"])
app.include_router(query.router,        prefix="/api/query",         tags=["Query"])
app.include_router(audit.router,        prefix="/api/audit",         tags=["Audit"])
app.include_router(orchestrator.router,prefix="/api/orchestrate",tags=["Orchestrator"])



@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}
