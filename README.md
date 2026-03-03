# FINORA 🏦
### Autonomous Financial Intelligence & Compliance Co-Pilot

> A multi-agent AI system that autonomously monitors regulatory compliance, detects transaction risk, analyzes market sentiment, and maintains a full audit trail — purpose-built for fintech.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   React Dashboard                   │
│         (Risk Feed | Compliance | Audit Log)        │
└───────────────────────┬─────────────────────────────┘
                        │ FastAPI
┌───────────────────────▼─────────────────────────────┐
│              LangGraph Orchestrator                 │
│  ┌─────────────┐  ┌──────────────┐                 │
│  │ Regulatory  │  │ Transaction  │                 │
│  │   Agent     │  │ Risk Agent   │                 │
│  └─────────────┘  └──────────────┘                 │
│  ┌─────────────┐  ┌──────────────┐                 │
│  │  Sentiment  │  │ Audit Trail  │                 │
│  │   Agent     │  │    Agent     │                 │
│  └─────────────┘  └──────────────┘                 │
└──────────┬───────────────────┬─────────────────────┘
           │                   │
    ┌──────▼──────┐    ┌───────▼──────┐
    │ PostgreSQL  │    │    Qdrant    │
    │ (Audit/Txn) │    │ (Reg. Docs)  │
    └─────────────┘    └──────────────┘
           │
    ┌──────▼──────┐
    │  LangSmith  │
    │(Observability)│
    └─────────────┘
```

## Tech Stack
| Layer | Tool | Why |
|---|---|---|
| Agent Orchestration | LangGraph | State machine, production-grade |
| Observability | LangSmith | Full trace logging |
| Backend | FastAPI | Async, fast, production-ready |
| Structured DB | PostgreSQL | Transactions, audit logs |
| Vector DB | Qdrant | Regulatory doc retrieval |
| RAG | LlamaIndex | Document ingestion & chunking |
| Sentiment | FinBERT | Finance-specific NLP model |
| Anomaly Detection | Isolation Forest + XGBoost | Classical ML + ensemble |
| Frontend | React | Clean component-based UI |
| Containerization | Docker | One-command deployment |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/FINORA.git
cd FINORA

# 2. Set up environment
cp .env.example .env
# Fill in your API keys in .env

# 3. Spin up all services
docker-compose up --build

# 4. Visit
# Backend API docs: http://localhost:8000/docs
# Frontend:         http://localhost:3000
# Qdrant UI:        http://localhost:6333/dashboard
```

## Project Status
- [x] Project scaffolding & Docker setup
- [ ] Synthetic data generation
- [ ] ML anomaly detection model
- [ ] Agent 1: Regulatory Watch
- [ ] Agent 2: Transaction Risk
- [ ] Agent 3: Market Sentiment
- [ ] Agent 4: Audit Trail
- [ ] LangGraph Orchestrator
- [ ] React Dashboard
- [ ] Production deployment
