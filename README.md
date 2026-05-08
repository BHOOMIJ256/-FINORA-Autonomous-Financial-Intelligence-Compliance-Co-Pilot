<div align="center">

# 🏦 FINORA
### Autonomous Financial Intelligence & Compliance Co-Pilot

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-FF6B35?style=flat)](https://langchain-ai.github.io/langgraph)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A production-grade multi-agent AI system for Indian fintech companies.**
Autonomously monitors regulatory compliance, detects transaction fraud,
analyzes market sentiment, and maintains a complete audit trail — all in real time.

[Features](#features) · [Architecture](#architecture) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [ML Model](#ml-model) · [Contributing](#contributing)

</div>

---

## The Problem

Financial companies in India — payment aggregators, NBFCs, fintechs — face three simultaneous operational burdens every day:

- **Compliance teams** manually read RBI and SEBI circulars, checking if internal policies are up to date. One missed circular can mean regulatory penalties.
- **Risk teams** manually review suspicious transactions. At scale, this is impossible.
- **Analysts** try to correlate market news with client exposure — with no systematic way to do it.

FINORA automates all three with a coordinated multi-agent AI system.

---

## Features

### 🤖 Four Specialized AI Agents

| Agent | Responsibility | Key Technology |
|-------|---------------|----------------|
| **Agent 1 — Regulatory Watch** | Monitors RBI/SEBI circulars, identifies compliance gaps against internal policies | LangChain RAG + Qdrant + Groq LLM |
| **Agent 2 — Transaction Risk** | Scores transactions for fraud risk, explains decisions in plain English | Isolation Forest + XGBoost + Groq LLM |
| **Agent 3 — Market Sentiment** | Analyzes financial news, maps impact to client exposure | FinBERT + NewsAPI + Groq LLM |
| **Agent 4 — Audit Trail** | Persists every agent decision with full reasoning, queryable in natural language | PostgreSQL + Groq LLM |

### 🧠 LangGraph Orchestrator
- Routes natural language queries to the right agent(s) automatically
- **Cross-agent triggering**: when a transaction is flagged BLOCK/HOLD, the orchestrator automatically invokes the Regulatory Agent to surface the relevant compliance requirement
- Synthesizes multi-agent outputs into a single cohesive intelligence report

### 📊 React Dashboard
- **Command bar**: type any question in natural language, get an instant intelligence report
- **Live risk feed**: real-time transaction decisions with full reasoning
- **Compliance center**: regulatory gap analysis with internal policy cross-referencing
- **Market intelligence**: sector sentiment visualization + client exposure mapping
- **Audit drawer**: full queryable decision history with NL search

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     React Dashboard (Next.js)                   │
│        Command Bar │ Risk Feed │ Compliance │ Market Intel       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ FastAPI
┌───────────────────────────────▼─────────────────────────────────┐
│                    LangGraph Orchestrator                        │
│                                                                  │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│   │  Agent 1    │  │   Agent 2    │  │      Agent 3        │   │
│   │ Regulatory  │  │ Transaction  │  │  Market Sentiment   │   │
│   │   Watch     │  │    Risk      │  │  FinBERT + NewsAPI  │   │
│   │  RAG+LLM    │  │  ML + LLM    │  │      + LLM          │   │
│   └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘   │
│          │                │                      │              │
│          └────────────────┴──────────────────────┘              │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │   Agent 4   │                              │
│                    │ Audit Trail │                              │
│                    └─────────────┘                              │
└──────────────────────────────────────────────────────────────────┘
          │                    │                    │
   ┌──────▼──────┐    ┌────────▼────────┐   ┌──────▼──────┐
   │ PostgreSQL  │    │     Qdrant      │   │  LangSmith  │
   │ Audit Logs  │    │ Regulatory Docs │   │Observability│
   │ Transactions│    │ (1,395 chunks)  │   └─────────────┘
   │   Clients   │    └─────────────────┘
   └─────────────┘
```

### Data Flow

```
RBI/SEBI RSS Feed ──→ regulatory_fetcher.py ──→ Qdrant (embeddings)
                                                       │
User Query ──→ Orchestrator ──→ Agent 1 ──→ RAG retrieval + LLM
                           ──→ Agent 2 ──→ ML inference + LLM
                           ──→ Agent 3 ──→ FinBERT + LLM
                           ──→ Agent 4 ──→ PostgreSQL read/write
                                ↓
                          Synthesis (Groq LLM)
                                ↓
                         Dashboard / API Response
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- Docker Desktop (optional, for containerized deployment)

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/FINORA.git
cd FINORA
```

### 2. Configure Environment

```bash
cp .env.example backend/.env
```

Edit `backend/.env`:

```env
# Database
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/finora_db

# LLM — Groq (free at console.groq.com)
GROQ_API_KEY=gsk_...

# LangSmith observability (free at smith.langchain.com)
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=FINORA

# News API (free at newsapi.org)
NEWS_API_KEY=...

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Install Backend Dependencies

```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 4. Setup PostgreSQL

Create the database in pgAdmin or via psql:

```sql
CREATE DATABASE finora_db;
```

### 5. Start Qdrant

```bash
# Download from github.com/qdrant/qdrant/releases
cd path/to/qdrant
./qdrant.exe    # Windows
# ./qdrant      # Mac/Linux
```

### 6. Initialize Data

```bash
cd backend

# Generate synthetic data
python scripts/generate_finora_data.py

# Seed PostgreSQL
python scripts/seed_db.py

# Install FinBERT (one-time, ~440MB)
python scripts/install_finbert.py

# Bootstrap regulatory documents into Qdrant
python scripts/regulatory_fetcher.py --mode both

# Train the ML risk model
python scripts/train_risk_model.py
```

### 7. Start Backend

```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### 8. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard runs at `http://localhost:3000`

---

## ML Model

### Architecture

FINORA uses an **Isolation Forest + XGBoost ensemble** for transaction anomaly detection.

| Component | Purpose | Why |
|-----------|---------|-----|
| **Isolation Forest** | Unsupervised anomaly detection | Learns normal behaviour without labels; works on production data with unknown fraud patterns |
| **XGBoost** | Supervised classification | Uses labeled training data for high-precision fraud detection |
| **Ensemble (35/65 weight)** | Combined score | Complementary strengths — IF catches statistical outliers, XGBoost catches known patterns |

### Features Engineered

| Feature | Description | Anomaly Caught |
|---------|-------------|----------------|
| `amount_zscore` | Standard deviations from user's 90-day average | Amount outlier |
| `amount_ratio` | Transaction amount / user average | Amount outlier |
| `is_known_mcc` | Is merchant category in user's history | MCC mismatch |
| `is_home_city` | Transaction city == user's home city | Geo mismatch |
| `is_foreign` | Merchant country outside India | Foreign country |
| `ip_mismatch` | IP country outside India | IP anomaly |
| `high_risk_mcc` | Wire Transfer / Gambling / FX (MCC 4829/7995/6051) | High-risk merchant |
| `risk_signal_count` | Count of simultaneous risk signals | Compound fraud |

### Performance

```
Dataset:     10,000 synthetic transactions (5% anomaly rate)
Train/Val:   80/20 split, stratified

ROC-AUC:     0.87
Precision:   1.00  ← zero false positives at BLOCK threshold
Recall:      0.45
F1 Score:    0.62

Action Distribution (validation set):
  BLOCK    2.2%  — highest certainty fraud
  HOLD     1.5%  — manual review required
  REVIEW  15.6%  — monitoring flag
  CLEAR   80.8%  — normal transactions
```

**Why Precision 1.0 matters:** In fintech, false positives (blocking legitimate transactions) destroy customer trust and drive churn. FINORA achieves zero false positives at the BLOCK level — every hard block is a genuine threat. Missed anomalies at BLOCK fall into HOLD/REVIEW where human analysts make the final call.

### Thresholds

Calibrated to actual score distribution (range: 0.25–0.77 on synthetic data):

```python
THRESHOLD_BLOCK  = 0.65   # Top 2.2% — block immediately
THRESHOLD_HOLD   = 0.50   # Top 3.7% — manual review
THRESHOLD_REVIEW = 0.38   # Top 19.3% — flag for monitoring
```

---

## Regulatory Intelligence

### Data Sources

| Source | Type | Update Frequency |
|--------|------|-----------------|
| RBI Notifications RSS | Live circulars | Daily poll |
| SEBI RSS Feed | Live circulars + press releases | Daily poll |
| RBI Master Directions | Foundational documents | Bootstrap once |

### Ingestion Pipeline

```
RSS Feed / Master Direction URL
        ↓
HTTP fetch → extract text → clean HTML
        ↓
SHA256 hash → deduplication check
        ↓
Chunk text (500 chars, 100 overlap)
        ↓
Embed with sentence-transformers/all-MiniLM-L6-v2 (384 dims)
        ↓
Upsert into Qdrant collection: regulatory_docs
        ↓
Available for Agent 1 RAG retrieval
```

**Current index:** 1,395 chunks from RBI Master Directions + recent circulars

### Internal Policies Covered

| Policy ID | Name | Category |
|-----------|------|----------|
| KYC-001 | Customer Onboarding & KYC Policy | KYC |
| KYC-002 | Business KYC & Merchant Verification | KYC |
| AML-001 | Anti-Money Laundering & CFT Policy | AML |
| AML-002 | Suspicious Transaction Monitoring | AML |
| LEND-001 | Digital Lending Policy | Lending |
| LEND-002 | Credit Risk Assessment Policy | Lending |
| LEND-003 | FLDG & Co-lending Arrangement | Lending |
| PAY-001 | Payment Aggregator Merchant Onboarding | Payments |
| PAY-002 | UPI Transaction Limits & Controls | Payments |
| PAY-003 | Prepaid Payment Instrument (PPI) Policy | Payments |
| GRIEV-001 | Customer Grievance Redressal Policy | Grievance |
| GRIEV-002 | RBI Integrated Ombudsman Compliance | Grievance |
| DATA-001 | Data Privacy & Localisation Policy | Data Governance |
| FRAUD-001 | Fraud Prevention & Response Policy | AML |
| RISK-001 | Liquidity Risk Management Policy | Risk Management |

---

## API Reference

All endpoints available at `http://localhost:8000/docs` (Swagger UI).

### Orchestrator

```
POST /api/orchestrate/ask
Body: { "query": "Analyze transaction TXN000050" }

Routes to the right agent(s) automatically. Returns synthesis + detailed results.
```

### Compliance (Agent 1)

```
POST /api/compliance/analyze
Body: { "query": "RBI KYC requirements for payment aggregators" }

GET  /api/compliance/policies
GET  /api/compliance/health
```

### Transactions (Agent 2)

```
POST /api/transactions/analyze
Body: { "transaction_id": "TXN000042" }

POST /api/transactions/batch
Body: { "transaction_ids": ["TXN000042", "TXN000043"] }

GET  /api/transactions/flagged
GET  /api/transactions/sample-anomalies
GET  /api/transactions/health
```

### Market Sentiment (Agent 3)

```
GET  /api/query/sentiment/scan
GET  /api/query/sentiment/clients/{client_id}
GET  /api/query/health
```

### Audit Trail (Agent 4)

```
GET  /api/audit/logs
GET  /api/audit/logs/{log_id}
GET  /api/audit/summary
POST /api/audit/query
Body: { "question": "What did FINORA block today?" }
POST /api/audit/log
GET  /api/audit/health
```

---

## Tech Stack

### Backend

| Layer | Technology | Version |
|-------|-----------|---------|
| Web Framework | FastAPI | 0.115 |
| Agent Orchestration | LangGraph | 0.2.28 |
| LLM Observability | LangSmith | 0.1.129 |
| LLM Provider | Groq (Llama-3.3-70b) | — |
| Vector Database | Qdrant | 1.11.1 |
| RAG Framework | LlamaIndex | 0.11.14 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | — |
| Sentiment Model | ProsusAI/FinBERT | — |
| Anomaly Detection | Isolation Forest (scikit-learn) | 1.5.2 |
| Classification | XGBoost | 2.1.1 |
| Structured DB | PostgreSQL + SQLAlchemy | 15 / 2.0.35 |
| Async Driver | asyncpg | 0.29.0 |
| Containerization | Docker | — |

### Frontend

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 14 |
| Styling | Tailwind CSS |
| Components | shadcn/ui |
| Charts | Recharts |
| Data Fetching | TanStack Query |
| Icons | Lucide React |

---

## Project Structure

```
FINORA/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py         ← LangGraph multi-agent orchestrator
│   │   ├── regulatory_agent.py     ← Agent 1: RAG compliance analysis
│   │   ├── risk_agent.py           ← Agent 2: ML + LLM fraud detection
│   │   ├── sentiment_agent.py      ← Agent 3: FinBERT market intelligence
│   │   └── audit_agent.py          ← Agent 4: audit persistence + NL query
│   ├── api/routes/
│   │   ├── compliance.py           ← Agent 1 endpoints
│   │   ├── transactions.py         ← Agent 2 endpoints
│   │   ├── query.py                ← Agent 3 endpoints
│   │   ├── audit.py                ← Agent 4 endpoints
│   │   └── orchestrator_routes.py  ← Unified orchestrator endpoint
│   ├── core/
│   │   ├── config.py               ← pydantic-settings configuration
│   │   ├── database.py             ← async SQLAlchemy setup
│   │   └── vector_store.py         ← Qdrant client + collection init
│   ├── models/
│   │   ├── transaction.py          ← SQLAlchemy ORM models
│   │   ├── user_profile.py
│   │   ├── internal_policy.py
│   │   ├── client.py
│   │   └── audit_log.py
│   ├── scripts/
│   │   ├── generate_finora_data.py ← synthetic data generator
│   │   ├── seed_db.py              ← PostgreSQL seeder
│   │   ├── install_finbert.py      ← FinBERT download + verification
│   │   ├── regulatory_fetcher.py   ← RBI/SEBI RSS ingestion pipeline
│   │   └── train_risk_model.py     ← ML model trainer
│   ├── data/raw/                   ← generated CSV/JSON data
│   ├── models/saved/               ← trained ML model artifacts
│   ├── main.py                     ← FastAPI application
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx                ← main dashboard
│   ├── components/
│   │   ├── CommandBar.tsx
│   │   ├── StatCards.tsx
│   │   ├── RiskFeed.tsx
│   │   ├── ComplianceCenter.tsx
│   │   ├── MarketIntelligence.tsx
│   │   └── AuditDrawer.tsx
│   └── lib/
│       └── api.ts                  ← typed API client
├── docker-compose.yml
└── README.md
```

---

## Design Decisions

### Why Groq over OpenAI?
Groq runs Llama-3.3-70b at 5-10x faster inference than OpenAI's API. In a multi-agent system where 3-4 agents chain LLM calls, speed directly affects demo quality and user experience. Groq is also free at development scale.

### Why Isolation Forest + XGBoost ensemble?
They have complementary strengths. Isolation Forest is unsupervised — it detects statistical outliers without needing labeled data, which matters in production where fraud patterns evolve constantly. XGBoost uses supervised labels for high-precision classification on known patterns. Combined, they're more robust than either alone.

### Why FinBERT over general sentiment models?
General sentiment models (VADER, generic BERT) are trained on product reviews and social media. They don't understand financial language. "RBI tightens liquidity norms" gets neutral or mildly negative from a general model. FinBERT correctly classifies it as strongly negative (0.885 confidence) because it was trained on financial news and SEC filings.

### Why RSS feeds over PDF downloads?
RBI and SEBI publish official RSS feeds specifically for programmatic consumption. RSS gives us real-time updates without scraping, with no maintenance overhead. Master Directions are bootstrapped once; amendments arrive automatically via RSS daily.

### Why ML model for decisions, LLM for explanations?
LLMs are expensive and slow — you don't run one on every transaction. The ML model makes the binary decision (block/hold/review/clear) in milliseconds at zero marginal cost. The LLM only runs when the ML model flags something (rare), producing the human-readable explanation for the compliance team. This is how production AI systems at serious fintech companies work.

---

## Deployment

### Local Development (Recommended)

```bash
# Terminal 1 — Qdrant
./qdrant.exe

# Terminal 2 — Backend
cd backend && uvicorn main:app --reload

# Terminal 3 — Frontend
cd frontend && npm run dev
```

### Docker (Production)

```bash
docker-compose up --build
```

Services:
- `finora_backend` → `http://localhost:8000`
- `finora_frontend` → `http://localhost:3000`
- `finora_postgres` → port 5432
- `finora_qdrant` → `http://localhost:6333`

### Cloud Deployment

| Service | Platform |
|---------|---------|
| Backend | Railway or Render |
| Frontend | Vercel |
| PostgreSQL | Railway managed Postgres or Supabase |
| Qdrant | Qdrant Cloud (free tier) |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ for Indian Fintech

**Bhoomi Jain** · [LinkedIn](https://linkedin.com/in/bhoomij256) · [GitHub](https://github.com/bhoomij256)

</div>
