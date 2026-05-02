"""
FINORA — Agent 3: Market Sentiment & Client Impact Agent
=========================================================
Fetches financial news via NewsAPI, runs FinBERT sentiment
analysis, maps results to client exposure data, and generates
a structured market intelligence report via Groq LLM.

Five core functions:
  1. fetch_financial_news()     — NewsAPI → filtered articles
  2. analyze_sentiment()        — FinBERT → sentiment scores
  3. map_to_clients()           — news → affected clients
  4. generate_impact_report()   — Groq LLM → per-client explanation
  5. run_sentiment_scan()       — orchestrates all four

Output:
{
  "scan_timestamp":      str,
  "total_scanned":       int,
  "relevant_articles":   int,
  "high_impact_clients": [...],
  "sector_summary":      {...},
  "market_brief":        str
}
"""

import os
import re
import json
import uuid
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Any

import httpx
import torch
import numpy as np
from loguru import logger
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

# ── Config ────────────────────────────────────────────────
NEWS_API_KEY  = os.getenv("NEWS_API_KEY", "")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL     = "llama-3.3-70b-versatile"
FINBERT_MODEL = "ProsusAI/finbert"

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# NewsAPI endpoint
NEWS_API_URL = "https://newsapi.org/v2/everything"

# FinBERT label mapping
LABEL_MAP = {
    "LABEL_0": "positive",
    "LABEL_1": "negative",
    "LABEL_2": "neutral",
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral"
}

# Sentiment score weights for impact calculation
SENTIMENT_WEIGHTS = {"negative": -1.0, "positive": 1.0, "neutral": 0.0}

# Impact thresholds
HIGH_IMPACT_THRESHOLD   = -0.45
MEDIUM_IMPACT_THRESHOLD = -0.20

# ── Singletons ────────────────────────────────────────────
_finbert_tokenizer = None
_finbert_model     = None
_llm               = None


def get_finbert():
    global _finbert_tokenizer, _finbert_model
    if _finbert_tokenizer is None:
        logger.info("[Agent3] Loading FinBERT...")
        _finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        _finbert_model     = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_MODEL
        )
        _finbert_model.eval()
        logger.success("[Agent3] FinBERT ready")
    return _finbert_tokenizer, _finbert_model


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
# COMPONENT 1 — LOAD STATIC DATA
# ─────────────────────────────────────────────────────────

def load_clients() -> list[dict]:
    """Load clients.csv into memory."""
    path = DATA_DIR / "clients.csv"
    if not path.exists():
        logger.warning(f"[Agent3] clients.csv not found at {path}")
        return []
    clients = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clients.append({
                "client_id":                row["client_id"].strip(),
                "client_name":              row["client_name"].strip(),
                "sector":                   row["sector"].strip(),
                "product_type":             row["product_type"].strip(),
                "exposure_short_term":      float(row["exposure_short_term"] or 0),
                "exposure_long_term":       float(row["exposure_long_term"] or 0),
                "credit_line_total":        float(row["credit_line_total"] or 0),
                "credit_line_utilized_pct": float(row["credit_line_utilized_pct"] or 0),
                "collateral_type":          row["collateral_type"].strip(),
                "risk_rating":              row["risk_rating"].strip(),
                "last_reviewed_date":       row["last_reviewed_date"].strip(),
                "payment_history":          row["payment_history"].strip(),
                "regulatory_sensitivity":   row["regulatory_sensitivity"].strip(),
            })
    logger.info(f"[Agent3] Loaded {len(clients)} clients")
    return clients


def load_sector_mapping() -> list[dict]:
    """Load sector_news_mapping.json."""
    path = DATA_DIR / "sector_news_mapping.json"
    if not path.exists():
        logger.warning(f"[Agent3] sector_news_mapping.json not found")
        return []
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    logger.info(f"[Agent3] Loaded {len(mapping)} sector mappings")
    return mapping


# ─────────────────────────────────────────────────────────
# COMPONENT 2 — NEWS FETCHING
# ─────────────────────────────────────────────────────────

def build_news_query(sector_mapping: list[dict]) -> str:
    """
    Build a combined NewsAPI query from all sector keywords.
    NewsAPI supports OR queries — we pick top keywords per sector.
    """
    priority_keywords = [
    "RBI", "SEBI", "fintech India", "payment gateway India",
    "NBFC regulation", "digital payment India", "UPI",
    "financial regulation India", "banking India", "lending India"
]
    return " OR ".join(f'"{kw}"' for kw in priority_keywords[:8])


def fetch_financial_news(
    sector_mapping: list[dict],
    max_articles: int = 30
) -> list[dict]:
    """
    Fetch relevant financial news from NewsAPI.
    Returns list of article dicts with title, description,
    source, url, publishedAt.
    """
    if not NEWS_API_KEY:
        logger.warning("[Agent3] NEWS_API_KEY not set — using mock articles")
        return _get_mock_articles()

    query = build_news_query(sector_mapping)

    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": max_articles,
        "apiKey":   NEWS_API_KEY
    }

    try:
        logger.info(f"[Agent3] Fetching news from NewsAPI...")
        response = httpx.get(NEWS_API_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("articles", []):
            title = item.get("title", "") or ""
            desc  = item.get("description", "") or ""

            # Skip removed articles
            if "[Removed]" in title:
                continue

            articles.append({
                "title":        title.strip(),
                "description":  desc.strip(),
                "source":       item.get("source", {}).get("name", "Unknown"),
                "url":          item.get("url", ""),
                "published_at": item.get("publishedAt", ""),
                "full_text":    f"{title}. {desc}".strip()
            })

        logger.info(f"[Agent3] Fetched {len(articles)} articles")
        return articles

    except httpx.HTTPError as e:
        logger.error(f"[Agent3] NewsAPI error: {e}")
        return _get_mock_articles()
    except Exception as e:
        logger.error(f"[Agent3] Unexpected fetch error: {e}")
        return _get_mock_articles()


def _get_mock_articles() -> list[dict]:
    """
    Fallback mock articles when NewsAPI is unavailable.
    Covers all major sectors in sector_news_mapping.json.
    """
    mock = [
        {
            "title": "RBI tightens liquidity norms for NBFCs amid rising credit risk",
            "description": "The Reserve Bank of India has issued new directions requiring NBFCs to maintain higher liquidity coverage ratios, effective immediately.",
            "source": "Economic Times",
            "url": "https://economictimes.indiatimes.com/mock",
            "published_at": datetime.utcnow().isoformat(),
            "full_text": "RBI tightens liquidity norms for NBFCs amid rising credit risk. The Reserve Bank of India has issued new directions requiring NBFCs to maintain higher liquidity coverage ratios, effective immediately."
        },
        {
            "title": "UPI transactions hit record 18 billion in March 2026",
            "description": "India's Unified Payments Interface processed a record 18 billion transactions in March 2026, signaling strong digital payment adoption.",
            "source": "Mint",
            "url": "https://livemint.com/mock",
            "published_at": datetime.utcnow().isoformat(),
            "full_text": "UPI transactions hit record 18 billion in March 2026. India's Unified Payments Interface processed a record 18 billion transactions in March 2026."
        },
        {
            "title": "SEBI tightens rules for investment advisers and portfolio managers",
            "description": "SEBI has issued a master circular updating compliance requirements for registered investment advisers and portfolio management services.",
            "source": "Business Standard",
            "url": "https://business-standard.com/mock",
            "published_at": datetime.utcnow().isoformat(),
            "full_text": "SEBI tightens rules for investment advisers and portfolio managers. SEBI has issued a master circular updating compliance requirements."
        },
        {
            "title": "Digital lending platforms face higher scrutiny under new RBI framework",
            "description": "RBI's updated digital lending guidelines increase compliance burden for BNPL and digital credit platforms operating in India.",
            "source": "Financial Express",
            "url": "https://financialexpress.com/mock",
            "published_at": datetime.utcnow().isoformat(),
            "full_text": "Digital lending platforms face higher scrutiny under new RBI framework. RBI's updated digital lending guidelines increase compliance burden."
        },
        {
            "title": "Housing finance companies report stress in affordable segment",
            "description": "Several housing finance companies flagged rising NPAs in the affordable housing segment amid slower recoveries and delayed project completions.",
            "source": "Mint",
            "url": "https://livemint.com/mock2",
            "published_at": datetime.utcnow().isoformat(),
            "full_text": "Housing finance companies report stress in affordable segment. Several housing finance companies flagged rising NPAs."
        },
        {
            "title": "Microfinance sector recovers with strong Q4 collections",
            "description": "The microfinance industry reported improved collection efficiency and disbursement growth in Q4 FY26, reversing a two-quarter slowdown.",
            "source": "Economic Times",
            "url": "https://economictimes.indiatimes.com/mock2",
            "published_at": datetime.utcnow().isoformat(),
            "full_text": "Microfinance sector recovers with strong Q4 collections. The microfinance industry reported improved collection efficiency."
        },
        {
            "title": "Payment aggregators face new escrow and settlement requirements",
            "description": "RBI's revised guidelines on payment aggregators mandate stricter escrow account maintenance and T+1 settlement for all verified merchants.",
            "source": "Business Standard",
            "url": "https://business-standard.com/mock2",
            "published_at": datetime.utcnow().isoformat(),
            "full_text": "Payment aggregators face new escrow and settlement requirements. RBI's revised guidelines mandate stricter escrow account maintenance."
        },
        {
            "title": "Mutual fund industry AUM crosses ₹60 lakh crore milestone",
            "description": "India's mutual fund industry crossed ₹60 lakh crore in assets under management, driven by strong SIP inflows and equity market performance.",
            "source": "Mint",
            "url": "https://livemint.com/mock3",
            "published_at": datetime.utcnow().isoformat(),
            "full_text": "Mutual fund industry AUM crosses ₹60 lakh crore milestone. India's mutual fund industry crossed ₹60 lakh crore in assets under management."
        }
    ]
    logger.info(f"[Agent3] Using {len(mock)} mock articles")
    return mock


# ─────────────────────────────────────────────────────────
# COMPONENT 3 — SENTIMENT ANALYSIS (FinBERT)
# ─────────────────────────────────────────────────────────

def analyze_sentiment(articles: list[dict]) -> list[dict]:
    """
    Run FinBERT on each article's full_text.
    Returns articles enriched with sentiment data.
    """
    if not articles:
        return []

    tokenizer, model = get_finbert()
    texts            = [a["full_text"][:512] for a in articles]

    logger.info(f"[Agent3] Running FinBERT on {len(texts)} articles...")

    # Batch inference — process in chunks of 8
    BATCH_SIZE = 8
    all_results = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for prob in probs:
            scores    = prob.tolist()
            label_idx = prob.argmax().item()
            raw_label = model.config.id2label[label_idx]
            label     = LABEL_MAP.get(raw_label, raw_label).lower()

            # Weighted sentiment score: +1 positive, -1 negative, 0 neutral
            weighted = (scores[0] * 1.0) + (scores[1] * -1.0) + (scores[2] * 0.0)

            all_results.append({
                "sentiment":        label,
                "confidence":       round(max(scores), 4),
                "sentiment_score":  round(weighted, 4),
                "scores": {
                    "positive": round(scores[0], 4),
                    "negative": round(scores[1], 4),
                    "neutral":  round(scores[2], 4)
                }
            })

    # Merge sentiment back into articles
    enriched = []
    for article, sentiment_data in zip(articles, all_results):
        enriched.append({**article, **sentiment_data})

    neg_count = sum(1 for a in enriched if a["sentiment"] == "negative")
    pos_count = sum(1 for a in enriched if a["sentiment"] == "positive")
    neu_count = sum(1 for a in enriched if a["sentiment"] == "neutral")

    logger.info(
        f"[Agent3] Sentiment breakdown — "
        f"neg: {neg_count} | pos: {pos_count} | neu: {neu_count}"
    )

    return enriched


# ─────────────────────────────────────────────────────────
# COMPONENT 4 — CLIENT IMPACT MAPPING
# ─────────────────────────────────────────────────────────

def map_to_clients(
    articles_with_sentiment: list[dict],
    clients: list[dict],
    sector_mapping: list[dict]
) -> dict:
    """
    Cross-reference sentiment-enriched articles against
    client sectors to identify which clients are affected.

    Returns dict keyed by client_id with:
      - relevant_articles: list of matching articles
      - weighted_sentiment: exposure-weighted sentiment score
      - impact_level: High / Medium / Low / None
    """
    # Build sector → keywords lookup
    sector_keywords = {}
    for sm in sector_mapping:
        sector_keywords[sm["sector"]] = [
            kw.lower() for kw in sm["keywords"]
        ]

    # Build sector → clients lookup
    sector_clients = {}
    for client in clients:
        sec = client["sector"]
        if sec not in sector_clients:
            sector_clients[sec] = []
        sector_clients[sec].append(client)

    # Match articles to sectors
    sector_articles = {sm["sector"]: [] for sm in sector_mapping}
    for article in articles_with_sentiment:
        text = article["full_text"].lower()
        for sector, keywords in sector_keywords.items():
            if any(kw in text for kw in keywords):
                sector_articles[sector].append(article)

    for sector, keywords in sector_keywords.items():
        if not sector_articles[sector]:
            for article in articles_with_sentiment:
                title = article["title"].lower()
                if any(kw in title for kw in keywords[:3]):
                    sector_articles[sector].append(article)

    # Build client impact map
    client_impact = {}
    for client in clients:
        sector           = client["sector"]
        relevant_news    = sector_articles.get(sector, [])

        if not relevant_news:
            client_impact[client["client_id"]] = {
                "client":            client,
                "relevant_articles": [],
                "weighted_sentiment": 0.0,
                "avg_sentiment":     0.0,
                "impact_level":      "None",
                "negative_count":    0,
                "positive_count":    0
            }
            continue

        # Weight sentiment by client's regulatory sensitivity
        sensitivity_multiplier = {
            "High": 1.5, "Medium": 1.0, "Low": 0.7
        }.get(client["regulatory_sensitivity"], 1.0)

        # Weight by credit utilization — highly utilized clients
        # are more vulnerable to regulatory changes
        utilization_factor = 1 + (client["credit_line_utilized_pct"] / 100)

        scores      = [a["sentiment_score"] for a in relevant_news]
        avg_score   = sum(scores) / len(scores)
        weighted    = avg_score * sensitivity_multiplier * utilization_factor
        weighted    = max(-1.0, min(1.0, weighted))

        neg_count = sum(1 for a in relevant_news if a["sentiment"] == "negative")
        pos_count = sum(1 for a in relevant_news if a["sentiment"] == "positive")

        # Impact level based on weighted score
        if weighted <= HIGH_IMPACT_THRESHOLD:
            impact_level = "High"
        elif weighted <= MEDIUM_IMPACT_THRESHOLD:
            impact_level = "Medium"
        elif weighted >= 0.20:
            impact_level = "Positive"
        else:
            impact_level = "Low"

        client_impact[client["client_id"]] = {
            "client":             client,
            "relevant_articles":  relevant_news,
            "weighted_sentiment": round(weighted, 4),
            "avg_sentiment":      round(avg_score, 4),
            "impact_level":       impact_level,
            "negative_count":     neg_count,
            "positive_count":     pos_count
        }

    high_impact = sum(
        1 for v in client_impact.values()
        if v["impact_level"] == "High"
    )
    logger.info(f"[Agent3] Client mapping done | high impact: {high_impact}")
    return client_impact


# ─────────────────────────────────────────────────────────
# COMPONENT 5 — LLM IMPACT REPORT GENERATION
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FINORA's Market Intelligence Agent for an Indian fintech 
risk team. You analyze how financial news impacts specific client relationships.

Be specific about: exact exposure amounts, regulatory implications, 
what actions the risk team should take and by when.
Respond with valid JSON only. No markdown, no preamble.
"""


def generate_client_impact(client_data: dict) -> dict:
    """Generate LLM impact report for a single affected client."""
    from langchain.schema import HumanMessage, SystemMessage

    client   = client_data["client"]
    articles = client_data["relevant_articles"]
    weighted = client_data["weighted_sentiment"]
    impact   = client_data["impact_level"]

    # Format news for prompt
    news_summary = ""
    for i, a in enumerate(articles[:4], 1):
        news_summary += (
            f"\n{i}. [{a['sentiment'].upper()} | "
            f"score: {a['sentiment_score']}] "
            f"{a['title']}"
        )

    total_exposure = (
        client["exposure_short_term"] + client["exposure_long_term"]
    )

    prompt = f"""CLIENT PROFILE:
  Name:              {client['client_name']}
  Sector:            {client['sector']}
  Product:           {client['product_type']}
  Total Exposure:    ₹{total_exposure:,.0f}
  Short-term:        ₹{client['exposure_short_term']:,.0f}
  Long-term:         ₹{client['exposure_long_term']:,.0f}
  Credit Utilization:{client['credit_line_utilized_pct']}%
  Risk Rating:       {client['risk_rating']}
  Payment History:   {client['payment_history']}
  Reg Sensitivity:   {client['regulatory_sensitivity']}

RELEVANT NEWS ({len(articles)} articles):
{news_summary}

SENTIMENT ANALYSIS:
  Weighted Score:    {weighted} (-1=very negative, +1=very positive)
  Impact Level:      {impact}
  Negative Articles: {client_data['negative_count']}
  Positive Articles: {client_data['positive_count']}

Respond ONLY with this JSON:
{{
  "explanation": "2-3 sentences on how this news specifically impacts this client given their exposure and sector",
  "key_risks": ["specific risk 1", "specific risk 2"],
  "recommended_action": "specific action the risk team should take",
  "urgency": "Immediate | This Week | This Month | Monitor",
  "confidence": 0.0
}}"""

    try:
        llm      = get_llm()
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        raw = re.sub(r"^```json\s*", "", response.content.strip())
        raw = re.sub(r"\s*```$", "", raw).strip()
        return json.loads(raw)

    except Exception as e:
        logger.error(f"[Agent3] LLM error for {client['client_id']}: {e}")
        return {
            "explanation": f"News sentiment is {impact} for {client['sector']} sector.",
            "key_risks":   [f"Regulatory changes affecting {client['sector']}"],
            "recommended_action": "Schedule review call with client relationship manager.",
            "urgency":     "This Week",
            "confidence":  0.5
        }


# ─────────────────────────────────────────────────────────
# COMPONENT 6 — MARKET BRIEF GENERATOR
# ─────────────────────────────────────────────────────────

def generate_market_brief(
    articles: list[dict],
    sector_summary: dict,
    high_impact_count: int
) -> str:
    """Generate a 3-4 sentence morning market brief for the risk team."""
    from langchain.schema import HumanMessage, SystemMessage

    top_negative = [
        a for a in articles if a["sentiment"] == "negative"
    ][:3]
    top_positive = [
        a for a in articles if a["sentiment"] == "positive"
    ][:2]

    neg_headlines = "\n".join(f"- {a['title']}" for a in top_negative)
    pos_headlines = "\n".join(f"- {a['title']}" for a in top_positive)

    sector_lines = "\n".join(
        f"- {s}: {d['sentiment']} (score: {d['score']}, "
        f"affected clients: {d['affected_clients']})"
        for s, d in sector_summary.items()
        if d["affected_clients"] > 0
    )

    prompt = f"""Generate a 3-4 sentence morning market intelligence brief 
for a fintech risk team. Be direct and actionable.

TOP NEGATIVE NEWS:
{neg_headlines}

TOP POSITIVE NEWS:
{pos_headlines}

SECTOR IMPACT:
{sector_lines}

HIGH IMPACT CLIENTS: {high_impact_count}

Write the brief as plain text (no JSON, no bullet points).
Start with the most important risk, end with recommended focus areas."""

    try:
        llm      = get_llm()
        response = llm.invoke([
            SystemMessage(content="You are a financial risk analyst. "
                         "Write concise, actionable market briefs."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()
    except Exception as e:
        logger.error(f"[Agent3] Brief generation error: {e}")
        return (
            f"Market scan complete. {high_impact_count} clients flagged "
            f"for elevated risk. Review high-impact client list immediately."
        )


# ─────────────────────────────────────────────────────────
# AUDIT LOG BUILDER
# ─────────────────────────────────────────────────────────

def build_audit_entry(result: dict) -> dict:
    """Build audit log entry for Agent 4."""
    high_clients = [
        c["client_name"]
        for c in result.get("high_impact_clients", [])
    ][:5]

    return {
        "log_id":             str(uuid.uuid4()),
        "agent_name":         "market_sentiment_agent",
        "trigger_type":       "scheduled_scan",
        "input_summary":      (
            f"Scanned {result.get('total_scanned', 0)} articles | "
            f"relevant: {result.get('relevant_articles', 0)}"
        ),
        "output_summary":     (
            f"High impact clients: {len(high_clients)} | "
            f"Names: {', '.join(high_clients)}"
        ),
        "full_reasoning":     result.get("market_brief", "")[:1000],
        "confidence_score":   0.8,
        "action_taken":       "REPORT_GENERATED",
        "timestamp":          datetime.utcnow().isoformat(),
        "related_entity_id":  None,
        "related_entity_type":"market_scan",
        "query_text":         None
    }


# ─────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────

def run_sentiment_scan() -> dict[str, Any]:
    """
    Main entry point for Agent 3.
    Orchestrates all five components and returns
    the full structured market intelligence report.
    """
    logger.info("\n[Agent3] === Starting Market Sentiment Scan ===")
    scan_start = time.time()

    # ── Load static data ──────────────────────────────────
    clients        = load_clients()
    sector_mapping = load_sector_mapping()

    if not clients:
        return {"error": "No client data found", "scan_timestamp": datetime.utcnow().isoformat()}

    # ── Fetch news ────────────────────────────────────────
    articles = fetch_financial_news(sector_mapping, max_articles=30)
    if not articles:
        return {"error": "No articles fetched", "scan_timestamp": datetime.utcnow().isoformat()}

    # ── Sentiment analysis ────────────────────────────────
    articles_with_sentiment = analyze_sentiment(articles)

    # ── Map to clients ────────────────────────────────────
    client_impact = map_to_clients(
        articles_with_sentiment, clients, sector_mapping
    )

    # ── Build sector summary ──────────────────────────────
    sector_summary = {}
    for sm in sector_mapping:
        sector  = sm["sector"]
        impacts = [
            v for k, v in client_impact.items()
            if v["client"]["sector"] == sector
            and v["relevant_articles"]
        ]
        if not impacts:
            continue

        avg_score        = (
            sum(i["weighted_sentiment"] for i in impacts) / len(impacts)
        )
        affected_clients = len(impacts)
        neg_count        = sum(i["negative_count"] for i in impacts)

        if avg_score <= -0.3:   sentiment_label = "Negative"
        elif avg_score >= 0.2:  sentiment_label = "Positive"
        else:                   sentiment_label = "Neutral"

        sector_summary[sector] = {
            "sentiment":        sentiment_label,
            "score":            round(avg_score, 4),
            "affected_clients": affected_clients,
            "negative_articles":neg_count
        }

    # ── Generate LLM impact reports for affected clients ──
    high_impact_clients   = []
    medium_impact_clients = []

    for client_id, data in client_impact.items():
        if data["impact_level"] in ("High", "Medium") and data["relevant_articles"]:
            logger.info(
                f"[Agent3] Generating impact report for "
                f"{data['client']['client_name']}..."
            )
            llm_output = generate_client_impact(data)

            report = {
                "client_id":           data["client"]["client_id"],
                "client_name":         data["client"]["client_name"],
                "sector":              data["client"]["sector"],
                "risk_rating":         data["client"]["risk_rating"],
                "total_exposure":      round(
                    data["client"]["exposure_short_term"] +
                    data["client"]["exposure_long_term"], 0
                ),
                "credit_utilized_pct": data["client"]["credit_line_utilized_pct"],
                "impact_level":        data["impact_level"],
                "weighted_sentiment":  data["weighted_sentiment"],
                "relevant_news": [
                    {
                        "title":           a["title"],
                        "sentiment":       a["sentiment"],
                        "sentiment_score": a["sentiment_score"],
                        "source":          a["source"],
                        "url":             a["url"]
                    }
                    for a in data["relevant_articles"][:4]
                ],
                "explanation":         llm_output.get("explanation", ""),
                "key_risks":           llm_output.get("key_risks", []),
                "recommended_action":  llm_output.get("recommended_action", ""),
                "urgency":             llm_output.get("urgency", "Monitor"),
                "confidence":          llm_output.get("confidence", 0.5)
            }

            if data["impact_level"] == "High":
                high_impact_clients.append(report)
            else:
                medium_impact_clients.append(report)

    # Sort by total exposure descending
    high_impact_clients.sort(key=lambda x: x["total_exposure"], reverse=True)
    medium_impact_clients.sort(key=lambda x: x["total_exposure"], reverse=True)

    # ── Generate market brief ─────────────────────────────
    market_brief = generate_market_brief(
        articles_with_sentiment,
        sector_summary,
        len(high_impact_clients)
    )

    # ── Build final result ────────────────────────────────
    scan_time = round(time.time() - scan_start, 2)

    result = {
        "scan_timestamp":      datetime.utcnow().isoformat(),
        "scan_duration_sec":   scan_time,
        "total_scanned":       len(articles),
        "relevant_articles":   sum(
            1 for a in articles_with_sentiment
            if a["sentiment"] != "neutral"
        ),
        "high_impact_clients":   high_impact_clients,
        "medium_impact_clients": medium_impact_clients,
        "sector_summary":        sector_summary,
        "market_brief":          market_brief,
        "all_articles": [
            {
                "title":           a["title"],
                "sentiment":       a["sentiment"],
                "sentiment_score": a["sentiment_score"],
                "source":          a["source"],
                "published_at":    a["published_at"]
            }
            for a in articles_with_sentiment
        ]
    }

    logger.success(
        f"[Agent3] Scan complete in {scan_time}s | "
        f"high impact: {len(high_impact_clients)} clients | "
        f"medium: {len(medium_impact_clients)} clients"
    )

    return result


# ─────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    print("\n[Agent3] Running standalone sentiment scan...\n")
    result = run_sentiment_scan()

    print(f"\n{'='*55}")
    print(f"  Scan complete in {result.get('scan_duration_sec')}s")
    print(f"  Articles scanned:    {result.get('total_scanned')}")
    print(f"  High impact clients: {len(result.get('high_impact_clients', []))}")
    print(f"  Medium impact:       {len(result.get('medium_impact_clients', []))}")
    print(f"\n  MARKET BRIEF:")
    print(f"  {result.get('market_brief', '')}")
    print(f"\n  SECTOR SUMMARY:")
    for sector, data in result.get("sector_summary", {}).items():
        print(f"  {sector:<22} {data['sentiment']:<10} "
              f"score: {data['score']:>6} | "
              f"clients: {data['affected_clients']}")

    print(f"\n  HIGH IMPACT CLIENTS:")
    for c in result.get("high_impact_clients", []):
        print(f"\n  [{c['impact_level']}] {c['client_name']} ({c['sector']})")
        print(f"  Exposure: ₹{c['total_exposure']:,.0f} | "
              f"Risk: {c['risk_rating']} | "
              f"Urgency: {c['urgency']}")
        print(f"  {c['explanation'][:120]}...")
    print(f"{'='*55}\n")