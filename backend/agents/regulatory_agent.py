"""
FINORA — Agent 1: Regulatory Watch Agent
==========================================
File location: backend/agents/regulatory_agent.py

RAG-based agent that:
1. Retrieves relevant regulatory chunks from Qdrant
2. Cross-references against internal_policies.csv
3. Generates a structured compliance gap report via Groq LLM

Output schema:
{
  "summary":           str,
  "affected_policies": [{"policy_id": str, "policy_name": str, "reason": str}],
  "gap_report":        str,
  "risk_level":        "High" | "Medium" | "Low",
  "retrieved_sources": [{"title": str, "circular_ref": str, "source": str,
                         "url": str, "score": float}],
  "confidence":        float   (0.0 – 1.0)
}

Confirmed payload keys in Qdrant (from audit run 2026-04-23):
  title, chunk_index, total_chunks, text, content_hash,
  source, circular_ref, pub_date, url, indexed_at
"""

import os
import json
import csv
import re
from pathlib import Path
from typing import Any
from datetime import datetime, UTC
import uuid

from loguru import logger
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

# ── Config ────────────────────────────────────────────────
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
COLLECTION_NAME = "regulatory_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
TOP_K           = 8       # default chunks to retrieve
MIN_SCORE       = 0.30    # drop chunks below this relevance score
DATA_DIR        = Path(__file__).parent.parent / "data" / "raw"


# ── Singletons ────────────────────────────────────────────
# Loaded once on first call, reused for every subsequent request.
# Avoids reloading the 90MB embedding model on every API call.

_embedding_model: SentenceTransformer | None = None
_qdrant_client:   QdrantClient | None        = None
_llm:             ChatGroq | None            = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("[Agent1] Loading embedding model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.success("[Agent1] Embedding model loaded")
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"[Agent1] Qdrant client connected: {QDRANT_HOST}:{QDRANT_PORT}")
    return _qdrant_client


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.1,
            max_tokens=2048
        )
        logger.info(f"[Agent1] LLM ready: {LLM_MODEL}")
    return _llm


# ─────────────────────────────────────────────────────────
# INTERNAL POLICIES LOADER
# ─────────────────────────────────────────────────────────

def load_internal_policies() -> list[dict]:
    """Load internal_policies.csv as a list of dicts."""
    path = DATA_DIR / "internal_policies.csv"
    if not path.exists():
        logger.warning(f"[Agent1] internal_policies.csv not found at {path}")
        return []

    policies = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            policies.append({
                "policy_id":             row.get("policy_id", "").strip(),
                "policy_name":           row.get("policy_name", "").strip(),
                "policy_category":       row.get("policy_category", "").strip(),
                "regulated_entity_type": row.get("regulated_entity_type", "").strip(),
                "policy_content":        row.get("policy_content", "").strip(),
                "status":                row.get("status", "").strip(),
                "owner_team":            row.get("owner_team", "").strip(),
            })

    logger.info(f"[Agent1] Loaded {len(policies)} internal policies")
    return policies


def format_policies_for_prompt(policies: list[dict]) -> str:
    """
    Format all policies into a compact string for the LLM prompt.
    No truncation — policy content is short enough to include in full,
    and the LLM needs specific thresholds (₹10,000 / re-KYC periods etc.)
    to generate meaningful gap analysis.
    """
    lines = []
    for p in policies:
        lines.append(
            f"[{p['policy_id']}] {p['policy_name']} "
            f"({p['policy_category']}, {p['regulated_entity_type']}) "
            f"— {p['policy_content']}"
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────
# QDRANT RETRIEVAL
# ─────────────────────────────────────────────────────────

def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and retrieve top-k relevant chunks from Qdrant.

    Key behaviours:
    - Uses confirmed payload field names from audit (2026-04-23)
    - Drops chunks below MIN_SCORE (0.35) to filter noise
    - Deduplicates by content_hash within a single retrieval
    - Falls back across text/page_content/content field names for
      compatibility if older chunks have different keys

    Returns list of dicts ready for prompt injection.
    """
    model  = get_embedding_model()
    client = get_qdrant_client()

    query_vector = model.encode(query).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )

    chunks      = []
    seen_hashes = set()

    for hit in results:
        # Drop low-relevance chunks — these are typically noise from
        # calendar pages, navigation menus, and unrelated RBI content
        if hit.score < MIN_SCORE:
            logger.debug(
                f"[Agent1] Dropped low-score chunk "
                f"(score={hit.score:.3f}): {hit.payload.get('title', '')[:40]}"
            )
            continue

        payload = hit.payload

        # Deduplicate by content_hash — prevents the same regulatory
        # paragraph appearing twice if it was indexed in multiple runs
        content_hash = payload.get("content_hash", "")
        if content_hash and content_hash in seen_hashes:
            continue
        if content_hash:
            seen_hashes.add(content_hash)

        # Field name fallback chain — "text" is the confirmed key from
        # audit, but fallbacks protect against any older indexed chunks
        text = (
            payload.get("text") or
            payload.get("page_content") or
            payload.get("content") or ""
        )

        if not text.strip():
            continue

        chunks.append({
            "text":         text,
            "title":        payload.get("title") or payload.get("source", "Unknown"),
            "circular_ref": payload.get("circular_ref", ""),
            "source":       payload.get("source", ""),
            "pub_date":     payload.get("pub_date", ""),
            "url":          payload.get("url", ""),
            "score":        round(hit.score, 4)
        })

    logger.info(
        f"[Agent1] Retrieved {len(chunks)} chunks "
        f"(score >= {MIN_SCORE}) for query: '{query[:60]}'"
    )
    return chunks


# ─────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FINORA's Regulatory Watch Agent — a compliance intelligence
system for Indian fintech companies. Your role is to analyze regulatory updates from
RBI and SEBI, cross-reference them against internal company policies, and generate
actionable compliance gap reports.

You must always respond with a valid JSON object — no markdown, no preamble,
no explanation outside the JSON. Just the raw JSON object.
"""

def build_analysis_prompt(
    query: str,
    chunks: list[dict],
    policies: list[dict]
) -> str:
    """
    Build the LLM prompt.

    Note: retrieved_sources is NOT included in the JSON schema here.
    It is injected by analyze_regulation() after parsing the LLM response.
    This avoids f-string + json.dumps fragility when source fields
    contain quotes or special characters.
    """
    # Format retrieved regulatory context with source attribution
    reg_context = ""
    for i, chunk in enumerate(chunks, 1):
        reg_context += (
            f"\n--- Source {i}: {chunk['title']} "
            f"[{chunk['source']} | ref: {chunk['circular_ref']}] "
            f"(relevance score: {chunk['score']}) ---\n"
            f"{chunk['text']}\n"
        )

    policies_context = format_policies_for_prompt(policies)

    return f"""QUERY: {query}

=== RETRIEVED REGULATORY CONTENT ===
{reg_context}

=== INTERNAL COMPANY POLICIES ===
{policies_context}

=== TASK ===
Analyze the retrieved regulatory content above. Cross-reference it against the
internal policies. Identify specific compliance gaps — places where the regulation
imposes requirements that the current policy does not fully address, or where
thresholds, timelines, or procedures differ.

Respond ONLY with this exact JSON structure (no markdown, no extra text):
{{
  "summary": "2-3 sentence plain English summary of what this regulation says and why it matters for the company",
  "affected_policies": [
    {{
      "policy_id": "exact policy ID from the list above e.g. KYC-001",
      "policy_name": "exact policy name",
      "reason": "specific reason why this regulation impacts this policy — include thresholds or requirements where relevant"
    }}
  ],
  "gap_report": "Detailed analysis of gaps between current internal policies and the regulation. Be specific: name the thresholds affected, what the regulation requires vs what the policy currently says, and what action each team must take.",
  "risk_level": "High",
  "confidence": 0.0
}}

Rules:
- risk_level must be exactly one of: High, Medium, Low
- confidence is a float 0.0-1.0: 1.0 = retrieved content directly answers the query, 0.5 = partial match, 0.2 = weak match
- affected_policies must only reference policy_ids that appear in the internal policies list above
- If the retrieved content does not relate to the query at all, set confidence to 0.1 and say so in summary
"""


# ─────────────────────────────────────────────────────────
# CORE ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────

def analyze_regulation(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """
    Main entry point for Agent 1.

    Args:
        query:  Natural language query about a regulation
                e.g. "Latest RBI guidelines on KYC for payment aggregators"
        top_k:  Number of chunks to retrieve from Qdrant (default 6, max 15)

    Returns:
        Structured compliance report dict matching the output schema
        at the top of this file.
    """
    logger.info(f"\n[Agent1] ═══ Analyzing: '{query}' ═══")

    # Step 1 — retrieve relevant chunks from Qdrant
    chunks = retrieve_chunks(query, top_k=top_k)

    if not chunks:
        logger.warning("[Agent1] No chunks retrieved above score threshold")
        return {
            "summary": (
                "No relevant regulatory content found for this query. "
                "The knowledge base may not contain circulars matching this topic. "
                "Try running the seed or poll mode to ingest more content."
            ),
            "affected_policies":  [],
            "gap_report":         "Unable to generate gap report — no matching regulatory content retrieved.",
            "risk_level":         "Unknown",
            "retrieved_sources":  [],
            "confidence":         0.0,
            "error":              "NO_CHUNKS_FOUND"
        }

    # Step 2 — load internal policies
    policies = load_internal_policies()
    if not policies:
        logger.warning("[Agent1] No internal policies loaded — gap analysis will be limited")

    # Step 3 — build prompt and call LLM
    llm    = get_llm()
    prompt = build_analysis_prompt(query, chunks, policies)

    logger.info(
        f"[Agent1] Sending to LLM — "
        f"{len(chunks)} chunks, {len(policies)} policies, "
        f"~{len(prompt)//4} tokens estimated"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    try:
        response = llm.invoke(messages)
        raw_text = response.content.strip()

        # Strip accidental markdown code fences
        # LLMs occasionally wrap JSON in ```json ... ``` despite instructions
        raw_text = re.sub(r"^```json\s*", "", raw_text)
        raw_text = re.sub(r"^```\s*",     "", raw_text)
        raw_text = re.sub(r"\s*```$",     "", raw_text)
        raw_text = raw_text.strip()

        result = json.loads(raw_text)

        # Inject retrieved_sources in Python after parsing —
        # avoids f-string + json.dumps fragility in the prompt template
        result["retrieved_sources"] = [
            {
                "title":        c["title"][:80],
                "circular_ref": c["circular_ref"],
                "source":       c["source"],
                "url":          c["url"],
                "score":        c["score"]
            }
            for c in chunks
        ]

        logger.success(
            f"[Agent1] Analysis complete | "
            f"risk={result.get('risk_level')} | "
            f"confidence={result.get('confidence')} | "
            f"affected_policies={len(result.get('affected_policies', []))}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"[Agent1] JSON parse error: {e}")
        logger.debug(f"[Agent1] Raw LLM response (first 500 chars): {raw_text[:500]}")
        return {
            "summary":           raw_text[:500],
            "affected_policies": [],
            "gap_report":        "JSON parsing failed — raw LLM response stored in summary field.",
            "risk_level":        "Unknown",
            "retrieved_sources": [
                {
                    "title":        c["title"][:80],
                    "circular_ref": c["circular_ref"],
                    "source":       c["source"],
                    "url":          c["url"],
                    "score":        c["score"]
                }
                for c in chunks
            ],
            "confidence": 0.0,
            "error":      "JSON_PARSE_ERROR"
        }

    except Exception as e:
        logger.error(f"[Agent1] LLM call failed: {e}")
        return {
            "summary":           f"Agent error: {str(e)}",
            "affected_policies": [],
            "gap_report":        "",
            "risk_level":        "Unknown",
            "retrieved_sources": [],
            "confidence":        0.0,
            "error":             str(e)
        }


# ─────────────────────────────────────────────────────────
# AUDIT LOG HELPER
# ─────────────────────────────────────────────────────────

def build_audit_entry(query: str, result: dict) -> dict:
    """
    Build a structured audit log entry for Agent 4 to persist.
    Called automatically after every analyze_regulation() call
    via the FastAPI BackgroundTasks mechanism in compliance.py.

    Uses timezone-aware UTC timestamp (datetime.now(UTC)) —
    avoids the deprecated datetime.utcnow() pattern.
    """
    return {
        "log_id":              str(uuid.uuid4()),
        "agent_name":          "regulatory_watch_agent",
        "trigger_type":        "manual_query",
        "input_summary":       query[:200],
        "output_summary":      result.get("summary", "")[:300],
        "full_reasoning":      result.get("gap_report", "")[:1000],
        "confidence_score":    result.get("confidence", 0.0),
        "risk_level":          result.get("risk_level", "Unknown"),
        "affected_policy_ids": [
            p.get("policy_id", "")
            for p in result.get("affected_policies", [])
        ],
        "action_taken":        "REPORT_GENERATED",
        "timestamp":           datetime.now(UTC).isoformat(),
        "related_entity_type": "regulation",
        "query_text":          query,
        "error":               result.get("error")
    }


# ─────────────────────────────────────────────────────────
# QUICK TEST — run directly to verify pipeline end-to-end
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    TEST_QUERIES = [
        "RBI KYC guidelines for payment aggregators customer due diligence",
        "RBI digital lending FLDG co-lending requirements for NBFCs",
        "RBI payment aggregator merchant onboarding escrow settlement",
    ]

    for query in TEST_QUERIES:
        print("\n" + "═" * 70)
        print(f"QUERY: {query}")
        print("═" * 70)

        result = analyze_regulation(query, top_k=6)

        print(f"\nRisk level  : {result.get('risk_level')}")
        print(f"Confidence  : {result.get('confidence')}")
        print(f"Error       : {result.get('error')}")
        print(f"\nSummary:\n{result.get('summary')}")
        print(f"\nAffected policies:")
        for p in result.get("affected_policies", []):
            print(f"  {p.get('policy_id')} — {p.get('reason', '')[:80]}")
        print(f"\nSources retrieved:")
        for s in result.get("retrieved_sources", []):
            print(f"  [{s.get('score')}] {s.get('title', '')[:60]}")
        print(f"\nGap report (first 400 chars):")
        print(result.get("gap_report", "")[:400])

        # Passing criteria — print clearly so you know if it's working
        confidence  = result.get("confidence", 0.0)
        policies    = result.get("affected_policies", [])
        risk        = result.get("risk_level", "Unknown")
        error       = result.get("error")

        print("\n── Test result ──")
        print(f"  confidence > 0.4  : {'PASS' if confidence > 0.4 else 'FAIL'} ({confidence})")
        print(f"  policies found    : {'PASS' if policies else 'FAIL'} ({len(policies)} found)")
        print(f"  risk level valid  : {'PASS' if risk in ('High','Medium','Low') else 'FAIL'} ({risk})")
        print(f"  no error          : {'PASS' if not error else 'FAIL'} ({error})")