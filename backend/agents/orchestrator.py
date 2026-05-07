"""
FINORA — LangGraph Orchestrator
=================================
Coordinates all four agents into a unified intelligent system.

Graph structure:
  router → [regulatory | risk | sentiment | audit | multi] → synthesis → audit_log

State flows through every node — each agent reads what it needs
and writes its result back. LangGraph handles merging.
"""

import os
import re
import json
import uuid
from datetime import datetime
from typing import TypedDict, Annotated, Any
import operator

from loguru import logger
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.3-70b-versatile"

# ── Singleton LLM ─────────────────────────────────────────
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.1,
            max_tokens=1500
        )
    return _llm


# ─────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────

class FINORAState(TypedDict):
    # ── Input ─────────────────────────────────────────────
    query:              str
    query_type:         str          # regulatory|risk|sentiment|audit|multi
    transaction_id:     str | None

    # ── Agent outputs ─────────────────────────────────────
    regulatory_result:  dict | None
    risk_result:        dict | None
    sentiment_result:   dict | None
    audit_result:       dict | None

    # ── Orchestrator metadata ─────────────────────────────
    agents_invoked:     list[str]
    synthesis:          str
    final_response:     dict
    error:              str | None
    session_id:         str


# ─────────────────────────────────────────────────────────
# QUERY CLASSIFIER (ROUTER)
# ─────────────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """
    Rule-based query classifier.
    Fast and free — no LLM needed for clear queries.
    Falls back to 'multi' for ambiguous queries.
    """
    q = query.lower()

    # Transaction / Risk signals
    if any(w in q for w in [
        "txn", "transaction", "block", "fraud", "flagged",
        "risk score", "suspicious", "payment of", "amount"
    ]):
        return "risk"

    # Regulatory / Compliance signals
    if any(w in q for w in [
        "rbi", "sebi", "regulation", "regulatory", "compliance",
        "policy", "kyc", "aml", "circular", "direction", "guideline",
        "norms", "mandatory", "filing", "reporting requirement"
    ]):
        return "regulatory"

    # Market / Sentiment signals
    if any(w in q for w in [
        "market", "news", "sentiment", "client", "sector",
        "nbfc", "exposure", "impact", "fintech sector", "briefing"
    ]):
        return "sentiment"

    # Audit / History signals
    if any(w in q for w in [
        "audit", "log", "history", "decided", "what did",
        "how many", "show me all", "last decision", "recent"
    ]):
        return "audit"

    # Full assessment / broad queries → invoke multiple agents
    if any(w in q for w in [
        "full", "complete", "overall", "everything", "assess",
        "summary", "briefing", "dashboard", "report"
    ]):
        return "multi"

    return "multi"


def extract_transaction_id(query: str) -> str | None:
    """Extract TXN ID from query if present."""
    match = re.search(r'TXN\d{6}', query, re.IGNORECASE)
    return match.group(0).upper() if match else None


# ─────────────────────────────────────────────────────────
# NODE 1 — ROUTER
# ─────────────────────────────────────────────────────────

def router_node(state: FINORAState) -> FINORAState:
    """
    Classifies the query and sets routing metadata.
    This is always the first node in the graph.
    """
    query          = state["query"]
    query_type     = classify_query(query)
    transaction_id = extract_transaction_id(query)

    logger.info(
        f"[Orchestrator] Router → query_type={query_type} | "
        f"txn_id={transaction_id} | query={query[:60]}"
    )

    return {
        **state,
        "query_type":     query_type,
        "transaction_id": transaction_id,
        "agents_invoked": [],
        "error":          None
    }


# ─────────────────────────────────────────────────────────
# NODE 2 — REGULATORY AGENT NODE
# ─────────────────────────────────────────────────────────

def regulatory_node(state: FINORAState) -> FINORAState:
    """Invokes Agent 1 — Regulatory Watch Agent."""
    from agents.regulatory_agent import analyze_regulation

    logger.info("[Orchestrator] → Invoking Regulatory Agent")

    try:
        result = analyze_regulation(state["query"])
        agents_invoked = state.get("agents_invoked", []) + ["regulatory"]
        logger.success(
            f"[Orchestrator] Regulatory done | "
            f"risk={result.get('risk_level')} | "
            f"policies={len(result.get('affected_policies', []))}"
        )
        return {
            **state,
            "regulatory_result": result,
            "agents_invoked":    agents_invoked
        }
    except Exception as e:
        logger.error(f"[Orchestrator] Regulatory node error: {e}")
        return {
            **state,
            "regulatory_result": {"error": str(e)},
            "agents_invoked":    state.get("agents_invoked", []) + ["regulatory"]
        }


# ─────────────────────────────────────────────────────────
# NODE 3 — RISK AGENT NODE
# ─────────────────────────────────────────────────────────

def risk_node(state: FINORAState) -> FINORAState:
    """
    Invokes Agent 2 — Transaction Risk Agent.
    Requires a transaction_id — if none found in query,
    fetches a sample anomalous transaction for demo.
    """
    import asyncio
    from agents.risk_agent import analyze_transaction, engineer_features, run_ml_inference, detect_anomaly_types
    import pandas as pd
    from pathlib import Path

    logger.info("[Orchestrator] → Invoking Risk Agent")

    transaction_id = state.get("transaction_id")

    # If no transaction ID in query, use offline ML inference on sample data
    if not transaction_id:
        try:
            data_dir = Path(__file__).parent.parent / "data" / "raw"
            txn_df   = pd.read_csv(data_dir / "transactions.csv")
            prof_df  = pd.read_csv(data_dir / "user_profiles.csv")

            # Pick a sample anomalous transaction
            sample   = txn_df[txn_df["is_anomaly"] == True].iloc[0].to_dict()
            profile  = prof_df[
                prof_df["user_id"] == sample["user_id"]
            ].iloc[0].to_dict()

            features      = engineer_features(sample, profile)
            ml_result     = run_ml_inference(features["vector"])
            anomaly_types = detect_anomaly_types(features["raw"], sample)

            result = {
                "transaction_id": sample["transaction_id"],
                "user_id":        sample["user_id"],
                "action":         ml_result["action"],
                "risk_score":     ml_result["risk_score"],
                "anomaly_types":  anomaly_types,
                "explanation":    (
                    f"Sample transaction ₹{sample['amount']:,.2f} at "
                    f"{sample['merchant_name']} flagged for "
                    f"{', '.join(anomaly_types)}."
                ),
                "evidence": [
                    f"Risk score: {ml_result['risk_score']}",
                    f"Anomaly types: {', '.join(anomaly_types)}"
                ],
                "policy_refs":    [],
                "note":           "No transaction ID provided — showing sample analysis"
            }

            # Cross-agent trigger: if high risk, also get regulatory context
            if ml_result["action"] in ("BLOCK", "HOLD") and not state.get("regulatory_result"):
                logger.info(
                    "[Orchestrator] High-risk transaction detected — "
                    "triggering Regulatory Agent for compliance context"
                )
                from agents.regulatory_agent import analyze_regulation
                reg_query  = (
                    f"AML and fraud compliance requirements for "
                    f"{', '.join(anomaly_types)} transaction patterns"
                )
                reg_result = analyze_regulation(reg_query)
                return {
                    **state,
                    "risk_result":       result,
                    "regulatory_result": reg_result,
                    "agents_invoked":    state.get("agents_invoked", []) + ["risk", "regulatory"]
                }

        except Exception as e:
            logger.error(f"[Orchestrator] Risk node offline error: {e}")
            result = {"error": str(e), "note": "No transaction ID provided"}

    else:
        # Transaction ID found — but we can't use async DB here
        # Use offline ML inference with CSV lookup instead
        try:
            data_dir = Path(__file__).parent.parent / "data" / "raw"
            txn_df   = pd.read_csv(data_dir / "transactions.csv")
            prof_df  = pd.read_csv(data_dir / "user_profiles.csv")

            txn_rows = txn_df[txn_df["transaction_id"] == transaction_id]
            if txn_rows.empty:
                result = {"error": f"Transaction {transaction_id} not found"}
            else:
                sample  = txn_rows.iloc[0].to_dict()
                prof_rows = prof_df[prof_df["user_id"] == sample["user_id"]]
                profile = prof_rows.iloc[0].to_dict() if not prof_rows.empty else {}

                features      = engineer_features(sample, profile)
                ml_result     = run_ml_inference(features["vector"])
                anomaly_types = detect_anomaly_types(features["raw"], sample)

                result = {
                    "transaction_id": transaction_id,
                    "user_id":        sample["user_id"],
                    "action":         ml_result["action"],
                    "risk_score":     ml_result["risk_score"],
                    "anomaly_types":  anomaly_types,
                    "explanation": (
                        f"Transaction ₹{sample['amount']:,.2f} at "
                        f"{sample['merchant_name']} — "
                        f"action: {ml_result['action']}, "
                        f"score: {ml_result['risk_score']}"
                    ),
                    "evidence": [
                        f"Risk score: {ml_result['risk_score']}",
                        f"Amount ratio: {features['raw']['amount_ratio']}x average",
                        f"Anomaly types: {', '.join(anomaly_types)}"
                    ],
                    "user_context": {
                        "home_city":  profile.get("home_city", ""),
                        "risk_tier":  profile.get("risk_tier", ""),
                        "avg_amount": features["raw"]["user_avg"]
                    },
                    "policy_refs": []
                }

                # Cross-agent trigger for high-risk transactions
                if ml_result["action"] in ("BLOCK", "HOLD"):
                    logger.info(
                        f"[Orchestrator] {ml_result['action']} detected on "
                        f"{transaction_id} — triggering Regulatory Agent"
                    )
                    from agents.regulatory_agent import analyze_regulation
                    reg_query  = (
                        f"Compliance requirements for "
                        f"{', '.join(anomaly_types)} — "
                        f"AML fraud prevention policies"
                    )
                    reg_result = analyze_regulation(reg_query)
                    return {
                        **state,
                        "risk_result":       result,
                        "regulatory_result": reg_result,
                        "agents_invoked":    state.get("agents_invoked", []) + ["risk", "regulatory"]
                    }

        except Exception as e:
            logger.error(f"[Orchestrator] Risk node error: {e}")
            result = {"error": str(e)}

    agents_invoked = state.get("agents_invoked", []) + ["risk"]
    logger.success(
        f"[Orchestrator] Risk done | "
        f"action={result.get('action', 'N/A')} | "
        f"score={result.get('risk_score', 'N/A')}"
    )

    return {**state, "risk_result": result, "agents_invoked": agents_invoked}


# ─────────────────────────────────────────────────────────
# NODE 4 — SENTIMENT AGENT NODE
# ─────────────────────────────────────────────────────────

def sentiment_node(state: FINORAState) -> FINORAState:
    """Invokes Agent 3 — Market Sentiment Agent."""
    from agents.sentiment_agent import run_sentiment_scan

    logger.info("[Orchestrator] → Invoking Sentiment Agent")

    try:
        result         = run_sentiment_scan()
        agents_invoked = state.get("agents_invoked", []) + ["sentiment"]
        logger.success(
            f"[Orchestrator] Sentiment done | "
            f"high_impact={len(result.get('high_impact_clients', []))}"
        )
        return {
            **state,
            "sentiment_result": result,
            "agents_invoked":   agents_invoked
        }
    except Exception as e:
        logger.error(f"[Orchestrator] Sentiment node error: {e}")
        return {
            **state,
            "sentiment_result": {"error": str(e)},
            "agents_invoked":   state.get("agents_invoked", []) + ["sentiment"]
        }


# ─────────────────────────────────────────────────────────
# NODE 5 — AUDIT NODE
# ─────────────────────────────────────────────────────────

def audit_node(state: FINORAState) -> FINORAState:
    """
    Handles audit queries — returns summary of recent decisions.
    Does not use DB directly (orchestrator is sync) — returns
    a structured response pointing to the audit API.
    """
    logger.info("[Orchestrator] → Invoking Audit Agent (summary mode)")

    agents_invoked = state.get("agents_invoked", []) + ["audit"]

    # Build a helpful response directing to audit endpoints
    audit_result = {
        "message": "Audit trail is available via the dedicated audit endpoints.",
        "endpoints": {
            "all_logs":   "GET /api/audit/logs",
            "summary":    "GET /api/audit/summary",
            "nl_query":   "POST /api/audit/query",
            "by_agent":   "GET /api/audit/logs?agent_name=transaction_risk_agent",
            "by_action":  "GET /api/audit/logs?action_taken=BLOCK",
            "last_24h":   "GET /api/audit/logs?since_hours=24"
        },
        "note": (
            "For natural language audit queries like 'what did FINORA block today?' "
            "use POST /api/audit/query directly for the richest response."
        )
    }

    return {
        **state,
        "audit_result":  audit_result,
        "agents_invoked": agents_invoked
    }


# ─────────────────────────────────────────────────────────
# NODE 6 — MULTI AGENT NODE
# ─────────────────────────────────────────────────────────

def multi_node(state: FINORAState) -> FINORAState:
    """
    Invokes all three main agents for broad queries.
    Used for 'full assessment', 'morning briefing', 'dashboard' queries.
    Runs agents sequentially and collects all results.
    """
    logger.info("[Orchestrator] → Multi-agent mode (all three agents)")

    updated_state = state.copy()

    # Agent 1 — Regulatory
    try:
        from agents.regulatory_agent import analyze_regulation
        reg_result = analyze_regulation(
            f"Key compliance risks and regulatory updates relevant to: {state['query']}"
        )
        updated_state["regulatory_result"] = reg_result
        updated_state["agents_invoked"]    = updated_state.get("agents_invoked", []) + ["regulatory"]
        logger.success("[Orchestrator] Multi → Regulatory done")
    except Exception as e:
        logger.error(f"[Orchestrator] Multi → Regulatory error: {e}")
        updated_state["regulatory_result"] = {"error": str(e)}

    # Agent 2 — Risk (sample transaction)
    try:
        import pandas as pd
        from pathlib import Path
        from agents.risk_agent import engineer_features, run_ml_inference, detect_anomaly_types

        data_dir  = Path(__file__).parent.parent / "data" / "raw"
        txn_df    = pd.read_csv(data_dir / "transactions.csv")
        prof_df   = pd.read_csv(data_dir / "user_profiles.csv")
        sample    = txn_df[txn_df["is_anomaly"] == True].iloc[0].to_dict()
        profile   = prof_df[prof_df["user_id"] == sample["user_id"]].iloc[0].to_dict()
        features  = engineer_features(sample, profile)
        ml_result = run_ml_inference(features["vector"])
        anomaly_types = detect_anomaly_types(features["raw"], sample)

        updated_state["risk_result"] = {
            "transaction_id": sample["transaction_id"],
            "action":         ml_result["action"],
            "risk_score":     ml_result["risk_score"],
            "anomaly_types":  anomaly_types,
            "explanation": (
                f"Latest flagged transaction: ₹{sample['amount']:,.2f} "
                f"at {sample['merchant_name']} — {ml_result['action']} "
                f"(score: {ml_result['risk_score']})"
            )
        }
        updated_state["agents_invoked"] = updated_state.get("agents_invoked", []) + ["risk"]
        logger.success("[Orchestrator] Multi → Risk done")
    except Exception as e:
        logger.error(f"[Orchestrator] Multi → Risk error: {e}")
        updated_state["risk_result"] = {"error": str(e)}

    # Agent 3 — Sentiment
    try:
        from agents.sentiment_agent import run_sentiment_scan
        sent_result = run_sentiment_scan()
        updated_state["sentiment_result"] = sent_result
        updated_state["agents_invoked"]   = updated_state.get("agents_invoked", []) + ["sentiment"]
        logger.success("[Orchestrator] Multi → Sentiment done")
    except Exception as e:
        logger.error(f"[Orchestrator] Multi → Sentiment error: {e}")
        updated_state["sentiment_result"] = {"error": str(e)}

    return updated_state


# ─────────────────────────────────────────────────────────
# NODE 7 — SYNTHESIS NODE
# ─────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """You are FINORA's Chief Intelligence Officer.
You synthesize outputs from multiple AI agents into a single, 
clear, actionable intelligence report for fintech risk teams.

Be specific, cite agent findings, and always end with
clear recommended actions. Respond in plain text — no JSON.
"""

def synthesis_node(state: FINORAState) -> FINORAState:
    """
    Combines all agent outputs into one cohesive response.
    Uses Groq LLM to write the synthesis narrative.
    """
    from langchain.schema import HumanMessage, SystemMessage

    logger.info("[Orchestrator] → Synthesizing results")

    agents_invoked   = state.get("agents_invoked", [])
    regulatory_result = state.get("regulatory_result")
    risk_result       = state.get("risk_result")
    sentiment_result  = state.get("sentiment_result")
    audit_result      = state.get("audit_result")

    # Build context from available results
    context_parts = []

    if regulatory_result and not regulatory_result.get("error"):
        context_parts.append(
            f"REGULATORY ANALYSIS:\n"
            f"  Summary: {regulatory_result.get('summary', '')}\n"
            f"  Risk Level: {regulatory_result.get('risk_level', '')}\n"
            f"  Affected Policies: "
            f"{[p['policy_id'] for p in regulatory_result.get('affected_policies', [])]}\n"
            f"  Gap Report: {regulatory_result.get('gap_report', '')[:300]}"
        )

    if risk_result and not risk_result.get("error"):
        context_parts.append(
            f"TRANSACTION RISK ANALYSIS:\n"
            f"  Transaction: {risk_result.get('transaction_id', 'N/A')}\n"
            f"  Action: {risk_result.get('action', 'N/A')}\n"
            f"  Risk Score: {risk_result.get('risk_score', 'N/A')}\n"
            f"  Anomaly Types: {risk_result.get('anomaly_types', [])}\n"
            f"  Explanation: {risk_result.get('explanation', '')}"
        )

    if sentiment_result and not sentiment_result.get("error"):
        high_impact = sentiment_result.get("high_impact_clients", [])
        sector_summary = sentiment_result.get("sector_summary", {})
        neg_sectors = [
            s for s, d in sector_summary.items()
            if d.get("sentiment") == "Negative"
        ]
        context_parts.append(
            f"MARKET INTELLIGENCE:\n"
            f"  Articles Scanned: {sentiment_result.get('total_scanned', 0)}\n"
            f"  High Impact Clients: {len(high_impact)}\n"
            f"  Negative Sectors: {neg_sectors}\n"
            f"  Market Brief: {sentiment_result.get('market_brief', '')[:300]}"
        )

    if audit_result:
        context_parts.append(
            f"AUDIT TRAIL: {audit_result.get('message', '')}"
        )

    if not context_parts:
        synthesis = (
            "No agent results available to synthesize. "
            "Please try a more specific query."
        )
    else:
        full_context = "\n\n".join(context_parts)
        prompt = (
            f"ORIGINAL QUERY: {state['query']}\n\n"
            f"AGENTS INVOKED: {', '.join(agents_invoked)}\n\n"
            f"{full_context}\n\n"
            f"Write a 3-5 sentence synthesis that directly answers the query, "
            f"highlights the most important findings, and ends with 1-3 "
            f"specific recommended actions."
        )

        try:
            llm      = get_llm()
            response = llm.invoke([
                SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            synthesis = response.content.strip()
        except Exception as e:
            logger.error(f"[Orchestrator] Synthesis LLM error: {e}")
            synthesis = " | ".join(
                part.split("\n")[0] for part in context_parts
            )

    logger.success(f"[Orchestrator] Synthesis complete | {len(synthesis)} chars")

    return {**state, "synthesis": synthesis}


# ─────────────────────────────────────────────────────────
# NODE 8 — FINAL RESPONSE BUILDER
# ─────────────────────────────────────────────────────────

def build_response_node(state: FINORAState) -> FINORAState:
    """
    Packages everything into the final API response structure.
    This is the last node before END.
    """
    final_response = {
        "session_id":      state.get("session_id", str(uuid.uuid4())),
        "query":           state["query"],
        "query_type":      state.get("query_type", "unknown"),
        "agents_invoked":  state.get("agents_invoked", []),
        "synthesis":       state.get("synthesis", ""),
        "timestamp":       datetime.utcnow().isoformat(),
        "results": {}
    }

    # Include non-null agent results
    if state.get("regulatory_result"):
        final_response["results"]["regulatory"] = state["regulatory_result"]
    if state.get("risk_result"):
        final_response["results"]["risk"]        = state["risk_result"]
    if state.get("sentiment_result"):
        # Only include summary for multi responses (full result is large)
        sent = state["sentiment_result"]
        final_response["results"]["sentiment"] = {
            "total_scanned":       sent.get("total_scanned"),
            "high_impact_clients": sent.get("high_impact_clients", []),
            "sector_summary":      sent.get("sector_summary", {}),
            "market_brief":        sent.get("market_brief", "")
        }
    if state.get("audit_result"):
        final_response["results"]["audit"] = state["audit_result"]

    return {**state, "final_response": final_response}


# ─────────────────────────────────────────────────────────
# ROUTING FUNCTION
# ─────────────────────────────────────────────────────────

def route_query(state: FINORAState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node based on query_type.
    """
    query_type = state.get("query_type", "multi")
    routes = {
        "regulatory": "regulatory_node",
        "risk":        "risk_node",
        "sentiment":   "sentiment_node",
        "audit":       "audit_node",
        "multi":       "multi_node"
    }
    return routes.get(query_type, "multi_node")


# ─────────────────────────────────────────────────────────
# GRAPH COMPILATION
# ─────────────────────────────────────────────────────────

def build_graph():
    """
    Compile the LangGraph state machine.
    Returns a compiled, runnable graph.
    """
    graph = StateGraph(FINORAState)

    # Add all nodes
    graph.add_node("router_node",       router_node)
    graph.add_node("regulatory_node",   regulatory_node)
    graph.add_node("risk_node",         risk_node)
    graph.add_node("sentiment_node",    sentiment_node)
    graph.add_node("audit_node",        audit_node)
    graph.add_node("multi_node",        multi_node)
    graph.add_node("synthesis_node",    synthesis_node)
    graph.add_node("build_response_node", build_response_node)

    # Entry point
    graph.set_entry_point("router_node")

    # Router → agent nodes (conditional routing)
    graph.add_conditional_edges(
        "router_node",
        route_query,
        {
            "regulatory_node": "regulatory_node",
            "risk_node":       "risk_node",
            "sentiment_node":  "sentiment_node",
            "audit_node":      "audit_node",
            "multi_node":      "multi_node"
        }
    )

    # All agent nodes → synthesis
    for node in ["regulatory_node", "risk_node", "sentiment_node",
                 "audit_node", "multi_node"]:
        graph.add_edge(node, "synthesis_node")

    # Synthesis → final response → END
    graph.add_edge("synthesis_node",      "build_response_node")
    graph.add_edge("build_response_node", END)

    return graph.compile()


# Compile graph at module load — reused across all requests
finora_graph = build_graph()
logger.info("[Orchestrator] LangGraph compiled successfully")


# ─────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────

def run_orchestrator(query: str) -> dict:
    """
    Main entry point for the orchestrator.

    Args:
        query: Natural language query from user or API

    Returns:
        Full structured response with synthesis + agent results
    """
    session_id = str(uuid.uuid4())
    logger.info(f"\n[Orchestrator] === New Session {session_id[:8]} ===")
    logger.info(f"[Orchestrator] Query: {query}")

    initial_state: FINORAState = {
        "query":              query,
        "query_type":         "",
        "transaction_id":     None,
        "regulatory_result":  None,
        "risk_result":        None,
        "sentiment_result":   None,
        "audit_result":       None,
        "agents_invoked":     [],
        "synthesis":          "",
        "final_response":     {},
        "error":              None,
        "session_id":         session_id
    }

    try:
        final_state = finora_graph.invoke(initial_state)
        result = final_state.get("final_response", {})
        logger.success(
            f"[Orchestrator] Session {session_id[:8]} complete | "
            f"agents={result.get('agents_invoked')} | "
            f"synthesis_len={len(result.get('synthesis', ''))}"
        )
        return result

    except Exception as e:
        logger.error(f"[Orchestrator] Graph error: {e}")
        return {
            "session_id":     session_id,
            "query":          query,
            "error":          str(e),
            "agents_invoked": [],
            "synthesis":      f"Orchestrator error: {str(e)}",
            "results":        {}
        }


# ─────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    test_queries = [
        "What does RBI say about KYC compliance for payment aggregators?",
        "Analyze transaction TXN000050",
        "Give me a full intelligence briefing",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print("="*60)
        result = run_orchestrator(query)
        print(f"Type:    {result.get('query_type')}")
        print(f"Agents:  {result.get('agents_invoked')}")
        print(f"\nSYNTHESIS:")
        print(result.get("synthesis", ""))
        print()