"""
FINORA — Regulatory Fetcher
============================
Autonomously fetches RBI & SEBI regulatory circulars,
chunks them, embeds them, and upserts into Qdrant.

Two modes:
  1. BOOTSTRAP  — crawls historical circulars (run once)
  2. POLL       — checks RSS feeds for new circulars (run daily)

Usage:
  python scripts/regulatory_fetcher.py --mode bootstrap
  python scripts/regulatory_fetcher.py --mode poll
  python scripts/regulatory_fetcher.py --mode both   ← default

APScheduler runs poll() automatically every 24h inside FastAPI.
"""

import os
import sys
import re
import hashlib
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, UpdateStatus
)
from sentence_transformers import SentenceTransformer

# ── Path setup ────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

# ── Config ────────────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME   = "regulatory_docs"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM        = 384          # all-MiniLM-L6-v2 output size
CHUNK_SIZE        = 500          # characters per chunk
CHUNK_OVERLAP     = 100          # overlap between chunks

# RSS feeds — official, no scraping
RSS_SOURCES = [
    {
        "name": "RBI",
        "url":  "https://rbi.org.in/notifications_rss.xml",
        "type": "rss"
    },
    {
        "name": "SEBI",
        "url":  "https://www.sebi.gov.in/sebirss.xml",
        "type": "rss"
    }
]

# Bootstrap — RBI notifications listing page (clean HTML, one-time crawl)
RBI_BOOTSTRAP_URL = "https://rbi.org.in/Scripts/NotificationUser.aspx"
BOOTSTRAP_LIMIT   = 50   # how many historical circulars to fetch on bootstrap

# Keywords to filter relevant circulars (skip irrelevant ones)
RELEVANCE_KEYWORDS = [
    "kyc", "aml", "anti-money laundering", "payment", "digital lending",
    "nbfc", "prepaid", "fintech", "upi", "fraud", "compliance", "know your customer",
    "transaction monitoring", "suspicious", "grievance", "data", "credit",
    "lending", "aggregator", "wallet", "escrow", "settlement", "risk",
    "sebi", "circular", "direction", "regulation", "amendment"
]

# ─────────────────────────────────────────────────────────
# CORE UTILITIES
# ─────────────────────────────────────────────────────────

def get_content_hash(text: str) -> str:
    """SHA256 hash of text — used for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean_html(html: str) -> str:
    """Strip HTML tags and clean whitespace from circular text."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove unicode artifacts
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Tries to split on sentence boundaries for cleaner chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to find a sentence boundary near the end
        if end < len(text):
            # Look for period, question mark, or newline near end
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("\n", start, end)
            )
            if boundary > start + (chunk_size // 2):
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def is_relevant(title: str, text: str) -> bool:
    """Filter circulars to only keep compliance/fintech relevant ones."""
    combined = (title + " " + text[:500]).lower()
    return any(kw in combined for kw in RELEVANCE_KEYWORDS)


# ─────────────────────────────────────────────────────────
# QDRANT SETUP
# ─────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient):
    """Create Qdrant collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE
            )
        )
        logger.info(f"[Qdrant] Created collection: {COLLECTION_NAME}")
    else:
        logger.info(f"[Qdrant] Collection exists: {COLLECTION_NAME}")


def doc_exists(client: QdrantClient, content_hash: str) -> bool:
    """Check if a circular is already in Qdrant by its content hash."""
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [{
                "key": "content_hash",
                "match": {"value": content_hash}
            }]
        },
        limit=1
    )
    return len(results[0]) > 0


def upsert_circular(
    client: QdrantClient,
    model: SentenceTransformer,
    title: str,
    text: str,
    metadata: dict
) -> int:
    """
    Chunk a circular, embed each chunk, upsert into Qdrant.
    Returns number of chunks upserted.
    """
    content_hash = get_content_hash(text)

    # Deduplication check
    if doc_exists(client, content_hash):
        logger.debug(f"  [skip] Already indexed: {title[:60]}")
        return 0

    chunks = chunk_text(text)
    points = []

    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()
        point_id = abs(hash(f"{content_hash}_{i}")) % (2**31)

        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "title":        title,
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "text":         chunk,
                "content_hash": content_hash,
                "source":       metadata.get("source", ""),
                "circular_ref": metadata.get("circular_ref", ""),
                "pub_date":     metadata.get("pub_date", ""),
                "url":          metadata.get("url", ""),
                "indexed_at":   datetime.utcnow().isoformat()
            }
        ))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.success(
            f"  [+] {len(chunks)} chunks | {title[:55]}..."
        )

    return len(chunks)


# ─────────────────────────────────────────────────────────
# RSS POLL MODE
# ─────────────────────────────────────────────────────────

def parse_rss_items(xml_text: str, source_name: str) -> list[dict]:
    """Parse RSS XML and extract circular items."""
    items = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
        channel = root.find("channel")
        if channel is None:
            return items

        for item in channel.findall("item"):
            title_el = item.find("title")
            desc_el  = item.find("description")
            link_el  = item.find("link")
            date_el  = item.find("pubDate")

            title = title_el.text if title_el is not None else ""
            desc  = desc_el.text  if desc_el  is not None else ""
            link  = link_el.text  if link_el  is not None else ""
            date  = date_el.text  if date_el  is not None else ""

            # Clean HTML from description — full text is embedded in RSS
            clean = clean_html(desc)

            # Extract circular reference number from text
            # RBI format: RBI/2025-26/225
            # SEBI format: SEBI/HO/...
            rbi_match  = re.search(r'RBI/\d{4}-\d{2}/\d+', clean)
            sebi_match = re.search(r'SEBI/[A-Z/]+/\d+', clean)
            ref = (rbi_match or sebi_match)
            circular_ref = ref.group(0) if ref else ""

            items.append({
                "title":        title.strip(),
                "text":         clean,
                "url":          link.strip(),
                "pub_date":     date.strip(),
                "circular_ref": circular_ref,
                "source":       source_name
            })

    except ET.ParseError as e:
        logger.error(f"RSS parse error for {source_name}: {e}")

    return items


def poll_rss(client: QdrantClient, model: SentenceTransformer) -> int:
    """Fetch both RSS feeds and ingest new circulars."""
    total_new = 0

    for source in RSS_SOURCES:
        logger.info(f"\n📡 Polling {source['name']} RSS...")
        try:
            response = httpx.get(source["url"], timeout=30, follow_redirects=True)
            response.raise_for_status()
            items = parse_rss_items(response.text, source["name"])
            logger.info(f"  Found {len(items)} items in feed")

            for item in items:
                if not item["text"] or len(item["text"]) < 100:
                    continue
                if not is_relevant(item["title"], item["text"]):
                    logger.debug(f"  [skip] Not relevant: {item['title'][:50]}")
                    continue

                chunks_added = upsert_circular(
                    client, model,
                    title=item["title"],
                    text=item["text"],
                    metadata={
                        "source":       item["source"],
                        "circular_ref": item["circular_ref"],
                        "pub_date":     item["pub_date"],
                        "url":          item["url"]
                    }
                )
                total_new += chunks_added

        except httpx.HTTPError as e:
            logger.error(f"  HTTP error for {source['name']}: {e}")
        except Exception as e:
            logger.error(f"  Unexpected error for {source['name']}: {e}")

    return total_new


# ─────────────────────────────────────────────────────────
# BOOTSTRAP MODE — historical RBI circulars
# ─────────────────────────────────────────────────────────

def fetch_rbi_circular_text(url: str) -> str:
    """Fetch text content from a single RBI circular page."""
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return clean_html(resp.text)
    except Exception as e:
        logger.warning(f"  Could not fetch {url}: {e}")
        return ""

# Master directions to seed — maps directly to your 15 internal policies
MASTER_DIRECTIONS = [
    {
        "name":   "KYC Master Direction",
        "url":    "https://rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566",
        "source": "RBI",
        "ref":    "RBI/DNBR/2016/MD/KYC",
        "tags":   ["KYC-001", "AML-001", "AML-002"]
    },
    {
        "name":   "Digital Lending Guidelines",
        "url":    "https://rbi.org.in/Scripts/NotificationUser.aspx?Id=12382",
        "source": "RBI",
        "ref":    "RBI/2022-23/111",
        "tags":   ["LEND-001", "LEND-002", "LEND-003"]
    },
    {
        "name":   "Payment Aggregator Master Directions",
        "url":    "https://rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12156",
        "source": "RBI",
        "ref":    "RBI/DPSS/2020-21/73",
        "tags":   ["PAY-001", "PAY-002", "PAY-003"]
    },
    {
        "name":   "Integrated Ombudsman Scheme",
        "url":    "https://rbi.org.in/Scripts/BS_ViewMasCirculardetails.aspx?id=12099",
        "source": "RBI",
        "ref":    "RBI/2021-22/174",
        "tags":   ["GRIEV-001", "GRIEV-002"]
    },
    {
        "name":   "Payment System Data Storage",
        "url":    "https://rbi.org.in/Scripts/NotificationUser.aspx?Id=11244",
        "source": "RBI",
        "ref":    "RBI/DPSS/2018-19/104",
        "tags":   ["DATA-001"]
    },
    {
        "name":   "Cyber Security Framework",
        "url":    "https://rbi.org.in/Scripts/NotificationUser.aspx?Id=10435",
        "source": "RBI",
        "ref":    "RBI/2015-16/418",
        "tags":   ["FRAUD-001"]
    },
    {
        "name":   "SFB Concentration Risk Directions",
        "url":    "https://rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12601",
        "source": "RBI",
        "ref":    "RBI/DoR/2024-25/SFB",
        "tags":   ["RISK-001"]
    },
]

def bootstrap_rbi(client: QdrantClient, model: SentenceTransformer,
                  limit: int = BOOTSTRAP_LIMIT) -> int:
    """
    One-time bootstrap: crawl RBI notifications listing page,
    extract circular links, fetch and ingest relevant ones.
    """
    logger.info(f"\n🚀 Bootstrap: fetching last {limit} RBI circulars...")
    total = 0

    try:
        # RBI notifications listing — clean table with circular links
        resp = httpx.get(RBI_BOOTSTRAP_URL, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find all notification links
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "NotificationUser.aspx" in href and "Id=" in href:
                full_url = f"https://rbi.org.in/scripts/{href}" \
                    if not href.startswith("http") else href
                title = a.get_text(strip=True)
                if title and len(title) > 10:
                    links.append({"url": full_url, "title": title})

        # Deduplicate by URL
        seen_urls = set()
        unique_links = []
        for link in links:
            if link["url"] not in seen_urls:
                seen_urls.add(link["url"])
                unique_links.append(link)

        logger.info(f"  Found {len(unique_links)} unique circulars on listing page")
        logger.info(f"  Processing first {min(limit, len(unique_links))}...")

        for i, link in enumerate(unique_links[:limit]):
            title = link["title"]

            # Pre-filter by title before fetching full text
            if not is_relevant(title, ""):
                continue

            logger.info(f"  [{i+1}/{min(limit, len(unique_links))}] {title[:60]}")
            text = fetch_rbi_circular_text(link["url"])

            if not text or len(text) < 200:
                continue

            chunks_added = upsert_circular(
                client, model,
                title=title,
                text=text,
                metadata={
                    "source":       "RBI",
                    "circular_ref": "",
                    "pub_date":     "",
                    "url":          link["url"]
                }
            )
            total += chunks_added

    except Exception as e:
        logger.error(f"Bootstrap error: {e}")

    return total


def bootstrap_sebi(client: QdrantClient, model: SentenceTransformer) -> int:
    """
    SEBI circulars come through RSS fully — just poll with higher limit.
    SEBI RSS contains recent circulars with full text, so RSS poll covers it.
    """
    logger.info("\n📋 SEBI: covered by RSS poll (full text in feed)")
    return 0

# ─────────────────────────────────────────────────────────
# SEED MASTER DIRECTIONS — targeted ingestion
# ─────────────────────────────────────────────────────────

def fetch_pdf_from_page(page_url: str) -> str:
    """
    Follow an RBI page link, find the embedded PDF link,
    extract full text. Falls back to page HTML text if no PDF found.
    This is the key difference from fetch_rbi_circular_text() —
    it specifically hunts for and reads the actual PDF document.
    """
    try:
        resp = httpx.get(page_url, timeout=30, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        # If the URL itself is a PDF
        if "application/pdf" in resp.headers.get("content-type", ""):
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)

        # It's an HTML page — find the PDF link inside it
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                pdf_url = (
                    href if href.startswith("http")
                    else "https://rbi.org.in" + href
                )
                try:
                    import pdfplumber, io
                    pdf_resp = httpx.get(
                        pdf_url, timeout=40, follow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with pdfplumber.open(io.BytesIO(pdf_resp.content)) as pdf:
                        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                        if len(text) > 300:
                            logger.debug(f"    PDF extracted: {len(text)} chars from {pdf_url}")
                            return text
                except Exception as e:
                    logger.warning(f"    PDF read failed ({pdf_url}): {e}")

        # No PDF found — fall back to page HTML text
        logger.warning(f"    No PDF found on page — using HTML text: {page_url}")
        return clean_html(resp.text)

    except Exception as e:
        logger.warning(f"    Could not fetch {page_url}: {e}")
        return ""


def seed_master_directions(client: QdrantClient,
                           model: SentenceTransformer) -> int:
    """
    One-time targeted ingestion of the 7 master directions that your
    15 internal policies are built on. Covers all policy gaps identified
    in the audit (KYC, AML, Digital Lending, Payment Aggregator,
    Ombudsman, Data Storage, Cyber Security, SFB Concentration Risk).

    Uses fetch_pdf_from_page() instead of fetch_rbi_circular_text()
    to ensure actual PDF text is extracted — not just the HTML wrapper.

    Safe to re-run — upsert_circular() deduplicates by content hash.
    """
    logger.info("\n── SEED MASTER DIRECTIONS ──────────────────────────")
    logger.info(f"  Targeting {len(MASTER_DIRECTIONS)} documents covering all 15 policies")
    total = 0

    for i, doc in enumerate(MASTER_DIRECTIONS, 1):
        logger.info(
            f"\n  [{i}/{len(MASTER_DIRECTIONS)}] {doc['name']}"
        )
        logger.info(f"    Covers: {doc['tags']}")
        logger.info(f"    URL: {doc['url']}")

        text = fetch_pdf_from_page(doc["url"])

        if not text or len(text) < 300:
            logger.warning(f"    SKIPPED — could not extract text (got {len(text)} chars)")
            continue

        logger.info(f"    Extracted {len(text):,} chars of regulatory text")

        chunks_added = upsert_circular(
            client, model,
            title=doc["name"],
            text=text,
            metadata={
                "source":       doc["source"],
                "circular_ref": doc["ref"],
                "pub_date":     "",
                "url":          doc["url"]
            }
        )
        total += chunks_added

    logger.success(f"\n  Seed complete — {total} new chunks added")
    return total

# ─────────────────────────────────────────────────────────
# AUDIT — check payload keys and noise ratio
# ─────────────────────────────────────────────────────────

def audit_collection(client: QdrantClient, delete_noise: bool = False):
    """
    Audit the Qdrant collection:
    1. Print actual payload keys (so regulatory_agent.py field names can
       be verified before running the agent)
    2. Count and optionally delete noise chunks

    Args:
        delete_noise: set True only after confirming the noise count looks right
    """
    logger.info("\n── AUDIT ────────────────────────────────────────────")

    # Step 1 — show real payload keys from a live document
    sample = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=3,
        with_payload=True,
        with_vectors=False
    )
    if not sample[0]:
        logger.error("Collection is empty — run bootstrap first")
        return

    logger.info(f"ACTUAL PAYLOAD KEYS : {list(sample[0][0].payload.keys())}")
    logger.info(f"SAMPLE PAYLOAD      : {sample[0][0].payload}")

    # Step 2 — scan all chunks and classify noise vs good
    all_results = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=2000,
        with_payload=True,
        with_vectors=False
    )

    NOISE_PHRASES = [
        "January February March", "All Months",
        "Podcasts Publications", "Denominational numeral",
        "Swachh Bharat", "Working Papers RBI Bulletin",
        "Half-Yearly Quarterly Bi-monthly",
        "Language panel", "Motif of Sanchi"
    ]

    noise_ids, good_ids = [], []
    for point in all_results[0]:
        # Use same fallback chain as regulatory_agent.py retrieve_chunks()
        text = (
            point.payload.get("text") or
            point.payload.get("page_content") or
            point.payload.get("content") or ""
        )
        is_noise = len(text) < 150 or any(p in text for p in NOISE_PHRASES)
        (noise_ids if is_noise else good_ids).append(point.id)

    logger.info(f"\nTotal chunks : {len(all_results[0])}")
    logger.info(f"Good chunks  : {len(good_ids)}")
    logger.info(f"Noise chunks : {len(noise_ids)}")

    if noise_ids and delete_noise:
        from qdrant_client.models import PointIdsList
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=noise_ids)
        )
        logger.success(f"Deleted {len(noise_ids)} noise chunks")
    elif noise_ids:
        logger.warning(
            f"Run with --mode audit_clean to delete {len(noise_ids)} noise chunks"
        )


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def run(mode: str = "both"):
    logger.info("\n" + "="*55)
    logger.info("  FINORA Regulatory Fetcher")
    logger.info("="*55)

    logger.info("\n📦 Connecting to Qdrant...")
    client = get_qdrant_client()
    ensure_collection(client)

    logger.info(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.success("  Model loaded")

    total_chunks = 0

    # Addition 1 — audit mode (read-only, no embedding needed)
    if mode == "audit":
        audit_collection(client, delete_noise=False)
        return

    # Addition 2 — audit_clean mode (deletes noise after confirmation)
    if mode == "audit_clean":
        audit_collection(client, delete_noise=True)
        return

    if mode in ("bootstrap", "both"):
        logger.info("\n── BOOTSTRAP MODE ──────────────────────────────────")
        total_chunks += bootstrap_rbi(client, model, limit=BOOTSTRAP_LIMIT)
        # Addition 3 — seed master directions runs as part of bootstrap
        total_chunks += seed_master_directions(client, model)
        total_chunks += bootstrap_sebi(client, model)

    if mode in ("poll", "both"):
        logger.info("\n── POLL MODE ────────────────────────────────────────")
        total_chunks += poll_rss(client, model)

    # Addition 4 — seed mode alone (without full bootstrap crawl)
    if mode == "seed":
        total_chunks += seed_master_directions(client, model)

    collection_info = client.get_collection(COLLECTION_NAME)
    total_in_qdrant = collection_info.points_count

    logger.info("\n" + "="*55)
    logger.success(f"  Done! +{total_chunks} new chunks ingested")
    logger.success(f"  Total chunks in Qdrant: {total_in_qdrant}")
    logger.info("="*55 + "\n")

# ── APScheduler hook (called from FastAPI lifespan) ───────
def schedule_daily_poll():
    """
    Call this from main.py lifespan to set up daily polling.
    Runs poll() every 24 hours automatically.
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    def _poll_job():
        logger.info("[Scheduler] Running daily regulatory poll...")
        client = get_qdrant_client()
        model  = SentenceTransformer(EMBEDDING_MODEL)
        new_chunks = poll_rss(client, model)
        logger.info(f"[Scheduler] Done. +{new_chunks} new chunks")

    scheduler = BackgroundScheduler()
    scheduler.add_job(_poll_job, "interval", hours=24, id="regulatory_poll")
    scheduler.start()
    logger.info("[Scheduler] Daily regulatory poll scheduled (every 24h)")
    return scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FINORA Regulatory Fetcher")
    parser.add_argument(
    "--mode",
    choices=["bootstrap", "poll", "both", "audit", "audit_clean", "seed"],
    default="both",
    help=(
        "bootstrap = historical crawl + seed master directions | "
        "poll      = RSS feeds only | "
        "both      = bootstrap + poll | "
        "audit     = check payload keys and noise count | "
        "audit_clean = audit + delete noise chunks | "
        "seed      = ingest 7 master directions only"
    )
)
    args = parser.parse_args()
    run(mode=args.mode)