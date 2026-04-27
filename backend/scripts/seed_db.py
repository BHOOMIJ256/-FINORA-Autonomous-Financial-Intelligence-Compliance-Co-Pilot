"""
FINORA — Database Seeder
=========================
Reads generated CSVs from data/raw/ and loads them into PostgreSQL.

Run from backend/ directory:
    python scripts/seed_db.py
"""

import csv
import json
import os
import uuid
import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "").replace(
    "postgresql://", "postgresql+asyncpg://"
)

# ── Import models ─────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.transaction     import Transaction
from models.user_profile    import UserProfile
from models.internal_policy import InternalPolicy
from models.client          import Client
from models.audit_log       import AuditLog
from models.base            import Base

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_bool(val: str) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")

def parse_float(val: str):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

# ─────────────────────────────────────────────
# SEEDERS
# ─────────────────────────────────────────────

async def seed_user_profiles(session: AsyncSession):
    path = DATA_DIR / "user_profiles.csv"
    if not path.exists():
        print(f"  ⚠️  {path} not found — skipping")
        return 0

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(UserProfile(
                user_id                 = row["user_id"].strip(),
                user_name               = row["user_name"].strip(),
                home_city               = row["home_city"].strip(),
                home_state              = row["home_state"].strip(),
                account_type            = row["account_type"].strip(),
                avg_txn_amount_90d      = parse_float(row["avg_txn_amount_90d"]),
                stddev_txn_amount_90d   = parse_float(row["stddev_txn_amount_90d"]),
                top_merchant_categories = row["top_merchant_categories"].strip(),
                active_since            = row["active_since"].strip(),
                risk_tier               = row["risk_tier"].strip(),
            ))

    session.add_all(rows)
    await session.flush()
    print(f"  ✅ user_profiles       → {len(rows)} rows")
    return len(rows)


async def seed_transactions(session: AsyncSession):
    path = DATA_DIR / "transactions.csv"
    if not path.exists():
        print(f"  ⚠️  {path} not found — skipping")
        return 0

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(Transaction(
                transaction_id          = row["transaction_id"].strip(),
                user_id                 = row["user_id"].strip(),
                timestamp               = row["timestamp"].strip(),
                amount                  = parse_float(row["amount"]),
                currency                = row["currency"].strip(),
                merchant_name           = row["merchant_name"].strip(),
                merchant_category_code  = row["merchant_category_code"].strip(),
                merchant_city           = row["merchant_city"].strip(),
                merchant_country        = row["merchant_country"].strip(),
                channel                 = row["channel"].strip(),
                device_id               = row["device_id"].strip(),
                ip_country              = row["ip_country"].strip(),
                is_anomaly              = parse_bool(row["is_anomaly"]),
                anomaly_type            = row["anomaly_type"].strip() or None,
                anomaly_reason          = row["anomaly_reason"].strip() or None,
            ))

    # Batch insert in chunks of 500 for performance
    CHUNK = 500
    total = 0
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i+CHUNK]
        session.add_all(chunk)
        await session.flush()
        total += len(chunk)
        print(f"  ✅ transactions        → {total}/{len(rows)} inserted...", end="\r")

    print(f"  ✅ transactions        → {len(rows)} rows                    ")
    return len(rows)


async def seed_internal_policies(session: AsyncSession):
    path = DATA_DIR / "internal_policies.csv"
    if not path.exists():
        print(f"  ⚠️  {path} not found — skipping")
        return 0

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(InternalPolicy(
                policy_id               = row["policy_id"].strip(),
                policy_name             = row["policy_name"].strip(),
                policy_category         = row["policy_category"].strip(),
                regulated_entity_type   = row["regulated_entity_type"].strip(),
                policy_version          = row["policy_version"].strip(),
                effective_date          = row["effective_date"].strip(),
                last_reviewed_date      = row["last_reviewed_date"].strip(),
                policy_content          = row["policy_content"].strip(),
                status                  = row["status"].strip(),
                owner_team              = row["owner_team"].strip(),
            ))

    session.add_all(rows)
    await session.flush()
    print(f"  ✅ internal_policies   → {len(rows)} rows")
    return len(rows)


async def seed_clients(session: AsyncSession):
    path = DATA_DIR / "clients.csv"
    if not path.exists():
        print(f"  ⚠️  {path} not found — skipping")
        return 0

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(Client(
                client_id                   = row["client_id"].strip(),
                client_name                 = row["client_name"].strip(),
                sector                      = row["sector"].strip(),
                product_type                = row["product_type"].strip(),
                exposure_short_term         = parse_float(row["exposure_short_term"]),
                exposure_long_term          = parse_float(row["exposure_long_term"]),
                credit_line_total           = parse_float(row["credit_line_total"]),
                credit_line_utilized_pct    = parse_float(row["credit_line_utilized_pct"]),
                collateral_type             = row["collateral_type"].strip(),
                risk_rating                 = row["risk_rating"].strip(),
                last_reviewed_date          = row["last_reviewed_date"].strip(),
                payment_history             = row["payment_history"].strip(),
                regulatory_sensitivity      = row["regulatory_sensitivity"].strip(),
            ))

    session.add_all(rows)
    await session.flush()
    print(f"  ✅ clients             → {len(rows)} rows")
    return len(rows)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    print("\n🌱 FINORA Database Seeder")
    print("─" * 40)

    if not DATABASE_URL:
        print("❌ DATABASE_URL not found in .env — aborting")
        return

    print(f"📡 Connecting to: {DATABASE_URL[:40]}...")

    engine = create_async_engine(DATABASE_URL, echo=False)
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    # Create all tables
    print("\n📋 Creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  ✅ Tables created (or already exist)")

    # Seed all tables
    print("\n📥 Seeding data...")
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await seed_user_profiles(session)
            await seed_internal_policies(session)
            await seed_clients(session)
            await seed_transactions(session)   # Last — largest table

    print("\n" + "="*40)
    print("  ✅ FINORA database seeded successfully!")
    print("  Open pgAdmin → finora_db to verify.")
    print("="*40 + "\n")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
