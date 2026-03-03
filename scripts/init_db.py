"""
Run this once after docker-compose up to verify DB + Qdrant are ready.
Usage: python scripts/init_db.py
"""
import asyncio
from core.database import init_db
from core.vector_store import init_collections

async def main():
    print("Initializing PostgreSQL tables...")
    await init_db()
    print("Initializing Qdrant collections...")
    init_collections()
    print("All done. FINORA is ready.")

if __name__ == "__main__":
    asyncio.run(main())
