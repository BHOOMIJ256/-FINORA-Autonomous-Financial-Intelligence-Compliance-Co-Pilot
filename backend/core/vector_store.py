from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from core.config import settings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def init_collections():
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION_REGULATORY not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_REGULATORY,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"[Qdrant] Created collection: {settings.QDRANT_COLLECTION_REGULATORY}")
    else:
        print(f"[Qdrant] Collection already exists: {settings.QDRANT_COLLECTION_REGULATORY}")
