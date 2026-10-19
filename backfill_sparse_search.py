"""Add lexical vectors without regenerating or overwriting existing dense vectors.

Run a dry run first; --apply explicitly enables writes. Run after switching
indexing workers to embeddings_service, then restart web replicas to rediscover
the collection schema. Concurrent legacy writers can remove sparse vectors.
"""
import argparse
import os
from qdrant_client import QdrantClient, models
from fast_search import SPARSE_NAME, lexical_vector


def backfill(client, collection, *, apply=False, batch_size=128):
    info = client.get_collection(collection)
    if apply and SPARSE_NAME not in (info.config.params.sparse_vectors or {}):
        if not hasattr(client, "create_vector_name"):
            raise RuntimeError("Adding a sparse field requires qdrant-client>=1.18 and Qdrant server>=1.18")
        client.create_vector_name(collection, SPARSE_NAME, models.SparseVectorNameConfig(
            sparse=models.SparseVectorConfig(modifier=models.Modifier.IDF)), wait=True)
    offset, scanned, updated = None, 0, 0
    while True:
        points, offset = client.scroll(collection, offset=offset, limit=batch_size,
                                       with_payload=["text"], with_vectors=False)
        vectors = [models.PointVectors(id=p.id, vector={SPARSE_NAME: lexical_vector((p.payload or {}).get("text", ""))})
                   for p in points]
        if apply and vectors:
            client.update_vectors(collection, points=vectors, wait=True)
            updated += len(vectors)
        scanned += len(points)
        print(f"scanned={scanned} updated={updated} apply={apply}", flush=True)
        if offset is None:
            return {"scanned": scanned, "updated": updated}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=30)
    try:
        backfill(client, os.getenv("SEGMENTS_COLLECTION", "video_transcript_segments"), apply=args.apply)
    finally:
        client.close()
