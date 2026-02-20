"""
Simple script to recreate Qdrant collection with 3072 dimensions.
Run this ONCE on Railway using: python migrate_to_3072.py
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff, PayloadSchemaType
import os

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "video_transcript_segments"

print("=" * 70)
print("RECREATE QDRANT COLLECTION WITH 3072 DIMENSIONS")
print("=" * 70)

if not QDRANT_URL or not QDRANT_API_KEY:
    print("ERROR: QDRANT_URL and QDRANT_API_KEY environment variables required")
    exit(1)

print(f"\nQdrant URL: {QDRANT_URL[:50]}...")
print(f"Collection: {COLLECTION_NAME}")
print(f"New Dimension: 3072")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# Check current status
try:
    info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"\nCurrent: {info.config.params.vectors.size} dims, {info.points_count} points")
    print("Deleting existing collection...")
    client.delete_collection(collection_name=COLLECTION_NAME)
    print("✓ Deleted")
except:
    print("\nNo existing collection found")

# Create with 3072 dimensions
print("Creating collection with 3072 dimensions...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    optimizers_config=OptimizersConfigDiff(indexing_threshold=1000)
)
print("✓ Created")

# Create indexes
print("Creating indexes...")
indexes = [
    ("video_id", PayloadSchemaType.INTEGER),
    ("speaker", PayloadSchemaType.KEYWORD),
    ("video_title", PayloadSchemaType.TEXT),
    ("language", PayloadSchemaType.KEYWORD),
    ("start_time", PayloadSchemaType.FLOAT),
]

for field, schema in indexes:
    try:
        client.create_payload_index(COLLECTION_NAME, field, schema)
        print(f"  ✓ {field}")
    except Exception as e:
        print(f"  - {field}: {str(e)}")

# Verify
info = client.get_collection(collection_name=COLLECTION_NAME)
print(f"\n✅ SUCCESS!")
print(f"Vector size: {info.config.params.vectors.size} dimensions")
print(f"\nYou can now run your app with text-embedding-3-large!")
