import sys
import json
import asyncio
from embeddings_test import search

async def main():
    try:
        response = await search(query="Hafiz Naeem ur Rehman Speech on Federalism")
        print(f"Total videos: {response.get('totalVideos', 0)}")
        print(f"Total segments: {response.get('totalSegments', 0)}")
        if response.get('videos'):
            for v in response['videos'][:2]:
                print(f"- {v.get('title', 'Unknown')} (Score: {v.get('max_score', 0)})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
