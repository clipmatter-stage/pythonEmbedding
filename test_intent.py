import sys
import json
import asyncio
from embeddings_test import analyze_query_intent

async def main():
    query = "speech on electric bill  and load shedding"
    result = await analyze_query_intent(query)
    print(json.dumps(result, indent=2))

asyncio.run(main())
