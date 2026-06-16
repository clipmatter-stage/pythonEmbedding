import os
import logging
from redis import Redis
from rq import Worker, Queue, Connection
from dotenv import load_dotenv

# Ensure we have the application environment loaded (e.g. Qdrant URLs, OpenAI keys)
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
logger.info(f"Connecting to Redis at {redis_url[:20]}...")

redis_conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    # We must import embeddings_test here so the worker has access to the task function and context
    # This ensures that qdrant_client, get_fastembed_model, etc. are properly initialized
    logger.info("Initializing worker application context...")
    import embeddings_test
    
    logger.info("Starting RQ worker listening on 'video_processing' queue...")
    with Connection(redis_conn):
        worker = Worker(['video_processing'])
        worker.work()
