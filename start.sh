#!/bin/bash
# Startup script for Railway deployment
# Railway provides PORT as an environment variable

echo "Starting RQ Background Worker..."
# Start the worker process in the background
python worker.py &

echo "Starting FastAPI application..."
# Run Python directly - it will read PORT from environment
# The exec command replaces the shell, running the API in the foreground
exec python embeddings_test.py
