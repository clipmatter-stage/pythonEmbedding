#!/bin/bash
# Startup script for Railway deployment
# Handles PORT environment variable properly

# Set default port if PORT is not set
PORT=${PORT:-9000}

echo "Starting uvicorn on port $PORT..."

# Start uvicorn with the port from environment
exec uvicorn embeddings_test:app --host 0.0.0.0 --port "$PORT"
