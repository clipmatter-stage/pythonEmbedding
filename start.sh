#!/bin/bash
# Startup script for Railway deployment
# Railway provides PORT as an environment variable

echo "Starting application with uvicorn..."

# Use uvicorn ASGI server - reads PORT from environment, defaults to 9000
exec uvicorn embeddings_test:app --host 0.0.0.0 --port ${PORT:-9000}
