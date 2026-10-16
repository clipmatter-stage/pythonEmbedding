#!/bin/bash
# Startup script for Railway deployment
# Railway provides PORT as an environment variable

echo "Starting application..."

# Use the new implementation; keep embeddings_test.py as the legacy service.
exec python -u embeddings_service.py
