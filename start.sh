#!/bin/bash
# Startup script for Railway deployment
# Railway provides PORT as an environment variable

echo "Starting application..."

# Run Python directly - it will read PORT from environment
exec python embeddings_test.py
