# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY embeddings_test.py .
COPY migrate_to_3072.py .
COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Pre-download FastEmbed model during build to avoid slow startup and HuggingFace rate limits at runtime
RUN python -c "from fastembed import TextEmbedding; model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); list(model.embed(['warmup'])); print('FastEmbed model pre-downloaded successfully')"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Railway will provide PORT environment variable)
EXPOSE 9000

# Health check removed - Railway has built-in health monitoring

# Start command - Railway will inject environment variables at runtime
# Python script will read PORT from environment and default to 9000
CMD ["python", "embeddings_test.py"]
