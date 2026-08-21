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
COPY semantic_query_decomposition.py .
COPY migrate_to_3072.py .
COPY start.sh .

# Fail the image build immediately if the semantic helper is missing,
# malformed, or accidentally contains a self-import. This prevents Railway
# from deploying an image that can only crash-loop at container startup.
RUN python -c "from semantic_query_decomposition import build_structured_rerank_fallback, decompose_semantic_query, extract_meaningful_query_terms, has_conceptual_topic_evidence, has_complete_facet_coverage, passes_structured_topic_validation, recover_empty_structured_rerank; assert all(callable(fn) for fn in (build_structured_rerank_fallback, decompose_semantic_query, extract_meaningful_query_terms, has_conceptual_topic_evidence, has_complete_facet_coverage, passes_structured_topic_validation, recover_empty_structured_rerank))"

# Make startup script executable
RUN chmod +x start.sh

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Railway will provide PORT environment variable)
EXPOSE 9000

# Health check removed - Railway has built-in health monitoring

# Start command - Railway will inject environment variables at runtime
# Python script will read PORT from environment and default to 9000
CMD ["python", "embeddings_test.py"]
