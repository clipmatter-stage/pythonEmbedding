# Use Python 3.11 slim image
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-rerank.txt ./
ARG INSTALL_RERANKING=false

# Install Python dependencies
RUN if [ "$INSTALL_RERANKING" = "true" ]; then \
      pip install --no-cache-dir -r requirements-rerank.txt; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application code
COPY embeddings_test.py .
COPY embeddings_service.py contextual_search.py fast_search.py worker.py intent_retrieval.py .
COPY backfill_sparse_search.py .
RUN python -m py_compile embeddings_service.py contextual_search.py fast_search.py worker.py backfill_sparse_search.py
COPY semantic_query_decomposition.py .
COPY semantic_passage_evidence.py .
COPY transcript_search_contract.py .
COPY exact_search_cursor.py exact_lexical.py grounded_validation.py grounded_search.py search_observability.py .
COPY grounded_title_search.py grounded_summary_search.py search_snapshots.py manual_metadata.py asr_search.py .
COPY callback_transport.py .
COPY index_operations.py speaker_identity.py .
COPY index_recovery_contract.py .
COPY search_contract_v1.json search_snapshot_migration.json .
COPY backfill_transcript_contract.py .
RUN python -c "from transcript_search_contract import match_spans; assert match_spans('I.P.P.', 'IPP')"
COPY migrate_to_3072.py .
COPY start.sh .

# Fail the image build immediately if the semantic helper is missing,
# malformed, or accidentally contains a self-import. This prevents Railway
# from deploying an image that can only crash-loop at container startup.
RUN python -c "from semantic_query_decomposition import build_structured_rerank_fallback, decompose_semantic_query, extract_meaningful_query_terms, has_conceptual_topic_evidence, has_complete_facet_coverage, passes_structured_topic_validation, recover_empty_structured_rerank; assert all(callable(fn) for fn in (build_structured_rerank_fallback, decompose_semantic_query, extract_meaningful_query_terms, has_conceptual_topic_evidence, has_complete_facet_coverage, passes_structured_topic_validation, recover_empty_structured_rerank))"

RUN python -c "from semantic_passage_evidence import validate_passages, BoundedRetrieval, requested_speaker_names; assert all(callable(fn) for fn in (validate_passages, BoundedRetrieval, requested_speaker_names))"

# Fail the image build if spoken-intent retrieval is absent. fast_search.py
# imports this at module load, and embeddings_service.py imports proposition
# when building a grounded plan. A missing file only showed up as a Railway
# crash-loop after the image had already built.
RUN python -c "from intent_retrieval import proposition, compose_retrieval_queries, alignment_score, keyword_vote_weight; assert all(callable(fn) for fn in (proposition, compose_retrieval_queries, alignment_score, keyword_vote_weight))"

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
