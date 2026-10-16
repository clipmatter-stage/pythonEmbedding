# Contextual video transcript search

FastAPI service using OpenAI dense embeddings to retrieve transcript candidates,
then **Qdrant native MaxSim multivector search** to rank contextual passages.
The local `BAAI/bge-m3` model generates multilingual token vectors; Qdrant computes
relevance. Semantic search has no GPT scoring prompt, weighted score blend, title
score floor, or mandatory query-word overlap.

## Setup

Use Python 3.11 or 3.12 with a supported PyTorch installation:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in credentials and export/load .env in your deployment environment.
uvicorn embeddings_service:app --host 0.0.0.0 --port 9000
```

Run Redis and the existing RQ video-processing worker for `/embed-video` jobs.
Docker/Railway and the Procfile use `start.sh`, which starts
`embeddings_service.py` with unbuffered logging. Run `python worker.py` as a
separate worker process with the same environment variables.
After pulling this change, rebuild and redeploy the image so the renamed module
and new dependencies are installed. Startup does not run a vector migration.
`embeddings_test.py` retains the legacy upstream implementation. Previously queued
`embeddings_test.process_video_task` jobs still resolve to that implementation;
new jobs use `embeddings_service.process_video_task`.
Loading `.env` is the responsibility of the shell or deployment; starting the
Python module does not load it automatically.

The default text model is `gpt-6-luna` for focused query understanding, translation,
and ingestion summaries. Verify access in your OpenAI account, or override
`OPENAI_TEXT_MODEL` with a compatible Chat Completions model. Requests use
`max_completion_tokens` and `reasoning_effort="none"`, without sampling parameters.
An override must support these parameters. `text-embedding-3-large` remains the
most capable OpenAI embedding model, so its 3072-dimensional default is retained.
See the [OpenAI model catalog](https://developers.openai.com/api/docs/models)
and [embedding documentation](https://developers.openai.com/api/docs/models/text-embedding-3-large).

## Search behavior

1. Parse the full query meaning and an explicit spoken-by constraint. Known
   speaker-plus-topic phrases are decomposed conservatively before LLM analysis.
2. Retrieve candidates using OpenAI embeddings of the topic and, when available,
   its translation. English queries do not automatically exclude Urdu videos.
3. Enforce searchable-video eligibility, video/language/date/time constraints,
   and explicit speaker/title constraints. Known speaker variants are filtered
   before retrieval; arbitrary fuzzy speaker/title constraints are post-filtered
   within the bounded candidate pool.
4. Build a passage from the focus transcript plus the nearest preceding/following
   same-speaker segments within `SEARCH_CONTEXT_SECONDS`. Context is restricted
   to the same eligible video and caller filters. Titles are excluded as evidence.
   The focus comes first so token truncation cannot discard it in favor of context.
   Unknown-speaker segments do not inherit other turns.
5. Encode passages with BGE-M3, cache token vectors by content/model/settings in a
   separate collection, and ask Qdrant to rank only current source candidate IDs.
   Deleted or changed source passages cannot reappear through the vector cache.
6. Rank videos by their best passage and keep video groups together for pagination.
   Text and timestamps refer to the focus segment; `context_text` exposes nearby
   evidence separately. Contextual relevance can come from this nearby evidence.

All nonnumeric semantic queries, including single words and default `top_k=20`,
follow this pipeline. `/search/incremental` delegates to it and caches the ranked
set for pagination. `/search-multi-video` uses it with video-ID filters.
Numeric-only queries retain literal simple text search. `search_mode="simple"`
retains structured/fuzzy filters and does not call an LLM reranker.

```json
{
  "query": "Hafiz Naeem speech on children's rights",
  "search_mode": "semantic",
  "top_k": 20,
  "min_score": 0.35
}
```

`min_score` is the **initial dense retrieval threshold**. Response `dense_score`
retains that similarity; `score` and `rerank_score` are BGE-M3 mean MaxSim scores
(Qdrant sum divided by query-token count). They are **not probabilities or accuracy
percentages**. `ranking` reports the model, candidate count, score type, context,
and query-understanding status. `SEMANTIC_CANDIDATE_LIMIT` bounds both inference
work and the total reranked candidate set. `top_k` is an output maximum; fewer
results may be returned. Increase candidate depth if constrained-speaker recall
is insufficient and your hardware can support it.

The deprecated `USE_RERANKING`, `USE_SIMPLE_RERANKING`, `USE_QUERY_EXPANSION`,
`OPENAI_ONLY_SEARCH`, and `query_decomposition_mode` switches no longer control
semantic ranking. `USE_LLM_UNDERSTANDING=false` disables LLM interpretation and
translation but retains native ranking. Existing simple filter parameters remain.

## Deployment and existing data

- Use a Qdrant server supporting multivectors (1.10+; 1.17+ recommended), with the
  client version in `requirements.txt`. No provider key is needed for BGE-M3.
  [Native ranking documentation](https://qdrant.tech/documentation/search/hybrid-queries/)
  and [BGE-M3 usage and MIT license](https://huggingface.co/BAAI/bge-m3).
- BGE-M3 requires a multi-GB model download and substantial RAM. CPU cold inference
  can be slow; GPU inference is recommended for sustained traffic. On a CUDA host,
  configure suitable PyTorch and `QDRANT_RERANK_DEVICE=cuda:0`. Batch/token limits
  bound memory; inference is serialized per process. Multiple API workers each
  load their own model copy.
- The Qdrant key must create/read/write the rerank collection. Token vectors use
  an on-disk configuration without HNSW, since only known candidates are rescored.
  They are generated on demand: **no destructive schema migration or full backfill
  is required to start**.
- The persistent vector cache grows with distinct content/settings. To reclaim
  stale vectors, stop API workers, recreate **only the rerank cache collection**,
  then restart. It rebuilds on demand. Do not delete source data for maintenance.
- New ingestion embeds same-speaker contextual passages. To improve initial
  retrieval for existing short fragments, back up your source collection, then
  invoke authenticated `POST /re-embed-all`. It updates dense vectors in place
  and can incur embedding costs. It is not run automatically. Reranking cannot
  recover passages that were never retrieved.
- Keep the source embedding model and dimensions consistent. Changing either
  requires a full rebuild, preferably into a new `SEGMENTS_COLLECTION`. OpenAI
  failures propagate instead of padding unrelated FastEmbed vectors into the
  OpenAI vector space. Rebuild any older fallback vectors with the OpenAI model.
- These changes do not guarantee correctness. Before rollout, measure relevance
  and latency on labeled English, Urdu, and Roman Urdu queries, including compound
  concepts and no-match cases. Set `QDRANT_RERANK_MIN_SCORE` from those measurements.
  No universal confidence cutoff is assumed. Without a cutoff, the API returns
  the best matches above the dense threshold even if none fully answers the query.

## Validation

```sh
python -m unittest discover -s tests -v
```

Tests use real local Qdrant with deterministic token vectors and a mocked OpenAI
transport. They cover ordering, default/single-word searches, cross-language
filters, eligibility, context isolation, cache freshness, provider errors, Chat
Completions parameters, and cursor pagination. They do not measure model semantic
accuracy or call live OpenAI/Qdrant services.
