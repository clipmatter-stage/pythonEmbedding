# Contextual video transcript search

FastAPI service using OpenAI dense embeddings to retrieve transcript candidates,
then **Qdrant native MaxSim multivector search** to rank contextual passages.
The local `BAAI/bge-m3` model generates multilingual token vectors; Qdrant computes
relevance. The upstream evidence validator then requires complete-topic quotes and
an independent verification pass before a passage can be returned. There is no
weighted score blend, title score floor, or mandatory query-word overlap.

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
   evidence separately. Context helps retrieval/ranking, but may not supply missing
   evidence: the validator judges only the complete displayed transcript text.
7. Apply the upstream bounded evidence validator and independent verifier to the
   original request. Each required facet needs an exact transcript quote. Provider
   failures return retryable 503 errors; partial validation is reported explicitly.
   Only validated passages are returned, ranked by Qdrant MaxSim within video groups.

All nonnumeric semantic queries, including single words and default `top_k=20`,
follow this pipeline. `/search/incremental` delegates to it and caches the ranked
set for pagination. `/search-multi-video` uses it with video-ID filters.
Numeric-only queries use simple transcript search. `search_mode="simple"` uses the
upstream contracts for text, title, and summary; speaker/date/language/video filters
retain their existing behavior. Summary searches retrieve transcript candidates and
require LLM evidence verification, so they incur embedding and validation calls.
Literal/normalized/alias transcript and title searches do not require an LLM.

```json
{
  "query": "Hafiz Naeem speech on children's rights",
  "search_mode": "semantic",
  "top_k": 20,
  "min_score": 0.35
}
```

`min_score` is the **initial dense retrieval threshold**. Response `dense_score`
retains that similarity. For semantic results, `score` and `llm_relevance_score`
retain the upstream verified 0–1 relevance score; `rerank_score` holds BGE-M3 mean
MaxSim (Qdrant sum divided by query-token count). Neither is a probability or
accuracy percentage. Qdrant determines ordering; evidence validation determines
admission. `ranking` reports the model, candidate count, score type, context,
and query-understanding status. `SEMANTIC_CANDIDATE_LIMIT` bounds both inference
work and the total reranked candidate set. `top_k` is an output maximum; fewer
results may be returned. Increase candidate depth if constrained-speaker recall
is insufficient and your hardware can support it.

The deprecated `USE_RERANKING`, `USE_SIMPLE_RERANKING`, `USE_QUERY_EXPANSION`,
`OPENAI_ONLY_SEARCH`, and `query_decomposition_mode` switches no longer control
semantic ranking. `USE_LLM_UNDERSTANDING=false` disables LLM interpretation and
translation but retains native ranking and mandatory passage validation. Existing
simple filter parameters remain. These settings cannot disable evidence verification.

## Preserved upstream contracts and optimizations

- `transcript_contract="literal"`, `"normalized"` (default), or `"alias"` preserves
  source spans, occurrence counts, utterance identities, and timestamp intervals.
- `summary_candidates` accepts up to 60 client-selected videos with
  `matched_summary_fields`. Summary metadata alone cannot establish a passage hit.
- `search_contract` and `metadata.passage_validation` report authority, counts,
  completeness, warnings, diagnostics, and retryability. Pagination carries them
  forward unchanged, together with the ranking metadata.
- Cursor caches are bound to the original query, filters, transcript contract and
  supplied summary candidates. Changed sessions and expired/malformed cursors
  return 409 instead of silently restarting or returning mismatched cached results.
- `/embed-video` accepts `embedding_model`, `embedding_dimensions`, `run_id`, and
  `timestamp_unit`. Model/dimension mismatches, nonseconds/invalid intervals, and
  invalid batch numbering are rejected before queueing. Source payloads retain
  normalized-index fields, transcript speaker labels and generation metadata.
- Indexing stages batches in Redis by run ID and commits only the active completed
  generation. The existing index survives embedding failures; obsolete IDs are
  removed only after all replacement upserts succeed. Supply globally unique
  `segment_index` values across batches. Legacy jobs without run IDs remain accepted.
- `/operations/points` and `/operations/patch-timestamps` retain upstream strict API
  authentication, bounded snapshots, hash preconditions, and rollback behavior.
- Source reads retain the 35-second/12-call budget. Context reads are grouped per
  video and incomplete context is reported. The shared validator retains its hard
  60-candidate/6-call/30-second limits, including independent verification.
- Complete validation results are cached for five minutes (128 entries) by model,
  original query, speaker constraints and candidate content. Source eligibility is
  checked on every fresh search. Changed content invalidates the validation cache;
  partial/error responses are never cached. Existing cursor paging reuses its
  validated snapshot and does not revalidate each page.

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
  No universal MaxSim confidence cutoff is assumed. The upstream evidence gate
  requires every returned passage to support the complete request, but model
  judgments still need evaluation against human-labeled examples.

## Laravel caller alignment

The `pakistan-clip-matters` caller now forwards original queries, date filters,
transcript contracts and evidence without replacing Python ranking. Its React
results preserve source occurrences and display partial/error states. Semantic
pagination uses the Python cursor directly.

Deploy both projects together. Laravel requires the generation-tracking migration,
frontend rebuild, refreshed config and restarted queue workers. Set its
`EMBEDDING_SERVICE_URL` to this API and `EMBEDDING_API_KEY` to the same `API_KEY`
used by this API and the RQ worker. Its `EMBEDDING_MODEL` and
`EMBEDDING_DIMENSIONS` must match `OPENAI_EMBEDDING_MODEL` and
`EMBEDDING_DIMENSION` here. A new model/dimension still requires a compatible
source collection; environment changes alone do not rebuild vectors.

Laravel submits a generation ID, global segment indices, seconds timestamps and
an HTTPS callback URL (`EMBEDDING_CALLBACK_URL`, defaulting to its `APP_URL` plus
`/api/embeddings/callback`). The RQ worker authenticates callbacks using
`X-API-Key`. Intermediate batches report processing; completed is sent only
after the full generation commits, with its final count and collection. Laravel
ignores stale generation callbacks. HTTP 202 means queued, not indexed.
Let existing legacy jobs finish before switching the two deployments. See the
Laravel project's `docs/embedding-service.md` for commands and environment sample.

## Validation

```sh
python -m unittest discover -s tests -v
```

Tests use real local Qdrant with deterministic token vectors and a mocked OpenAI
transport. They cover ordering, default/single-word searches, cross-language
filters, eligibility, context isolation, cache freshness, provider errors, Chat
Completions parameters, and cursor pagination. They do not measure model semantic
accuracy or call live OpenAI/Qdrant services.
