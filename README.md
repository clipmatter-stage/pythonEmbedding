# Video transcript search

`embeddings_service.py` preserves the legacy API and indexing contracts while
using fast vector retrieval by default, with optional grounded verification. `embeddings_test.py` remains available for legacy jobs and endpoints. The Docker image retains its legacy default command; the Railway API
uses the modern service when its custom start command is `bash start.sh`.

## Setup

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in credentials and export/load .env in your shell or deployment.
uvicorn embeddings_service:app --host 0.0.0.0 --port 9000
```

Run Redis and `python worker.py` separately for queued embedding jobs. The
worker needs the same environment as the web service. Existing queued
`embeddings_test.process_video_task` jobs still resolve to the legacy module;
new jobs use `embeddings_service.process_video_task`. Finish legacy jobs before
backfilling the sparse index because legacy upserts do not preserve sparse data.

The Docker image includes both implementations and still defaults to the legacy
script. To select the new service, set Railway's web Custom Start Command to
`bash start.sh`, or run that command locally. Rebuild the image first. Leave the
worker's separate RQ command in place. Verify `GET /` reports version `8.2` and
`ai_features.search_pipeline: fast`. Sparse-index migrations are explicit operations; pushing to a Railway-linked
branch can deploy automatically according to that service's settings.

## Ordinary search and optional verification

`SEARCH_PIPELINE=fast` is the default: one cached query embedding, bounded Qdrant
retrieval, and no mandatory passage judging. `POST /search`, `/search/incremental`
and `/search-multi-video` accept `verify_relevance: true` to select grounded
checking explicitly. This runs a new search and can take tens of seconds.
Laravel forwards this flag and the results page offers **Verify relevance**.
The selected mode is retained in subsequent pages and bound to the cursor;
changing it requires a fresh search. Metadata/exact searches retain their routes.
Summary search keeps its existing evidence requirements and can still be slow.

## Grounded contextual verification (optional)

`SEARCH_PIPELINE=grounded` uses a cached, six-second-budget translation plan to
retrieve the proposition (search instructions such as "find the clip" removed)
plus short spoken Urdu, Roman Urdu and English forms of that same proposition,
falling back to faithful translations of the complete query. At most four query
embeddings are requested concurrently. A shared audience word does not outrank a
passage that covers the proposition; the original query is still what the judge
checks. It combines dense retrieval with the existing sparse keyword index, or an
indexed `text` keyword lane when sparse vectors are unavailable. If neither
keyword index is available, the response reports `keyword_index_unavailable`; it
does not claim hybrid coverage. `verify_relevance: true` selects this path on
`/search` and `/search/incremental` even when `SEARCH_PIPELINE` is `fast`.
`min_score` filters the judge's relevance score after retrieval. It is not a
dense-vector cutoff. Grounded positives already require a judge score of at
least 0.65, so a UI floor of 0.50 does not drop them.
Qdrant groups each retrieval lane by video before truncation, retaining at most
two passages per video per lane. Rank fusion combines votes at video level,
including when different languages retrieve different passages of the same
video, and verification
gives each candidate video a turn before checking second passages. This improves
bounded candidate coverage; it does not establish exhaustive archive recall.

This adds a cached query-planning call and up to three extra query embeddings on
a cold query. No new environment variables, vector migration, speaker repair or
model download are required. Fast mode retains its existing retrieval behavior.
Translation failures retain original-query retrieval with an explicit warning.
Successful query-only plans are shared across replicas in Redis for 24 hours,
bound to the query, model and planner version. A translation or translated-embedding
failure cannot be cached as a completed no-match: verified matches may still be
returned as partial results, while an empty response becomes a retryable error.
The verifier still receives the original intent and applies the same evidence,
role, relationship, quotation, eligibility and authorization checks.

Grounded search expands nearby original
turns only within the same source speaker label and a bounded time window. One
stable-ID structured judge both ranks and checks the complete query intent with
quotes mapped back to original transcript evidence. Only affected missing or
malformed judgments are retried; verified peers survive with explicit partial
status. Provider/protocol failure is not a completed no-match or an unverified
retrieval fallback. Candidate IDs, stage scores and bounded reason codes support
diagnostics without printing transcripts or credentials.

`SEARCH_JUDGE_MODEL` optionally overrides `OPENAI_TEXT_MODEL` for grounded search;
the default remains `gpt-6-luna` after the controlled Phase 4 comparison. No
embedding model/dimension change or reranking-weight download is required. The
old `SEARCH_RERANKING`/`SEARCH_VALIDATE_RESULTS` flags govern `fast` mode, not this
grounded path. `SEARCH_PIPELINE=grounded` can still make checking mandatory for
an entire deployment. Set `verified` for the older independently checked MaxSim path.

Grounded verification uses low reasoning effort on supported reasoning models.
General action queries do not acquire an invented actor or recipient; explicit
actors, audiences, negation and direction still require source support. Exact
quotations and all existing acceptance checks remain mandatory. Changing these
rules invalidates old grounded judgments and search sessions. The explicit
read-only evaluation in `tests/eval_grounded_roles.py` compares repeated judgments
against synthetic controls and the previously authorized live video-592 passage;
it is not run by ordinary unit tests and requires production/provider credentials.
Validated individual judgments are shared in Redis for 15 minutes, independently
of title/summary scan completion. Keys bind the query, exact source passage,
speaker labels, index revision, model, citation mode, reasoning effort and rule
versions. Retrieval and eligibility filters run before lookup, and cached
judgments must pass the same score, role and original-quotation checks again.
Cache outages fall back to judging; invalid judgments and failed expansion are
not cached. The response reports `judgment_cache_hits` and actual provider calls.

Summary candidate selection remains in Laravel with shared multilingual rules.
The full authorized candidate registry (up to 10,000 videos) enters a single
bounded semantic retrieval instead of truncating the first 60 candidate videos.
Bounded semantic evaluation is explicit; summary text is never passage evidence.
Grounded result caches bind query/filter/authorization, source revision, rules,
judge and embedding contract. Cached verified results incur no new judge calls.

Exact search can use versioned keyword-array payload indexes after proving
complete eligible field coverage. It still checks original transcript spans;
missing coverage or unavailable schema falls back to complete continuation.
Literal matching has a separate case-sensitive token field; punctuation-only
queries without usable tokens retain complete traversal. Payload-only backfills require
explicit video IDs, reviewed checkpoints and post-state verification; no vector,
source text, timestamp or producer metadata changes are made. The private
checkpoint is required for guarded rollback. Do not clear uncertain-write state
without inspecting and verifying the affected operation.

## Phase 3 exact search and worker callbacks

The deterministic `/search/exact` path uses `exact_cursor_v1` with the shared
`transcript_contract_v4` matching rules. Continue using the opaque returned
cursor until `counts_final=true`; an empty partial page is not a final no-match.
Cursor expiry, authorization changes and index changes require a new search.
This path makes no embedding or generative model requests.

Phase 3 requires no new Railway environment variable. In **both** API and RQ
worker services, configure the same `API_KEY`, `REDIS_URL`, `QDRANT_URL`,
`QDRANT_API_KEY`, `OPENAI_API_KEY`, `USE_OPENAI_EMBEDDINGS=true`,
`OPENAI_EMBEDDING_MODEL=text-embedding-3-large` and `EMBEDDING_DIMENSION=3072`.
Use the current index's compatible model/dimensions; these are not migration
instructions. `API_KEY` must match Laravel's `EMBEDDING_API_KEY`. Variables on
one Railway service are not proof that the other service receives them.

New jobs carry generation-scoped callback credentials issued by the API.
Legacy queued jobs still need the worker's shared `API_KEY`. The read-only
`/operations/callback-probe` queues the real processing worker and distinguishes
legacy shared-key delivery from job-token delivery without embedding a video.
`CALLBACK_ALLOWED_HOSTS=clip.digitalmatters.pk` is optional and already the code
default; change it only when the callback host changes. Laravel's
`EMBEDDING_EXACT_CURSOR_ENABLED` setting belongs in Laravel, not Python Railway.
Deploy/restart both services after changing their variables.

## Fast search (explicit rollback mode)

1. Resolve known speaker aliases and conservatively decompose explicit speaker
   plus topic requests locally. A mere name mention does not impose a speaker filter.
2. Generate one cached query embedding using the existing stored embedding model
   and dimensions. Do not change either without rebuilding the dense index.
3. Submit one bounded Qdrant query. If the optional `lexical_tf_v1` sparse index
   exists, Qdrant retrieves dense and sparse candidates and combines them with
   native Reciprocal Rank Fusion. Otherwise it retrieves existing dense vectors
   and reports `sparse_index_unavailable_using_dense` in ranking warnings.
4. Apply searchable-video eligibility and video/language/date/time restrictions
   in Qdrant. Apply supported known-speaker filters there too. Speaker/title
   constraints are also checked against returned payloads. Arbitrary fuzzy names
   and titles are post-filtered inside the bounded candidate pool; sparse or
   keyword indexes do not guarantee exhaustive fuzzy recall.
5. Return source text, real segment IDs, speakers and seek timestamps, grouping
   videos by their best-ranked passage. No context scans, model downloads,
   generative query rewriting, or LLM validation run on the default semantic path.

`FAST_SEARCH_CANDIDATE_LIMIT=100` bounds retrieval. `top_k` is an output maximum;
requests larger than the candidate limit may return fewer matches. Cursor paging
reuses its cached snapshot, including across replicas through Redis, without
repeating embedding or ranking. `/search-multi-video` uses the same engine and
candidate budget scoped to the requested IDs; it does not guarantee per-video
coverage if one video dominates the bounded pool.

Cursor snapshots use a bounded Redis timeout (`SEARCH_CACHE_TIMEOUT_SECONDS=3`).
If the initial shared-cache write fails, the response includes the entire bounded
snapshot with no continuation cursor (`metadata.pagination_mode=inline_snapshot`),
so another replica cannot strand the remaining matches. A cache-read outage
returns retryable 503 `search_session_unavailable`; it does not expire the cursor.
A genuinely missing/expired snapshot still returns 409 and requires a fresh search.

Fast results are **ranked search matches, not independently verified answers**.
`metadata.passage_validation` is null. `score_type` is `cosine_similarity` for
dense retrieval or `rrf` for native hybrid retrieval; neither is a probability.
Fusion scores are not comparable with cosine or LLM relevance scores.
Legacy `min_score` is a verified relevance cutoff, so it is not applied to these
retrieval scores and a nonzero supplied value produces an explicit warning.
Use `SEARCH_MIN_DENSE_SCORE` to tune the dense branch (default 0); it cannot
exclude sparse matches. Calibrate it on real queries rather than assuming 0.5
means 50% relevance.

`ranking.timings` separates understanding, embedding, retrieval, reranking,
validation and total time. The earlier 1–3 second latency estimate is a target,
not a measured production guarantee. Benchmark relevance, no-match cases and
latency on English, Urdu and Roman Urdu before switching production.

## Optional native hybrid index

Sparse lexical vectors are normalized word frequencies with stable hashed
32-bit indices; Qdrant supplies IDF. This is TF-IDF, not BM25 or a learned model.
Unicode/Urdu character normalization is deterministic. Lexical search does not
translate Roman Urdu into Urdu; multilingual dense embeddings still provide
cross-language retrieval. Hash collisions are possible. No extra account or
model download is required.

Adding a sparse field to an existing collection requires Qdrant server 1.18+
and the client in `requirements.txt`. Older servers can continue with dense
search. The backfill uses the additive named-vector schema API and
`update_vectors`, retaining every existing dense vector and payload. It does not
regenerate embeddings. Switch indexing workers to the new service first, take
an ordinary database backup, and run:

```sh
python backfill_sparse_search.py          # read-only dry run
python backfill_sparse_search.py --apply  # add sparse field and backfill
```

Restart web replicas after completion so they discover the new schema. While a
backfill is partial, dense retrieval remains available but sparse coverage is
incomplete. The command can resume by rerunning. Pause concurrent transcript
writes during backfill so lexical vectors cannot be derived from stale text.
New ingestion, recovery and `/re-embed-all` maintain sparse vectors when the
field exists. Neither application startup nor a search changes the source schema.
See [Qdrant vector schema updates](https://qdrant.tech/documentation/manage-data/collections/#update-vector-schema)
and [native hybrid fusion](https://qdrant.tech/documentation/search/hybrid-queries/).

## Optional expensive stages

- `SEARCH_LLM_UNDERSTANDING=true`: add the existing LLM query interpretation and
  translation. The older `USE_LLM_UNDERSTANDING` setting does not override fast
  mode's default, so existing deployment variables do not silently enable chat calls.
- `SEARCH_RERANKING=true`: BGE-M3/Qdrant MaxSim reranks at most 20 retrieved
  passages. Install `pip install -r requirements-rerank.txt` first. CPU model
  startup/inference may add substantial latency; use only after measuring it.
  For Docker, build with `--build-arg INSTALL_RERANKING=true` to include these
  optional dependencies; the default image omits them.
- `SEARCH_VALIDATE_RESULTS=true`: invoke bounded complete-topic evidence judging
  and independent verification on the best candidates. `min_score` applies to
  verified relevance again. Results failing evidence validation are omitted.
- `SEARCH_PIPELINE=verified`: retain the previous contextual BGE-M3 plus mandatory
  evidence pipeline, including neighboring source passages and optional LLM
  understanding via `USE_LLM_UNDERSTANDING`. Requires optional rerank dependencies.

Simple speaker/date/language/video filters, literal/normalized/alias transcript
search, title search, suggestions, CRUD/admin operations and indexing methods
remain available. Simple summary search still uses its existing evidence
contract and can invoke embedding and LLM validation calls. These optional or
explicit summary operations are not covered by the default fast latency target.

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
- The optional verified pipeline retains the 35-second/12-call source-read budget. Context reads are grouped per
  video and incomplete context is reported. The shared validator retains its hard
  60-candidate/6-call/30-second limits, including independent verification.
- Complete validation results are cached for five minutes (128 entries) by model,
  original query, speaker constraints and candidate content. Source eligibility is
  checked on every fresh search. Changed content invalidates the validation cache;
  partial/error responses are never cached. Existing cursor paging reuses its
  ranked snapshot and does not revalidate each page.

## Deployment and existing data

- Optional reranking needs Qdrant multivectors (1.10+); adding sparse fields needs 1.18+. Use the
  client version in `requirements.txt`. No provider key is needed for BGE-M3.
  [Native ranking documentation](https://qdrant.tech/documentation/search/hybrid-queries/)
  and [BGE-M3 usage and MIT license](https://huggingface.co/BAAI/bge-m3).
- Optional BGE-M3 requires a multi-GB model download and substantial RAM. CPU cold inference
  can be slow; GPU inference is recommended for sustained traffic. CPU inference threads are
  capped to the container quota (at most 8 by default), with tokenizer thread
  parallelism disabled by default. `QDRANT_RERANK_CPU_THREADS` can lower this cap. On a CUDA host,
  configure suitable PyTorch and `QDRANT_RERANK_DEVICE=cuda:0`. Batch/token limits
  bound memory; inference is serialized per process. Multiple API workers each
  load their own model copy.
- When reranking is enabled, the Qdrant key must create/read/write the rerank collection. Token vectors use
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
  No universal MaxSim confidence cutoff is assumed. When enabled, the evidence gate
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


## Small grounding with separate speaker constraints

With `SEARCH_PIPELINE=grounded` (or `verify_relevance=true`), fast Qdrant
retrieval is followed by a 12-candidate grounding stage, in parallel batches of
six. If fewer than ten videos pass (or the requested `top_k` when smaller), one
more stage is allowed, up to 24 source candidates and a shared 25-second context
and validation budget. This is not a hard HTTP latency limit. Set
`GROUNDING_INITIAL_CANDIDATES` and `GROUNDING_MAX_CANDIDATES` to tune these bounds
(`1 <= initial <= maximum <= 32`). It does not change the default fast pipeline.

Only when that strict pass ends cleanly with zero results (no title, summary or
manual-metadata hit, no budget or context warning) and retrieved candidates are
still unjudged, one extra batch of `GROUNDING_INITIAL_CANDIDATES` passages is
judged. It is taken, in order, from the first Urdu-script expansion's own
retrieval lane, skipping passages already judged. Same judge, prompt, quote
checks, thresholds, batch size and six-call budget. This lets a Roman Urdu or
English request reach the Urdu-script passage that its fused ranking placed just
below the cut. Any search with at least one result never runs it, and its output
is unchanged. The response records the attempt in
`passage_validation.empty_result_rerank`. Kill switch:
`GROUNDED_EMPTY_RESULT_URDU_RERANK=false` (default on); with it off, saved
results that only this pass found are not replayed.

If the strict pass and that extra pass both still end with zero results, the
response may list up to 3 videos (best passage per video) that the judge
already scored at least 0.4 but did not verify (reason `related_only`,
`specificity`, `insufficient_evidence` or `role_mismatch`; never `negation`).
No extra judge call is made. Each such row carries
`verification_status: "related_unverified"`, `evidence_verified: false`,
`transcript_verified: false` and `match_types: ["related_unverified"]`; the
response carries `fallback: "related_not_verified"` at the top level, in
`metadata.passage_validation` (with `related_unverified_count`) and in
`ranking`. `passage_validation.status` stays `no_matches`. These responses
are never saved, and a saved response carrying the flag is never replayed, so
a later verified answer is never blocked. Searches with at least one verified,
title, summary or manual-metadata result never use it. Kill switch:
`GROUNDED_RELATED_UNVERIFIED_FALLBACK=false` (default on) restores the empty
answer exactly.

Saved results are keyed by a hash of the search-rule source files. When a
reviewed change only affects searches that end with zero results, list the
previous hash under `persistent_carry_over` in `search_snapshot_migration.json`
(bound to the new hash), so existing non-empty saved results keep being served.
`tests/test_empty_result_rerank.py` fails if the rules change without that review.

Explicit known-speaker/action requests, including
`hafiz naeem nojawano ko namaz ki talqeen`, separate the speaker constraint from
the action. Retrieval matches configured names; grounding accepts only exact
normalized named speaker aliases from source metadata and checks the remaining
complete action, audience, direction and negation against transcript quotes.
The person need not introduce themselves in the transcript. Anonymous labels
such as `SPEAKER_02` cannot establish that person. A named source-label match is
reported as `named_source_label_match`, not authenticated identity; incorrect
source tagging still requires correction at ingestion.

Grounding joins adjacent original turns from the same speaker and diarization
label, with a one-second maximum gap, within bounded 40-second / 2,400-character
passages. It does not use titles or other speakers to invent missing evidence.
The response reports the original and grounded queries, speaker constraint,
checked candidate counts, and whether retrieved candidates remain unchecked.
Whole-query results remain cached against source revision, scope, model and
rules. Existing pagination pages the evaluated result snapshot; it does not
lazily judge unchecked candidates on View More, and counts are not collection-wide.
No-match results never fall back to unchecked semantic hits. Simple topic
searches without an attributed speaker continue to ground the complete query.


## Titles in grounded AI search

Grounded AI search now also retrieves lexical candidates from `video_title`.
It uses the existing Qdrant title text index when available, or a scoped,
payload-only scan bounded to 10,000 source points / approximately three seconds.
It does not add or rebuild an index at startup. `metadata.title_retrieval`
reports the method, checked coverage, warnings and matched video count; bounded
coverage is not an exhaustive collection-wide count. Eligibility, allowed video
IDs, language, date and time restrictions apply to both candidate sources.

An exact configured person alias such as `HNR` expands to known full-name title
variants. Name-only lookup returns title matches without embedding or grounding
calls. A compound query such as `HNR namaz` retains the topic in every expanded
title query; the alias cannot turn a compound request into a name-only search.
Other queries combine title matches with existing grounded transcript retrieval.

Title-only results have `is_video_only=true`, empty `text`, `evidence_source=title`
and `transcript_verified=false`. They never contain fabricated timestamps or
transcript quotes and do not confirm who actually spoke. Results supported by
both sources keep their verified passages and add title evidence, without a
second video-only result. Verified passages are ranked before title-only videos;
title lexical scores and grounded relevance scores are not compared. The
response-wide `ranking.evidence_verified` is false when title-only rows are present.
The existing Laravel presenter already recognizes video-only title matches.
The separate **Search by Title** category continues to use its existing route.

### Phase 5 identity and citation contract
Keep Railway `SEARCH_PIPELINE=grounded` on API and workers; capabilities report the actual default (`fast` when unset) and whether grounded configuration is intact. Laravel requests verified relevance explicitly.

Speaker names require `speaker_identity_v2` provenance: reviewed manual evidence, or accepted enrollment/calibration/run metadata. Anonymous labels remain separate; unsupported legacy predictions display as Unknown Speaker. Explicit Unknown filtering is separate from all speakers. Search suggestions and named filters do not promote unsupported names.

`/operations/patch-speaker-identities` accepts guarded explicit identity metadata only (1–50 IDs), supports dry run and guarded rollback, and never writes vectors/text/timestamps. Legacy producer provenance blocks assigning a different named person; negative/Unknown quarantine keeps the raw legacy prediction as audit data. Source/database changes use the paired Laravel private checkpoint workflow.

`GROUNDING_CITATION_MODE=span_id` is the Phase 5 default adapter: the model selects source span IDs, which are resolved to original excerpts. `quote` restores the earlier adapter; both retain complete-intent validation and affected-only retries. Cache bindings include citation/identity versions. Completion-based scheduling retains at most two calls and the existing bounded budgets.

## Summary candidates in grounded AI search

Grounded AI search now includes stored summaries alongside titles and transcript
passages. It checks `video_summary`, `video_summary_english`,
`video_summary_urdu`, `summary_en`, and `summary_ur` using normalized lexical
matching and the same explicit person-alias expansion as title retrieval.
Complete query terms must match within one summary field; unrelated fields
cannot supply separate pieces of a compound request. This does not create
summary embeddings or change the document embedding space.

Title and summary discovery run in parallel. Summary retrieval uses available
Qdrant text indexes, then checks unindexed legacy fields through a scoped scan,
sharing a budget of 10,000 source-point reads and approximately three seconds.
Each Qdrant request also has a transport timeout; this is not a hard HTTP latency
guarantee. `metadata.summary_retrieval` reports indexed/searched fields, scanned
points, matching videos, warnings and whether the lookup exhausted its scope.
Missing or bounded coverage must not be interpreted as collection-wide absence.
Existing eligibility, authorized video IDs, language/date/time and explicit
speaker/title constraints apply to this source too. No index backfill or extra
account is required; availability depends on summaries already stored in Qdrant.

Summary-only matches are video-only results with empty transcript text,
`evidence_source=summary`, `transcript_verified=false`, matched summary fields
and bounded excerpts with original character offsets. They do not invent spoken
quotes, timestamps, or speaker identities. Videos found in multiple sources are
merged; verified transcript passages retain their evidence and receive matching
title/summary metadata. Verified passages stay before metadata-only videos.
Pure-name queries such as `HNR` now search both titles and summaries without a
model call. Continuation pages retain summary coverage metadata, and the cache
and cursor strategy version changes so older title/transcript sessions cannot
be reused under the new behavior.

The separate **Search in Summary** category retains its existing behavior:
summary candidate videos must still supply supporting transcript evidence.
Laravel application code is unchanged for this handoff. Apply the instructions
in the Laravel project’s `SUMMARY_SEARCH_INTEGRATION.md` to label summary-only
hits correctly and preserve all matched fields before rolling out the UI.
