# Grounding reliability regression: 23 October 2026

The deployed verifier sometimes rejected video 592's prayer-guidance passage as
`role_mismatch` or `related_only` for `namaz ki talqeen`. The source passage and
its candidate hash were identical between accepted and rejected requests.

The final implementation clarifies that general action queries do not imply a
particular actor or recipient. Explicit roles, recipient groups, negation and
direction still require evidence. It retains the same model (`gpt-6-luna`), uses
low reasoning effort for grounding, and preserves exact-quote, relevance-score,
completeness, role and relationship checks. No video ID forces a production match.

Validated positive and negative judgments can be reused for 15 minutes. Entries
bind the exact query, original passage, speaker labels, index revision, model,
citation mode, reasoning effort and rule versions. Retrieval and authorization
filters run before cache lookup, and cached entries pass the same validator again.
Protocol errors and invalid quotations are never cached. Cache availability does
not determine whether an otherwise verified result survives.

## Measurements

These are fixed, scoped checks, not an archive-wide accuracy guarantee. Provider
protocol retries were allowed by the existing bounds; valid negatives were not
retried until they became positive. Production caches and application files were
not modified by the evaluation processes.

| Check | Deployed baseline | Final reasoning configuration |
| --- | --- | --- |
| Relevant video 592, failed website batch, five fixed arrangements | 2/5 accepted | 5/5 accepted |
| Labeled synthetic controls plus live reference in mixed batches | 22 positive and 30 negative decisions correct | 22 positive and 30 negative decisions correct |
| Five fresh end-to-end searches with response cache disabled | Not a baseline claim | 5/5 returned video 592; 13.69–29.26 seconds |
| Individual judgment reuse, full response cache disabled | Unavailable | Cold 18.37 seconds; repeats 2.49 and 2.47 seconds; 24 cached judgments and zero grounding calls on repeats |
| Deployed public `/search/incremental`, three fresh requests | Not a baseline claim | 3/3 returned video 592; first 28.29 seconds, repeats 7.97 and 10.04 seconds; 24 shared cached judgments and zero grounding calls on repeats |

The isolated cache test used in-memory storage. The public figures include real
shared Redis access, metadata scans and HTTPS; the 2.5-second isolated timings
are not a production latency promise.

A prompt-only attempt missed video 592 in three of five website-batch repeats
and all five fresh end-to-end repetitions, so it was not deployed. Its results are retained
alongside the successful configuration rather than discarded.

Evidence:

- [Prompt-only comparison](evidence/grounding-prompt-comparison.json)
- [Reasoning comparison](evidence/grounding-reasoning-comparison.json)
- [Uncached end-to-end searches](evidence/grounding-reasoning-end-to-end.json)
- [Judgment-cache end-to-end searches](evidence/grounding-judgment-cache-end-to-end.json)
- [Deployed public endpoint verification](evidence/grounding-production-verification.json)

`tests/eval_grounded_roles.py` runs the explicit read-only provider evaluation in
Railway. It requires the service's normal credentials and authorization to send
the query's production candidate excerpts to its configured grounding provider.
It uses cached failed-request source IDs when available, otherwise the saved
ID/hash fixture. Changed or ineligible source passages fail the frozen-source
check; the script never stores production transcript text in repository fixtures.

The evaluation design follows the [official OpenAI evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices): use task-specific controls, include production failures, compare fixed runs and retain failures. Model judgments remain probabilistic; these measurements establish the observed regression result, not universal determinism.

Final validation: 132 targeted unit/integration tests passed in an isolated
Railway directory with external services mocked, including cache corruption,
invalid quotes/roles, source and model changes, eligibility, and SDK parameters.

Deployed through Railway CLI as `19232944-82b3-47f7-85b6-a461d79e0740`.
The public checks confirmed `grounded_passage_v4_cached_query_reasoning` on every
response. Source changes remain uncommitted; no stash or Git history change was
performed. The Laravel project and legacy `embeddings_test.py` were not changed
for this correction.
