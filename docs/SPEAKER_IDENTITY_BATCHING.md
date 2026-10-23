# Speaker identity batch writes

`POST /operations/patch-speaker-identities` retains its 1–50 point limit,
authentication, shared per-video Redis lock, source/payload hash checks,
identity-only field allowlist, manual identity provenance checks and final
payload-hash verification.

After validating every change, the endpoint uses Qdrant `batch_update_points`
with ordered SetPayload/DeletePayload operations, `wait=True`, and a 60-second
write timeout. For 50 points requiring both set and delete, this replaces 100
separate write round trips with one batch containing 100 operations. Including
initial and verification reads, that is 3 Qdrant requests instead of 102.
No vectors, source text or anonymous speaker labels are replaced.

The batch is not an atomic transaction. Failed or uncertain requests must retain
checkpoints. Existing expected/post hashes allow retries to skip fully applied
points; intermediate/conflicting payloads still require inspection. Do not
blindly retry without saved post hashes. No fallback to slow individual writes
is attempted after a batch error.

Validation:

    PYTHONPATH=.:tests python -m unittest test_speaker_identity_batch test_speaker_identity test_speaker_identity_api

Tests cover 50 points, actual SDK HTTP serialization/request count, zero-write
dry runs and stale-input rejection, explicit null deletion, preserved vectors
and transcript fields, idempotent replay, partial failure recovery and final
verification failures. These are local/transport tests, not a production
throughput benchmark.

Deploy the Python service before increasing Laravel's live batch size from its
temporary safe override of 5. Start with a representative 50-point request and
measure endpoint latency and error rates; do not assume round-trip reduction is
an equivalent end-to-end speedup. The write timeout remains below the endpoint's
180-second lock TTL; investigate any operation that approaches that TTL rather
than increasing timeouts without coordinating lock ownership.
