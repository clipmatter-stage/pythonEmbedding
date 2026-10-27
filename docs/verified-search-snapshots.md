# Verified search snapshots

Completed grounded-search results are stored without expiration in the private
Qdrant collection `verified_search_snapshots_v1`. This derived collection is
separate from transcript vectors and does not advance their index revision.
The existing snapshot lock coordinates concurrent requests across replicas.
Redis remains temporary storage for pagination sessions and individual judgments.

The snapshot key includes the complete search parameters, filters, authorized
video IDs, permission/scope revision, source index revision, search model and
embedding contracts, and a fingerprint of search code and rules. Changed content,
permissions, models or search logic selects a new snapshot. An unrelated deploy
or elapsed time does not. Existing snapshots remain stored for their original
revision and scope; they are not used to grant access to changed/private videos.

Only successful completed/no-match evaluations are saved. Bounded candidate and
metadata coverage stays explicit; a saved result is not a claim of exhaustive
corpus coverage. Transient provider failures and partial validations remain
retryable. Storage read/write failures fail the request instead of silently
rerunning AI and changing a previously verified answer.

Storage persistence and backups follow the existing Qdrant deployment. Removing
that collection intentionally resets verified search history.
