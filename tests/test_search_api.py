"""Endpoint integration without external services or model downloads."""
import asyncio
import importlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import httpx
from openai import OpenAI
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contextual_search import ContextualSearch, QdrantReranker, SearchSettings


class MemoryRedis:
    """Redis byte responses and locks for indexing generation integration tests."""
    def __init__(self):
        self.values = {}
        self.hashes = {}

    def lock(self, *args, **kwargs):
        return nullcontext()

    def set(self, key, value, nx=False, **kwargs):
        if nx and key in self.values:
            return False
        self.values[key] = str(value).encode()
        return True

    def get(self, key):
        return self.values.get(key)

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[str(field).encode()] = value.encode()

    def hgetall(self, key):
        return self.hashes.get(key, {}).copy()

    def expire(self, *args):
        return True


class SearchApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = QdrantClient(":memory:")
        stub = types.ModuleType("fastembed")
        stub.TextEmbedding = type("TextEmbedding", (), {})
        environment = {
            "QDRANT_URL": "http://localhost:6333", "QDRANT_API_KEY": "test",
            "OPENAI_API_KEY": "test", "EMBEDDING_DIMENSION": "2",
            "ENSURE_INDEXES_ON_STARTUP": "false", "API_KEY": "",
        }
        with patch.dict(sys.modules, {"fastembed": stub}), patch.dict(os.environ, environment), patch(
            "qdrant_client.QdrantClient", return_value=cls.database,
        ):
            cls.api = importlib.import_module("embeddings_service")

    @classmethod
    def tearDownClass(cls):
        cls.database.close()

    def setUp(self):
        self.shared_cache = MemoryRedis()
        cache_patch = patch.object(self.api, "search_cache_redis", self.shared_cache)
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.api.search_result_cache.clear()
        self.api.embedding_cache.clear()
        self.api.passage_validation_cache.clear()
        self.api.get_contextual_search.cache_clear()

    def put(self, identity, text, video_id=9000, **extra):
        self.database.upsert(self.api.SEGMENTS_COLLECTION, [models.PointStruct(
            id=identity, vector=[1.0, 0.0], payload={
                "video_id": video_id, "text": text, "speaker": "Alice",
                "start_time": 10, "end_time": 20, "timestamp_unit": "seconds",
                "processing_status": "completed", "approval_status": "approved", "is_archived": False,
                **extra,
            },
        )], wait=True)

    def production_service(self):
        with patch.dict(os.environ, {"QDRANT_RERANK_DIMENSION": "2", "SEARCH_CONTEXT_SECONDS": "0"}):
            service = self.api.get_contextual_search()
        service.embed = lambda query: [1.0, 0.0]
        service.understand = lambda query: {"semantic_query": query}
        service.reranker.encoder = lambda texts: [[[1.0, 0.0]] for text in texts]
        return service

    @staticmethod
    def evidence(query, candidates, timeout, facets=None):
        facets = facets or ["complete_topic"]
        return {"required_facets": facets, "passages": [
            {"index": index, "score": .9, "complete": "protection" in row["text"],
             "evidence": {facet: row["text"] for facet in facets}}
            for index, row in enumerate(candidates)
        ]}

    def test_production_semantic_pipeline_requires_independent_displayed_evidence(self):
        self.put(9001, "Children attended a rally.")
        self.put(9002, "Children deserve education and protection.")
        service = self.production_service()
        self.assertIsNotNone(service.validator)
        with patch.object(self.api, "judge_passage_batch", side_effect=self.evidence), patch.object(
            self.api, "verify_passage_batch", side_effect=self.evidence,
        ):
            response = asyncio.run(self.api.search(self.api.SearchRequest(
                query="children's rights", video_id=9000,
            ), True))
        self.assertEqual([9002], [row["id"] for row in response["results"]])
        self.assertTrue(response["results"][0]["independently_verified"])
        self.assertEqual(.9, response["results"][0]["score"])
        self.assertEqual(1.0, response["results"][0]["rerank_score"])
        self.assertTrue(response["metadata"]["passage_validation"]["independent_verification"])

    def test_verification_outage_returns_retryable_503(self):
        self.put(9003, "Children deserve protection.", video_id=9003)
        self.production_service()
        with patch.object(self.api, "judge_passage_batch", side_effect=self.evidence), patch.object(
            self.api, "verify_passage_batch", side_effect=TimeoutError,
        ):
            with self.assertRaises(self.api.HTTPException) as error:
                asyncio.run(self.api.search(self.api.SearchRequest(query="rights", video_id=9003), True))
        self.assertEqual(503, error.exception.status_code)
        self.assertTrue(error.exception.detail["retryable"])
        self.assertEqual("passage_validation_timeout", error.exception.detail["code"])

    def test_verified_judgments_cached_only_for_identical_source_content(self):
        self.put(9004, "Children deserve protection.", video_id=9004)
        self.production_service()
        request = self.api.SearchRequest(query="rights", video_id=9004)
        with patch.object(self.api, "judge_passage_batch", side_effect=self.evidence) as judge, patch.object(
            self.api, "verify_passage_batch", side_effect=self.evidence,
        ):
            first = asyncio.run(self.api.search(request, True))
            second = asyncio.run(self.api.search(request, True))
            self.assertEqual(1, judge.call_count)
            self.assertTrue(second["metadata"]["passage_validation"]["cache_hit"])
            self.assertEqual(0, second["metadata"]["passage_validation"]["provider_calls"])
            self.put(9004, "The protection of children is essential.", video_id=9004)
            asyncio.run(self.api.search(request, True))
            self.assertEqual(2, judge.call_count)
        self.assertFalse(first["metadata"]["passage_validation"]["cache_hit"])

    def test_completed_verification_is_reused_after_local_replica_cache_is_empty(self):
        self.put(9048, "Children deserve protection.", video_id=9048)
        self.production_service()
        request = self.api.SearchRequest(query="rights", video_id=9048)
        with patch.object(self.api, "judge_passage_batch", side_effect=self.evidence) as judge, patch.object(
            self.api, "verify_passage_batch", side_effect=self.evidence,
        ):
            first = asyncio.run(self.api.search(request, True))
            self.api.passage_validation_cache.clear()
            second = asyncio.run(self.api.search(request, True))
        self.assertEqual(1, judge.call_count)
        self.assertEqual(first["results"][0]["id"], second["results"][0]["id"])
        self.assertTrue(second["metadata"]["passage_validation"]["cache_hit"])
        self.assertEqual(0, second["metadata"]["passage_validation"]["provider_calls"])

    def test_search_session_survives_replica_local_cache_miss(self):
        request = self.api.IncrementalSearchRequest(query="rights")
        rows = [{"id": 1, "video_id": 1, "score": .9}, {"id": 2, "video_id": 2, "score": .8}]
        self.api.cache_search_results("shared-session", rows, {"contract_binding": self.api.search_session_binding(request)})
        self.api.search_result_cache.clear()
        cursor = self.api.encode_cursor(1, .9, 0, "shared-session")
        result = asyncio.run(self.api.search_incremental(request.model_copy(update={"cursor": cursor}), True))
        self.assertEqual(2, result["results"][0]["id"])
        self.assertTrue(result["metadata"]["cache_hit"])

    def test_shared_cache_outage_does_not_fail_search(self):
        from unittest.mock import Mock
        self.put(9049, "Children deserve protection.", video_id=9049)
        self.production_service()
        unavailable = Mock()
        unavailable.get.side_effect = ConnectionError("offline")
        unavailable.set.side_effect = ConnectionError("offline")
        with patch.object(self.api, "search_cache_redis", unavailable), patch.object(
            self.api, "judge_passage_batch", side_effect=self.evidence,
        ), patch.object(self.api, "verify_passage_batch", side_effect=self.evidence):
            result = asyncio.run(self.api.search(self.api.SearchRequest(query="rights", video_id=9049), True))
        self.assertEqual(1, len(result["results"]))
        self.assertTrue(result["results"][0]["independently_verified"])

    def test_default_validation_pool_is_bounded_and_shared_cache_never_stores_partial(self):
        service = self.production_service()
        candidates = [{"id": str(i), "video_id": i, "text": "Children deserve protection.", "score": .8,
                       "start_time": 0, "end_time": 5, "speaker": "Alice"} for i in range(30)]
        with patch.object(self.api, "judge_passage_batch", side_effect=self.evidence) as judge, patch.object(
            self.api, "verify_passage_batch", side_effect=self.evidence,
        ):
            results, metadata = service.validator("rights", candidates, None)
        self.assertEqual(20, len(results))
        self.assertEqual(30, metadata["native_candidate_count"])
        self.assertEqual(20, metadata["validation_candidate_limit"])
        self.assertEqual(1, judge.call_count)
        self.assertEqual("completed", metadata["status"])

    def test_literal_normalized_alias_contracts_reach_new_endpoint(self):
        self.put(9010, "IPP I.P.P. آئی پی پی ipp", video_id=9010)
        for contract, count in [("literal", 1), ("normalized", 3), ("alias", 4)]:
            response = asyncio.run(self.api.search(self.api.SearchRequest(
                query="IPP", video_id=9010, search_mode="simple", filter_type="text", transcript_contract=contract,
            ), True))
            self.assertEqual(count, response["search_contract"]["counts"]["matched_transcript_occurrences"])
            self.assertEqual(contract, response["search_contract"]["contract"])
            for row in response["results"]:
                for span in row["match_spans"]:
                    self.assertEqual(span["text"], row["text"][span["start"]:span["end"]])

    def test_title_contract_returns_video_without_invented_passage(self):
        self.put(9011, "Unrelated content.", video_id=9011, video_title="Graduation ceremony")
        response = asyncio.run(self.api.search(self.api.SearchRequest(
            query="Graduation ceremony", video_id=9011, search_mode="simple", filter_type="title",
        ), True))
        self.assertEqual(1, response["returned"])
        self.assertTrue(response["results"][0]["is_video_only"])
        self.assertEqual("", response["results"][0]["text"])

    def test_client_summary_candidates_need_transcript_evidence(self):
        self.put(9012, "Children deserve protection.", video_id=9012)
        request = self.api.SearchRequest(query="rights", search_mode="simple", filter_type="summary",
            summary_candidates=[{"video_id": 9012, "matched_summary_fields": ["video_summary_english"]}])
        with patch.object(self.api, "get_semantic_query_embedding", return_value=[1.0, 0.0]), patch.object(
            self.api, "judge_passage_batch", side_effect=self.evidence,
        ), patch.object(self.api, "verify_passage_batch", side_effect=self.evidence):
            response = asyncio.run(self.api.search(request, True))
        self.assertEqual(9012, response["results"][0]["video_id"])
        self.assertEqual(["video_summary_english"], response["results"][0]["matched_summary_fields"])
        self.assertTrue(response["results"][0]["passage_evidence"])

    def test_cursor_rejects_changed_query_contract_and_expired_cache(self):
        rows = [{"id": index, "video_id": index, "score": .9} for index in [9013, 9014]]
        self.api.cache_search_results("binding-session", rows, {
            "contract_binding": self.api.search_session_binding(self.api.IncrementalSearchRequest(query="IPP")),
        })
        cursor = self.api.encode_cursor(9013, .9, 0, "binding-session")
        for changes in [{"query": "ipp"}, {"query": "IPP", "transcript_contract": "literal"}]:
            with self.assertRaises(self.api.HTTPException) as error:
                asyncio.run(self.api.search_incremental(self.api.IncrementalSearchRequest(
                    cursor=cursor, **changes,
                ), True))
            self.assertEqual("search_session_mismatch", error.exception.detail["code"])
        self.api.search_result_cache.clear()
        self.shared_cache.values.clear()
        with self.assertRaises(self.api.HTTPException) as error:
            asyncio.run(self.api.search_incremental(self.api.IncrementalSearchRequest(query="IPP", cursor=cursor), True))
        self.assertEqual("search_session_expired", error.exception.detail["code"])

    def test_ingestion_rejects_model_dimension_timestamp_and_interval_mismatches_before_queue(self):
        from unittest.mock import Mock
        queue = Mock()
        for changes in [{"embedding_model": "another-model"}, {"embedding_dimensions": 999},
                        {"timestamp_unit": "milliseconds"},
                        {"identification_segments": [{"text": "valid text", "start": 8, "end": 1}]}]:
            values = {"video_id": 9020, "identification_segments": [{"text": "valid text", "start": 1, "end": 8}], **changes}
            with patch.object(self.api, "task_queue", queue), self.assertRaises(self.api.HTTPException) as error:
                asyncio.run(self.api.embed_video(self.api.EmbedVideoRequest(**values), True))
            self.assertEqual(422, error.exception.status_code)
        queue.enqueue.assert_not_called()

    def test_staged_generation_preserves_old_index_until_all_batches_succeed(self):
        redis = MemoryRedis()
        redis.set("index-active:9021", "generation")
        self.put(9021, "Old indexed text.", video_id=9021)
        with patch.object(self.api, "redis_conn", redis), patch.object(
            self.api, "get_openai_embeddings_batch", side_effect=lambda texts: [[1.0, 0.0] for text in texts],
        ):
            for number in [1, 2]:
                self.api.process_video_task({
                    "video_id": 9021, "run_id": "generation", "timestamp_unit": "seconds",
                    "batch_info": {"batch_number": number, "total_batches": 2},
                    "identification_segments": [{"segment_index": number, "text": f"New batch {number}",
                                                 "start": number * 10, "end": number * 10 + 5}],
                })
                points, _ = self.database.scroll(self.api.SEGMENTS_COLLECTION, scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="video_id", match=models.MatchValue(value=9021)),
                ]), with_payload=True)
                if number == 1:
                    self.assertEqual(["Old indexed text."], [p.payload["text"] for p in points])
                else:
                    self.assertEqual({"New batch 1", "New batch 2"}, {p.payload["text"] for p in points})
                    self.assertTrue(all(p.payload["indexing_run_id"] == "generation" for p in points))
                    self.assertTrue(all("transcript_normalized_v1" in p.payload for p in points))

    def test_worker_provider_failure_preserves_existing_generation(self):
        self.put(9022, "Existing source text.", video_id=9022)
        with patch.object(self.api, "get_openai_embeddings_batch", side_effect=RuntimeError("provider failed")):
            with self.assertRaises(RuntimeError):
                self.api.process_video_task({"video_id": 9022,
                    "identification_segments": [{"text": "Replacement source", "start": 0, "end": 5}]})
        self.assertEqual("Existing source text.", self.database.retrieve(self.api.SEGMENTS_COLLECTION, [9022])[0].payload["text"])

    def test_admin_operation_routes_and_upstream_request_fields_are_preserved(self):
        import ast
        legacy = ast.parse((Path(__file__).resolve().parents[1] / "embeddings_test.py").read_text())
        for name in ["EmbedVideoRequest", "SearchRequest", "IncrementalSearchRequest"]:
            node = next(n for n in legacy.body if isinstance(n, ast.ClassDef) and n.name == name)
            fields = {n.target.id for n in node.body if isinstance(n, ast.AnnAssign)}
            self.assertTrue(fields.issubset(getattr(self.api, name).model_fields), name)
        paths = {route.path for route in self.api.app.routes}
        self.assertTrue({"/operations/points", "/operations/patch-timestamps"}.issubset(paths))

    def test_native_high_score_cannot_bypass_independent_rejection(self):
        self.put(9030, "Children deserve protection.", video_id=9030)
        self.production_service()

        def reject(*args):
            response = self.evidence(*args)
            for row in response["passages"]:
                row["complete"] = False
            return response

        with patch.object(self.api, "judge_passage_batch", side_effect=self.evidence), patch.object(
            self.api, "verify_passage_batch", side_effect=reject,
        ):
            response = asyncio.run(self.api.search(self.api.SearchRequest(query="rights", video_id=9030), True))
        self.assertEqual([], response["results"])
        self.assertEqual("no_matches", response["metadata"]["passage_validation"]["status"])

    def test_invalid_evidence_is_partial_and_not_cached_as_no_matches(self):
        self.put(9040, "Children deserve protection.", video_id=9040)
        self.production_service()

        def fabricate(*args):
            response = self.evidence(*args)
            response["passages"][0]["evidence"] = {"complete_topic": "An invented quotation"}
            return response

        with patch.object(self.api, "judge_passage_batch", side_effect=fabricate) as judge:
            for _ in range(2):
                response = asyncio.run(self.api.search(self.api.SearchRequest(query="rights", video_id=9040), True))
                self.assertEqual([], response["results"])
                self.assertEqual("partial", response["metadata"]["passage_validation"]["status"])
                self.assertTrue(response["metadata"]["passage_validation"]["retryable"])
            self.assertEqual(2, judge.call_count)

    def test_eligibility_changes_cannot_reuse_a_cached_positive(self):
        self.put(9031, "Children deserve protection.", video_id=9031)
        self.production_service()
        request = self.api.SearchRequest(query="rights", video_id=9031)
        with patch.object(self.api, "judge_passage_batch", side_effect=self.evidence), patch.object(
            self.api, "verify_passage_batch", side_effect=self.evidence,
        ):
            self.assertEqual(1, asyncio.run(self.api.search(request, True))["returned"])
            self.put(9031, "Children deserve protection.", video_id=9031, approval_status="pending")
            self.assertEqual(0, asyncio.run(self.api.search(request, True))["returned"])

    def test_verifier_role_contract_and_current_model_parameters(self):
        captured = []

        def transport(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "chat-evidence", "object": "chat.completion", "created": 0, "model": "gpt-6-luna",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps({
                    "required_facets": ["topic"], "passages": [{"index": 0, "score": .99, "complete": True,
                        "roles_supported": False, "relationship_supported": True, "specificity_supported": True,
                        "evidence": [{"facet": "topic", "quote": "Children deserve protection."}]}],
                })}}],
            })

        client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(transport)))
        with patch.object(self.api, "openai_client", client):
            response = self.api.verify_passage_batch("rights", [{"text": "Children deserve protection."}], 5, ["topic"])
        self.assertFalse(response["passages"][0]["complete"])
        self.assertEqual("gpt-6-luna", captured[0]["model"])
        self.assertEqual("none", captured[0]["reasoning_effort"])
        self.assertEqual("json_schema", captured[0]["response_format"]["type"])
        self.assertNotIn("temperature", captured[0])
        client.close()

    def test_incremental_contract_metadata_survives_cache_and_contract_changes_fail(self):
        self.put(9032, "IPP IPP", video_id=9032)
        self.put(9033, "IPP", video_id=9033)
        request = self.api.IncrementalSearchRequest(query="IPP", search_mode="simple", filter_type="text", batch_size=1)
        first = asyncio.run(self.api.search_incremental(request, True))
        second = asyncio.run(self.api.search_incremental(request.model_copy(update={"cursor": first["cursor"]["next"]}), True))
        self.assertTrue(second["metadata"]["cache_hit"])
        self.assertEqual(first["metadata"]["search_contract"], second["metadata"]["search_contract"])
        with self.assertRaises(self.api.HTTPException) as error:
            asyncio.run(self.api.search_incremental(request.model_copy(update={
                "cursor": first["cursor"]["next"], "transcript_contract": "literal",
            }), True))
        self.assertEqual(409, error.exception.status_code)

    def test_enqueue_registers_generation_and_preserves_new_job_module(self):
        from unittest.mock import Mock
        redis = MemoryRedis()
        queue = Mock()
        queue.enqueue.return_value.id = "job-1"
        request = self.api.EmbedVideoRequest(video_id=9034, run_id="generation-1",
            embedding_model=self.api.OPENAI_EMBEDDING_MODEL, embedding_dimensions=2,
            identification_segments=[{"text": "New source text", "start": 1, "end": 5}])
        with patch.object(self.api, "redis_conn", redis), patch.object(self.api, "task_queue", queue):
            response = asyncio.run(self.api.embed_video(request, True))
        self.assertEqual("queued", response["status"])
        self.assertEqual(b"generation-1", redis.get("index-active:9034"))
        self.assertEqual("embeddings_service.process_video_task", queue.enqueue.call_args.args[0])
        self.assertEqual("generation-1", queue.enqueue.call_args.args[1]["run_id"])

    def test_stale_generation_does_not_overwrite_current_index(self):
        redis = MemoryRedis()
        redis.set("index-active:9035", "current-generation")
        self.put(9035, "Current source text.", video_id=9035)
        with patch.object(self.api, "redis_conn", redis), patch.object(
            self.api, "get_openai_embeddings_batch", return_value=[[1.0, 0.0]],
        ):
            response = self.api.process_video_task({"video_id": 9035, "run_id": "stale-generation",
                "identification_segments": [{"text": "Stale replacement", "start": 0, "end": 5}]})
        self.assertEqual({"status": "stale"}, response)
        self.assertEqual("Current source text.", self.database.retrieve(self.api.SEGMENTS_COLLECTION, [9035])[0].payload["text"])

    def test_legacy_ui_cutoff_does_not_discard_low_cosine_verified_passages(self):
        self.put(9047, "Children deserve protection.", video_id=9047)
        service = self.production_service()
        service.embed = lambda query: [.25, .96824584]
        with patch.object(self.api, "judge_passage_batch", side_effect=self.evidence), patch.object(
            self.api, "verify_passage_batch", side_effect=self.evidence,
        ):
            request = self.api.SearchRequest(query="rights", video_id=9047, min_score=.5)
            response = asyncio.run(self.api.search(request, True))
            strict = asyncio.run(self.api.search(request.model_copy(update={"min_score": .95}), True))
        self.assertEqual(1, len(response["results"]))
        self.assertLess(response["results"][0]["dense_score"], .35)
        self.assertTrue(response["results"][0]["independently_verified"])
        self.assertEqual([], strict["results"])

    def test_roman_urdu_speaker_is_resolved_before_language_understanding(self):
        query = "hafiz naeem nojawan ko namaz ki talqeen karte huwe"
        with patch.object(self.api, "understand_query", return_value={"semantic_query": "نوجوانوں کو نماز کی تلقین", "query_type": "general"}) as analyze:
            service = self.api.get_contextual_search()
            intent = service.understand(query)
        analyze.assert_called_once_with("nojawan ko namaz ki talqeen karte huwe")
        self.assertEqual("speaker_search", intent["query_type"])
        self.assertEqual("Hafiz Naeem Ur Rehman", intent["extracted_speaker"])

    def test_callback_sends_shared_api_key(self):
        from unittest.mock import Mock
        response = Mock(status_code=200, text='acknowledged')
        with patch.object(self.api, "API_KEY", "shared-secret"), patch.object(
            self.api.requests, "post", return_value=response,
        ) as post:
            self.api.send_webhook_with_retry("https://laravel.test/api/embeddings/callback", {"status": "processing"})
        self.assertEqual({"X-API-Key": "shared-secret"}, post.call_args.kwargs["headers"])

    def test_callbacks_complete_only_after_all_generation_batches_commit(self):
        redis = MemoryRedis()
        redis.set("index-active:9045", "generation-callback")
        base = {"video_id": 9045, "run_id": "generation-callback", "timestamp_unit": "seconds",
            "webhook_url": "https://laravel.test/api/embeddings/callback"}
        with patch.object(self.api, "redis_conn", redis), patch.object(
            self.api, "get_openai_embeddings_batch", return_value=[[1.0, 0.0]],
        ), patch.object(self.api, "send_webhook_with_retry") as notify:
            for batch in (1, 2):
                self.api.process_video_task({**base, "batch_info": {"batch_number": batch, "total_batches": 2},
                    "identification_segments": [{"text": "Children deserve protection.", "start": batch * 10,
                        "end": batch * 10 + 5, "segment_index": batch - 1}]})
            first, final = [call.args[1] for call in notify.call_args_list]
        self.assertEqual(("processing", 0), (first["status"], first["segments_embedded"]))
        self.assertEqual(("completed", 2), (final["status"], final["segments_embedded"]))
        self.assertEqual("generation-callback", final["run_id"])
        self.assertEqual(self.api.SEGMENTS_COLLECTION, final["collection"])
        count = self.database.count(self.api.SEGMENTS_COLLECTION, count_filter=models.Filter(
            must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=9045))],
        )).count
        self.assertEqual(2, count)

    def test_failed_callback_retains_generation_identity(self):
        with patch.object(self.api, "get_openai_embeddings_batch", side_effect=RuntimeError("provider unavailable")), patch.object(
            self.api, "send_webhook_with_retry",
        ) as notify:
            with self.assertRaises(RuntimeError):
                self.api.process_video_task({"video_id": 9046, "run_id": "failed-generation",
                    "webhook_url": "https://laravel.test/api/embeddings/callback",
                    "identification_segments": [{"text": "Source text.", "start": 0, "end": 5}]})
        self.assertEqual("failed-generation", notify.call_args.args[1]["run_id"])
        self.assertEqual("failed", notify.call_args.args[1]["status"])

    def test_invalid_cursor_returns_409_without_restarting_search(self):
        with self.assertRaises(self.api.HTTPException) as error:
            asyncio.run(self.api.search_incremental(self.api.IncrementalSearchRequest(query="IPP", cursor="bad-cursor"), True))
        self.assertEqual(409, error.exception.status_code)

    def test_chat_migration_sends_supported_parameters(self):
        captured = []

        def transport(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "chat-test", "object": "chat.completion", "created": 0,
                "model": "gpt-6-luna", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps({
                        "semantic_query": "children's rights", "detected_language": "english",
                    })}}],
            })

        client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(transport)))
        with patch.object(self.api, "openai_client", client):
            self.assertEqual("children's rights", self.api.understand_query("children's rights")["semantic_query"])
        self.assertEqual("gpt-6-luna", captured[0]["model"])
        self.assertEqual("none", captured[0]["reasoning_effort"])
        self.assertIn("max_completion_tokens", captured[0])
        self.assertNotIn("temperature", captured[0])
        self.assertNotIn("max_tokens", captured[0])
        client.close()

    def test_provider_failure_never_calls_an_incompatible_embedding_model(self):
        from unittest.mock import Mock
        provider = Mock()
        provider.embeddings.create.side_effect = RuntimeError("provider unavailable")
        tokenizer = types.SimpleNamespace(encode=lambda text, **kw: list(text), decode=lambda ids: "".join(ids))
        with patch.object(self.api, "openai_client", provider), patch.object(
            self.api.tiktoken, "get_encoding", return_value=tokenizer,
        ), patch.object(self.api, "get_fastembed_embedding") as fallback:
            with self.assertRaises(self.api.HTTPException) as error:
                self.api.get_openai_embedding("A meaningful query")
            self.assertEqual(503, error.exception.status_code)
            fallback.assert_not_called()

    def test_long_query_cache_keys_cannot_collide_at_character_500(self):
        from unittest.mock import Mock
        embed = Mock(side_effect=[[1.0, 0.0], [0.0, 1.0]])
        with patch.object(self.api, "get_openai_embedding", embed):
            self.assertNotEqual(self.api.get_cached_embedding("a" * 500 + " children"),
                                self.api.get_cached_embedding("a" * 500 + " workers"))

    def test_incremental_words_only_query_is_not_duplicated(self):
        from unittest.mock import Mock
        service = Mock()
        service.search.return_value = {"results": [], "ranking": {"engine": "qdrant_maxsim"}}
        with patch.object(self.api, "get_contextual_search", return_value=service):
            asyncio.run(self.api.search_incremental(self.api.IncrementalSearchRequest(words=["children", "rights"]), True))
        delegated = service.search.call_args.args[0]
        self.assertEqual("", delegated.query)
        self.assertEqual(["children", "rights"], delegated.words)

    def test_simple_search_does_not_require_openai(self):
        with patch.object(self.api, "openai_client", None):
            response = asyncio.run(self.api.search(self.api.SearchRequest(
                search_mode="simple", filter_type="video", video_id=9999,
            ), True))
        self.assertEqual(0, response["returned"])

    def test_default_search_and_incremental_cursor_use_native_ranking(self):
        source = self.api.SEGMENTS_COLLECTION
        for identity in [1, 2, 3]:
            self.database.upsert(source, [models.PointStruct(
                id=identity, vector=[1.0, 0.0], payload={
                    "video_id": identity, "text": "Every child deserves protection.",
                    "speaker": "Alice", "start_time": 0, "end_time": 5,
                    "processing_status": "completed", "approval_status": "approved", "is_archived": False,
                },
            )], wait=True)
        ranking = QdrantReranker(
            self.database, SearchSettings(dimension=2, context_seconds=0),
            lambda texts: [[[1.0, 0.0]] for text in texts],
        )
        service = ContextualSearch(self.database, source, lambda text: [1.0, 0.0],
                                   lambda text: {"semantic_query": text},
                                   lambda *args: True, lambda *args: True, ranking)
        with patch.object(self.api, "get_contextual_search", return_value=service):
            result = asyncio.run(self.api.search(self.api.SearchRequest(query="rights"), True))
            self.assertEqual("qdrant_maxsim", result["ranking"]["engine"])
            first = asyncio.run(self.api.search_incremental(
                self.api.IncrementalSearchRequest(query="rights", batch_size=1), True,
            ))
            second = asyncio.run(self.api.search_incremental(self.api.IncrementalSearchRequest(
                query="rights", batch_size=1, cursor=first["cursor"]["next"],
            ), True))
        self.assertTrue(second["metadata"]["cache_hit"])
        self.assertNotEqual(first["results"][0]["video_id"], second["results"][0]["video_id"])
        self.assertEqual("qdrant_maxsim", first["metadata"]["ranking"]["engine"])


if __name__ == "__main__":
    unittest.main()
