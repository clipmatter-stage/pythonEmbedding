"""Endpoint integration without external services or model downloads."""
import asyncio
import importlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contextual_search import ContextualSearch, QdrantReranker, SearchSettings


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
        self.api.search_result_cache.clear()
        self.api.embedding_cache.clear()

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
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                self.api.get_openai_embedding("A meaningful query")
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
