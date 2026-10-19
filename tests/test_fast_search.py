"""Exercise actual Qdrant dense/sparse fusion and additive vector migration."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qdrant_client import QdrantClient, models
from backfill_sparse_search import backfill
from fast_search import FastSemanticSearch, FastSearchSettings, SPARSE_NAME, lexical_vector, with_sparse_vectors


def request(**changes):
    return SimpleNamespace(**{**dict(query="نماز", words=[], word=None, top_k=20,
        min_score=.5, speaker=None, title=None, language=None, video_id=None,
        time_range=None, filter_date=None, filter_year=None, filter_month=None), **changes})


class FastSearchTest(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection("source", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        self.embed = Mock(return_value=[1.0, 0.0])
        self.service = FastSemanticSearch(self.client, "source", self.embed,
            lambda query: {"semantic_query": query, "understanding_status": "disabled"},
            lambda query, value: query.casefold() in value.casefold(),
            lambda query, value: query.casefold() in value.casefold())
        self.put(1, "education", [1.0, 0.0])
        self.put(2, "نماز کی تلقین", [-1.0, 0.0], speaker="Bob")

    def tearDown(self):
        self.client.close()

    def put(self, identity, text, vector, **changes):
        payload = dict(video_id=identity, text=text, speaker="Alice", start_time=1, end_time=3,
                       processing_status="completed", approval_status="approved", is_archived=False,
                       video_title="Speech", language="ur")
        self.client.upsert("source", [models.PointStruct(id=identity, vector=vector, payload={**payload, **changes})])

    def test_dense_is_one_query_embedding_without_reranker_or_validator(self):
        with patch.object(self.client, "query_points", wraps=self.client.query_points) as lookup:
            response = self.service.search(request())
        self.assertEqual(1, lookup.call_count)
        self.assertEqual([1], [r["id"] for r in response["results"]])
        self.embed.assert_called_once_with("نماز")
        self.assertEqual("qdrant_dense", response["ranking"]["engine"])
        self.assertIsNone(response["metadata"]["passage_validation"])
        self.assertIsNone(response["results"][0]["llm_relevance_score"])
        self.assertIn("legacy_min_score_not_applied_to_retrieval_scores", response["ranking"]["retrieval_warnings"])

    def test_native_hybrid_recovers_literal_passage_excluded_by_dense_floor(self):
        backfill(self.client, "source", apply=True)
        self.service.sparse_enabled = True
        self.service.settings = FastSearchSettings(minimum_dense_score=.1)
        response = self.service.search(request(speaker="Bob"))
        self.assertEqual([2], [r["id"] for r in response["results"]])
        self.assertEqual("qdrant_hybrid_rrf", response["ranking"]["engine"])
        self.assertIsNone(response["results"][0]["dense_score"])
        self.assertEqual("rrf", response["results"][0]["score_type"])

    def test_ineligible_sparse_hits_and_other_video_ids_cannot_escape_filters(self):
        self.put(3, "نماز", [1.0, 0.0], approval_status="pending")
        self.put(4, "نماز", [1.0, 0.0], is_archived=True)
        self.put(5, "نماز", [1.0, 0.0], processing_status="processing")
        backfill(self.client, "source", apply=True)
        self.service.sparse_enabled = True
        self.assertEqual({1, 2}, {r["id"] for r in self.service.search(request())["results"]})
        self.assertEqual([2], [r["id"] for r in self.service.search(request(), [2])["results"]])
        self.assertEqual([], self.service.search(request(language="en"))["results"])

    def test_backfill_dry_run_does_not_change_schema_and_apply_preserves_dense(self):
        before = self.client.retrieve("source", [1, 2], with_vectors=True)
        backfill(self.client, "source")
        self.assertNotIn(SPARSE_NAME, self.client.get_collection("source").config.params.sparse_vectors or {})
        backfill(self.client, "source", apply=True, batch_size=1)
        after = self.client.retrieve("source", [1, 2], with_vectors=True)
        for original, migrated in zip(before, after):
            self.assertEqual(original.vector, migrated.vector[""])
            self.assertEqual(original.payload, migrated.payload)
            self.assertIn(SPARSE_NAME, migrated.vector)
        backfill(self.client, "source", apply=True)
        self.assertEqual(2, self.client.count("source").count)

    def test_new_indexing_keeps_both_vector_types(self):
        backfill(self.client, "source", apply=True)
        point = models.PointStruct(id=6, vector=[1., 0.], payload={"text": "نماز"})
        self.client.upsert("source", with_sparse_vectors([point], True))
        saved = self.client.retrieve("source", [6], with_vectors=True)[0]
        self.assertEqual([1., 0.], saved.vector[""])
        self.assertEqual(lexical_vector("نماز"), saved.vector[SPARSE_NAME])

    def test_sparse_normalizes_arabic_urdu_forms_and_is_stable(self):
        self.assertEqual(lexical_vector("عربي كتاب"), lexical_vector("عربی کتاب"))
        self.assertEqual([], lexical_vector("!!!").indices)

    def test_service_keeps_every_legacy_route(self):
        root = Path(__file__).resolve().parents[1]
        def routes(filename):
            tree = ast.parse((root / filename).read_text())
            return {(d.func.attr, d.args[0].value) for fn in tree.body
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                    for d in fn.decorator_list if isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Attribute) and isinstance(d.func.value, ast.Name)
                    and d.func.value.id == "app" and d.args and isinstance(d.args[0], ast.Constant)}
        self.assertTrue(routes("embeddings_test.py") <= routes("embeddings_service.py"))


if __name__ == "__main__":
    unittest.main()
