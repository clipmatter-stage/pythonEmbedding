"""Real Qdrant local integration tests with deterministic token vectors.

These verify ranking mechanics and access filters, not model semantic quality.
No API keys, model downloads, or external database are used.
"""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient, models
from contextual_search import (
    ContextualSearch, QdrantReranker, SearchInputError, SearchSettings,
    nearby_passage, source_passage, timestamp_url, cpu_thread_budget,
)


def request(**changes):
    values = dict(query="children's rights", words=[], word=None, top_k=20,
                  min_score=0.0, speaker=None, title=None, language=None,
                  video_id=None, time_range=None, filter_date=None,
                  filter_year=None, filter_month=None)
    return SimpleNamespace(**{**values, **changes})


class TokenEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.extend(texts)
        return [[[1.0, 0.0]] if "education and protection" in text or text in {
            "children's rights", "rights", "بچوں کے حقوق",
        } else [[0.0, 1.0]] for text in texts]


class ContextualSearchTest(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            "source", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        self.encoder = TokenEncoder()
        self.settings = SearchSettings(dimension=2, context_seconds=0)
        self.reranker = QdrantReranker(self.client, self.settings, self.encoder)
        self.service = ContextualSearch(
            self.client, "source", lambda query: [1.0, 0.0],
            lambda query: {"semantic_query": query},
            lambda name, value: name.casefold() in value.casefold(),
            lambda name, value: name.casefold() in value.casefold(), self.reranker,
        )
        self.put(1, "Children attended the rally.", title="Children's rights", vector=[1.0, 0.0])
        self.put(2, "Every child deserves education and protection.", vector=[0.8, 0.6])

    def tearDown(self):
        self.client.close()

    def put(self, identity, text, title="Speech", vector=None, **overrides):
        payload = dict(text=text, video_id=identity, speaker="Alice", start_time=10,
                       end_time=20, video_title=title, language="ur",
                       processing_status="completed", approval_status="approved",
                       is_archived=False)
        payload.update(overrides)
        self.client.upsert("source", [models.PointStruct(
            id=identity, vector=vector or [1.0, 0.0], payload=payload,
        )], wait=True)

    def test_default_first_page_is_reranked_and_title_cannot_override_meaning(self):
        response = self.service.search(request())
        self.assertEqual([2, 1], [r["id"] for r in response["results"]])
        self.assertEqual(2, response["ranking"]["candidates_reranked"])
        self.assertEqual(1.0, response["results"][0]["rerank_score"])
        self.assertNotIn("Children's rights", " ".join(self.encoder.calls))

    def test_literal_topic_can_enter_native_reranking_below_dense_candidate_pool(self):
        self.reranker.settings = replace(self.settings, candidate_limit=2)
        self.put(3, "نماز education and protection", vector=[-1.0, 0.0])
        self.service.understand = lambda query: {"semantic_query": query, "retrieval_terms": ["نماز"]}
        response = self.service.search(request())
        self.assertEqual(3, response["results"][0]["id"])
        self.assertIsNone(response["results"][0]["dense_score"])
        self.assertEqual(1, response["ranking"]["lexical_candidates"])
        self.assertEqual(2, response["ranking"]["candidates_reranked"])

    def test_single_word_is_also_reranked(self):
        response = self.service.search(request(query="rights"))
        self.assertEqual(2, response["results"][0]["id"])

    def test_cross_language_does_not_auto_filter_urdu(self):
        self.assertEqual(2, self.service.search(request())["returned"])
        self.assertEqual(0, self.service.search(request(language="en"))["returned"])

    def test_ineligible_sources_never_enter_rerank_cache(self):
        self.put(3, "education and protection", approval_status="pending")
        self.put(4, "education and protection", is_archived=True)
        self.put(5, "education and protection", processing_status="processing")
        response = self.service.search(request())
        self.assertEqual({1, 2}, {r["id"] for r in response["results"]})

    def test_explicit_speaker_title_and_video_filters(self):
        self.put(3, "education and protection", speaker="Bob")
        response = self.service.search(request(speaker="Bob"))
        self.assertEqual([3], [r["id"] for r in response["results"]])
        self.assertEqual([1], [r["id"] for r in self.service.search(request(title="rights"))["results"]])
        self.assertEqual([2], [r["id"] for r in self.service.search(request(video_id=2))["results"]])

    def test_inferred_spoken_by_constraint_is_enforced(self):
        self.service.understand = lambda query: {
            "semantic_query": "children's rights", "query_type": "speaker_search",
            "extracted_speaker": "Nobody",
        }
        self.assertEqual(0, self.service.search(request())["returned"])

    def test_topic_mentions_do_not_become_speaker_filters(self):
        self.service.understand = lambda query: {
            "semantic_query": "children's rights", "query_type": "topic_search",
            "extracted_speaker": "Nobody",
        }
        self.assertEqual(2, self.service.search(request())["returned"])

    def test_translation_candidates_deduplicate_before_reranking(self):
        self.service.understand = lambda query: {
            "semantic_query": query, "semantic_query_translated": "بچوں کے حقوق",
        }
        self.assertEqual(2, self.service.search(request())["returned"])
        self.assertEqual(3, len(self.encoder.calls))  # two documents, one ranking query

    def test_cached_vectors_reused_and_deleted_source_cannot_reappear(self):
        self.service.search(request())
        previous_calls = len(self.encoder.calls)
        self.service.search(request())
        self.assertEqual(previous_calls + 1, len(self.encoder.calls))
        self.client.delete("source", models.PointIdsList(points=[2]), wait=True)
        response = self.service.search(request())
        self.assertEqual([1], [r["id"] for r in response["results"]])

    def test_updated_passage_gets_fresh_vectors(self):
        self.service.search(request())
        self.put(1, "education and protection")
        response = self.service.search(request())
        self.assertEqual(1.0, next(r for r in response["results"] if r["id"] == 1)["score"])

    def test_configured_threshold_can_return_no_results_without_padding(self):
        self.reranker.settings = replace(self.settings, minimum_rerank_score=1.1)
        self.assertEqual([], self.service.search(request())["results"])

    def test_model_failure_does_not_silently_return_unreranked_results(self):
        def broken(texts):
            raise RuntimeError("model unavailable")
        self.reranker.encoder = broken
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            self.service.search(request())

    def test_wrong_collection_schema_is_rejected(self):
        self.client.create_collection(self.settings.rerank_collection,
                                     vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            self.service.search(request())

    def test_video_scope_and_time_range(self):
        self.assertEqual([2], [r["id"] for r in self.service.search(request(), [2])["results"]])
        self.assertEqual([], self.service.search(request(time_range={"start": 30}))["results"])

    def test_dates_are_applied_in_qdrant(self):
        self.put(1, "rally", video_created_at="2026-10-16T12:00:00Z")
        self.put(2, "education and protection", video_created_at="2026-09-16T12:00:00Z")
        results = self.service.search(request(filter_year=2026, filter_month=10))["results"]
        self.assertEqual([1], [r["id"] for r in results])
        with self.assertRaises(SearchInputError):
            self.service.search(request(filter_month=10))

    def test_full_passage_is_given_to_encoder_not_200_character_prefix(self):
        text = "An introduction. " * 30 + "education and protection"
        self.put(1, text)
        self.service.search(request())
        self.assertIn(text, self.encoder.calls)

    def test_videos_remain_contiguous_for_pagination(self):
        self.put(3, "education and protection", video_id=1)
        results = self.service.search(request())["results"]
        ids = [r["video_id"] for r in results]
        self.assertEqual([2, 1, 1], ids)

    def test_mean_maxsim_is_normalized_by_query_token_count(self):
        self.reranker.encoder = lambda texts: [[[1.0, 0.0], [1.0, 0.0]] for text in texts]
        response = self.service.search(request())
        self.assertEqual(1.0, response["results"][0]["score"])

    def test_live_context_respects_source_eligibility(self):
        self.reranker.settings = replace(self.settings, context_seconds=20)
        self.put(1, "That protects them.", start_time=20, end_time=30)
        self.put(3, "education and protection", video_id=1, start_time=10, end_time=20)
        self.put(4, "An unapproved assertion", video_id=1, start_time=19, end_time=20,
                 approval_status="pending")
        results = self.service.search(request(video_id=1))["results"]
        focus = next(r for r in results if r["id"] == 1)
        self.assertIn("education and protection", focus["context_text"])
        self.assertNotIn("unapproved", focus["context_text"])
        self.assertEqual("That protects them.", focus["text"])
        self.assertEqual(20, focus["start_time"])

    def test_validator_receives_adjacent_source_evidence_and_real_seek_times(self):
        self.reranker.settings = replace(self.settings, context_seconds=20)
        self.put(1, "They should pray.", video_id=1, start_time=20, end_time=25)
        self.put(3, "Young people, listen.", video_id=1, start_time=15, end_time=20)
        captured = []
        def validate(query, candidates, speaker):
            captured.extend(candidates)
            row = next(r for r in candidates if r["id"] == 1)
            return [{**row, "score": .9, "match_types": ["validated_passage"]}], {}
        self.service.validator = validate
        result = self.service.search(request(video_id=1))["results"][0]
        self.assertEqual("Young people, listen. They should pray.", result["text"])
        self.assertEqual([3, 1], result["segment_ids"])
        self.assertEqual(15, result["start_time"])
        self.assertEqual(25, result["end_time"])
        self.assertEqual("Young people, listen.", result["source_utterances"][0]["text"])
        self.assertNotIn("Preceding context:", result["text"])
        self.assertNotIn("_source_passage", result)

    def test_known_speaker_prefilter_retrieves_within_the_speaker_subset(self):
        self.service.speaker_values = lambda name: [name]
        self.put(3, "education and protection", speaker="Bob")
        response = self.service.search(request(speaker="Bob"))
        self.assertEqual([3], [r["id"] for r in response["results"]])

    def test_speaker_prefilter_uses_existing_text_index_without_keyword_requirement(self):
        from unittest.mock import patch
        self.service.speaker_values = lambda name: [name]
        self.service.speaker_index_types = {"speaker": "text"}
        self.put(3, "education and protection", speaker="Bob")
        with patch.object(self.client, "query_points", wraps=self.client.query_points) as queries:
            response = self.service.search(request(speaker="Bob"))
        self.assertEqual([3], [r["id"] for r in response["results"]])
        condition = queries.call_args_list[0].kwargs["query_filter"].must[-1].should[0]
        self.assertIsInstance(condition.match, models.MatchText)

    def test_words_contribute_once_to_semantic_meaning(self):
        captured = []
        self.service.understand = lambda query: captured.append(query) or {"semantic_query": query}
        self.service.search(request(query="", words=["children", "rights"]))
        self.assertEqual(["children rights"], captured)

    def test_context_fetches_are_grouped_once_per_video(self):
        from unittest.mock import patch
        self.reranker.settings = replace(self.settings, context_seconds=20)
        self.put(3, "education and protection", video_id=1, start_time=21, end_time=25)
        with patch.object(self.client, "scroll", wraps=self.client.scroll) as reads:
            self.service.search(request(video_id=1))
        self.assertEqual(1, reads.call_count)

    def test_context_read_budget_is_reported_instead_of_unbounded_scans(self):
        self.reranker.settings = replace(self.settings, context_seconds=20)
        for identity in range(3, 20):
            self.put(identity, "education and protection")
        response = self.service.search(request())
        self.assertIn("context_retrieval_budget_reached", response["ranking"]["retrieval_warnings"])


class ContextWindowTest(unittest.TestCase):
    def test_inference_thread_budget_respects_cpu_quota_instead_of_host_core_count(self):
        from unittest.mock import patch
        with patch("contextual_search.os.cpu_count", return_value=48), patch(
            "contextual_search.Path.read_text", return_value="800000 100000",
        ), patch.dict("os.environ", {"QDRANT_RERANK_CPU_THREADS": "48"}):
            self.assertEqual(8, cpu_thread_budget())
        with patch("contextual_search.os.cpu_count", return_value=48), patch(
            "contextual_search.Path.read_text", return_value="50000 100000",
        ):
            self.assertEqual(1, cpu_thread_budget())


    def test_source_passage_does_not_join_other_speakers_or_noncontiguous_turns(self):
        focus = dict(id=1, video_id=1, speaker="Alice", start_time=20, end_time=25, text="Pray regularly.")
        wrong = {**focus, "id": 2, "speaker": "Bob", "start_time": 15, "end_time": 20, "text": "Young people."}
        distant = {**focus, "id": 3, "start_time": 1, "end_time": 5, "text": "Young people."}
        self.assertEqual({}, source_passage(focus, [wrong, distant], 20))
        unknown = {**focus, "speaker": "Unknown Speaker"}
        self.assertEqual({}, source_passage(unknown, [{**wrong, "speaker": "Unknown Speaker"}], 20))


    def test_context_preserves_focus_and_excludes_other_speakers_videos_and_distant_text(self):
        focus = dict(video_id=1, speaker="Alice", start_time=20, end_time=30, text="That protects them.")
        before = dict(video_id=1, speaker="Alice", start_time=10, end_time=20, text="We discussed children's rights.")
        wrong = [dict(before, speaker="Bob", text="Bob's claim"),
                 dict(before, video_id=2, text="Another video"),
                 dict(before, start_time=-90, end_time=-80, text="Distant claim")]
        passage = nearby_passage(focus, [before] + wrong)
        self.assertTrue(passage.startswith(focus["text"]))
        self.assertIn(before["text"], passage)
        self.assertNotIn("claim", passage)
        self.assertNotIn("Another video", passage)

    def test_unknown_speaker_does_not_inherit_nearby_statements(self):
        focus = dict(video_id=1, speaker="UNKNOWN", text="Unattributed fragment")
        self.assertEqual(focus["text"], nearby_passage(focus, []))

    def test_timestamp_retains_existing_query_parameters(self):
        self.assertEqual("https://youtube.com/watch?v=123&t=20",
                         timestamp_url("https://youtube.com/watch?v=123&t=1", 20))


if __name__ == "__main__":
    unittest.main()
