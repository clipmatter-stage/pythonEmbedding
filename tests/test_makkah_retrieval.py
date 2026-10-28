"""Makkah-agreement retrieval: later sentence in the same speech, phonetic Roman.

Video 1087 stands in for the Peshawar clip. The best dense hit mentions Makkah
and does not answer a request that also names Iran and Turkey. That sentence
is a later passage of the same video. Exact alias lookup stays unchanged.
"""
import os
import pathlib
import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from fast_search import FastSearchSettings, FastSemanticSearch
from grounded_search import GroundedSearch, validation_rows
from grounded_validation import candidate_id
from test_fast_search import request
from transcript_search_contract import CONFIG, match_spans, normalize, query_variants


LONG = "مکہ معاہدے میں ایران اور ترکی کو بھی شامل کیا جائے"
BEST = "مکہ میں ایک عمومی بات ہوئی"
EVIDENCE = "مکہ معاہدے میں ایران اور ترکی کو بھی شامل کرنے کی بات ہوئی"
BRIDGE = "اس دفاعی تعاون پر تفصیل سے بات ہوئی"
PARTIES = "ایران اور ترکی کو بھی اس معاہدے میں شامل کیا جائے"


def _accepts(query, text):
    folded = query.casefold()
    # The phonetic token has to be rewritten before the judge. A literal
    # "makh" is not evidence that the passage is about Makkah.
    if any(token in folded.split() for token in ("makh", "makka", "makkha", "muahida", "mohaida", "muaahida")):
        return False
    parties = "ایران" in query or "iran" in folded
    agreement = "معاہد" in text or "ایگریمنٹ" in text
    makkah = "مکہ" in text or "مکۃ" in text
    if parties:
        return agreement and makkah and "ایران" in text and "ترکی" in text
    topic = "معاہد" in query or "mohida" in folded or "makkah" in folded or "مکہ" in query
    return bool(topic and agreement and makkah)


class SelectionTests(unittest.TestCase):
    def test_spare_slots_keep_the_next_passage_of_a_selected_video(self):
        pool = [
            {"id": "best", "video_id": 1087, "score": 0.9, "retrieval_score": 0.9},
            {"id": "later", "video_id": 1087, "score": 0.2, "retrieval_score": 0.2},
        ]
        for index in range(15):
            pool.append({"id": f"d{index}", "video_id": 2000 + index, "score": 0.5, "retrieval_score": 0.5})
        chosen = validation_rows(pool, 8, 16)
        self.assertIn("later", {row["id"] for row in chosen})
        self.assertLess(
            [row["id"] for row in chosen].index("best"),
            [row["id"] for row in chosen].index("later"),
        )


class MakkahRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            "source", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        self.cache = {}
        self.judged = []
        self.base = FastSemanticSearch(
            self.client, "source", lambda query: [1.0, 0.0],
            lambda query: {"semantic_query": query, "understanding_status": "disabled"},
            lambda query, value: True, lambda query, value: True,
            settings=FastSearchSettings(candidate_limit=100),
        )
        self.engine = GroundedSearch(
            self.base, self._judge, "fixture-model", lambda: "epoch",
            self.cache.get, lambda key, value, ttl: self.cache.update({key: value}),
            ("same-model", 2),
        )
        self._put(7101, BEST, 1087, [1.0, 0.0], 10, 20)
        self._put(7103, EVIDENCE, 1087, [0.3, 0.95], 31, 77)
        for index in range(15):
            self._put(8000 + index, "غیر متعلق سیاسی تقریر", 3000 + index, [0.9, 0.2], 1, 5)

    def tearDown(self):
        self.client.close()

    def _put(self, identity, text, video_id, vector, start, end):
        self.client.upsert("source", [models.PointStruct(id=identity, vector=vector, payload={
            "video_id": video_id, "text": text, "speaker": "Unverified Label",
            "transcript_speaker": "A", "start_time": start, "end_time": end,
            "processing_status": "completed", "approval_status": "approved",
            "is_archived": False,
        })])

    def _judge(self, query, rows, timeout):
        self.judged.append(query)
        judgments = []
        for row in rows:
            yes = _accepts(query, row["text"])
            quote = "مکہ معاہدے میں ایران اور ترکی" if yes else ""
            judgments.append({
                "candidate_id": candidate_id(row),
                "score": 0.9 if yes else 0.2,
                "relevant": yes, "complete": yes,
                "roles_supported": yes, "relationship_supported": yes,
                "quotes": [quote] if yes else [],
                "reason": "relevant" if yes else "related_only",
            })
        return {"judgments": judgments}

    def _search(self, query):
        self.judged.clear()
        self.cache.clear()
        with patch.dict(os.environ, {"GROUNDING_INITIAL_CANDIDATES": "8", "GROUNDING_MAX_CANDIDATES": "16"}):
            return self.engine.search(request(query=query, min_score=0.5, top_k=5))

    def _clip(self, result):
        matches = [row for row in result["results"] if row["video_id"] == 1087]
        self.assertTrue(matches, result["metadata"]["passage_validation"]["status"])
        return matches[0]

    def test_short_makkah_agreement_still_returns_the_clip(self):
        result = self._search("مکہ معاہدہ")
        clip = self._clip(result)
        self.assertEqual(31, clip["start_time"])
        self.assertEqual(77, clip["end_time"])
        self.assertIn("معاہد", clip["text"])
        self.assertIsNone(result["speaker_filter"])
        self.assertIsNone(result["metadata"]["passage_validation"]["speaker_constraint"])

    def test_iran_turkey_request_uses_the_later_passage_not_a_speaker_constraint(self):
        result = self._search(LONG)
        clip = self._clip(result)
        self.assertEqual(31, clip["start_time"])
        self.assertEqual(77, clip["end_time"])
        self.assertIn("ایران", clip["text"])
        self.assertIn("ترکی", clip["text"])
        self.assertTrue(self.judged)
        self.assertEqual({LONG}, set(self.judged))
        validation = result["metadata"]["passage_validation"]
        self.assertIsNone(result["speaker_filter"])
        self.assertIsNone(validation["speaker_constraint"])
        self.assertEqual(LONG, validation["grounding_query"])
        self.assertEqual(LONG, validation["original_query"])

    def test_phonetic_roman_reaches_the_same_agreement_clip(self):
        for query in ("makh mohida", "makka mohida", "makh muahida", "Makh Mohaida", "makkah mohida"):
            with self.subTest(query=query):
                result = self._search(query)
                clip = self._clip(result)
                self.assertEqual(1087, clip["video_id"])
                self.assertEqual(31, clip["start_time"])
                self.assertEqual(77, clip["end_time"])
                self.assertEqual("makkah mohida", self.judged[0])
                self.assertEqual(query, result["query"])
                self.assertIsNone(result["metadata"]["passage_validation"]["speaker_constraint"])

    def test_exact_alias_contract_does_not_grow_a_phrase_dictionary(self):
        self.assertEqual([normalize("makh mohida")], query_variants("makh mohida", "alias"))
        self.assertNotIn("makh", [item for values in CONFIG["aliases"].values() for item in values])
        self.assertFalse(match_spans(EVIDENCE, "makh mohida", "alias"))
        self.assertFalse(match_spans(EVIDENCE, "makh mohida", "literal"))
        self.assertFalse(match_spans(EVIDENCE, LONG, "literal"))
        source = pathlib.Path(__file__).resolve().parents[1].joinpath("asr_search.py").read_text()
        self.assertNotIn("makh", source)
        self.assertNotIn("ایران اور ترکی", source)
        self.assertNotIn(LONG, source)


GAZA = "غزہ کی جنگ میں اسرائیل کو ہتھیار پہنچائے گئے"
GAZA_BEST = "غزہ میں صورتحال پر بات ہوئی"
GAZA_LATER = "غزہ کی جنگ کے دوران اسرائیل کو ہتھیار پہنچائے گئے"


class GeneralConjunctionTests(unittest.TestCase):
    """Same selection path on a different topic. Not a Makkah fixture."""

    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            "source", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        self.cache = {}
        self.judged = []
        self.base = FastSemanticSearch(
            self.client, "source", lambda query: [1.0, 0.0],
            lambda query: {"semantic_query": query, "understanding_status": "disabled"},
            lambda query, value: True, lambda query, value: True,
            settings=FastSearchSettings(candidate_limit=100),
        )
        self.engine = GroundedSearch(
            self.base, self._judge, "fixture-model", lambda: "epoch",
            self.cache.get, lambda key, value, ttl: self.cache.update({key: value}),
            ("same-model", 2),
        )
        self._put(7201, GAZA_BEST, 880, [1.0, 0.0], 10, 20)
        self._put(7203, GAZA_LATER, 880, [0.3, 0.95], 31, 77)
        for index in range(15):
            self._put(8200 + index, "غیر متعلق سیاسی تقریر", 6000 + index, [0.9, 0.2], 1, 5)

    def tearDown(self):
        self.client.close()

    def _put(self, identity, text, video_id, vector, start, end):
        self.client.upsert("source", [models.PointStruct(id=identity, vector=vector, payload={
            "video_id": video_id, "text": text, "speaker": "Unverified Label",
            "transcript_speaker": "A", "start_time": start, "end_time": end,
            "processing_status": "completed", "approval_status": "approved",
            "is_archived": False,
        })])

    def _judge(self, query, rows, timeout):
        self.judged.append(query)
        folded = query.casefold()
        unresolved = any(token in folded.split() for token in ("gazza", "ghazza", "ghazah"))
        judgments = []
        for row in rows:
            yes = (not unresolved and ("gaza" in folded or "غزہ" in query or "ہتھیار" in query)
                   and "ہتھیار" in row["text"] and "غزہ" in row["text"] and "اسرائیل" in row["text"])
            judgments.append({
                "candidate_id": candidate_id(row),
                "score": 0.9 if yes else 0.2,
                "relevant": yes, "complete": yes,
                "roles_supported": yes, "relationship_supported": yes,
                "quotes": ["اسرائیل کو ہتھیار"] if yes else [],
                "reason": "relevant" if yes else "related_only",
            })
        return {"judgments": judgments}

    def _search(self, query):
        self.judged.clear()
        self.cache.clear()
        with patch.dict(os.environ, {"GROUNDING_INITIAL_CANDIDATES": "8", "GROUNDING_MAX_CANDIDATES": "16"}):
            return self.engine.search(request(query=query, min_score=0.5, top_k=5))

    def test_multi_clause_gaza_query_uses_the_later_passage(self):
        result = self._search(GAZA)
        clips = [row for row in result["results"] if row["video_id"] == 880]
        self.assertTrue(clips)
        self.assertEqual(31, clips[0]["start_time"])
        self.assertEqual(77, clips[0]["end_time"])
        self.assertIn("ہتھیار", clips[0]["text"])
        self.assertEqual({GAZA}, set(self.judged))
        self.assertIsNone(result["speaker_filter"])
        self.assertIsNone(result["metadata"]["passage_validation"]["speaker_constraint"])
        self.assertNotIn(1087, {row["video_id"] for row in result["results"]})

    def test_near_miss_roman_on_another_family_uses_the_same_path(self):
        result = self._search("gazza hathyar")
        clips = [row for row in result["results"] if row["video_id"] == 880]
        self.assertTrue(clips)
        self.assertEqual(31, clips[0]["start_time"])
        self.assertEqual(77, clips[0]["end_time"])
        self.assertEqual("gaza hathyar", self.judged[0])
        self.assertEqual("gazza hathyar", result["query"])
        self.assertIsNone(result["metadata"]["passage_validation"]["speaker_constraint"])


class ContinuousSpeechTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            "source", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        self.base = FastSemanticSearch(
            self.client, "source", lambda query: [1.0, 0.0],
            lambda query: {"semantic_query": query, "understanding_status": "disabled"},
            lambda query, value: True, lambda query, value: True,
            settings=FastSearchSettings(candidate_limit=100),
        )
        self.engine = GroundedSearch(
            self.base, self._judge, "fixture-model", lambda: "epoch",
            lambda key: None, lambda key, value, ttl: None, ("same-model", 2),
        )
        # A short focus at the start of a continuous speech. The Iran sentence
        # begins after the old ±20s window and inside the 60s join span.
        self._put(1, "مکہ میں دفاعی ایگریمنٹ طے پایا", [1.0, 0.0], 31, 36)
        self._put(2, BRIDGE, [0.0, 1.0], 36, 56)
        self._put(3, PARTIES, [0.0, 1.0], 57, 77)
        for index in range(15):
            self._put(100 + index, "غیر متعلق", [0.85, 0.3], 1, 4, video_id=4000 + index)

    def tearDown(self):
        self.client.close()

    def _put(self, identity, text, vector, start, end, video_id=1087):
        self.client.upsert("source", [models.PointStruct(id=identity, vector=vector, payload={
            "video_id": video_id, "text": text, "speaker": "Unverified Label",
            "transcript_speaker": "A", "start_time": start, "end_time": end,
            "processing_status": "completed", "approval_status": "approved",
            "is_archived": False,
        })])

    def _judge(self, query, rows, timeout):
        judgments = []
        for row in rows:
            yes = "ایران" in row["text"] and "ترکی" in row["text"] and ("ایگریمنٹ" in row["text"] or "معاہد" in row["text"])
            judgments.append({
                "candidate_id": candidate_id(row),
                "score": 0.9 if yes else 0.2,
                "relevant": yes, "complete": yes,
                "roles_supported": yes, "relationship_supported": yes,
                "quotes": ["ایران اور ترکی کو بھی اس معاہدے میں شامل"] if yes else [],
                "reason": "relevant" if yes else "related_only",
            })
        return {"judgments": judgments}

    def test_continuous_speech_joins_the_later_sentence_into_the_31_to_77_clip(self):
        with patch.dict(os.environ, {"GROUNDING_INITIAL_CANDIDATES": "8", "GROUNDING_MAX_CANDIDATES": "16"}):
            result = self.engine.search(request(query=LONG, min_score=0.5, top_k=5))
        clips = [row for row in result["results"] if row["video_id"] == 1087 and row["start_time"] <= 36 and row["end_time"] >= 70]
        self.assertTrue(clips)
        self.assertIn("ایران", clips[0]["text"])
        self.assertIn("ایگریمنٹ", clips[0]["text"])
        self.assertIsNone(result["metadata"]["passage_validation"]["speaker_constraint"])


if __name__ == "__main__":
    unittest.main()
