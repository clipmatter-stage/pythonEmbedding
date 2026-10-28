"""Speaker-prefixed grounded search keeps the topic and the speaker's clips.

Topic-only paraphrases still use spoken-intent retrieval. A known speaker
prefix must not wipe those hits, and speech boilerplate such as "bayan karte
huye" must return that speaker's turns. Exact alias contracts stay unchanged.
"""
import os
import sys
import types
import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient, models

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("EMBEDDING_DIMENSION", "2")
os.environ.setdefault("ENSURE_INDEXES_ON_STARTUP", "false")
os.environ.setdefault("API_KEY", "")
os.environ.setdefault("SEARCH_LLM_UNDERSTANDING", "false")

from fast_search import FastSearchSettings, FastSemanticSearch
from grounded_search import GroundedSearch
from grounded_validation import candidate_id
from speaker_constraint import (
    constraint_names, is_about_person, label_matches, passes_constraint,
    refine_speaker_query, same_name, speech_wrapper_only,
)
from speaker_identity import VERSION, view
from test_fast_search import request
from transcript_search_contract import match_spans, query_variants


PRAYER = "نوجوانوں کو نماز پڑھنے کی تلقین کی اور مسجد جانے کی دعوت دی"
ECONOMY = "بجلی کے بل اور مہنگائی پر لمبی بات ہوئی آج کی نشست میں"
SPEECH = "آج میں آپ سے چند اہم باتیں کرنا چاہتا ہوں جو معاشرے کے لیے مفید ہیں"
TOPIC = "nojawano ko namaz ki talqeen"
SPEAKER_TOPIC = "Hafiz Naeem nojawano ko namaz ki talqeen"
SPEAKER_ONLY = "Hafiz Naeem bayan karte huye"


def _embed(text):
    folded = (text or "").casefold()
    if any(token in folded for token in ("namaz", "talqeen", "نماز", "تلقین", "nojawan", "naujawan", "نوجوان", "encourag", "pray")):
        return [1.0, 0.0]
    if any(token in folded for token in ("bijli", "bill", "بجلی", "مہنگائی")):
        return [-1.0, 0.0]
    return [0.0, -1.0]


def _plan(query):
    if any(token in (query or "").casefold() for token in ("namaz", "talqeen", "نماز", "تلقین", "nojawan", "نوجوان", "encourag", "pray")):
        return {
            "queries": ["نماز کی تلقین", "nojawano ko namaz ki talqeen", "encouraging youth to pray"],
            "spoken_queries": ["نوجوانوں کو نماز کی تلقین", "nojawano ko namaz ki talqeen", "encourage the youth to pray"],
            "retrieval_terms": ["نماز", "talqeen"],
        }
    return {"queries": [query], "spoken_queries": [], "retrieval_terms": []}


class SpeakerConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub = types.ModuleType("fastembed")
        stub.TextEmbedding = type("TextEmbedding", (), {})
        database = QdrantClient(":memory:")
        with patch.dict(sys.modules, {"fastembed": stub}), patch("qdrant_client.QdrantClient", return_value=database):
            import embeddings_service as api
        cls.api = api
        cls.names = constraint_names("hafiz_naeem", api.PERSON_ALIASES)
        cls.canonical = api.PERSON_ALIASES["hafiz_naeem"]["canonical"]

    def test_speech_boilerplate_is_not_a_topic_and_commentary_stays_put(self):
        self.assertTrue(speech_wrapper_only("bayan karte huye"))
        self.assertTrue(speech_wrapper_only("ki taqreer"))
        self.assertTrue(speech_wrapper_only("کا بیان"))
        self.assertFalse(speech_wrapper_only("nojawano ko namaz ki talqeen"))
        self.assertTrue(is_about_person("ki talqeen par tanqeed"))
        bare = self.api.decompose_semantic_query("Hafiz Naeem", self.canonical, self.names)
        self.assertFalse(refine_speaker_query("Hafiz Naeem", bare, self.canonical, self.names)["decomposed"])
        commentary = self.api.decompose_semantic_query(
            "hafiz naeem ki talqeen par tanqeed", self.canonical, self.names)
        refined = refine_speaker_query(
            "hafiz naeem ki talqeen par tanqeed", commentary, self.canonical, self.names)
        self.assertFalse(refined["decomposed"])
        wrapped = refine_speaker_query(
            "show me Hafiz Naeem bayan", commentary, self.canonical, self.names)
        self.assertEqual("speaker_only", wrapped["intent"])

    def test_labels_accept_curated_forms_and_reject_other_people(self):
        self.assertTrue(same_name("Hafez Naeem", "hafiz naeem"))
        self.assertTrue(same_name("حافظ نعیم", "حافظ نعیم الرحمن"))
        self.assertFalse(same_name("Hafiz Naeemullah", "hafiz naeem"))
        self.assertFalse(same_name("Hafiz Ali", "hafiz naeem"))
        self.assertTrue(label_matches("Hafiz Naeem", self.names))
        self.assertTrue(label_matches("حافظ نعیم الرحمن", self.names))
        self.assertTrue(label_matches("Hafez Naeem", self.names))
        self.assertFalse(label_matches("Other Leader", self.names))
        self.assertFalse(label_matches("Unknown Speaker", self.names))
        self.assertFalse(label_matches("Hafiz Naeemullah", self.names))
        verified = {
            "speaker": "SPEAKER_01", "identity_version": VERSION, "identity_status": "verified",
            "identity_source": "manual", "identity_key": "hafiz_naeem_ur_rehman",
            "identity_review": {"review_id": "r", "reviewer_id": "u", "reviewed_at": "2026-10-20",
                                "evidence_sha256": "a" * 64},
        }
        self.assertTrue(passes_constraint(view(verified), self.canonical, self.names))
        labeled = view({"speaker": "Hafiz Naeem"})
        self.assertEqual("Unknown Speaker", labeled["speaker"])
        self.assertEqual("Hafiz Naeem", labeled["source_speaker_label"])
        self.assertTrue(passes_constraint(labeled, self.canonical, self.names))
        self.assertFalse(passes_constraint(view({"speaker": "Other Leader"}), self.canonical, self.names))
        self.assertIn("حافظ نعیم", self.names)
        self.assertNotIn("hafiz", [name.casefold() for name in self.names if " " not in name and name.casefold() == "hafiz"])

    def test_exact_alias_contract_is_unchanged(self):
        self.assertIn("hafiz naeem", query_variants("HNR", "alias"))
        self.assertTrue(match_spans("حافظ نعیم الرحمن نے کہا", "HNR", "alias"))
        self.assertFalse(match_spans(PRAYER, SPEAKER_TOPIC, "alias"))
        self.assertFalse(match_spans(SPEECH, SPEAKER_ONLY, "normalized"))


class SpeakerSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub = types.ModuleType("fastembed")
        stub.TextEmbedding = type("TextEmbedding", (), {})
        database = QdrantClient(":memory:")
        with patch.dict(sys.modules, {"fastembed": stub}), patch("qdrant_client.QdrantClient", return_value=database):
            import embeddings_service as api
        cls.api = api
        cls.service = api.get_contextual_search("fast")

    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection("source", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        self.client.create_payload_index("source", "text", models.TextIndexParams(type="text", tokenizer="word", lowercase=True))
        self.client.create_payload_index("source", "speaker", models.PayloadSchemaType.KEYWORD)
        self.expanded = []
        self.embedded = []
        self.judged = []
        verified = dict(
            identity_version=VERSION, identity_status="verified", identity_source="manual",
            identity_key="hafiz_naeem_ur_rehman",
            identity_review={"review_id": "r", "reviewer_id": "u", "reviewed_at": "2026-10-20",
                             "evidence_sha256": "a" * 64},
        )
        self.put(1, PRAYER, "Hafiz Naeem")
        self.put(2, PRAYER, "حافظ نعیم الرحمن")
        self.put(3, PRAYER, "SPEAKER_01", **verified)
        self.put(4, PRAYER, "Other Leader")
        self.put(5, ECONOMY, "Hafiz Naeem")
        self.put(6, ECONOMY, "Other Leader")
        self.put(7, SPEECH, "Hafez Naeem")
        self.put(8, SPEECH, "حافظ نعیم")
        info = self.client.get_collection("source")

        def embed(text):
            self.embedded.append(text)
            return _embed(text)

        def expand(query):
            self.expanded.append(query)
            return _plan(query)

        self.base = FastSemanticSearch(
            self.client, "source", embed, self.service.understand,
            lambda query, value: True, lambda query, value: True,
            speaker_values=self.service.speaker_values,
            speaker_index_types={key: getattr(value, "data_type", value) for key, value in (info.payload_schema or {}).items()},
            settings=FastSearchSettings(candidate_limit=50),
            retrieval_expander=expand, diversify_candidates=True, lexical_indexed=True,
        )
        self.cache = {}
        self.engine = GroundedSearch(
            self.base, self._judge, "fixture-model", lambda: "epoch",
            self.cache.get, lambda key, value, ttl: self.cache.update({key: value}),
            ("same-model", 2), title_intent=lambda query: {"queries": [query], "person_only": False},
        )

    def tearDown(self):
        self.client.close()

    def put(self, identity, text, speaker, **extra):
        self.client.upsert("source", [models.PointStruct(id=identity, vector=_embed(text), payload={
            "video_id": identity, "text": text, "speaker": speaker, "start_time": 10.0, "end_time": 40.0,
            "processing_status": "completed", "approval_status": "approved", "is_archived": False,
            "video_title": "Weekly program", **extra,
        })])

    def _judge(self, query, rows, timeout):
        self.judged.append(query)
        judgments = []
        for row in rows:
            yes = ("تلقین" in row["text"] and "نماز" in row["text"]) if any(
                token in query.casefold() for token in ("namaz", "talqeen", "نماز", "تلقین", "pray", "encourag", "nojawan", "نوجوان")
            ) else ("بجلی" in row["text"] or "مہنگائی" in row["text"])
            judgments.append({
                "candidate_id": candidate_id(row), "score": 0.84 if yes else 0.1,
                "relevant": yes, "complete": yes, "roles_supported": yes, "relationship_supported": yes,
                "quotes": [row["text"][:80]] if yes else [], "reason": "relevant" if yes else "related_only",
                "asr_interpretation": None,
            })
        return {"judgments": judgments}

    def _ids(self, query):
        self.judged.clear()
        self.expanded.clear()
        self.embedded.clear()
        self.cache.clear()
        result = self.engine.search(request(query=query, min_score=0.5, top_k=10))
        return {row["video_id"] for row in result["results"]}, result

    def test_topic_only_still_reaches_intent_retrieval_and_the_prayer_clips(self):
        ids, result = self._ids(TOPIC)
        self.assertTrue({1, 2, 3, 4} <= ids)
        self.assertNotIn(5, ids)
        self.assertIn(TOPIC, self.expanded)
        self.assertEqual(TOPIC, self.judged[0])
        self.assertEqual("completed", result["metadata"]["passage_validation"]["status"])

    def test_speaker_plus_topic_keeps_his_clips_in_urdu_roman_and_reviewed_identity(self):
        intent = self.service.understand(SPEAKER_TOPIC)
        self.assertEqual("speaker_search", intent["query_type"])
        self.assertEqual(TOPIC, intent["semantic_query"])
        self.assertEqual("Hafiz Naeem Ur Rehman", intent["extracted_speaker"])
        ids, result = self._ids(SPEAKER_TOPIC)
        self.assertTrue({1, 2, 3} <= ids)
        self.assertNotIn(4, ids)
        self.assertNotIn(5, ids)
        self.assertNotIn(6, ids)
        self.assertIn(TOPIC, self.expanded)
        self.assertNotIn(SPEAKER_TOPIC, self.expanded)
        self.assertEqual(TOPIC, self.judged[0])
        self.assertEqual("Hafiz Naeem Ur Rehman", result["metadata"]["passage_validation"]["speaker_constraint"])
        urdu = self.service.understand("حافظ نعیم نوجوانوں کو نماز کی تلقین")
        self.assertEqual("speaker_search", urdu["query_type"])
        self.assertIn("نماز", urdu["semantic_query"])
        self.assertNotIn("حافظ", urdu["semantic_query"])
        spelled = self.service.understand("Hafez Naeem nojawano ko namaz ki talqeen")
        self.assertEqual(TOPIC, spelled["semantic_query"])
        paraphrase = self.service.understand("Hafiz Naeem encouraging youth to pray")
        self.assertEqual("speaker_search", paraphrase["query_type"])
        self.assertEqual("encouraging youth to pray", paraphrase["semantic_query"])
        ids, _result = self._ids("Hafiz Naeem encouraging youth to pray")
        self.assertTrue({1, 2, 3} <= ids)
        self.assertNotIn(4, ids)
        commentary = self.service.understand("hafiz naeem ki talqeen par tanqeed")
        self.assertNotEqual("speaker_search", commentary.get("query_type"))

    def test_speaker_only_returns_his_speech_without_embedding_the_boilerplate(self):
        intent = self.service.understand(SPEAKER_ONLY)
        self.assertEqual("speaker_only", intent["query_type"])
        ids, result = self._ids(SPEAKER_ONLY)
        self.assertTrue({1, 5, 7, 8} <= ids)
        self.assertTrue({2, 3} <= ids)
        self.assertNotIn(4, ids)
        self.assertNotIn(6, ids)
        self.assertEqual([], self.embedded)
        self.assertEqual([], self.judged)
        self.assertEqual("speaker_speech_lookup", result["ranking"]["engine"])
        self.assertEqual("speaker_speech_lookup", result["metadata"]["passage_validation"]["reason"])
        speech = next(row for row in result["results"] if row["video_id"] == 7)
        self.assertIn("معاشرے", speech["text"])
        self.assertNotIn("bayan", speech["text"].casefold())
        self.assertEqual("unverified_source_label", speech["speaker_identity_status"])
        reviewed = next(row for row in result["results"] if row["video_id"] == 3)
        self.assertEqual("supported_identity", reviewed["speaker_identity_status"])
        hnr = self.service.understand("HNR bayan karte huye")
        self.assertEqual("speaker_only", hnr["query_type"])
        urdu_only = self.service.understand("حافظ نعیم کا بیان")
        self.assertEqual("speaker_only", urdu_only["query_type"])

    def test_unrelated_speaker_does_not_inherit_the_hafiz_constraint(self):
        intent = self.service.understand("Sara Ali nojawano ko namaz ki talqeen")
        self.assertNotEqual("speaker_search", intent.get("query_type"))
        self.assertNotEqual("Hafiz Naeem Ur Rehman", intent.get("extracted_speaker"))
        ids, _result = self._ids("Hafiz Naeem bijli ke bill")
        self.assertIn(5, ids)
        self.assertNotIn(6, ids)
        self.assertNotIn(4, ids)
        self.assertNotIn(1, ids)


if __name__ == "__main__":
    unittest.main()
