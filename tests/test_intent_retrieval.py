"""Paraphrase retrieval: same intent, separated intents, exact/alias unchanged.

Spoken forms stand in for the multilingual planner. Nothing here is a
phrase-to-keyword table; the ranker only sees those strings as retrieval text.
"""
import os
import pathlib
import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from fast_search import FastSearchSettings, FastSemanticSearch
from grounded_search import GroundedSearch
from grounded_validation import SYSTEM, candidate_id
from intent_retrieval import (
    alignment_score, compose_retrieval_queries, keyword_vote_weight, proposition,
)
from test_fast_search import request
from transcript_search_contract import match_spans, query_variants


CLIPS = {
    1: "Bachon ko nazam ki taraf bulao.",
    2: "Bachon mein janamaz taqseem karna.",
    3: "Naujawano ko nazam ki taraf bulana hai.",
    4: "Bachon ko naam lekar stage par bulaya, jaise Tubla.",
}
DISTRACTOR = "Bachon ko election campaign ke bare mein bataya."
ORGANIZE = "Leader ne bachon ko organize karne ki baat ki."
INVITE = "Find the clip where the leader invites young people to join the organization."
JANAMAZ = "Find the speech about distributing janamaz among children."
BY_NAME = "Find the moment where the leader calls a student by name."

ORGANIZE_PLAN = {
    "queries": [
        "بچوں کو تنظیم کی طرف بلاؤ",
        "bachon ko nazam ki taraf bulao",
        "call the children to join the organization",
    ],
    "spoken_queries": [
        "بچوں کو تنظیم کی طرف بلاؤ",
        "bachon ko nazam ki taraf bulao",
        "call the children to join the organization",
    ],
    "retrieval_terms": ["bachon"],
}
JANAMAZ_PLAN = {
    "queries": ["بچوں میں جانماز تقسیم کرو", "bachon mein janamaz taqseem karna"],
    "spoken_queries": [
        "بچوں میں جانماز تقسیم کرو",
        "bachon mein janamaz taqseem karna",
        "distribute prayer mats among the children",
    ],
    "retrieval_terms": ["bachon"],
}
NAME_PLAN = {
    "queries": ["بچوں کو نام لے کر اسٹیج پر بلایا", "bachon ko naam lekar stage par bulaya"],
    "spoken_queries": [
        "بچوں کو نام لے کر اسٹیج پر بلایا",
        "bachon ko naam lekar stage par bulaya",
        "called a student onto the stage by name",
    ],
    "retrieval_terms": ["bachon"],
}
PLANS = {ORGANIZE: ORGANIZE_PLAN, INVITE: ORGANIZE_PLAN, JANAMAZ: JANAMAZ_PLAN, BY_NAME: NAME_PLAN}


def _embed(text):
    folded = text.casefold()
    if any(token in folded for token in ("nazam", "tanzeem", "organize", "organization", "invites young", "تنظیم")):
        return [1.0, 0.0]
    if any(token in folded for token in ("janamaz", "janamaaz", "taqseem", "prayer mat", "جانماز")):
        return [0.0, 1.0]
    if any(token in folded for token in ("naam", "by name", "stage par", "نام")):
        return [0.6, 0.8]
    return [-1.0, 0.0]


class PropositionTests(unittest.TestCase):
    def test_search_boilerplate_is_removed_and_topics_are_kept(self):
        self.assertEqual("bachon ko organize karne", proposition(ORGANIZE))
        self.assertEqual("distributing janamaz among children", proposition(JANAMAZ))
        self.assertEqual(
            "the leader invites young people to join the organization",
            proposition(INVITE),
        )
        self.assertEqual("the leader calls a student by name", proposition(BY_NAME))
        self.assertEqual("namaz ki talqeen", proposition("namaz ki talqeen"))
        self.assertEqual("نماز کی تلقین", proposition("نماز کی تلقین"))
        self.assertEqual("HNR", proposition("HNR"))
        self.assertEqual("namaz ki talqeen", proposition("Find the speech about namaz ki talqeen"))

    def test_spoken_forms_fill_the_embedding_budget_ahead_of_the_question(self):
        queries = compose_retrieval_queries(ORGANIZE, ORGANIZE_PLAN)
        self.assertEqual(4, len(queries))
        self.assertEqual("bachon ko organize karne", queries[0])
        self.assertNotIn(ORGANIZE, queries)
        self.assertIn("bachon ko nazam ki taraf bulao", queries)

    def test_ranker_source_has_no_gold_phrase_table(self):
        source = pathlib.Path(__file__).resolve().parents[1].joinpath("intent_retrieval.py").read_text().casefold()
        for phrase in ("janamaz", "nazam", "organize", "tubla", "tanzeem"):
            self.assertNotIn(phrase, source)

    def test_asr_spelling_and_verb_endings_still_align(self):
        mats = "Bachon mein janamaz taqseem karna."
        folded = "bachon mein janamaaz taqseem karna"
        self.assertGreater(alignment_score(folded, mats), alignment_score(folded, CLIPS[1]))
        invite = "bachon ko nazam ki taraf bulao"
        self.assertGreater(alignment_score(invite, CLIPS[3]), alignment_score(invite, CLIPS[2]))
        self.assertGreater(alignment_score(invite, CLIPS[1]), 0.8)

    def test_same_intent_clusters_and_other_intents_stay_apart(self):
        organize = ORGANIZE_PLAN["spoken_queries"]
        self.assertGreater(alignment_score(organize, CLIPS[1]), alignment_score(organize, CLIPS[2]))
        self.assertGreater(alignment_score(organize, CLIPS[3]), alignment_score(organize, CLIPS[4]))
        self.assertGreater(alignment_score(organize, CLIPS[1]), alignment_score(organize, CLIPS[4]))
        mats = JANAMAZ_PLAN["spoken_queries"]
        self.assertGreater(alignment_score(mats, CLIPS[2]), alignment_score(mats, CLIPS[1]))
        self.assertGreater(alignment_score(mats, CLIPS[2]), alignment_score(mats, CLIPS[4]))
        named = NAME_PLAN["spoken_queries"]
        self.assertGreater(alignment_score(named, CLIPS[4]), alignment_score(named, CLIPS[1]))
        self.assertGreater(alignment_score(named, CLIPS[4]), alignment_score(named, CLIPS[2]))
        # An English paraphrase beside a Roman spoken form must not dilute the Roman hit.
        mixed = ["call the children to join the organization", "bachon ko nazam ki taraf bulao"]
        self.assertGreater(alignment_score(mixed, CLIPS[1]), 0.8)
        self.assertLess(alignment_score(["bachon", *organize], DISTRACTOR), 0.25)

    def test_audience_word_is_not_a_distinctive_keyword_vote(self):
        self.assertEqual(1.0, keyword_vote_weight(
            "نوجوانوں کو نماز پڑھنے کی تلقین کی",
            ["namaz ki talqeen", "نماز کی تلقین", "encouraging prayer"],
            ["نماز"],
        ))
        self.assertEqual(0.0, keyword_vote_weight(
            "Discussion of foreign policy and political meetings",
            ["namaz ki talqeen", "نماز کی تلقین"],
            ["نماز"],
        ))
        self.assertLess(keyword_vote_weight(DISTRACTOR, ORGANIZE_PLAN["spoken_queries"], ["bachon"]), 1)
        self.assertEqual(1.0, keyword_vote_weight(CLIPS[1], ORGANIZE_PLAN["spoken_queries"], ["bachon"]))
        self.assertEqual(1.0, keyword_vote_weight("bachon ke liye", ["bachon"], ["bachon"]))

    def test_exact_and_alias_contracts_ignore_open_paraphrases(self):
        self.assertIn("hafiz naeem", query_variants("HNR", "alias"))
        self.assertTrue(match_spans("حافظ نعیم الرحمن نے کہا", "HNR", "alias"))
        self.assertTrue(match_spans("فلسطین کے بارے میں", "palestine", "alias"))
        for clip in CLIPS.values():
            for query in (ORGANIZE, INVITE, JANAMAZ, BY_NAME):
                self.assertFalse(match_spans(clip, query, "normalized"))
                self.assertFalse(match_spans(clip, query, "alias"))
                self.assertFalse(match_spans(clip, query, "literal"))

    def test_judge_accepts_a_short_paraphrase_without_shared_audience_words(self):
        self.assertIn("same proposition", SYSTEM)
        self.assertIn("different action", SYSTEM)


class IntentRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection("source", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        self.cache = {}
        self.calls = []
        for video_id, text in CLIPS.items():
            self._put(video_id, text, _embed(text))
        for video_id in range(10, 30):
            self._put(video_id, DISTRACTOR, _embed(DISTRACTOR))
        self.client.create_payload_index(
            "source", "text", models.TextIndexParams(type="text", tokenizer="word", lowercase=True))
        self.base = FastSemanticSearch(
            self.client, "source", _embed,
            lambda query: {"semantic_query": query, "understanding_status": "disabled"},
            lambda query, value: True, lambda query, value: True,
            settings=FastSearchSettings(candidate_limit=50),
            retrieval_expander=lambda query: PLANS.get(query, {"queries": [], "retrieval_terms": ["bachon"]}),
            diversify_candidates=True, lexical_indexed=True,
        )
        self.engine = GroundedSearch(
            self.base, self._judge, "fixture-model", lambda: "epoch",
            self.cache.get, lambda key, value, ttl: self.cache.update({key: value}),
            ("same-model", 2),
        )

    def tearDown(self):
        self.client.close()

    def _put(self, video_id, text, vector):
        self.client.upsert("source", [models.PointStruct(id=video_id, vector=vector, payload={
            "video_id": video_id, "text": text, "speaker": "Leader", "transcript_speaker": "A",
            "start_time": float(video_id), "end_time": float(video_id) + 4,
            "processing_status": "completed", "approval_status": "approved", "is_archived": False,
        })])

    def _judge(self, query, rows, timeout):
        self.calls.append((query, [row["video_id"] for row in rows]))
        expected = {
            ORGANIZE: {1, 3},
            INVITE: {1, 3},
            JANAMAZ: {2},
            BY_NAME: {4},
        }.get(query, set())
        judgments = []
        for row in rows:
            yes = row["video_id"] in expected and row["text"] == CLIPS[row["video_id"]]
            judgments.append({
                "candidate_id": candidate_id(row),
                "score": 0.72 if yes else 0.1,
                "relevant": yes, "complete": yes, "roles_supported": yes,
                "relationship_supported": yes,
                "quotes": [CLIPS[row["video_id"]]] if yes else [],
                "reason": "relevant" if yes else "related_only",
                "asr_interpretation": None,
            })
        return {"judgments": judgments}

    def _accepted(self, query, **changes):
        self.calls.clear()
        params = {"query": query, "min_score": 0.50, "top_k": 10}
        params.update(changes)
        with patch.dict(os.environ, {"GROUNDING_INITIAL_CANDIDATES": "8", "GROUNDING_MAX_CANDIDATES": "16"}):
            result = self.engine.search(request(**params))
        return result

    def _first_stage(self):
        seen = []
        for _query, videos in self.calls:
            seen.extend(videos)
            if len(seen) >= 8:
                break
        return set(seen[:8])

    def test_tied_dense_scores_still_rank_the_proposition_above_shared_bachon(self):
        self.base.embed = lambda _text: [1.0, 0.0]
        for video_id, text in {**CLIPS, **{video_id: DISTRACTOR for video_id in range(10, 30)}}.items():
            self.client.update_vectors("source", [models.PointVectors(id=video_id, vector=[1.0, 0.0])])
        result = self.base.search(request(query=ORGANIZE, top_k=30))
        order = []
        for row in result["results"]:
            if row["video_id"] not in order:
                order.append(row["video_id"])
        self.assertIn(order[0], {1, 3})
        self.assertTrue({1, 3} <= set(order[:8]))
        self.assertGreater(order.index(2), max(order.index(1), order.index(3)))

    def test_same_intent_different_wording_returns_the_same_clips(self):
        organize = {row["video_id"] for row in self._accepted(ORGANIZE)["results"]}
        self.assertEqual({1, 3}, organize)
        self.assertTrue({1, 3} <= self._first_stage())
        self.assertEqual(ORGANIZE, self.calls[0][0])
        invite = {row["video_id"] for row in self._accepted(INVITE)["results"]}
        self.assertEqual(organize, invite)
        self.assertEqual(INVITE, self.calls[0][0])

    def test_different_intents_do_not_cross_accept(self):
        self.assertEqual({2}, {row["video_id"] for row in self._accepted(JANAMAZ)["results"]})
        self.assertEqual({4}, {row["video_id"] for row in self._accepted(BY_NAME)["results"]})

    def test_ui_min_score_filters_judge_scores_and_not_dense_candidates(self):
        kept = self._accepted(ORGANIZE)
        self.assertEqual({1, 3}, {row["video_id"] for row in kept["results"]})
        self.assertTrue(all(row["score"] >= 0.50 for row in kept["results"]))
        judged = [video for _query, videos in self.calls for video in videos]
        self.assertTrue({1, 3} <= set(judged))
        self.cache.clear()
        dropped = self._accepted(ORGANIZE, min_score=0.90)
        self.assertEqual([], dropped["results"])
        judged_again = [video for _query, videos in self.calls for video in videos]
        self.assertTrue({1, 3} <= set(judged_again))
        self.assertEqual("no_matches", dropped["metadata"]["passage_validation"]["status"])


if __name__ == "__main__":
    unittest.main()
