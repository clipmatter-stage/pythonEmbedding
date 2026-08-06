import pathlib
import sys
import unittest


SERVICE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from semantic_query_decomposition import decompose_semantic_query, has_minimum_topic_evidence


class SemanticQueryDecompositionTest(unittest.TestCase):
    def test_decomposes_client_speaker_topic_phrase(self):
        result = decompose_semantic_query(
            "Hafiz Naeem ur Rehman Speech on Constitution of Pakistan",
            "Hafiz Naeem Ur Rehman",
        )

        self.assertTrue(result["decomposed"])
        self.assertEqual("Constitution of Pakistan", result["retrieval_query"])
        self.assertEqual("spoken_by", result["relation"])

    def test_decomposes_natural_question(self):
        result = decompose_semantic_query(
            "What did Hafiz Naeem say about electoral reforms?",
            "Hafiz Naeem Ur Rehman",
        )

        self.assertTrue(result["decomposed"])
        self.assertEqual("electoral reforms", result["topic"])

    def test_decomposes_talked_about_phrase(self):
        result = decompose_semantic_query(
            "Hafiz Naeem talked about electricity bills",
            "Hafiz Naeem Ur Rehman",
        )

        self.assertTrue(result["decomposed"])
        self.assertEqual("electricity bills", result["retrieval_query"])

    def test_does_not_strip_unknown_out_of_domain_person(self):
        query = "Bill Gates talked about weather"
        result = decompose_semantic_query(query)

        self.assertFalse(result["decomposed"])
        self.assertEqual(query, result["retrieval_query"])

    def test_leaves_topic_only_query_unchanged(self):
        query = "constitutional rights and public mandate"
        result = decompose_semantic_query(query, "Hafiz Naeem Ur Rehman")

        self.assertFalse(result["decomposed"])
        self.assertEqual(query, result["retrieval_query"])

    def test_rejects_tiny_transcript_fragments(self):
        self.assertFalse(has_minimum_topic_evidence("اور"))
        self.assertFalse(has_minimum_topic_evidence("پاکستان کے سیاسی نظام یہی"))

    def test_accepts_passage_with_enough_topic_evidence(self):
        self.assertTrue(
            has_minimum_topic_evidence(
                "ملک میں جمہوریت اور آئین کی بالادستی کے علاوہ کوئی دوسرا حل نہیں ہے"
            )
        )

    def test_rejects_transcription_placeholder(self):
        self.assertFalse(has_minimum_topic_evidence("پ..."))


if __name__ == "__main__":
    unittest.main()
