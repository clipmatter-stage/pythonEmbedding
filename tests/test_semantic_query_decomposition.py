import pathlib
import sys
import unittest


SERVICE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from semantic_query_decomposition import (
    decompose_semantic_query,
    has_conceptual_topic_evidence,
    has_minimum_topic_evidence,
)


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

    def test_parliament_rejects_incidental_job_title(self):
        self.assertFalse(
            has_conceptual_topic_evidence(
                "Parliament",
                "Congratulations to our brother, who is the youngest parliamentarian.",
            )
        )

    def test_parliament_accepts_substantive_institutional_discussion(self):
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Parliament",
                "The parliament must debate legislation and hold the government accountable.",
            )
        )
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Parliament",
                "قومی اسمبلی اور سینیٹ میں قانون سازی اور عوامی نمائندگی پر بحث ہونی چاہیے",
            )
        )

    def test_other_topics_are_not_affected_by_institutional_guard(self):
        self.assertTrue(has_conceptual_topic_evidence("Rule of Law", "any passage"))

    def test_leadership_development_requires_connected_development_evidence(self):
        self.assertFalse(
            has_conceptual_topic_evidence(
                "Leadership Development",
                "The city needs development and, much later, the election needs a leader.",
            )
        )
        self.assertFalse(
            has_conceptual_topic_evidence(
                "Leadership Development",
                "Leadership must be honest and work with a strong team.",
            )
        )
        self.assertFalse(
            has_conceptual_topic_evidence(
                "Leadership Development",
                "کراچی شہر میں development چاہیے، شہر آگے بڑھے گا لیکن اس کے لیے اس کو لیڈر چاہیے",
            )
        )
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Leadership Development",
                "We train and mentor young people to develop their leadership skills.",
            )
        )
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Leadership Development",
                "نوجوانوں کی تربیت اور صلاحیت کے ذریعے نئی قیادت تیار کرنا ضروری ہے",
            )
        )

    def test_farmers_rights_requires_farmer_domain_and_welfare_evidence(self):
        self.assertFalse(
            has_conceptual_topic_evidence(
                "Farmers Rights",
                "Every worker deserves social security, a bonus, and other legal rights.",
            )
        )
        self.assertFalse(
            has_conceptual_topic_evidence(
                "Farmers Rights",
                "جو محنت کرے اس کو پھل ملے، ای او بی آئی، سوشل سیکیورٹی اور بونس کے حقوق ملیں",
            )
        )
        self.assertFalse(
            has_conceptual_topic_evidence(
                "Farmers Rights",
                "People must be counted so resources can be distributed fairly.",
            )
        )
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Farmers Rights",
                "Farmers deserve a fair crop price, subsidies, and protection from exploitation.",
            )
        )
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Farmers Rights",
                "کسان پریشان ہیں کیونکہ فصل کی مناسب قیمت اور معاوضہ نہیں ملتا",
            )
        )


if __name__ == "__main__":
    unittest.main()
