import pathlib
import sys
import unittest


SERVICE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from semantic_query_decomposition import (
    build_structured_rerank_fallback,
    decompose_semantic_query,
    has_conceptual_topic_evidence,
    has_complete_facet_coverage,
    has_minimum_topic_evidence,
    passes_structured_topic_validation,
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

    def test_structured_validation_requires_complete_non_incidental_topic(self):
        valid = {
            "llm_relevance_score": 0.8,
            "llm_complete_topic": True,
            "llm_incidental_match": False,
            "text": "Public institutions must be accountable for misuse of authority.",
        }
        self.assertTrue(passes_structured_topic_validation(valid, "Accountability"))

        for override in (
            {"llm_complete_topic": False},
            {"llm_incidental_match": True},
            {"llm_relevance_score": 0.6},
            {"llm_complete_topic": None},
        ):
            candidate = {**valid, **override}
            self.assertFalse(
                passes_structured_topic_validation(candidate, "Accountability")
            )

    def test_complete_facet_coverage_is_computed_fail_closed(self):
        self.assertTrue(has_complete_facet_coverage(["subject", "rights"], ["rights", "subject"]))
        self.assertFalse(has_complete_facet_coverage(["subject", "rights"], ["rights"]))
        self.assertFalse(has_complete_facet_coverage([], []))
        self.assertFalse(has_complete_facet_coverage(["topic"], None))

    def test_relational_rights_require_the_correct_beneficiary(self):
        generic_labor = "Pakistan labor laws provide these rights and social-security benefits."
        womens_rights = "Women and girls must receive equal rights, education, and protection."
        self.assertTrue(has_conceptual_topic_evidence("Labor Rights", generic_labor))
        self.assertFalse(has_conceptual_topic_evidence("Women's Rights", generic_labor))
        self.assertTrue(has_conceptual_topic_evidence("Women's Rights", womens_rights))
        self.assertFalse(has_conceptual_topic_evidence("Labor Rights", womens_rights))

    def test_urdu_relational_rights_do_not_cross_match_groups(self):
        women_only = "عورتوں کو بنیادی حقوق سے محروم کیا جاتا ہے اور لڑکیوں کو تعلیم نہیں ملتی"
        workers_only = "مزدوروں کو ان کے حقوق، اجرت اور سوشل سیکیورٹی ملنی چاہیے"
        self.assertTrue(has_conceptual_topic_evidence("Women's Rights", women_only))
        self.assertFalse(has_conceptual_topic_evidence("Labor Rights", women_only))
        self.assertTrue(has_conceptual_topic_evidence("Labor Rights", workers_only))
        self.assertFalse(has_conceptual_topic_evidence("Women's Rights", workers_only))

    def test_federalism_requires_a_power_or_resource_relationship(self):
        incidental = "The federal government and provincial government are both in power."
        substantive = "Powers and resources must be divided fairly between the federation and provinces."
        urdu_substantive = "وفاق کو صوبوں کے اختیارات اور وسائل کی منصفانہ تقسیم یقینی بنانی چاہیے"
        self.assertFalse(has_conceptual_topic_evidence("Federalism", incidental))
        self.assertTrue(has_conceptual_topic_evidence("Federalism", substantive))
        self.assertTrue(has_conceptual_topic_evidence("Federalism", urdu_substantive))

    def test_provincial_rights_exclude_rights_of_local_government(self):
        local_rights = "The provincial government refuses to transfer funds and powers to local councils."
        province_rights = "Each province has a constitutional right to autonomy and its resource share."
        self.assertFalse(has_conceptual_topic_evidence("Provincial Rights", local_rights))
        self.assertTrue(has_conceptual_topic_evidence("Provincial Rights", province_rights))

    def test_accountability_requires_answerability_or_oversight(self):
        incidental = "Everyone should perform their personal responsibility."
        substantive = "Institutions must be answerable and investigated for misuse of authority."
        urdu_substantive = "اداروں کو جواب دہ بنائیں اور اختیارات کے ناجائز استعمال کا احتساب کریں"
        self.assertFalse(has_conceptual_topic_evidence("Accountability", incidental))
        self.assertTrue(has_conceptual_topic_evidence("Accountability", substantive))
        self.assertTrue(has_conceptual_topic_evidence("Accountability", urdu_substantive))

    def test_reported_production_false_positives_are_rejected(self):
        cases = (
            (
                "Federalism",
                "وفاقی حکومت اور صوبائی حکومت، اور اب تو صوبائی حکومت بھی وفاقی حکومت میں شامل ہے",
            ),
            (
                "Provincial Rights",
                "پیپلز پارٹی صوبائی حکومت میں تھی، بلدیاتی ایکٹ میں ہمارے حقوق غصب اور بلدیاتی اختیارات کم کیے گئے",
            ),
            (
                "Labor Rights",
                "عورت کو بنیادی حقوق سے محروم کرتے ہیں اور لڑکیوں کو تعلیم اور صحت کے حقوق نہیں دیتے",
            ),
            (
                "Women's Rights",
                "پاکستان کے لیبر قوانین کے تحت مزدوروں کو سوشل سیکیورٹی اور تمام حقوق ملنے چاہیئے",
            ),
        )
        for topic, passage in cases:
            with self.subTest(topic=topic):
                self.assertFalse(has_conceptual_topic_evidence(topic, passage))

    def test_higher_education_excludes_generic_education(self):
        self.assertFalse(
            has_conceptual_topic_evidence(
                "Higher Education",
                "Every child has a right to the same standard of education.",
            )
        )
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Higher Education",
                "Universities and colleges need funding so degree education can be affordable.",
            )
        )
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Higher Education",
                "پاکستان کی جامعات اور یونیورسٹیوں کے طلبہ کو اعلیٰ تعلیم مفت ملنی چاہیے",
            )
        )

    def test_structured_fallback_is_transcript_backed_and_precision_first(self):
        results = [
            {
                "id": 1,
                "text": "Pakistan's Constitution guarantees education as a fundamental right for citizens.",
                "score": 0.61,
                "match_types": ["semantic", "query_term_present"],
            },
            {
                "id": 2,
                "text": "This passage is only semantically similar and has no explicit supporting term.",
                "score": 0.9,
                "match_types": ["semantic", "title_match"],
            },
        ]
        fallback = build_structured_rerank_fallback("Constitution of Pakistan", results)
        self.assertEqual([1], [item["id"] for item in fallback])
        self.assertTrue(fallback[0]["llm_complete_topic"])
        self.assertFalse(fallback[0]["llm_incidental_match"])
        self.assertIn("rerank_fallback_exact_evidence", fallback[0]["match_types"])

    def test_structured_fallback_preserves_compound_topic_guards(self):
        results = [{
            "id": 1,
            "text": "The federal government and provincial government are both in power today.",
            "score": 0.9,
            "match_types": ["semantic", "query_term_present"],
        }]
        self.assertEqual([], build_structured_rerank_fallback("Federalism", results))

    def test_democratic_struggle_requires_both_democracy_and_struggle(self):
        self.assertFalse(
            has_conceptual_topic_evidence("Democratic Struggle", "Democracy is important.")
        )
        self.assertFalse(
            has_conceptual_topic_evidence("Democratic Struggle", "Workers continued their struggle for wages.")
        )
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Democratic Struggle",
                "Citizens launched a democratic movement and protested to defend their voting rights.",
            )
        )
        self.assertTrue(
            has_conceptual_topic_evidence(
                "Democratic Struggle",
                "عوامی مینڈیٹ اور جمہوری حقوق کے لیے عوام نے احتجاج اور جدوجہد شروع کی",
            )
        )

    def test_final_structured_gate_rejects_production_fragment(self):
        fragment = {
            "llm_relevance_score": 0.8,
            "llm_complete_topic": True,
            "llm_incidental_match": False,
            "text": "جمہوریت ہے اور انہوں نے اس پورے",
        }
        self.assertFalse(
            passes_structured_topic_validation(fragment, "Democratic Struggle")
        )


if __name__ == "__main__":
    unittest.main()
