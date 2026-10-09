import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from semantic_passage_evidence import ValidationUnavailable, validation_error_detail

class ValidationErrorTests(unittest.TestCase):
    def test_evidence_failure_is_distinct_from_outage(self):
        for reason in ('quote_not_in_passage', 'missing_facet_evidence'):
            result = validation_error_detail(ValidationUnavailable('private', reason=reason, diagnostic_id='test'))
            self.assertEqual('passage_evidence_invalid', result['code'])
            self.assertEqual(reason, result['reason'])
            self.assertEqual('test', result['diagnostic_id'])
            self.assertNotIn('private', str(result))

    def test_timeout_protocol_and_outage_are_distinct(self):
        for reason, code in [('provider_timeout', 'passage_validation_timeout'),
                             ('incomplete_judgments', 'passage_validation_invalid_response'),
                             ('provider_connection', 'passage_validation_unavailable')]:
            self.assertEqual(code, validation_error_detail(ValidationUnavailable('', reason=reason))['code'])

    def test_arbitrary_reason_is_not_exposed(self):
        self.assertEqual('validation_failed', validation_error_detail(ValidationUnavailable('', reason='private payload'))['reason'])
