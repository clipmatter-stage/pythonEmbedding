"""Curated person/party retrieval aliases for exact alias search.

These are the sheet forms already used for Hafiz Naeem speaker matching, plus
the Jamaat-e-Islami forms named by the client. They are not the full
PERSON_ALIASES typo list, and evidence quotes stay original transcript slices.
"""
import hashlib, pathlib, re, sys, unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qdrant_client import QdrantClient, models
from exact_lexical import INDEX_VERSION, derived_fields
from exact_search_cursor import exact_page
from test_exact_cursor import Redis
from transcript_search_contract import CompiledMatcher, normalize, query_variants

HAFIZ = [
    'Hafiz Naeem',
    'Hafiz Naeem ur Rehman',
    'Hafiz Naeem-ur-Rehman',
    'Hafiz Naeem Rehman',
    'Hafiz Naeem Ur Rahman',
    'HNR',
    'H.N.R.',
    'حافظ نعیم',
    'حافظ نعیم الرحمن',
    'حافظ نعیم الرحمان',
]
JAMAAT = [
    'Jamaat Islami',
    'Jamaat-e-Islami',
    'Jamaat e Islami',
    'Jamat-e-Islami',
    'Jamat Islami',
    'JI',
    'جماعت اسلامی',
    'جماعتِ اسلامی',
]


def spans(query, text, contract='alias'):
    found = CompiledMatcher(query, contract).spans(text)
    for span in found:
        if span['text'] != text[span['start']:span['end']]:
            raise AssertionError(span)
    return found


class PersonPartyAliasTests(unittest.TestCase):
    def test_english_roman_and_abbreviations_share_urdu_retrieval(self):
        hafiz = query_variants('حافظ نعیم', 'alias')
        jamaat = query_variants('جماعت اسلامی', 'alias')
        self.assertIn('حافظ نعیم', hafiz)
        self.assertIn(normalize('Hafiz Naeem ur Rehman'), hafiz)
        self.assertIn('hnr', hafiz)
        self.assertIn('جماعت اسلامی', jamaat)
        self.assertIn(normalize('Jamaat-e-Islami'), jamaat)
        self.assertIn('ji', jamaat)
        for query in HAFIZ:
            self.assertEqual(hafiz, query_variants(query, 'alias'), query)
        for query in JAMAAT:
            self.assertEqual(jamaat, query_variants(query, 'alias'), query)
        self.assertTrue(set(hafiz).isdisjoint(jamaat))

    def test_cross_script_spans_keep_original_asr(self):
        urdu = 'آج حافظ نعیم نے جماعت اسلامی کے بارے میں بات کی'
        self.assertEqual(['حافظ نعیم'], [s['text'] for s in spans('HNR', urdu)])
        self.assertEqual(['حافظ نعیم'], [s['text'] for s in spans('Hafiz Naeem ur Rehman', urdu)])
        self.assertEqual(['جماعت اسلامی'], [s['text'] for s in spans('JI', urdu)])
        self.assertEqual(['جماعت اسلامی'], [s['text'] for s in spans('Jamaat-e-Islami', urdu)])
        english = "Hafiz Naeem ur Rehman said Jamaat-e-Islami's rally continues"
        self.assertEqual(['Hafiz Naeem ur Rehman'], [s['text'] for s in spans('حافظ نعیم', english)])
        self.assertEqual(['Jamaat-e-Islami'], [s['text'] for s in spans('جماعت اسلامی', english)])
        self.assertEqual(['JI', 'جماعتِ اسلامی'], [s['text'] for s in spans('Jamaat Islami', 'JI اور جماعتِ اسلامی')])
        dotted = spans('حافظ نعیم', 'H.N.R. announced')
        self.assertEqual(['H.N.R'], [s['text'] for s in dotted])
        self.assertNotIn('hafiz', dotted[0]['text'])

    def test_alias_inside_a_longer_query_does_not_rewrite_the_quote(self):
        text = 'حافظ نعیم federalism'
        found = spans('HNR federalism', text)
        self.assertEqual(['حافظ نعیم federalism'], [s['text'] for s in found])
        self.assertNotIn('hafiz_naeem', found[0]['text'])

    def test_short_forms_are_whole_tokens(self):
        for query, text in [
            ('JI', 'Jinnah addressed jihad and the hajj'),
            ('JI', 'J.I.P. met'),
            ('JI', 'Jamaat only'),
            ('HNR', 'HNRish label'),
            ('HNR', 'Hafiz Naeemullah'),
            ('Hafiz Naeem', 'Hafiz Naeemullah spoke'),
            ('JI', 'حافظ نعیم'),
            ('HNR', 'جماعت اسلامی'),
        ]:
            self.assertEqual([], spans(query, text), (query, text))
        # Standalone "ji" is the abbreviation, so the Roman honorific collides.
        # The quote is still the original token, not the party name.
        honorific = spans('جماعت اسلامی', 'sahib ji')
        self.assertEqual(['ji'], [s['text'] for s in honorific])

    def test_name_parts_and_unrelated_topics_are_not_aliases(self):
        for query in ['hafiz', 'naeem', 'rehman', 'ur', 'نعیم', 'jamaat', 'islami', 'e', 'ameer jamaat', 'hafiz sahab']:
            self.assertEqual([normalize(query)], query_variants(query, 'alias'), query)
            self.assertEqual([], spans(query, 'حافظ نعیم'), query)
            self.assertEqual([], spans(query, 'جماعت اسلامی'), query)
        self.assertEqual(query_variants('sood', 'alias'), query_variants('سود', 'alias'))
        self.assertEqual(query_variants('interest', 'alias'), query_variants('سود', 'alias'))
        self.assertEqual(query_variants('namaz', 'alias'), query_variants('نماز', 'alias'))
        self.assertEqual(query_variants('Quran', 'alias'), query_variants('قرآن', 'alias'))
        self.assertEqual(query_variants('Tax', 'alias'), query_variants('ٹیکس', 'alias'))
        self.assertEqual(query_variants('Elite', 'alias'), query_variants('اشرافیہ', 'alias'))
        self.assertEqual(query_variants('Ghaza', 'alias'), query_variants('غزہ', 'alias'))
        self.assertEqual(query_variants('Youth', 'alias'), query_variants('نوجوان', 'alias'))
        self.assertEqual(query_variants('KE', 'alias'), query_variants('کے الیکٹرک', 'alias'))
        self.assertIn('فلسطین', query_variants('falasteen', 'alias'))
        self.assertEqual(query_variants('IPP', 'alias'), query_variants('آئی پی پی', 'alias'))
        self.assertEqual(query_variants('IMF', 'alias'), query_variants('آئی ایم ایف', 'alias'))
        self.assertTrue(spans('interest', 'سود'))
        self.assertTrue(spans('Namaz', 'نماز'))
        self.assertEqual([], spans('Gaza', 'Palestine فلسطین'))
        self.assertEqual([], CompiledMatcher('JI', 'normalized').spans('جماعت اسلامی'))
        self.assertEqual([], CompiledMatcher('HNR', 'literal').spans('حافظ نعیم'))

    def test_exact_alias_prefilter_retrieves_urdu_from_english_without_rewriting(self):
        rows = {
            1: 'آج حافظ نعیم نے بات کی',
            2: 'Hafiz Naeem ur Rehman spoke',
            3: 'HNR said this',
            4: 'جماعت اسلامی نے اعلان کیا',
            5: 'Jamaat-e-Islami held a rally',
            6: 'The JI statement',
            7: 'Jinnah addressed the crowd',
            8: 'HNRish is not a name',
            9: 'budget discussion today',
            10: 'H.N.R. announced the plan',
        }
        client = QdrantClient(':memory:')
        self.addCleanup(client.close)
        client.create_collection('source', vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        points = []
        for identity, text in rows.items():
            fields = derived_fields(text) if identity == 10 else _stale_alias_fields(text)
            points.append(models.PointStruct(id=identity, vector=[.1, .2], payload={
                'video_id': 1, 'text': text, 'start_time': identity, 'end_time': identity + 1,
                'timestamp_unit': 'seconds', 'processing_status': 'completed',
                'approval_status': 'approved', 'is_archived': False, **fields,
            }))
        client.upsert('source', points)
        info = client.get_collection('source')
        info.payload_schema = {key: NS(data_type='keyword') for key in [
            'exact_terms_version', 'exact_normalized_terms_v1', 'exact_alias_terms_v1', 'exact_literal_terms_v1']}
        with patch.object(client, 'get_collection', return_value=info):
            hafiz = _page(client, 'HNR')
            party = _page(client, 'JI')
            urdu_party = _page(client, 'جماعت اسلامی')
            normalized = _page(client, 'HNR', 'normalized')
        self.assertTrue(hafiz['search_contract']['lexical_index_used'])
        self.assertLess(hafiz['search_contract']['scanned_segments'], len(rows))
        self.assertEqual({1, 2, 3, 10}, {int(row['id']) for row in hafiz['results']})
        urdu_hit = next(row for row in hafiz['results'] if row['id'] == '1')
        self.assertEqual(rows[1], urdu_hit['text'])
        self.assertEqual(['حافظ نعیم'], [span['text'] for span in urdu_hit['match_spans']])
        self.assertEqual(urdu_hit['match_spans'][0]['text'], urdu_hit['text'][urdu_hit['match_spans'][0]['start']:urdu_hit['match_spans'][0]['end']])
        dotted = next(row for row in hafiz['results'] if row['id'] == '10')
        self.assertEqual('H.N.R', dotted['match_spans'][0]['text'])
        self.assertEqual({4, 5, 6}, {int(row['id']) for row in party['results']})
        self.assertEqual({4, 5, 6}, {int(row['id']) for row in urdu_party['results']})
        normalized_ids = {row['id'] for row in normalized['results']}
        self.assertIn('3', normalized_ids)
        self.assertNotIn('1', normalized_ids)
        self.assertNotIn('2', normalized_ids)
        self.assertTrue(all(client.retrieve('source', [int(row['id'])])[0].payload['text'] == rows[int(row['id'])] for row in hafiz['results']))


def _page(client, query, contract='alias'):
    return exact_page({'query': query, 'transcript_contract': contract, 'allowed_video_ids': [1]}, client, Redis(), 'source', models)


def _stale_alias_fields(text):
    """Normalized tokens copied into alias terms, as stored before this family existed."""
    tokens = sorted(set(re.findall(r'\w+', normalize(text))))
    fields = derived_fields(text)
    fields['exact_alias_terms_v1'] = tokens
    fields['exact_terms_version'] = INDEX_VERSION
    fields['exact_source_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    return fields


if __name__ == '__main__':
    unittest.main()
