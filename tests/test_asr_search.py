import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from asr_search import retrieval_hints
from grounded_validation import validate_grounded,candidate_id
class AsrTests(unittest.TestCase):
 def test_makkah_variants_are_bounded_and_unrelated_entities_are_not_expanded(self):
  self.assertIn('مکت المکرمہ',retrieval_hints('Makkah agreement',['agreement']))
  self.assertIn('مکت المکرمہ',retrieval_hints('Mecca',[]))
  self.assertNotIn('مکت المکرمہ',retrieval_hints('Medina agreement',['مدینہ']))
  self.assertLessEqual(len(retrieval_hints('Makkah',['a','b','c','d'])),4)
 def test_reviewed_error_gets_a_bounded_validation_slot_not_automatic_acceptance(self):
  from asr_search import reserve_candidates
  rows=[{'id':str(i),'video_id':i,'text':'مکت المکرمہ میں معاہدہ','score':.01}for i in range(10)]
  self.assertEqual(4,len(reserve_candidates('Makkah',rows)))
  self.assertEqual([],reserve_candidates('Medina',rows))

 def test_canonicalization_does_not_remove_relationships_or_negations(self):
  from asr_search import canonical_query
  self.assertEqual('Makkah',canonical_query('makkah'))
  self.assertEqual('Makkah is not Medina',canonical_query('Makkah is not Medina'))
  self.assertEqual('makkah agreement with Iran',canonical_query('makkah agreement with Iran'))

 def test_contextual_interpretation_keeps_original_quote_and_timestamp(self):
  row={'id':'one','video_id':1087,'text':'مکت المکرمہ میں ہونے والے معاہدے پر بات ہوئی','start_time':31.,'end_time':46.,'score':.8}
  def judge(q,rows,t):return {'judgments':[{'candidate_id':candidate_id(row),'score':.9,'relevant':True,'complete':True,'roles_supported':True,'relationship_supported':True,'reason':'relevant','quotes':['مکت المکرمہ'],'asr_interpretation':'مکت المکرمہ likely refers to مکہ المکرمہ in this context.'}]}
  rows,_=validate_grounded('Makkah', [row],judge)
  self.assertEqual(row['text'],rows[0]['text']);self.assertEqual(31.,rows[0]['start_time'])
  self.assertEqual('مکت المکرمہ',rows[0]['evidence_spans'][0]['quote'])
  self.assertEqual('text_context_inference',rows[0]['asr_interpretation_authority'])
 def test_corrected_or_invented_quote_is_rejected(self):
  from semantic_passage_evidence import ValidationUnavailable
  row={'id':'one','video_id':1087,'text':'مکت المکرمہ کا ذکر','start_time':31.,'end_time':46.,'score':.8}
  def judge(q,rows,t):return {'judgments':[{'candidate_id':candidate_id(row),'score':.9,'relevant':True,'complete':True,'roles_supported':True,'relationship_supported':True,'reason':'relevant','quotes':['مکہ المکرمہ'],'asr_interpretation':'Likely typo'}]}
  with self.assertRaises(ValidationUnavailable):validate_grounded('Makkah',[row],judge)
