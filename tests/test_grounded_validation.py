import pathlib,sys,unittest,copy
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from grounded_validation import validate_grounded,candidate_id
from semantic_passage_evidence import ValidationUnavailable

def row(i):return {'id':str(i),'video_id':i,'text':'University tuition costs more than families can afford.','start_time':1,'end_time':5,'segment_ids':[str(i)]}
def judgment(r,positive=True):return {'candidate_id':candidate_id(r),'score':.9 if positive else .1,'relevant':positive,'complete':positive,'roles_supported':positive,'relationship_supported':positive,'quotes':['tuition costs more than families can afford']if positive else [],'reason':'relevant'if positive else 'related_only'}
class GroundedTests(unittest.TestCase):
 def test_semantic_evidence_without_literal_query_words(self):
  rows,meta=validate_grounded('students and high education fees',[row(1)],lambda q,b,t:{'judgments':[judgment(b[0])]});self.assertEqual(1,len(rows));self.assertIn('passage_evidence',rows[0]);self.assertEqual('completed',meta['status'])
 def test_only_missing_judgment_is_retried(self):
  calls=[]
  def judge(q,b,t):
   calls.append([r['id']for r in b]);return {'judgments':[judgment(r)for r in (b[:1]if len(calls)==1 else b)]}
  rows,meta=validate_grounded('fees',[row(1),row(2)],judge);self.assertEqual([['1','2'],['2']],calls);self.assertEqual(2,len(rows));self.assertEqual('completed',meta['status'])
 def test_duplicate_judgment_does_not_replace_verified_peer(self):
  calls=[]
  def judge(q,b,t):
   calls.append(b);items=[judgment(r)for r in b];return {'judgments':items+[items[-1]]if len(calls)==1 else items}
  rows,meta=validate_grounded('fees',[row(1),row(2)],judge);self.assertEqual(['2'],[r['id']for r in calls[1]]);self.assertEqual(2,len(rows))
 def test_unknown_id_is_explicit_partial(self):
  def judge(q,b,t):return {'judgments':[judgment(b[0]),{**judgment(b[0]),'candidate_id':'unknown'}]}
  rows,meta=validate_grounded('fees',[row(1)],judge);self.assertEqual('partial',meta['status']);self.assertIn('unknown_candidate_id',meta['failure_reasons'])
 def test_bad_quote_cannot_become_match_and_valid_peer_survives(self):
  def judge(q,b,t):return {'judgments':[{**judgment(r),'quotes':['Invented quotation']}if r['id']=='2'else judgment(r)for r in b]}
  rows,meta=validate_grounded('fees',[row(1),row(2)],judge);self.assertEqual(['1'],[r['id']for r in rows]);self.assertEqual('partial',meta['status'])
 def test_provider_failure_cannot_become_no_match(self):
  def judge(*a):raise TimeoutError()
  with self.assertRaises(ValidationUnavailable):validate_grounded('fees',[row(1)],judge)
 def test_recovered_timeout_and_valid_negative_is_true_no_match(self):
  calls=[]
  def judge(q,b,t):
   calls.append(1)
   if len(calls)==1:raise TimeoutError()
   return {'judgments':[judgment(r,False)for r in b]}
  rows,meta=validate_grounded('Minecraft',[row(1)],judge);self.assertEqual([],rows);self.assertEqual('no_matches',meta['status']);self.assertEqual(2,meta['provider_calls'])
 def test_roles_and_negation_cannot_be_ignored(self):
  def judge(q,b,t):return {'judgments':[{**judgment(b[0]),'roles_supported':False}]}
  with self.assertRaises(ValidationUnavailable):validate_grounded('fees',[row(1)],judge)

 def test_truncated_batch_is_split_within_budget(self):
  calls=[]
  def judge(q,b,t):
   calls.append(len(b))
   if len(b)>2:raise ValidationUnavailable('Incomplete provider response')
   return {'judgments':[judgment(r)for r in b]}
  rows,meta=validate_grounded('fees',[row(i)for i in range(4)],judge);self.assertEqual([4,2,2],calls);self.assertEqual(4,len(rows));self.assertEqual('completed',meta['status'])
