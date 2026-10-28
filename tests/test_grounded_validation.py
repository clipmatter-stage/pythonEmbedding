import pathlib,sys,unittest,copy
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from grounded_validation import validate_grounded,candidate_id
from semantic_passage_evidence import ValidationUnavailable

def row(i):return {'id':str(i),'video_id':i,'text':'University tuition costs more than families can afford.','start_time':1,'end_time':5,'segment_ids':[str(i)]}
def judgment(r,positive=True):return {'candidate_id':candidate_id(r),'score':.9 if positive else .1,'relevant':positive,'complete':positive,'roles_supported':positive,'relationship_supported':positive,'quotes':['tuition costs more than families can afford']if positive else [],'reason':'relevant'if positive else 'related_only'}
class GroundedTests(unittest.TestCase):
 def test_accepted_judgment_is_reused_and_rejection_is_not_cached(self):
  cache={}
  def read(q,r):return cache.get((q,candidate_id(r)))
  def write(q,r,j):
    self.assertTrue(j['relevant']);cache[(q,candidate_id(r))]=copy.deepcopy(j)
  calls=[]
  def judge(q,rs,timeout):
    calls.append([r['id']for r in rs]);return {'judgments':[judgment(r,r['id']=='1')for r in rs]}
  first,meta=validate_grounded('fees',[row(1),row(2)],judge,read_judgment=read,write_judgment=write)
  second,meta=validate_grounded('fees',[row(2),row(1)],judge,read_judgment=read,write_judgment=write)
  self.assertEqual([['1','2'],['2']],calls)
  self.assertEqual([r['validation_id']for r in first],[r['validation_id']for r in second])
  self.assertEqual(1,meta['provider_calls']);self.assertEqual(1,meta['judgment_cache_hits'])
  self.assertEqual(2,meta['evaluated_count'])
  self.assertEqual({('fees',candidate_id(row(1)))},set(cache))

 def test_cached_fabricated_quote_is_rejected_and_rejudged(self):
  bad={**judgment(row(1)),'quotes':['This quotation was invented']};calls=[]
  def judge(q,rs,t):calls.append(1);return {'judgments':[judgment(r,False)for r in rs]}
  accepted,meta=validate_grounded('fees',[row(1)],judge,read_judgment=lambda q,r:bad)
  self.assertEqual([],accepted);self.assertEqual(1,len(calls));self.assertEqual(0,meta['judgment_cache_hits'])

 def test_invalid_provider_judgment_is_never_cached(self):
  written=[]
  def judge(q,rs,t):return {'judgments':[{**judgment(r),'roles_supported':False}for r in rs]}
  with self.assertRaises(ValidationUnavailable):validate_grounded('fees',[row(1)],judge,write_judgment=lambda *args:written.append(args))
  self.assertFalse(written)

 def test_judgment_cache_failure_preserves_verified_results(self):
  def down(*args):raise ConnectionError('Cache unavailable')
  accepted,meta=validate_grounded('fees',[row(1)],lambda q,rs,t:{'judgments':[judgment(r)for r in rs]},read_judgment=down,write_judgment=down)
  self.assertEqual(1,len(accepted));self.assertEqual('completed',meta['status'])

 def test_reasoning_judge_does_not_send_incompatible_temperature(self):
  import json
  from types import SimpleNamespace as NS
  from unittest.mock import Mock
  from grounded_validation import provider_judge
  client=Mock();client.with_options.return_value.chat.completions.create.return_value=NS(
   choices=[NS(finish_reason='stop',message=NS(content=json.dumps({'judgments':[judgment(row(1))]}),refusal=None))],
   usage=NS(prompt_tokens=10,completion_tokens=20,total_tokens=30))
  result=provider_judge(client,'gpt-6-luna')('fees',[row(1)],5)
  kwargs=client.with_options.return_value.chat.completions.create.call_args.kwargs
  self.assertEqual('low',kwargs['reasoning_effort']);self.assertNotIn('temperature',kwargs)
  self.assertEqual('low',result['_usage']['reasoning_effort'])
  provider_judge(client,'gpt-4o-mini')('fees',[row(1)],5)
  kwargs=client.with_options.return_value.chat.completions.create.call_args.kwargs
  self.assertNotIn('reasoning_effort',kwargs);self.assertEqual(0,kwargs['temperature'])

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

 def test_bounded_concurrency_never_exceeds_two(self):
  import threading,time
  lock=threading.Lock();active=0;maximum=0
  def judge(q,b,t):
   nonlocal active,maximum
   with lock:active+=1;maximum=max(maximum,active)
   time.sleep(.01)
   with lock:active-=1
   return {'judgments':[judgment(r)for r in b]}
  rows,meta=validate_grounded('fees',[row(i)for i in range(24)],judge,concurrency=100)
  self.assertEqual(24,len(rows));self.assertLessEqual(maximum,2);self.assertEqual(2,meta['limits']['concurrency']);self.assertEqual(3,meta['provider_calls'])

class SourceSpanTests(unittest.TestCase):
 def test_source_spans_are_original_unicode_with_bounded_lengths(self):
  from grounded_validation import source_spans
  text=('طلبہ کی تعلیم اور فیس کے مسائل۔ '*30)+'Final evidence.'
  spans=source_spans(text);self.assertTrue(spans);self.assertEqual(''.join(text.split()),''.join(''.join(s['quote'].split())for s in spans))
  for s in spans:self.assertEqual(text[s['start']:s['end']],s['quote']);self.assertLessEqual(len(s['quote']),170)
 def test_fast_future_frees_slot_while_slow_peer_remains_inflight(self):
  import time
  started={};finished={}
  def judge(q,rs,timeout):
   i=int(rs[0]['id']);started[i]=time.monotonic();time.sleep(.10 if i==0 else .01);finished[i]=time.monotonic();return {'judgments':[judgment(r,yes=False)for r in rs]}
  rows=[row(i)for i in range(4)]
  # Use local schema helper to keep the concurrency test independent of provider transport.
  def judgment(r,yes=False):return {'candidate_id':candidate_id(r),'score':.1,'relevant':False,'complete':False,'roles_supported':False,'relationship_supported':False,'quotes':[],'reason':'related_only'}
  accepted,meta=validate_grounded('unrelated',rows,judge,batch_size=1,max_calls=4,concurrency=2)
  self.assertEqual([],accepted);self.assertEqual(4,meta['evaluated_count']);self.assertLess(started[2],finished[0])
