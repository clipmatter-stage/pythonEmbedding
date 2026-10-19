import pathlib,sys,unittest
from types import SimpleNamespace as NS
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from qdrant_client import QdrantClient,models
from fast_search import FastSemanticSearch,FastSearchSettings
from grounded_search import GroundedSearch,diverse
from grounded_validation import candidate_id
from test_fast_search import request
class SearchTests(unittest.TestCase):
 def setUp(self):
  self.c=QdrantClient(':memory:');self.c.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE));self.cache={};self.epoch='a';self.calls=0
  self.base=FastSemanticSearch(self.c,'source',lambda q:[1.,0.],lambda q:{'semantic_query':q},lambda q,v:True,lambda q,v:True,settings=FastSearchSettings(candidate_limit=100))
  self.engine=GroundedSearch(self.base,self.judge,'fixture-model',lambda:self.epoch,self.cache.get,lambda k,v,t:self.cache.update({k:v}),('same-model',2))
 def tearDown(self):self.c.close()
 def put(self,i,text,video=1,start=1,end=5,**extra):self.c.upsert('source',[models.PointStruct(id=i,vector=[1.,0.],payload={'video_id':video,'text':text,'speaker':'Unverified Label','transcript_speaker':'A','start_time':start,'end_time':end,'processing_status':'completed','approval_status':'approved','is_archived':False,**extra})])
 def judge(self,q,rows,t):
  self.calls+=1;out=[]
  for r in rows:
   yes='tuition costs' in r['text'] and 'cannot afford' in r['text'];out.append({'candidate_id':candidate_id(r),'score':.9 if yes else .1,'relevant':yes,'complete':yes,'roles_supported':yes,'relationship_supported':yes,'quotes':['tuition costs 100000 rupees','Families cannot afford it']if yes else [],'reason':'relevant'if yes else 'related_only'})
  return {'judgments':out}
 def test_original_turns_supply_context_without_cross_speaker_or_gap(self):
  self.put(1,'University tuition costs 100000 rupees',start=1,end=5);self.put(2,'Families cannot afford it',start=5,end=9);self.put(3,'unrelated filler',start=40,end=50)
  response=self.engine.search(request(query='students and high education fees'));self.assertTrue(response['ranking']['evidence_verified']);self.assertTrue(response['results']);row=response['results'][0];self.assertEqual([1,9],[row['start_time'],row['end_time']]);self.assertEqual({'1','2'},set(row['segment_ids']));self.assertNotIn('filler',row['text']);self.assertEqual('Unverified Label',row['speaker'])
 def test_scope_and_index_revision_bind_cached_verification(self):
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it',video=1);self.put(2,'University tuition costs 100000 rupees. Families cannot afford it',video=2)
  q=request(query='fees',allowed_video_ids=[1],authorization_scope='one',scope_revision='a');first=self.engine.search(q);n=self.calls;second=self.engine.search(q);self.assertEqual(n,self.calls);self.assertTrue(second['metadata']['passage_validation']['cache_hit']);self.assertEqual({1},{r['video_id']for r in first['results']});self.assertEqual({1},{r['video_id']for r in first['ranking']['candidate_trace']});self.epoch='b';self.engine.search(q);self.assertGreater(self.calls,n)
 def test_genuine_no_match_returns_no_unverified_dense_fallback(self):
  self.put(1,'Generic education and elections');r=self.engine.search(request(query='Minecraft'));self.assertEqual([],r['results']);self.assertEqual('no_matches',r['metadata']['passage_validation']['status'])
 def test_summary_uses_all_supplied_candidates_and_requires_original_evidence(self):
  from grounded_search import summary_search
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it',video=80)
  q=request(query='students education fees',allowed_video_ids=[80],summary_candidates=[{'video_id':i,'matched_summary_fields':['summary_urdu']}for i in range(1,84)])
  response=summary_search(self.engine,q);self.assertEqual(83,response['search_contract']['summary_candidate_videos']);self.assertEqual([80],[r['video_id']for r in response['results']]);self.assertTrue(response['results'][0]['passage_evidence']);self.assertEqual(['summary_urdu'],response['results'][0]['matched_summary_fields'])
 def test_diversity_selection_occurs_before_final_grouping(self):
  rows=[{'id':str(i),'video_id':1 if i<20 else 2,'score':1-i/100,'retrieval_score':1-i/100}for i in range(21)];self.assertEqual({1,2},{r['video_id']for r in diverse(rows,2)})

 def test_named_speaker_is_separate_from_action_evidence(self):
  self.base.understand=lambda q:{'semantic_query':'young people should pray','query_type':'speaker_search','extracted_speaker':'Hafiz Naeem Ur Rehman','query_decomposition':{'decomposed':True,'topic':'young people should pray'}}
  self.base.speaker_values=lambda q:['Hafiz Naeem Rehman']
  for i,speaker in enumerate(['Hafiz Naeem Rehman','SPEAKER_02','Hafiz Ali'],1):
   self.put(i,'University tuition costs 100000 rupees. Families cannot afford it',video=i,speaker=speaker)
  seen=[];judge=self.engine.judge
  def recording(query,rows,timeout):
   seen.append(query);return judge(query,rows,timeout)
  self.engine.judge=recording
  result=self.engine.search(request(query='Hafiz Naeem young people should pray'))
  self.assertEqual(['young people should pray'],seen)
  self.assertEqual([1],[r['video_id']for r in result['results']])
  self.assertEqual('named_source_label_match',result['results'][0]['speaker_identity_status'])
  self.assertEqual('source_metadata_only',result['metadata']['passage_validation']['speaker_identity_authority'])
 def test_context_chains_short_turns_but_stops_at_speaker_or_gap(self):
  from grounded_search import coherent_passage
  rows=[{'id':str(i),'video_id':1,'speaker':'Hafiz Naeem','text':str(i),'transcript_speaker':'A','start_time':i,'end_time':i+1}for i in range(1,9)]
  rows[5]['speaker']='Another person'
  result=coherent_passage(rows[2],rows)
  self.assertEqual({'1','2','3','4','5'},set(result['segment_ids']))
  self.assertEqual('1 2 3 4 5',result['text'])
 def test_first_stage_is_small_and_uses_parallel_sized_batches(self):
  for i in range(1,31):self.put(i,'University tuition costs 100000 rupees. Families cannot afford it',video=i)
  result=self.engine.search(request(query='education fees',top_k=10))
  meta=result['metadata']['passage_validation']
  self.assertEqual(12,meta['candidate_count']);self.assertEqual(2,meta['provider_calls'])
  self.assertTrue(meta['has_unchecked_candidates']);self.assertEqual(1,result['ranking']['context_reads'])
 def test_expand_when_first_stage_does_not_find_matches(self):
  for i in range(1,31):self.put(i,'Generic education',video=i)
  n=[0]
  def judge(q,rows,t):
   n[0]+=len(rows)
   return {'judgments':[{'candidate_id':candidate_id(r),'score':.1,'relevant':False,'complete':False,'roles_supported':False,'relationship_supported':False,'quotes':[],'reason':'related_only'}for r in rows]}
  self.engine.judge=judge
  result=self.engine.search(request(query='high fees'))
  self.assertEqual(24,n[0]);self.assertEqual(2,result['ranking']['context_reads'])
  self.assertEqual([],result['results']);self.assertTrue(result['metadata']['passage_validation']['has_unchecked_candidates'])

 def test_unknown_unlabeled_speakers_are_not_joined(self):
  from grounded_search import coherent_passage
  rows=[{'id':str(i),'video_id':1,'speaker':'Unknown','text':str(i),'start_time':i,'end_time':i+1}for i in range(1,4)]
  result=coherent_passage(rows[1],rows)
  self.assertEqual(['2'],result['segment_ids'])
 def test_context_does_not_bridge_long_gaps(self):
  from grounded_search import coherent_passage
  rows=[{'id':str(i),'video_id':1,'speaker':'Alice','text':str(i),'start_time':i*10,'end_time':i*10+1}for i in range(1,4)]
  self.assertEqual(['2'],coherent_passage(rows[1],rows)['segment_ids'])

 def test_grounded_session_binding_changes_with_limits(self):
  from unittest.mock import patch
  from transcript_search_contract import search_session_binding
  q=request(query='fees',verify_relevance=True)
  before=search_session_binding(q)
  with patch.dict('os.environ',{'GROUNDING_INITIAL_CANDIDATES':'8'}):
   self.assertNotEqual(before,search_session_binding(q))
 def test_title_only_candidate_is_not_claimed_as_spoken_evidence(self):
  self.put(1,'Unrelated transcript',video_title='Graduation ceremony')
  result=self.engine.search(request(query='Graduation ceremony'))
  self.assertEqual(1,result['returned']);row=result['results'][0]
  self.assertTrue(row['is_video_only']);self.assertEqual('',row['text'])
  self.assertEqual('title',row['evidence_source']);self.assertFalse(row['transcript_verified'])
  self.assertNotIn('passage_evidence',row);self.assertFalse(result['ranking']['evidence_verified'])
 def test_both_sources_merge_without_duplicate_video(self):
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it',video_title='Education fees')
  result=self.engine.search(request(query='Education fees'))
  self.assertEqual(1,result['returned']);row=result['results'][0]
  self.assertEqual('title_and_transcript',row['evidence_source'])
  self.assertTrue(row['transcript_verified']);self.assertIn('title_match',row['match_types'])
  self.assertTrue(row['passage_evidence']);self.assertNotIn('is_video_only',row)
 def test_name_only_alias_title_lookup_needs_no_embedding_or_judge(self):
  self.put(1,'Other content',video_title='Hafiz Naeem Rehman speaks',speaker='SPEAKER_02')
  self.engine.title_intent=lambda q:{'queries':['HNR','Hafiz Naeem Rehman'],'person_only':True}
  def unavailable(*args):raise AssertionError('No model calls for name-only title lookup')
  self.engine.judge=unavailable;self.base.embed=unavailable
  result=self.engine.search(request(query='HNR'))
  self.assertEqual(1,result['returned']);self.assertEqual(0,result['metadata']['passage_validation']['provider_calls'])
  self.assertEqual('not_established_by_title',result['results'][0]['speaker_identity_status'])
  self.assertFalse(result['ranking']['evidence_verified'])
 def test_title_source_obeys_eligibility_and_authorized_scope(self):
  for i in range(1,5):self.put(i,'Other',video=i,video_title='Education fees',approval_status='pending'if i==3 else 'approved',is_archived=i==4)
  result=self.engine.search(request(query='Education fees',allowed_video_ids=[1,3,4]))
  self.assertEqual([1],[r['video_id']for r in result['results']])
 def test_topic_is_not_discarded_when_name_expands_in_title_query(self):
  self.put(1,'Other',video_title='Hafiz Naeem Rehman budget discussion')
  self.engine.title_intent=lambda q:{'queries':['HNR namaz','Hafiz Naeem Rehman namaz'],'person_only':False}
  result=self.engine.search(request(query='HNR namaz'))
  self.assertEqual([],result['results'])
 def test_indexed_title_retrieval_is_scored_against_original_title(self):
  self.c.create_payload_index('source','video_title',models.TextIndexParams(type='text',tokenizer='word',lowercase=True))
  self.put(1,'Other',video_title='Education fees');self.engine.title_indexed=True
  result=self.engine.search(request(query='Education fees'))
  self.assertEqual(1,result['returned']);self.assertEqual('indexed_lexical',result['metadata']['title_retrieval']['method'])
