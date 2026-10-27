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
  response=self.engine.search(request(query='students and high education fees'));self.assertTrue(response['ranking']['evidence_verified']);self.assertTrue(response['results']);row=response['results'][0];self.assertEqual([1,9],[row['start_time'],row['end_time']]);self.assertEqual({'1','2'},set(row['segment_ids']));self.assertNotIn('filler',row['text']);self.assertEqual('Unknown Speaker',row['speaker'])
 def test_scope_and_index_revision_bind_cached_verification(self):
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it',video=1);self.put(2,'University tuition costs 100000 rupees. Families cannot afford it',video=2)
  q=request(query='fees',allowed_video_ids=[1],authorization_scope='one',scope_revision='a');first=self.engine.search(q);n=self.calls;second=self.engine.search(q);self.assertEqual(n,self.calls);self.assertTrue(second['metadata']['passage_validation']['cache_hit']);self.assertEqual({1},{r['video_id']for r in first['results']});self.assertEqual({1},{r['video_id']for r in first['ranking']['candidate_trace']});self.epoch='b';self.engine.search(q);self.assertGreater(self.calls,n)

 def test_bounded_summary_scan_keeps_verified_snapshot_across_fresh_searches(self):
  from unittest.mock import patch
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it',video=592)
  writes=[]
  def write(key,value,ttl):
   writes.append((key,ttl));self.cache[key]=value
  self.engine.write_cache=write
  metadata={'matching_videos':0,'elapsed_seconds':0,'warnings':['summary_retrieval_bounded'],'exhaustive':False}
  with patch('grounded_summary_search.summary_candidates',return_value=([],metadata)):
   first=self.engine.search(request(query='fees'));calls=self.calls
   second=self.engine.search(request(query='fees'))
   self.assertEqual(first['results'],second['results']);self.assertEqual(calls,self.calls)
   self.assertTrue(second['metadata']['passage_validation']['cache_hit'])
   self.assertFalse(second['metadata']['summary_retrieval']['exhaustive'])
   self.assertTrue(any(key.startswith('grounded:')and ttl is None for key,ttl in writes))
   self.epoch='changed-index';self.engine.search(request(query='fees'));self.assertGreater(self.calls,calls)

 def test_metadata_provider_failure_cannot_freeze_a_snapshot(self):
  from unittest.mock import patch
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it')
  metadata={'matching_videos':0,'elapsed_seconds':0,'warnings':['summary_retrieval_unavailable'],'exhaustive':False}
  with patch('grounded_summary_search.summary_candidates',return_value=([],metadata)):
   self.engine.search(request(query='fees'))
  self.assertFalse(any(key.startswith('grounded:')for key in self.cache))

 def test_verified_results_survive_loss_of_temporary_judgment_cache(self):
  from search_snapshots import SearchSnapshots
  snapshots=SearchSnapshots(self.c)
  self.engine.read_cache=lambda key:snapshots.read(key)if key.startswith('grounded:')else self.cache.get(key)
  self.engine.write_cache=lambda key,value,ttl:snapshots.write(key,value)if key.startswith('grounded:')else self.cache.update({key:value})
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it',video=1169)
  first=self.engine.search(request(query='fees'));calls=self.calls
  self.cache.clear()
  snapshots=SearchSnapshots(self.c)
  second=self.engine.search(request(query='fees'))
  self.assertEqual(calls,self.calls);self.assertEqual(first['results'],second['results'])
  self.assertTrue(second['metadata']['passage_validation']['cache_hit'])

 def test_unrelated_deployment_keeps_the_verified_snapshot(self):
  from unittest.mock import patch
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it')
  with patch.dict('os.environ',{'RAILWAY_GIT_COMMIT_SHA':'deployment-one'}):
   first=self.engine.search(request(query='fees'));calls=self.calls
  with patch.dict('os.environ',{'RAILWAY_GIT_COMMIT_SHA':'deployment-two'}):
   second=self.engine.search(request(query='fees'))
  self.assertEqual(calls,self.calls);self.assertEqual(first['results'],second['results'])
  self.assertTrue(second['metadata']['passage_validation']['cache_hit'])

 def test_search_rules_and_access_scope_invalidate_saved_results(self):
  from unittest.mock import patch
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it',video=1)
  with patch('grounded_search.search_rules_fingerprint',return_value='rules-one'):
   self.engine.search(request(query='fees',allowed_video_ids=[1],authorization_scope='one'))
  keys=set(self.cache)
  with patch('grounded_search.search_rules_fingerprint',return_value='rules-two'):
   self.engine.search(request(query='fees',allowed_video_ids=[1],authorization_scope='one'))
   self.assertGreater(len(set(self.cache)-keys),0)
   result=self.engine.search(request(query='fees',allowed_video_ids=[2],authorization_scope='two'))
  self.assertEqual([],result['results'])

 def test_equal_scores_have_a_stable_order(self):
  rows=[{'id':'b','video_id':2,'score':.9},{'id':'c','video_id':1,'score':.9},{'id':'a','video_id':1,'score':.9}]
  self.assertEqual(diverse(rows,3),diverse(list(reversed(rows)),3))

 def test_individual_judgments_reused_without_full_response_cache(self):
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it',video=592)
  self.engine.read_cache=lambda key:self.cache.get(key)if key.startswith('grounded_judgment:')else None
  first=self.engine.search(request(query='fees'));calls=self.calls
  second=self.engine.search(request(query='fees'))
  self.assertEqual(calls,self.calls);self.assertEqual([592],[r['video_id']for r in second['results']])
  self.assertEqual(0,second['metadata']['passage_validation']['provider_calls'])
  self.assertEqual(1,second['metadata']['passage_validation']['judgment_cache_hits'])
  self.engine.model='different-model';self.engine.search(request(query='fees'));self.assertGreater(self.calls,calls)
  calls=self.calls;self.engine.search(request(query='unaffordable education'));self.assertGreater(self.calls,calls)
  self.c.set_payload('source',{'text':'Unrelated political discussion'},points=[1])
  self.assertEqual([],self.engine.search(request(query='fees'))['results'])
  self.c.set_payload('source',{'approval_status':'pending'},points=[1]);calls=self.calls
  self.assertEqual([],self.engine.search(request(query='fees'))['results']);self.assertEqual(calls,self.calls)

 def test_full_response_and_judgments_bind_model_request_settings(self):
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it')
  def judge(query,rows,timeout):return self.judge(query,rows,timeout)
  judge.cache_contract=('fixture-model','span_id','none');self.engine.judge=judge
  self.engine.search(request(query='fees'));calls=self.calls
  self.engine.search(request(query='fees'));self.assertEqual(calls,self.calls)
  judge.cache_contract=('fixture-model','span_id','low')
  self.engine.search(request(query='fees'));self.assertGreater(self.calls,calls)
 def test_genuine_no_match_returns_no_unverified_dense_fallback(self):
  self.put(1,'Generic education and elections');r=self.engine.search(request(query='Minecraft'));self.assertEqual([],r['results']);self.assertEqual('no_matches',r['metadata']['passage_validation']['status'])

 def test_expansion_failure_cannot_be_cached_as_completed_no_match(self):
  from semantic_passage_evidence import ValidationUnavailable,validation_error_detail
  self.put(1,'Unrelated election discussion')
  self.base.retrieval_expander=lambda q:{'queries':[],'warnings':['multilingual_expansion_unavailable']}
  with self.assertRaises(ValidationUnavailable)as failure:
   self.engine.search(request(query='namaz ki talqeen'))
  self.assertEqual('query_expansion_unavailable',failure.exception.reason)
  self.assertTrue(validation_error_detail(failure.exception)['retryable'])
  self.assertFalse(self.cache)

 def test_expansion_failure_preserves_verified_matches_as_uncached_partial(self):
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it')
  self.base.retrieval_expander=lambda q:{'queries':[],'warnings':['multilingual_expansion_unavailable']}
  result=self.engine.search(request(query='education fees'))
  self.assertEqual(1,result['returned']);self.assertTrue(result['results'][0]['transcript_verified'])
  self.assertEqual('partial',result['metadata']['passage_validation']['status'])
  self.assertTrue(result['metadata']['passage_validation']['retryable']);self.assertFalse(self.cache)

 def test_legacy_query_words_are_not_repeated_for_retrieval_or_judgment(self):
  self.put(1,'Generic source text')
  retrieved=[];judged=[];understand=self.base.understand;judge=self.engine.judge
  self.base.understand=lambda query:(retrieved.append(query)or understand(query))
  self.engine.judge=lambda query,rows,timeout:(judged.append(query)or judge(query,rows,timeout))
  result=self.engine.search(request(query='namaz ki talqeen',words=['namaz','ki','talqeen']))
  self.assertEqual(['namaz ki talqeen'],retrieved)
  self.assertTrue(judged);self.assertEqual({'namaz ki talqeen'},set(judged))
  self.assertEqual('namaz ki talqeen',result['metadata']['passage_validation']['original_query'])
 def test_summary_uses_all_supplied_candidates_and_requires_original_evidence(self):
  from grounded_search import summary_search
  self.put(1,'University tuition costs 100000 rupees. Families cannot afford it',video=80)
  q=request(query='students education fees',allowed_video_ids=[80],summary_candidates=[{'video_id':i,'matched_summary_fields':['summary_urdu']}for i in range(1,84)])
  response=summary_search(self.engine,q);self.assertEqual(83,response['search_contract']['summary_candidate_videos']);self.assertEqual([80],[r['video_id']for r in response['results']]);self.assertTrue(response['results'][0]['passage_evidence']);self.assertEqual(['summary_urdu'],response['results'][0]['matched_summary_fields'])
 def test_diversity_selection_occurs_before_final_grouping(self):
  rows=[{'id':str(i),'video_id':1 if i<20 else 2,'score':1-i/100,'retrieval_score':1-i/100}for i in range(21)];self.assertEqual({1,2},{r['video_id']for r in diverse(rows,2)})

 def test_namaz_ki_talqeen_video_592_reaches_verification_across_all_videos(self):
  # Synthetic source fixture: this tests retrieval/verification integration,
  # not an assertion about the current live transcript contents of video 592.
  for i in range(1,111):self.put(i,'Unrelated election discussion',video=1,start=i*100,end=i*100+10)
  text='ہم نے نوجوانوں کو نماز پڑھنے کی تلقین کی اور مسجد جانے کی دعوت دی'
  self.put(592,text,video=592,start=1000,end=1015)
  self.c.update_vectors('source',[models.PointVectors(id=592,vector=[-1.,0.])])
  self.base.diversify_candidates=True;self.base.lexical_indexed=True
  self.base.retrieval_expander=lambda q:{'queries':['نماز کی تلقین','encouraging people to pray'],'retrieval_terms':['نماز']}
  self.base.embed=lambda q:[-1.,0.]if q=='نماز کی تلقین'else[1.,0.]
  seen=[]
  def judge(query,rows,timeout):
   self.assertEqual('namaz ki talqeen',query);out=[]
   for row in rows:
    seen.append(row['video_id']);yes=row['text']==text
    out.append({'candidate_id':candidate_id(row),'score':.9 if yes else .1,'relevant':yes,'complete':yes,'roles_supported':yes,'relationship_supported':yes,'quotes':['نوجوانوں کو نماز پڑھنے کی تلقین کی']if yes else [],'reason':'relevant'if yes else 'related_only'})
   return {'judgments':out}
  self.engine.judge=judge
  result=self.engine.search(request(query='namaz ki talqeen'))
  self.assertIn(592,seen);self.assertEqual([592],[r['video_id']for r in result['results']])
  self.assertTrue(result['results'][0]['transcript_verified'])
  self.c.set_payload('source',{'approval_status':'pending'},points=[592]);self.epoch='changed'
  self.assertEqual([],self.engine.search(request(query='namaz ki talqeen'))['results'])

 def test_named_speaker_is_separate_from_action_evidence(self):
  self.base.understand=lambda q:{'semantic_query':'young people should pray','query_type':'speaker_search','extracted_speaker':'Hafiz Naeem Ur Rehman','query_decomposition':{'decomposed':True,'topic':'young people should pray'}}
  self.base.speaker_values=lambda q:['Hafiz Naeem Rehman']
  for i,speaker in enumerate(['Hafiz Naeem Rehman','SPEAKER_02','Hafiz Ali'],1):
   self.put(i,'University tuition costs 100000 rupees. Families cannot afford it',video=i,speaker=speaker,identity_version='speaker_identity_v2',identity_status='verified',identity_source='manual',identity_key='hafiz_naeem_ur_rehman'if i==1 else 'fixture',identity_review={'review_id':'fixture','reviewer_id':'fixture','reviewed_at':'2026-10-20','evidence_sha256':'a'*64})
  seen=[];judge=self.engine.judge
  def recording(query,rows,timeout):
   seen.append(query);return judge(query,rows,timeout)
  self.engine.judge=recording
  result=self.engine.search(request(query='Hafiz Naeem young people should pray'))
  self.assertEqual(['young people should pray'],seen)
  self.assertEqual([1],[r['video_id']for r in result['results']])
  self.assertEqual('supported_identity',result['results'][0]['speaker_identity_status'])
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
 def test_adjacent_indexed_unknown_turns_supply_context_without_identity_claim(self):
  from grounded_search import coherent_passage
  rows=[{'id':str(i),'segment_index':i,'video_id':1,'speaker':'Unknown Speaker','source_identity_group':'same-source-group','text':str(i),'start_time':i*15,'end_time':(i+1)*15}for i in range(1,4)]
  result=coherent_passage(rows[1],rows)
  self.assertEqual(['1','2','3'],result['segment_ids']);self.assertEqual('Unknown Speaker',result['speaker'])
  self.assertEqual([1,2,3],[r['segment_index']for r in result['source_utterances']])
  rows[0]['segment_index']=99
  self.assertNotIn('1',coherent_passage(rows[1],rows)['segment_ids'])
  rows[0]['segment_index']=1;rows[2]['source_identity_group']='different-source-group'
  self.assertNotIn('3',coherent_passage(rows[1],rows)['segment_ids'])
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

 def test_ai_summary_only_match_is_not_spoken_evidence(self):
  self.put(301,'Unrelated transcript',video=301,video_summary_english='University tuition costs are expensive.')
  result=self.engine.search(request(query='tuition costs'))
  self.assertEqual(1,result['returned']);row=result['results'][0]
  self.assertEqual('summary',row['evidence_source']);self.assertTrue(row['is_video_only'])
  self.assertFalse(row['transcript_verified']);self.assertEqual('',row['text'])
  self.assertNotIn('passage_evidence',row);self.assertNotIn('start_time',row)
  self.assertEqual(['video_summary_english'],row['matched_summary_fields'])
  self.assertTrue(row['summary_evidence'][0]['excerpts'])
  self.assertIn('video_summary',result['ranking']['candidate_sources'])
 def test_ai_all_three_sources_merge_without_duplicate_video(self):
  self.put(302,'University tuition costs 100000 rupees. Families cannot afford it',video=302,
           video_title='Education fees',video_summary='Education fees are unaffordable.')
  result=self.engine.search(request(query='Education fees'))
  self.assertEqual(1,result['returned']);row=result['results'][0]
  self.assertEqual('title_and_summary_and_transcript',row['evidence_source'])
  self.assertTrue(row['transcript_verified']);self.assertTrue(row['passage_evidence'])
  self.assertEqual({'text','video_title','video_summary'},set(row['matched_fields']))
 def test_ai_summary_scope_eligibility_and_repeated_segments(self):
  for i in range(303,307):
   self.put(i,'Other',video=i,video_summary_urdu='تعلیم کے اخراجات',approval_status='pending'if i==305 else 'approved',is_archived=i==306)
  self.put(307,'Other',video=303,start=10,end=15,video_summary_urdu='تعلیم کے اخراجات')
  result=self.engine.search(request(query='تعلیم',allowed_video_ids=[303,305,306]))
  self.assertEqual([303],[r['video_id']for r in result['results']])
  self.assertEqual(1,len(result['results'][0]['summary_evidence']))
 def test_ai_name_alias_in_summary_needs_no_model_and_keeps_topic(self):
  self.put(308,'Other',video=308,video_summary='Hafiz Naeem Rehman discusses education.')
  self.engine.title_intent=lambda q:{'queries':['HNR','Hafiz Naeem Rehman'],'person_only':True}
  def unavailable(*args):raise AssertionError('No model for pure-name lookup')
  self.base.embed=unavailable;self.engine.judge=unavailable
  result=self.engine.search(request(query='HNR'))
  self.assertEqual([308],[r['video_id']for r in result['results']])
  self.assertEqual(0,result['metadata']['passage_validation']['provider_calls'])
 def test_ai_summary_partial_index_keeps_legacy_field_fallback(self):
  from grounded_summary_search import summary_candidates
  self.c.create_payload_index('source','video_summary_english',models.TextIndexParams(type='text',tokenizer='word',lowercase=True))
  self.put(309,'Other',video=309,video_summary_english='Education matters.')
  self.put(310,'Other',video=310,summary_ur='Education matters.')
  rows,metadata=summary_candidates(self.base,request(query='education'),['education'],indexed_fields=['video_summary_english'])
  self.assertEqual({309,310},{r['video_id']for r in rows})
  self.assertTrue(metadata['exhaustive']);self.assertIn('summary_ur',metadata['searched_fields'])
 def test_ai_summary_does_not_drop_terms_or_borrow_across_fields(self):
  self.put(311,'Other',video=311,video_summary='Education',video_summary_english='fees')
  result=self.engine.search(request(query='education fees'))
  self.assertEqual([],result['results'])
 def test_ai_separate_summary_category_still_requires_transcript_evidence(self):
  from grounded_search import summary_search
  self.put(312,'Unrelated transcript',video=312,video_summary='Education fees')
  q=request(query='education fees',search_mode='simple',filter_type='summary',summary_candidates=[{'video_id':312,'matched_summary_fields':['video_summary']}])
  result=summary_search(self.engine,q)
  self.assertEqual([],result['results'])

 def test_short_fragment_recovers_separate_complete_turn_without_inventing_identity(self):
  from grounded_search import contextual_candidate
  focus={'id':'focus','video_id':920,'text':'نماز ادا کریں','speaker':'Unknown Speaker','source_identity_group':'one','start_time':983.9,'end_time':986.0}
  neighbor={'id':'complete','video_id':920,'text':'ہم نے بچوں کو مسجد میں نماز باجماعت پڑھنے کی تلقین کی','speaker':'Unknown Speaker','source_identity_group':'two','start_time':986.0,'end_time':1001.3}
  result=contextual_candidate(focus,[focus,neighbor])
  self.assertEqual('complete',result['id']);self.assertEqual(['complete'],result['segment_ids'])
  self.assertEqual(neighbor['text'],result['text']);self.assertEqual(986.0,result['start_time'])
  self.assertEqual('focus',result['context_retrieval_anchor']['point_id'])
  self.assertEqual('Unknown Speaker',result['speaker'])
 def test_short_fragment_does_not_recover_other_named_actor_or_distant_turn(self):
  from grounded_search import contextual_candidate
  from transcript_search_contract import normalize
  focus={'id':'focus','video_id':1,'text':'Pray regularly','speaker':'Hafiz Naeem','start_time':10,'end_time':12}
  other={'id':'other','video_id':1,'text':'Children should perform prayers regularly every day at home','speaker':'Another person','start_time':12,'end_time':18}
  self.assertEqual('focus',contextual_candidate(focus,[focus,other],{normalize('Hafiz Naeem')})['id'])
  far={**other,'start_time':40,'end_time':50}
  self.assertEqual('focus',contextual_candidate(focus,[focus,far])['id'])
 def test_short_fragment_recovery_respects_eligibility_and_uses_real_quote(self):
  self.put(801,'Brief fragment',video=777,start=10,end=12,speaker='A')
  self.put(802,'University tuition costs 100000 rupees. Families cannot afford it',video=777,start=12,end=20,speaker='B')
  self.c.update_vectors('source',[models.PointVectors(id=802,vector=[-1.,0.])])
  self.base.settings=FastSearchSettings(candidate_limit=100,minimum_dense_score=.7)
  result=self.engine.search(request(query='education fees',allowed_video_ids=[777]))
  self.assertEqual(1,result['returned']);row=result['results'][0]
  self.assertEqual('802',str(row['id']));self.assertEqual(['802'],row['segment_ids'])
  self.assertTrue(row['passage_evidence']);self.assertEqual('801',row['context_retrieval_anchor']['point_id'])
  self.assertTrue(any(t['stage']=='nearby_original_turn'for t in result['ranking']['candidate_trace']))
  self.c.set_payload('source',{'approval_status':'pending'},points=[802]);self.epoch='next'
  self.assertEqual([],self.engine.search(request(query='education fees',allowed_video_ids=[777]))['results'])
