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
