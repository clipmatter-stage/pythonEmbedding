import pathlib,sys,unittest,copy
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from qdrant_client import QdrantClient,models
from exact_lexical import derived_fields,plan_video,apply_video,KEYS
from exact_search_cursor import exact_page
from test_exact_cursor import Redis
class LexicalTests(unittest.TestCase):
 def setUp(self):
  self.c=QdrantClient(':memory:');self.c.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
  self.c.upsert('source',[models.PointStruct(id=i,vector=[.1,.2],payload={'video_id':1,'text':'I.P.P. کے الیکٹرک'if i==2601 else 'other text','start_time':i,'end_time':i+1,'processing_status':'completed','approval_status':'approved','is_archived':False})for i in range(1,2603)])
 def tearDown(self):self.c.close()
 def test_checkpoint_backfill_preserves_vectors_text_and_timestamps_and_rolls_back(self):
  before=self.c.retrieve('source',[2601],with_vectors=True)[0];plan=plan_video(self.c,'source',1,models);self.assertTrue(apply_video(self.c,'source',plan,models)['verified']);apply_video(self.c,'source',plan,models)
  after=self.c.retrieve('source',[2601],with_vectors=True)[0];self.assertEqual(before.vector,after.vector);self.assertEqual(before.payload,{k:v for k,v in after.payload.items()if k not in KEYS});self.assertEqual([],plan_video(self.c,'source',1,models)['items']);apply_video(self.c,'source',plan,models,True);self.assertEqual(before.payload,self.c.retrieve('source',[2601])[0].payload)
 def test_changed_source_or_nonlexical_rollback_is_rejected(self):
  plan=plan_video(self.c,'source',1,models);self.c.set_payload('source',{'text':'changed'},[1])
  with self.assertRaises(ValueError):apply_video(self.c,'source',plan,models)
 def test_index_prefilter_keeps_complete_occurrences_beyond_2400(self):
  plan=plan_video(self.c,'source',1,models);apply_video(self.c,'source',plan,models)
  info=self.c.get_collection('source');info.payload_schema={k:NS(data_type='keyword')for k in ['exact_terms_version','exact_normalized_terms_v1','exact_alias_terms_v1','exact_literal_terms_v1']}
  with patch.object(self.c,'get_collection',return_value=info):
   for query,contract in [('IPP','normalized'),('k electric','alias'),('I.P.P.','literal'),('.P.','literal')]:
    response=exact_page({'query':query,'transcript_contract':contract,'allowed_video_ids':[1]},self.c,Redis(),'source',models);self.assertTrue(response['search_contract']['lexical_index_used']);self.assertEqual(1,response['search_contract']['scanned_segments']);self.assertEqual(1,response['search_contract']['counts']['matched_transcript_occurrences']);self.assertTrue(response['search_contract']['counts_final'])
 def test_missing_coverage_falls_back_without_losing_matches(self):
  self.c.set_payload('source',derived_fields('other text'),[1]);info=self.c.get_collection('source');info.payload_schema={k:NS(data_type='keyword')for k in ['exact_terms_version','exact_normalized_terms_v1','exact_alias_terms_v1','exact_literal_terms_v1']}
  with patch.object(self.c,'get_collection',return_value=info):
   r=exact_page({'query':'IPP','transcript_contract':'normalized','allowed_video_ids':[1]},self.c,Redis(),'source',models);self.assertFalse(r['search_contract']['lexical_index_used']);self.assertEqual('partial',r['search_contract']['status']);self.assertEqual(1000,r['search_contract']['scanned_segments'])

 def test_acronym_family_exact_page_counts_surfaces_and_skips_substrings(self):
  from transcript_search_contract import normalize
  self.assertEqual(['ipp'],derived_fields('آئی پی پی')['exact_normalized_terms_v1'])
  self.assertEqual(['imf'],derived_fields('آئی ایم ایف')['exact_normalized_terms_v1'])
  self.assertNotIn('ipp',derived_fields('IPProvider')['exact_normalized_terms_v1'])
  client=QdrantClient(':memory:');client.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
  rows=[(1,'IPP and I.P.P. and I P P and آئی پی پی and IPProvider'),(2,'IMF I.M.F. I M F آئی ایم ایف'),(3,'shipping skipper IPProvider confirmation IMForce')]
  client.upsert('source',[models.PointStruct(id=i,vector=[.1,.2],payload={'video_id':1,'text':text,'start_time':i,'end_time':i+1,'processing_status':'completed','approval_status':'approved','is_archived':False,**derived_fields(text)}) for i,text in rows])
  info=client.get_collection('source');info.payload_schema={k:NS(data_type='keyword')for k in ['exact_terms_version','exact_normalized_terms_v1','exact_alias_terms_v1','exact_literal_terms_v1']}
  try:
   with patch.object(client,'get_collection',return_value=info):
    for query in ['IPP','I.P.P.','I P P','آئی پی پی']:
     response=exact_page({'query':query,'transcript_contract':'normalized','allowed_video_ids':[1]},client,Redis(),'source',models)
     self.assertTrue(response['search_contract']['lexical_index_used'],query)
     self.assertEqual(4,response['search_contract']['counts']['matched_transcript_occurrences'],query)
     self.assertEqual(['1'],[row['id'] for row in response['results']])
     for row in response['results']:
      for span in row['match_spans']:
       self.assertEqual(span['text'],row['text'][span['start']:span['end']])
       self.assertEqual('ipp',normalize(span['text']))
       self.assertNotIn('IPProvider',span['text'])
    for query in ['IMF','I.M.F.','I M F','آئی ایم ایف']:
     response=exact_page({'query':query,'transcript_contract':'normalized','allowed_video_ids':[1]},client,Redis(),'source',models)
     self.assertEqual(4,response['search_contract']['counts']['matched_transcript_occurrences'],query)
     self.assertEqual(['2'],[row['id'] for row in response['results']])
     for span in response['results'][0]['match_spans']:
      self.assertEqual('imf',normalize(span['text']))
      self.assertEqual(span['text'],response['results'][0]['text'][span['start']:span['end']])
  finally:client.close()

 def test_managed_text_edit_refreshes_tokens_and_invalidates_cursors(self):
  from exact_search_cursor import RevisionedClient
  client=RevisionedClient(self.c,Redis());client.set_payload('source',{'text':'New topic IMF'},[1]);row=self.c.retrieve('source',[1])[0];self.assertIn('imf',row.payload['exact_normalized_terms_v1']);self.assertEqual(derived_fields(row.payload['text'])['exact_source_sha256'],row.payload['exact_source_sha256'])
