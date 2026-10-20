import unittest
from speaker_identity import *
from index_operations import payload_hash
from qdrant_client import QdrantClient,models
class IdentityTests(unittest.TestCase):
 def manual(self):return {'speaker':'Hafiz Naeem Rehman','identity_key':'hafiz_naeem_ur_rehman','identity_version':VERSION,'identity_status':'verified','identity_source':'manual','identity_review':{'review_id':'one','reviewer_id':'operator','reviewed_at':'2026-10-20','evidence_sha256':'a'*64}}
 def setUp(self):
  self.c=QdrantClient(':memory:');self.c.create_collection('s',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE));self.c.upsert('s',[models.PointStruct(id=1,vector=[1.,0.],payload={'video_id':1079,'speaker':'ELON MUSK','transcript_speaker':'A','text':'source stays intact','start_time':1.,'end_time':2.,'timestamp_unit':'seconds','indexing_run_id':'legacy','processing_status':'completed','approval_status':'approved','is_archived':False})])
 def change(self):
  p=self.c.retrieve('s',[1])[0];return {'point_id':1,'expected_payload_sha256':payload_hash(p.payload),'fields':{'identity_version':VERSION,'identity_status':'unknown','identity_source':'manual','identity_name':UNKNOWN,'identity_key':None,'identity_revision':'review-one'}}
 def test_legacy_and_unreviewed_names_are_unknown_but_reviewed_alias_is_canonical(self):
  for r in [{'speaker':'ELON MUSK'},{'speaker':'Hafiz Naeem Rehman','identity_source':'manual'},dict(self.manual(),identity_review={})]:self.assertEqual(UNKNOWN,view(r)['speaker'])
  self.assertEqual('Hafiz Naeem Ur Rehman',view(self.manual())['speaker'])
 def test_auto_requires_enrollment_calibration_and_run_proof(self):
  r=dict(self.manual(),identity_status='accepted',identity_source='audio');self.assertFalse(supported(r));r.update(identity_enrollment_fingerprint='a'*64,identity_calibration_sha256='b'*64,identity_run_id='one');self.assertTrue(supported(r))
 def test_quarantine_idempotence_and_rollback_preserve_vectors_text_and_labels(self):
  old=self.c.retrieve('s',[1],with_vectors=True)[0];change=self.change();dry=patch_identity(self.c,'s',1079,[change]);change['post_payload_sha256']=dry['after_hashes']['1'];patch_identity(self.c,'s',1079,[change],dry_run=False);self.assertEqual(0,patch_identity(self.c,'s',1079,[change],dry_run=False)['updated_points']);p=self.c.retrieve('s',[1],with_vectors=True)[0];self.assertEqual(old.vector,p.vector);self.assertEqual(UNKNOWN,view(p.payload)['speaker']);self.assertEqual('ELON MUSK',p.payload['speaker']);self.assertEqual('source stays intact',p.payload['text']);self.assertEqual('A',p.payload['transcript_speaker']);back={'point_id':1,'expected_payload_sha256':payload_hash(p.payload),'fields':{},'delete_fields':FIELDS,'post_payload_sha256':payload_hash(old.payload)};patch_identity(self.c,'s',1079,[back],rollback=True,dry_run=False);self.assertEqual(old.payload,self.c.retrieve('s',[1])[0].payload)
 def test_changed_run_rejects_before_any_write(self):
  change=self.change();self.c.set_payload('s',{'indexing_run_id':'newer'},[1]);
  with self.assertRaises(ValueError):patch_identity(self.c,'s',1079,[change],dry_run=False)
  self.assertNotIn('identity_version',self.c.retrieve('s',[1])[0].payload)
 def test_null_identity_key_is_explicitly_removed_for_remote_store_parity(self):
  self.c.set_payload('s',{'identity_key':'unreviewed-old-key'},[1]);change=self.change()
  dry=patch_identity(self.c,'s',1079,[change]);change['post_payload_sha256']=dry['after_hashes']['1']
  result=patch_identity(self.c,'s',1079,[change],dry_run=False)
  actual=self.c.retrieve('s',[1])[0].payload
  self.assertNotIn('identity_key',actual);self.assertEqual(payload_hash(actual),result['after_hashes']['1'])
 def test_text_edits_and_unproven_named_replacement_are_refused(self):
  change=self.change();change['fields']={'text':'forged'}
  with self.assertRaises(ValueError):patch_identity(self.c,'s',1079,[change],dry_run=False)
  change=self.change();change['fields']={**self.manual(),'identity_name':'Hafiz Naeem Ur Rehman'};change['fields'].pop('speaker')
  with self.assertRaises(ValueError):patch_identity(self.c,'s',1079,[change],dry_run=False)
 def test_rollback_cannot_overwrite_newer_correction(self):
  change=self.change();result=patch_identity(self.c,'s',1079,[change],dry_run=False);self.c.set_payload('s',{'identity_revision':'newer'},[1]);back={'point_id':1,'expected_payload_sha256':result['after_hashes']['1'],'fields':{},'delete_fields':FIELDS}
  with self.assertRaises(ValueError):patch_identity(self.c,'s',1079,[back],rollback=True,dry_run=False)
 def test_unknown_is_distinct_filter_with_authorized_scope(self):
  from test_fast_search import request
  q=request(speaker='Unknown Speaker',max_scanned=100,top_k=20,allowed_video_ids=[1079]);self.assertEqual(1,speaker_search(self.c,'s',q,lambda a,b:a in b)['returned']);q.speaker='ELON MUSK';self.assertEqual(0,speaker_search(self.c,'s',q,lambda a,b:a in b)['returned']);q.speaker='Unknown Speaker';q.allowed_video_ids=[999];self.assertEqual(0,speaker_search(self.c,'s',q,lambda a,b:a in b)['returned'])

class IdentityAliasTests(unittest.TestCase):
 def test_shared_first_name_does_not_merge_people(self):
  hf={'speaker':'Hafiz Naeem Ur Rehman','identity_key':'hafiz_naeem_ur_rehman','identity_version':VERSION,'identity_status':'accepted','identity_source':'audio','identity_enrollment_fingerprint':'a'*64,'identity_calibration_sha256':'b'*64,'identity_run_id':'fixture'}
  other={**hf,'speaker':'Hafiz Ali','identity_key':'another-person'}
  self.assertTrue(matches_identity('HNR',hf));self.assertFalse(matches_identity('Hafiz Ali',hf));self.assertFalse(matches_identity('Hafiz Naeem',other));self.assertTrue(matches_identity('Hafiz Ali',other))
 def test_unknown_names_preserve_separate_context_groups(self):
  a=view({'video_id':1,'speaker':'old A'});b=view({'video_id':1,'speaker':'old B'});self.assertEqual(UNKNOWN,a['speaker']);self.assertEqual(UNKNOWN,b['speaker']);self.assertNotEqual(a['source_identity_group'],b['source_identity_group']);self.assertEqual(a['source_identity_group'],view(a)['source_identity_group'])
