import os,unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient,models
from test_exact_cursor import Redis
from index_operations import register_index_operations,payload_hash
class IdentityApiTests(unittest.TestCase):
 def setUp(self):
  self.c=QdrantClient(':memory:');self.c.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE));self.c.upsert('source',[models.PointStruct(id=1,vector=[1.,0.],payload={'video_id':1079,'text':'fixture','speaker':'ELON MUSK'})]);a=FastAPI();register_index_operations(a,self.c,'source',Redis(),'fixture-key');self.api=TestClient(a);self.headers={'X-API-Key':'fixture-key'}
 def test_capability_exposes_fast_misconfiguration_and_observed_grounded_mode(self):
  self.assertEqual(401,self.api.get('/operations/search-capabilities').status_code)
  with patch.dict(os.environ,{'SEARCH_PIPELINE':'fast'}):
   d=self.api.get('/operations/search-capabilities',headers=self.headers).json();self.assertEqual('fast',d['pipeline']);self.assertFalse(d['grounded_configuration_ok']);self.assertEqual('speaker_identity_v2',d['speaker_identity_version'])
  with patch.dict(os.environ,{'SEARCH_PIPELINE':'grounded'}):self.assertTrue(self.api.get('/operations/search-capabilities',headers=self.headers).json()['grounded_configuration_ok'])
 def test_metadata_endpoint_requires_auth_and_never_allows_source_replacement(self):
  p=self.c.retrieve('source',[1])[0];body={'video_id':1079,'changes':[{'point_id':1,'expected_payload_sha256':payload_hash(p.payload),'fields':{'text':'forged'}}],'dry_run':False}
  self.assertEqual(401,self.api.post('/operations/patch-speaker-identities',json=body).status_code);self.assertEqual(409,self.api.post('/operations/patch-speaker-identities',json=body,headers=self.headers).status_code);self.assertEqual('fixture',self.c.retrieve('source',[1])[0].payload['text'])
