import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi import FastAPI,HTTPException,Request
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient,models
from exact_lexical import register_operations,digest,KEYS
from test_exact_cursor import Redis
class LexicalApiTests(unittest.TestCase):
 def setUp(self):
  self.db=QdrantClient(':memory:');self.db.create_collection('source',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE));self.db.upsert('source',[models.PointStruct(id=1,vector=[1.,0.],payload={'video_id':7,'text':'آئی پی پی','start_time':1,'end_time':5,'video_summary':'large summary remains untouched'})]);app=FastAPI()
  async def auth(request:Request):
   if request.headers.get('x-api-key')!='fixture':raise HTTPException(401)
  register_operations(app,self.db,Redis(),'source',models,auth);self.client=TestClient(app);self.headers={'X-API-Key':'fixture'}
 def tearDown(self):self.client.close();self.db.close()
 def test_authorization_guarded_apply_and_rollback_roundtrip(self):
  self.assertEqual(401,self.client.get('/operations/exact-lexical-plan?video_id=7').status_code)
  plan=self.client.get('/operations/exact-lexical-plan?video_id=7',headers=self.headers).json();self.assertNotIn('text',plan['items'][0]);self.assertEqual('exact_payload_guard_v1',plan['guard_version'])
  r=self.client.post('/operations/exact-lexical-apply',headers=self.headers,json={'checkpoint':plan});self.assertEqual(200,r.status_code);self.assertTrue(r.json()['verified']);self.assertEqual(0,r.json()['vectors_written']);self.assertEqual(0,self.client.post('/operations/exact-lexical-apply',headers=self.headers,json={'checkpoint':plan}).json()['updated_points'])
  r=self.client.post('/operations/exact-lexical-apply',headers=self.headers,json={'checkpoint':plan,'rollback':True});self.assertEqual(200,r.status_code);point=self.db.retrieve('source',[1],with_vectors=True)[0];self.assertEqual([1.,0.],point.vector);self.assertEqual('large summary remains untouched',point.payload['video_summary']);self.assertFalse(set(KEYS)&set(point.payload))
 def test_new_source_blocks_old_checkpoint(self):
  plan=self.client.get('/operations/exact-lexical-plan?video_id=7',headers=self.headers).json();self.db.set_payload('source',{'text':'Newer source'},[1]);self.assertEqual(409,self.client.post('/operations/exact-lexical-apply',headers=self.headers,json={'checkpoint':plan}).status_code);self.assertEqual('Newer source',self.db.retrieve('source',[1])[0].payload['text'])
 def test_rollback_cannot_write_source_fields(self):
  plan=self.client.get('/operations/exact-lexical-plan?video_id=7',headers=self.headers).json();plan['items'][0]['before_fields']['text']='forged';plan['plan_sha256']=digest(plan['items']);r=self.client.post('/operations/exact-lexical-apply',headers=self.headers,json={'checkpoint':plan,'rollback':True});self.assertEqual(409,r.status_code);self.assertEqual('آئی پی پی',self.db.retrieve('source',[1])[0].payload['text'])
