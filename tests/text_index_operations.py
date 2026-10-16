import pathlib,sys,unittest
from types import SimpleNamespace as NS
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from index_operations import payload_hash,patch_timestamps,IndexOperationError
class Client:
 def __init__(self):self.payload={'video_id':1,'text':'source','start_time':3000,'end_time':6000,'duration':3000};self.vector=[.1,.2];self.fail=False
 def retrieve(self,**kw):return [NS(id=1,payload=dict(self.payload))]
 def set_payload(self,**kw):
  self.payload.update(kw['payload'])
  if self.fail:self.fail=False;raise TimeoutError()
 def delete_payload(self,**kw):
  for k in kw['keys']:self.payload.pop(k,None)
def change(c):return [{'point_id':1,'expected_payload_sha256':payload_hash(c.payload),'payload':{'start_time':3.,'end_time':6.,'duration':3.,'timestamp_unit':'seconds'}}]
class IndexOperationsTest(unittest.TestCase):
 def test_dry_run_never_changes_index(self):
  c=Client();before=dict(c.payload);patch_timestamps(c,'fixture',1,change(c));self.assertEqual(before,c.payload)
 def test_payload_repair_preserves_vectors(self):
  c=Client();patch_timestamps(c,'fixture',1,change(c),dry_run=False);self.assertEqual(3,c.payload['start_time']);self.assertEqual([.1,.2],c.vector)
 def test_source_changed_is_rejected(self):
  c=Client();v=change(c);c.payload['text']='changed'
  with self.assertRaises(IndexOperationError):patch_timestamps(c,'fixture',1,v,dry_run=False)
 def test_wrong_video_is_rejected(self):
  c=Client()
  with self.assertRaises(IndexOperationError):patch_timestamps(c,'fixture',2,change(c),dry_run=False)
 def test_uncertain_write_is_rolled_back(self):
  c=Client();before=dict(c.payload);c.fail=True
  with self.assertRaises(IndexOperationError):patch_timestamps(c,'fixture',1,change(c),dry_run=False)
  self.assertEqual(before,c.payload)
