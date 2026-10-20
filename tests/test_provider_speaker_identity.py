import unittest
from speaker_identity import supported, view, unknown_label
class ProviderIdentityTests(unittest.TestCase):
 def test_current_provider_match_does_not_forge_calibration(self):
  row={'identity_version':'speaker_identity_v2','identity_status':'accepted','identity_source':'provider','identity_acceptance_policy':'pyannote_match_v1','identity_key':'person','identity_enrollment_fingerprint':'enrollment','identity_run_id':'run','identity_name':'Reviewed Person'}
  self.assertTrue(supported(row));self.assertEqual('Reviewed Person',view(row)['speaker']);self.assertNotIn('identity_calibration_sha256',row)
  self.assertFalse(supported({**row,'identity_source':'legacy'}))
 def test_unknown_groups_are_display_only(self):
  self.assertEqual('Unknown A',view({'speaker':'legacy person','transcript_speaker':'A'})['speaker_display_name']);self.assertEqual('Unknown B',unknown_label({'transcript_speaker':'SPEAKER_01'}));self.assertEqual('Unknown AA',unknown_label({'transcript_speaker':'SPEAKER_26'}));self.assertEqual('Unknown Speaker',view({'transcript_speaker':'B'})['speaker'])

 def test_provider_metadata_patch_keeps_vectors_and_requires_compatible_index(self):
  from qdrant_client import QdrantClient,models
  from speaker_identity import patch_identity
  from index_operations import payload_hash
  client=QdrantClient(':memory:');client.create_collection('fixture',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
  payload={'video_id':1,'speaker':'old prediction','text':'source text','embedding_input_contract':'transcript_context_text_only_v1'}
  client.upsert('fixture',[models.PointStruct(id=1,vector=[0.2,0.4],payload=payload)])
  baseline=client.retrieve('fixture',[1],with_vectors=True)[0]
  fields={'identity_version':'speaker_identity_v2','identity_status':'accepted','identity_source':'provider','identity_acceptance_policy':'pyannote_match_v1','identity_key':'person','identity_enrollment_fingerprint':'enrollment','identity_run_id':'run','identity_name':'Reviewed Person'}
  result=patch_identity(client,'fixture',1,[{'point_id':1,'expected_payload_sha256':payload_hash(baseline.payload),'fields':fields}],dry_run=False)
  after=client.retrieve('fixture',[1],with_vectors=True)[0]
  self.assertTrue(result['verified']);self.assertEqual(baseline.vector,after.vector);self.assertEqual('source text',after.payload['text']);self.assertTrue(supported(after.payload));client.close()
