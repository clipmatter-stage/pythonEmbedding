import pathlib,sys,unittest,hmac,hashlib
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from callback_transport import post_callback,worker_canary,deliver,issue_callback_token,attach_callback_token
class CallbackTests(unittest.TestCase):
 def test_transport_sends_authenticated_post_and_does_not_follow_cross_origin_redirect(self):
  with patch('callback_transport.requests.post',return_value=NS(status_code=200)) as send:
   post_callback('https://clip.digitalmatters.pk/api/embeddings/callback',{'run_id':'fixture'},'fixture-key');kw=send.call_args.kwargs;self.assertEqual('fixture-key',kw['headers']['X-API-Key']);self.assertFalse(kw['allow_redirects']);self.assertEqual('fixture',kw['json']['run_id'])
  with patch('callback_transport.requests.post',return_value=NS(status_code=302,headers={'Location':'https://untrusted.invalid/'})) as send:
   with self.assertRaises(ValueError):post_callback('https://clip.digitalmatters.pk/api/embeddings/callback',{},'fixture-key')
   self.assertEqual(1,send.call_count)
 def test_actual_worker_canary_returns_no_credentials_and_no_provider_calls(self):
  nonce='a'*32;proof=hmac.new(b'fixture-key',nonce.encode(),hashlib.sha256).hexdigest();data={'challenge':nonce,'expected_signature':proof,'callback_url':'https://clip.digitalmatters.pk/api/embeddings/callback/probe'}
  with patch('callback_transport.requests.post',return_value=NS(status_code=200,json=lambda:{'authenticated':True,'header_present':True,'body_proof_matches':True})):
   result=worker_canary(data,'fixture-key','fixture');self.assertTrue(result['configured_key_matches_expected']);self.assertEqual(200,result['http']);self.assertEqual(0,result['provider_calls']);self.assertNotIn('fixture-key',str(result))
  with patch('callback_transport.requests.post',return_value=NS(status_code=401,json=lambda:{'authenticated':False,'header_present':False})):
   result=worker_canary(data,'wrong','fixture');self.assertFalse(result['configured_key_matches_expected']);self.assertEqual(401,result['http'])
 def test_delivery_auth_rejection_is_not_provider_failure_or_silent_success(self):
  class DeliveryError(Exception):pass
  with patch('callback_transport.requests.post',return_value=NS(status_code=401)),patch('callback_transport.time.sleep') as sleep:
   with self.assertRaises(DeliveryError):deliver('https://clip.digitalmatters.pk/api/embeddings/callback',{},'wrong',DeliveryError)
   sleep.assert_not_called()
 def test_api_issues_narrow_token_without_master_key_in_job(self):
  data=attach_callback_token({'video_id':1,'run_id':'fixture','webhook_url':'https://clip.digitalmatters.pk/api/embeddings/callback'},'fixture-key');self.assertNotIn('fixture-key',str(data))
  with patch('callback_transport.requests.post',return_value=NS(status_code=200)) as send:
   post_callback(data['webhook_url'],{'video_id':1,'run_id':'fixture'},'different-worker-key',data['_callback_token']);self.assertTrue(send.call_args.kwargs['headers']['Authorization'].startswith('Bearer '));self.assertNotIn('X-API-Key',send.call_args.kwargs['headers'])
if __name__=='__main__':unittest.main()
