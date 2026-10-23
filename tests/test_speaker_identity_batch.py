"""Regression tests for bounded, payload-only speaker batching."""
import unittest
from unittest.mock import Mock
from qdrant_client import QdrantClient, models
from index_operations import payload_hash
from speaker_identity import patch_identity, VERSION, UNKNOWN


class IdentityBatchTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(':memory:')
        self.client.create_collection('source', vectors_config=models.VectorParams(
            size=2, distance=models.Distance.COSINE))
        self.client.upsert('source', [models.PointStruct(
            id=i, vector=[1., 0.], payload={
                'video_id': 10, 'text': f'Segment {i}', 'speaker': 'A',
                'identity_key': 'old', 'start_time': i, 'end_time': i + 1,
            }) for i in range(1, 51)])
        self.before = {p.id: p for p in self.client.retrieve(
            'source', list(range(1, 51)), with_vectors=True)}
        self.changes = [{
            'point_id': i, 'expected_payload_sha256': payload_hash(p.payload),
            'fields': {'identity_version': VERSION, 'identity_status': 'unknown',
                       'identity_name': UNKNOWN, 'identity_key': None},
        } for i, p in self.before.items()]
        self.spy = Mock(wraps=self.client)

    def tearDown(self):
        self.client.close()

    def plan(self):
        dry = patch_identity(self.spy, 'source', 10, self.changes)
        for change in self.changes:
            change['post_payload_sha256'] = dry['after_hashes'][str(change['point_id'])]
        return dry

    def test_fifty_points_use_one_write_request_and_resume_without_writes(self):
        self.plan()
        self.spy.batch_update_points.assert_not_called()
        result = patch_identity(self.spy, 'source', 10, self.changes, dry_run=False)
        self.assertTrue(result['verified'])
        self.spy.batch_update_points.assert_called_once()
        operations = self.spy.batch_update_points.call_args.kwargs['update_operations']
        self.assertEqual(100, len(operations))
        self.spy.set_payload.assert_not_called()
        self.spy.delete_payload.assert_not_called()
        for point in self.client.retrieve('source', list(self.before), with_vectors=True):
            self.assertEqual(self.before[point.id].vector, point.vector)
            self.assertEqual(self.before[point.id].payload['text'], point.payload['text'])
            self.assertEqual('A', point.payload['speaker'])
            self.assertNotIn('identity_key', point.payload)
        self.assertEqual(0, patch_identity(
            self.spy, 'source', 10, self.changes, dry_run=False)['updated_points'])
        self.spy.batch_update_points.assert_called_once()

    def test_remote_transport_uses_one_batch_write_between_two_reads(self):
        import copy
        import json
        import httpx
        payloads = {i: copy.deepcopy(p.payload) for i, p in self.before.items()}
        requests = []
        def respond(request):
            requests.append(request)
            body = json.loads(request.content)
            if request.url.path.endswith('/points/batch'):
                self.assertEqual('true', request.url.params['wait'])
                self.assertEqual(100, len(body['operations']))
                for operation in body['operations']:
                    if 'set_payload' in operation:
                        op = operation['set_payload']
                        for point_id in op['points']:
                            payloads[point_id].update(op['payload'])
                    else:
                        op = operation['delete_payload']
                        for point_id in op['points']:
                            for key in op['keys']:
                                payloads[point_id].pop(key, None)
                result = [{'operation_id': i, 'status': 'completed'} for i in range(100)]
            else:
                result = [{'id': i, 'payload': payloads[i]} for i in body['ids']]
            return httpx.Response(200, json={'result': result, 'status': 'ok', 'time': 0.01})
        remote = QdrantClient(url='http://qdrant.test:6333', check_compatibility=False,
                              transport=httpx.MockTransport(respond))
        try:
            result = patch_identity(remote, 'source', 10, self.changes, dry_run=False)
            self.assertTrue(result['verified'])
            self.assertEqual(3, len(requests))
        finally:
            remote.close()

    def test_stale_final_point_rejects_entire_request_before_any_write(self):
        self.changes[-1]['expected_payload_sha256'] = 'stale'
        with self.assertRaisesRegex(ValueError, 'Source/run/payload changed'):
            patch_identity(self.spy, 'source', 10, self.changes, dry_run=False)
        self.spy.batch_update_points.assert_not_called()
        self.assertEqual('old', self.client.retrieve('source', [1])[0].payload['identity_key'])

    def test_partial_batch_failure_can_resume_with_saved_post_hashes(self):
        self.plan()
        def interrupted(**kwargs):
            kwargs['update_operations'] = kwargs['update_operations'][:2]
            self.client.batch_update_points(**kwargs)
            raise TimeoutError('response lost')
        self.spy.batch_update_points.side_effect = interrupted
        with self.assertRaises(TimeoutError):
            patch_identity(self.spy, 'source', 10, self.changes, dry_run=False)
        self.spy.batch_update_points.side_effect = None
        result = patch_identity(self.spy, 'source', 10, self.changes, dry_run=False)
        self.assertEqual(49, result['updated_points'])
        self.assertTrue(result['verified'])

    def test_verification_rejects_acknowledged_but_missing_writes(self):
        self.plan()
        self.spy.batch_update_points.side_effect = lambda **kwargs: []
        with self.assertRaisesRegex(ValueError, 'Uncertain identity write'):
            patch_identity(self.spy, 'source', 10, self.changes, dry_run=False)

    def test_limit_still_rejects_more_than_fifty_points(self):
        with self.assertRaises(ValueError):
            patch_identity(self.spy, 'source', 10, self.changes + [self.changes[0]], dry_run=False)
        self.spy.batch_update_points.assert_not_called()
