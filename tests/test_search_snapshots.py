import pathlib,sys,unittest
from unittest.mock import Mock
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from qdrant_client import QdrantClient
from search_snapshots import SearchSnapshots,COLLECTION

class SnapshotTests(unittest.TestCase):
    def test_snapshot_survives_new_client_instance_without_cache_or_expiry(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            client=QdrantClient(path=directory)
            store=SearchSnapshots(client)
            result={'results':[{'video_id':1169,'text':'verified passage'}]}
            store.write('grounded:scope-and-index-one',result)
            client.close()
            client=QdrantClient(path=directory)
            self.assertEqual(result,SearchSnapshots(client).read('grounded:scope-and-index-one'))
            self.assertIsNone(SearchSnapshots(client).read('grounded:different-scope-or-index'))
            self.assertEqual(1,client.count(COLLECTION).count)
            client.close()

    def test_storage_read_failure_does_not_become_a_cache_miss(self):
        client=Mock();client.collection_exists.return_value=True
        client.retrieve.side_effect=RuntimeError('storage unavailable')
        with self.assertRaises(RuntimeError):SearchSnapshots(client).read('grounded:key')

    def test_writes_must_be_acknowledged(self):
        client=Mock();client.collection_exists.return_value=True
        SearchSnapshots(client).write('grounded:key',{'results':[]})
        self.assertTrue(client.upsert.call_args.kwargs['wait'])
        self.assertEqual(COLLECTION,client.upsert.call_args.kwargs['collection_name'])
