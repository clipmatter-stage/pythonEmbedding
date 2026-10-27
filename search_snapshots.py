"""Durable, scope-bound search snapshots. No expiration or Redis dependency."""
import copy
import json
import uuid
from qdrant_client import models

COLLECTION = 'verified_search_snapshots_v1'

class SearchSnapshots:
    def __init__(self, client):
        # Use the raw client: derived snapshots must not advance the SOURCE index revision.
        self.client = client
        self.ready = False

    def ensure(self):
        if self.ready:
            return
        if not self.client.collection_exists(COLLECTION):
            try:
                self.client.create_collection(collection_name=COLLECTION, vectors_config={})
            except Exception:
                # Another replica may have created the collection concurrently.
                if not self.client.collection_exists(COLLECTION):
                    raise
        self.ready = True

    @staticmethod
    def point_id(key):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, 'clipmatters:verified-search:' + key))

    def read(self, key):
        self.ensure()
        points = self.client.retrieve(collection_name=COLLECTION, ids=[self.point_id(key)],
                                      with_payload=True, with_vectors=False, timeout=5)
        if not points:
            return None
        payload = points[0].payload or {}
        if payload.get('key') != key or not isinstance(payload.get('result'), dict):
            raise ValueError('Invalid verified search snapshot')
        return copy.deepcopy(payload['result'])

    def write(self, key, result):
        self.ensure()
        self.client.upsert(collection_name=COLLECTION, points=[models.PointStruct(
            id=self.point_id(key), vector={}, payload={'key': key, 'result': json.loads(json.dumps(result,ensure_ascii=False,default=str))})],
            wait=True, timeout=10)
