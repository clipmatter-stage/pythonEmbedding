"""Exercise production functions without importing service/network startup."""
import ast
import pathlib
import unittest
from types import SimpleNamespace

SOURCE = pathlib.Path(__file__).resolve().parents[1] / 'embeddings_test.py'

class ProviderError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        super().__init__(detail)

class EmbeddingCompatibilityTest(unittest.TestCase):
    def scope(self, provider):
        tree = ast.parse(SOURCE.read_text())
        names = {'get_openai_embedding', 'get_openai_embeddings_batch', 'get_cached_embedding'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = dict(List=list, HTTPException=ProviderError, OPENAI_EMBEDDING_MODEL='text-embedding-3-large',
                     EMBEDDING_DIMENSION=3072, USE_OPENAI_EMBEDDINGS=True, embedding_cache={},
                     openai_client=provider, logger=SimpleNamespace(info=lambda *x: None, warning=lambda *x: None))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
        return scope

    def test_provider_failure_is_retryable_without_alternate_vectors(self):
        def fail(**kwargs): raise RuntimeError('provider outage')
        scope = self.scope(SimpleNamespace(embeddings=SimpleNamespace(create=fail)))
        for function, value in [('get_openai_embedding','query'), ('get_openai_embeddings_batch',['document'])]:
            with self.assertRaises(ProviderError) as error: scope[function](value)
            self.assertEqual(503, error.exception.status_code)
        self.assertEqual({}, scope['embedding_cache'])

    def test_query_and_document_use_same_model_and_dimensions(self):
        calls = []
        def embed(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(index=i,embedding=[0.1]*3072) for i in range(len(kwargs['input']))])
        scope = self.scope(SimpleNamespace(embeddings=SimpleNamespace(create=embed)))
        scope['get_cached_embedding']('query')
        scope['get_openai_embeddings_batch'](['document'])
        self.assertEqual(2,len(calls))
        self.assertTrue(all(c['model']=='text-embedding-3-large' and c['dimensions']==3072 for c in calls))

    def test_worker_does_not_delete_before_provider_generation(self):
        tree = ast.parse(SOURCE.read_text())
        worker = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_video_task')
        calls = [n for n in ast.walk(worker) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id=='delete_existing_embeddings' for n in calls))

class WorkerProviderFailureTest(unittest.TestCase):
    def test_document_provider_failure_preserves_index(self):
        import datetime
        tree=ast.parse(SOURCE.read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_video_task')
        writes=[]
        def fail(texts): raise ProviderError(503,'retryable outage')
        scope=dict(datetime=datetime.datetime, logger=SimpleNamespace(info=lambda *x:None,error=lambda *x:None,warning=lambda *x:None),
                   OPENAI_EMBEDDING_MODEL='text-embedding-3-large', USE_OPENAI_EMBEDDINGS=True, openai_client=object(), get_openai_embeddings_batch=fail,
                   WebhookDeliveryError=type('WebhookDeliveryError',(Exception,),{}),
                   qdrant_client=SimpleNamespace(upsert=lambda **k:writes.append(k),delete=lambda **k:writes.append(k)))
        exec(compile(ast.Module(body=[worker],type_ignores=[]),str(SOURCE),'exec'),scope)
        with self.assertRaises(ProviderError):
            scope['process_video_task']({'video_id':1,'run_id':'run','timestamp_unit':'seconds',
                'identification_segments':[{'start':0,'end':8,'text':'valid transcript'}]})
        self.assertEqual([],writes)
