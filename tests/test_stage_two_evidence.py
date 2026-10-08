import ast
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_passage_evidence import (ValidationUnavailable, consolidate_candidates,
    normalize_multilingual, requested_speaker_names, validate_passages)


def passage(index, text='Students face high education fees at university.', **extra):
    return {'id':str(index), 'video_id':index, 'start_time':8, 'end_time':18,
                'speaker':'Unknown Speaker', 'diarization_speaker':'A', 'text':text,
                'score':0.9, 'language':'ur', **extra}


def judgments(query, batch, timeout, facets):
    facets = facets or ['students_and_high_education_fees']
    return {'required_facets': facets, 'passages': [
        {'index': i, 'score': 0.9, 'complete': True,
         'evidence': {facet: r['text'] for facet in facets}} for i, r in enumerate(batch)]}


class StageTwoEvidenceTest(unittest.TestCase):
    def test_equivalent_queries_preserve_complete_intent_and_cross_language_candidates(self):
        queries = ['Students facing high education fees', 'طلبہ کو تعلیم کی زیادہ فیسوں کا سامنا',
                   'talaba ko taleem ki zyada fees ka samna']
        for query in queries:
            received = []
            def judge(q, batch, timeout, facets):
                received.append(q)
                return judgments(q, batch, timeout, facets)
            results, metadata = validate_passages(query, [passage(1, 'طلبہ مہنگی تعلیمی فیسوں سے پریشان ہیں۔')], judge)
            self.assertEqual([query], received)
            self.assertEqual('ur', results[0]['language'])
            self.assertEqual(8, results[0]['start_time'])
            self.assertEqual('completed', metadata['status'])

    def test_compound_requests_require_every_facet_with_own_passage_quotes(self):
        cases = [('youth encouraged to pray', ['youth', 'encouragement_toward_prayer'],
                  'Youth attend the political rally today.', 'Youth are encouraged to pray each day.'),
                 ('students facing high education fees', ['students', 'high_education_fees'],
                  'Students deserve good education and modern classrooms.', 'Students cannot afford high education fees.'),
                 ('youth supporting Palestine', ['youth_participation', 'Palestine_support'],
                  'The speaker condemns violence in Palestine.', 'Youth march in support of Palestine today.')]
        for query, facets, negative, positive in cases:
            def judge(q, batch, timeout, previous):
                return {'required_facets': facets, 'passages': [
                    {'index': i, 'score': 0.95, 'complete': i == 1,
                     'evidence': {f: r['text'] for f in facets}} for i, r in enumerate(batch)]}
            results, _ = validate_passages(query, [passage(1, negative), passage(2, positive)], judge)
            self.assertEqual(['2'], [r['id'] for r in results])
            self.assertEqual(facets, results[0]['llm_supported_facets'])

    def test_missing_facet_or_quote_from_title_cannot_bypass_validation(self):
        def bad(query, batch, timeout, facets):
            return {'required_facets': ['students', 'fees'], 'passages': [
                {'index': 0, 'score': 1, 'complete': True,
                 'evidence': {'students': 'Students', 'fees': 'high education fees'}}]}
        with self.assertRaises(ValidationUnavailable):
            validate_passages('students high fees', [passage(1, 'Students enjoy the library.', video_title='high education fees')], bad)

    def test_wrong_explicit_speaker_rejected_before_provider(self):
        calls = []
        def judge(*args): calls.append(args); return judgments(*args)
        results, meta = validate_passages('Hafiz Naeem talks about education',
            [passage(1)], judge, speaker_names=['Hafiz Naeem Ur Rehman'])
        self.assertEqual([], results); self.assertEqual([], calls)
        self.assertEqual('no_matches', meta['status'])

    def test_subject_does_not_invent_speaker_constraint(self):
        aliases = {'hnr': {'canonical': 'Hafiz Naeem Ur Rehman', 'aliases': ['hafiz naeem', 'hnr', 'naeem'],
                           'speaker_variants': ['حافظ نعیم']}}
        self.assertEqual([], requested_speaker_names('Youth supporting Palestine', aliases))
        self.assertEqual([], requested_speaker_names('naeem education policy', aliases))
        self.assertEqual([], requested_speaker_names('Students discuss Hafiz Naeem policies', aliases))
        self.assertIn('حافظ نعیم', requested_speaker_names('Hafiz Naeem encouraging young people to pray', aliases))

    def test_relevant_candidates_below_30_and_strict_budget(self):
        calls = []
        def judge(query, batch, timeout, facets):
            calls.append(len(batch)); result = judgments(query, batch, timeout, facets)
            for r, j in zip(batch, result['passages']): j['complete'] = int(r['id']) >= 35
            return result
        results, meta = validate_passages('education fees', [passage(i, score=.9-i*.001) for i in range(100)], judge)
        self.assertTrue(any(r['id'] == '35' for r in results))
        self.assertEqual([20,20,20], calls); self.assertEqual(60, meta['evaluated_count'])
        self.assertEqual('partial', meta['status'])

    def test_timeout_and_invalid_json_are_retryable_not_zero_matches(self):
        for error in (TimeoutError('provider timeout'), ValueError('invalid JSON')):
            def fail(*args): raise error
            with self.assertRaises(ValidationUnavailable): validate_passages('fees', [passage(1)], fail)
        results, _ = validate_passages('fees', [passage(1)], judgments)
        self.assertEqual(1, len(results))

    def test_later_provider_failure_preserves_only_verified_partial_results(self):
        calls = []
        def judge(*args):
            calls.append(1)
            if len(calls) == 2: raise TimeoutError()
            return judgments(*args)
        results, meta = validate_passages('fees', [passage(i) for i in range(40)], judge)
        self.assertEqual(20, len(results)); self.assertEqual('partial', meta['status'])
        self.assertTrue(meta['retryable']); self.assertEqual(20, meta['evaluated_count'])

    def test_no_genuine_match_returns_no_filler(self):
        def reject(*args):
            result=judgments(*args)
            for item in result['passages']: item['complete']=False
            return result
        results, meta = validate_passages('fees', [passage(1, 'Generic politics')], reject)
        self.assertEqual([], results); self.assertEqual('no_matches', meta['status'])
        self.assertFalse(meta['retryable'])

    def test_incomplete_or_duplicate_judgments_fail_explicitly(self):
        for response in ({'required_facets': ['fees'], 'passages': []},
                         {'required_facets': [], 'passages': []}):
            with self.assertRaises(ValidationUnavailable):
                validate_passages('fees', [passage(1)], lambda *args: response)

    def test_consolidation_preserves_speakers_distinct_overlap_ids_and_seconds(self):
        items=[passage(1), passage(1), {**passage(2), 'video_id':1, 'start_time':18, 'end_time':25},
               {**passage(3), 'video_id':1, 'start_time':25, 'end_time':30, 'diarization_speaker':'B'},
               {**passage(4), 'video_id':1, 'start_time':9, 'end_time':12}]
        results=consolidate_candidates(items)
        self.assertEqual(4, sum(len(r['segment_ids']) for r in results))
        self.assertTrue(any(r['id']=='4' for r in results))
        self.assertTrue(any(r['diarization_speaker']=='B' for r in results))
        self.assertEqual(8, min(r['start_time'] for r in results))

    def test_oversized_passages_are_partial_never_silently_truncated(self):
        results, meta=validate_passages('fees', [passage(1, 'x'*2401)], judgments)
        self.assertEqual([], results); self.assertEqual('partial', meta['status'])
        self.assertEqual(0, meta['provider_calls'])

    def test_deadline_prevents_another_call(self):
        ticks=iter([0,0,31,31])
        results, meta=validate_passages('fees', [passage(i) for i in range(40)], judgments, clock=lambda:next(ticks))
        self.assertEqual(20,len(results)); self.assertEqual(1,meta['provider_calls'])

    def test_unicode_preserves_meaning_and_matches_keyboard_and_punctuation_variants(self):
        self.assertEqual(normalize_multilingual('  تعليم، كی “فيس”؟ '), normalize_multilingual('تعلیم, کی "فیس"?'))
        self.assertNotEqual(normalize_multilingual('بھارت'),normalize_multilingual('بہارت'))

    def test_adapter_prompt_and_incremental_metadata_contract(self):
        source=(ROOT/'embeddings_test.py').read_text()
        self.assertNotIn('Auto-applied language filter: en',source)
        self.assertNotIn('Auto-set speaker_filter=',source)
        self.assertIn('passage_validation": query_params.get',source)
        self.assertIn('max_retries=0',source)
        tree=ast.parse(source)
        search=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='search')
        gate=next(n for n in ast.walk(search) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='validate_passages')
        self.assertIsInstance(gate.args[0],ast.BoolOp)
        self.assertEqual('raw_query_text',gate.args[0].values[0].id)

    def test_actual_judge_adapter_uses_full_query_bounded_timeout_and_no_title(self):
        from types import SimpleNamespace
        import json
        node=next(n for n in ast.parse((ROOT/'embeddings_test.py').read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='judge_passage_batch')
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(judgments('q',[passage(1)],5,None))))])
        options=[]
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client.with_options=lambda **kwargs: (options.append(kwargs) or client)
        scope={'openai_client':client,'ValidationUnavailable':ValidationUnavailable}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'judge_adapter','exec'),scope)
        query='Hafiz Naeem encouraging youth to pray'
        result=scope['judge_passage_batch'](query,[passage(1,video_title='unrelated title')],5)
        data=json.loads(calls[0]['messages'][1]['content'])
        self.assertEqual(query,data['query']);self.assertNotIn('title',data['passages'][0])
        self.assertEqual(5,calls[0]['timeout']);self.assertEqual([{'max_retries':0}],options)
        self.assertEqual('gpt-4o-mini',calls[0]['model'])
        self.assertEqual(1,len(result['passages']))

    def test_bounded_retrieval_uses_real_transport_timeout_contract_and_stops(self):
        from semantic_passage_evidence import BoundedRetrieval, RetrievalBudgetReached
        from types import SimpleNamespace
        calls=[]
        client=SimpleNamespace(scroll=lambda **kwargs:calls.append(kwargs))
        reader=BoundedRetrieval(client,max_calls=1,clock=lambda:0)
        reader.scroll(collection_name='isolated')
        self.assertEqual(10,calls[0]['timeout'])
        with self.assertRaises(RetrievalBudgetReached):reader.scroll(collection_name='isolated')
        self.assertEqual(1,len(calls))

    def test_query_embeddings_keep_document_model_space_with_explicit_retryable_failure(self):
        from types import SimpleNamespace
        node=next(n for n in ast.parse((ROOT/'embeddings_test.py').read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='get_semantic_query_embedding')
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[.1,.2,.3])])
        class HttpError(Exception):
            def __init__(self,**kwargs):self.status=kwargs['status_code']
        client=SimpleNamespace(embeddings=SimpleNamespace(create=create))
        client.with_options=lambda **kwargs:client
        scope={'OPENAI_EMBEDDING_MODEL':'configured-model','EMBEDDING_DIMENSION':3,
               'embedding_cache':{},'USE_OPENAI_EMBEDDINGS':True,'openai_client':client,'HTTPException':HttpError}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'query_embedding','exec'),scope)
        get=scope['get_semantic_query_embedding']
        self.assertEqual([.1,.2,.3],get('query',6));get('query',6)
        self.assertEqual(1,len(calls));self.assertEqual('configured-model',calls[0]['model'])
        self.assertEqual(3,calls[0]['dimensions']);self.assertEqual(6,calls[0]['timeout'])
        def fail(**kwargs):raise TimeoutError()
        client.embeddings.create=fail
        with self.assertRaises(HttpError) as caught:get('uncached',6)
        self.assertEqual(503,caught.exception.status)
        self.assertNotIn(('configured-model',3,'uncached'),scope['embedding_cache'])

    def test_video_groups_are_contiguous_and_source_labels_prevent_unknown_merge(self):
        candidates=[passage(1,score=.99),passage(2,score=.95),
                    {**passage(3,score=.8),'video_id':1,'start_time':40,'end_time':50}]
        results,_=validate_passages('fees',candidates,judgments)
        self.assertEqual([1,1,2],[r['video_id'] for r in results])
        a={**passage(1),'diarization_speaker':'','transcript_speaker':'A'}
        b={**passage(2),'video_id':1,'start_time':18,'end_time':25,'diarization_speaker':'','transcript_speaker':'B'}
        self.assertEqual(2,len(consolidate_candidates([a,b])))
        del a['transcript_speaker'];del b['transcript_speaker']
        self.assertEqual(2,len(consolidate_candidates([a,b])))

    def test_expired_cursor_fails_instead_of_skipping_into_a_different_result_set(self):
        from types import SimpleNamespace
        node=next(n for n in ast.walk(ast.parse((ROOT/'embeddings_test.py').read_text()))
                  if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp)
                  and isinstance(n.test.values[0],ast.Name) and n.test.values[0].id=='cursor'
                  and isinstance(n.test.values[1],ast.UnaryOp)
                  and isinstance(n.test.values[1].operand,ast.Name) and n.test.values[1].operand.id=='cached_data')
        class HttpError(Exception):
            def __init__(self,**kwargs):self.detail=kwargs
        scope={'cursor':{'index':20},'cached_data':None,'HTTPException':HttpError}
        with self.assertRaises(HttpError) as caught:
            exec(compile(ast.Module(body=[node],type_ignores=[]),'expired_cursor','exec'),scope)
        self.assertEqual(409,caught.exception.detail['status_code'])
        self.assertTrue(caught.exception.detail['detail']['retryable'])

    def test_actual_cursor_extraction_keeps_complete_video_groups_on_stable_pages(self):
        import base64, json, logging, typing
        names={'extract_batch_from_results','encode_cursor','decode_cursor'}
        nodes=[n for n in ast.parse((ROOT/'embeddings_test.py').read_text()).body
               if isinstance(n,ast.FunctionDef) and n.name in names]
        scope={'List':typing.List,'Dict':typing.Dict,'Optional':typing.Optional,'Tuple':typing.Tuple,
               'base64':base64,'json_module':json,'logger':logging.getLogger('cursor-test')}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_cursor','exec'),scope)
        candidates=[passage(1,score=.99),passage(2,score=.95),
                    {**passage(3,score=.8),'video_id':1,'start_time':40,'end_time':50}]
        results,_=validate_passages('fees',candidates,judgments)
        batch,cursor,more=scope['extract_batch_from_results'](results,None,1,'isolated')
        self.assertEqual([1,1],[r['video_id'] for r in batch]);self.assertTrue(more)
        second,next_cursor,more=scope['extract_batch_from_results'](results,scope['decode_cursor'](cursor),1,'isolated')
        self.assertEqual([2],[r['video_id'] for r in second]);self.assertFalse(more)
        self.assertIsNone(next_cursor)

    def test_validation_logs_safe_reason_and_correlatable_error_id(self):
        def fail(*args):
            raise TimeoutError('sk-private-api-key SECRET_TRANSCRIPT')
        with self.assertLogs('semantic_passage_evidence', level='INFO') as logs:
            with self.assertRaises(ValidationUnavailable) as caught:
                validate_passages('SECRET_QUERY', [passage(1,'SECRET_TRANSCRIPT')], fail)
        output='\n'.join(logs.output)
        self.assertIn('reason=provider_timeout',output)
        self.assertIn(caught.exception.diagnostic_id,output)
        self.assertEqual('provider_timeout',caught.exception.reason)
        for secret in ('sk-private-api-key','SECRET_TRANSCRIPT','SECRET_QUERY'):
            self.assertNotIn(secret,output)

    def test_protocol_failure_logs_the_exact_failed_check(self):
        def invalid(*args):
            return {'required_facets':['fees'],'passages':[]}
        with self.assertLogs('semantic_passage_evidence', level='ERROR') as logs:
            with self.assertRaises(ValidationUnavailable) as caught:
                validate_passages('fees',[passage(1)],invalid)
        self.assertEqual('incomplete_judgments',caught.exception.reason)
        self.assertIn('reason=incomplete_judgments','\n'.join(logs.output))

    def test_partial_provider_failure_is_logged_and_keeps_same_diagnostic_id(self):
        calls=[]
        def judge(*args):
            calls.append(1)
            if len(calls)==2:raise TimeoutError('secret')
            return judgments(*args)
        with self.assertLogs('semantic_passage_evidence',level='INFO') as logs:
            results,metadata=validate_passages('fees',[passage(i) for i in range(40)],judge)
        self.assertEqual(20,len(results));self.assertEqual('partial',metadata['status'])
        failure=next(line for line in logs.output if 'PASSAGE_VALIDATION_FAILED' in line)
        self.assertIn(metadata['diagnostic_id'],failure)
        self.assertIn('reason=provider_timeout',failure)
