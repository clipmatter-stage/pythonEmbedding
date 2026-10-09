"""Generic independent verification; fixtures are synthetic unless stated."""
import pathlib, sys, unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from semantic_passage_evidence import validate_passages, ValidationUnavailable

def candidate(i,text):
    return dict(id=str(i),video_id=i,text=text,start_time=1,end_time=10,score=.9,speaker='Known Speaker')
def primary(query,batch,timeout,facets):
    return {'required_facets':['complete_request'],'passages':[dict(index=i,score=.95,complete=True,evidence={'complete_request':r['text']})for i,r in enumerate(batch)]}
class RelationshipValidationTests(unittest.TestCase):
    def test_independent_rejection_applies_to_different_relationships(self):
        cases=[('guide youth to perform namaz','Pray for the protection of our young people.'),
               ('students facing high fees','Students enjoy modern classrooms.'),
               ('youth supporting Palestine','The speaker condemns violence in Palestine.'),
               ('workers supporting the policy','Workers reject the policy.')]
        for query,text in cases:
            def reject(q,b,t,f):
                response=primary(q,b,t,f)
                for j in response['passages']:j['complete']=False
                return response
            rows,meta=validate_passages(query,[candidate(1,text)],primary,verifier=reject)
            self.assertEqual([],rows);self.assertEqual('no_matches',meta['status'])
            self.assertEqual(2,meta['provider_calls']);self.assertEqual(1,meta['verification_rejected_count'])
    def test_verified_match_keeps_original_timestamp_and_evidence(self):
        rows,meta=validate_passages('students facing high fees',[candidate(1,'Students cannot afford high fees.')],primary,verifier=primary)
        self.assertTrue(rows[0]['independently_verified']);self.assertEqual(1,rows[0]['start_time'])
        self.assertTrue(meta['independent_verification'])
    def test_no_budget_for_verification_never_returns_primary_positive(self):
        rows,meta=validate_passages('any topic',[candidate(1,'A topic discussion.')],primary,verifier=primary,max_calls=1)
        self.assertEqual([],rows);self.assertEqual('partial',meta['status']);self.assertEqual(1,meta['provider_calls'])
    def test_verifier_failure_is_retryable_not_no_matches(self):
        def fail(*args):raise TimeoutError()
        with self.assertRaises(ValidationUnavailable):validate_passages('any topic',[candidate(1,'A topic discussion.')],primary,verifier=fail)
    def test_independent_quotes_must_belong_to_this_passage(self):
        def fabricate(q,b,t,f):
            response=primary(q,b,t,f);response['passages'][0]['evidence']={'complete_request':'unrelated invented quote'};return response
        with self.assertRaises(ValidationUnavailable):validate_passages('any topic',[candidate(1,'A topic discussion.')],primary,verifier=fabricate)
    def test_both_passes_share_six_call_limit(self):
        rows,meta=validate_passages('any topic',[candidate(i,'A topic discussion.')for i in range(100)],primary,verifier=primary)
        self.assertEqual(6,meta['provider_calls']);self.assertEqual(30,len(rows));self.assertEqual('partial',meta['status'])

class JudgeRoleContractTests(unittest.TestCase):
    def test_verification_requires_every_role_verdict(self):
        import ast,json
        from types import SimpleNamespace as NS
        source=(pathlib.Path(__file__).resolve().parents[1]/'embeddings_test.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='judge_passage_batch')
        for missing in ['roles_supported','relationship_supported','specificity_supported']:
            verdict=dict(index=0,score=.99,complete=True,evidence=[{'facet':'topic','quote':'A topic discussion.'}],roles_supported=True,relationship_supported=True,specificity_supported=True)
            verdict[missing]=False
            def create(**kwargs):return NS(choices=[NS(message=NS(content=json.dumps({'required_facets':['topic'],'passages':[verdict]})))])
            client=NS(chat=NS(completions=NS(create=create)));client.with_options=lambda **kw:client
            scope={'openai_client':client,'ValidationUnavailable':ValidationUnavailable}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'judge-test','exec'),scope)
            result=scope['judge_passage_batch']('any request',[candidate(1,'A topic discussion.')],5,['topic'],verification=True)
            self.assertFalse(result['passages'][0]['complete'])
