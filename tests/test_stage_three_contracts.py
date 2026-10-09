"""Synthetic transcript fixtures; no claims of human-verified footage."""
import pathlib,sys,unittest
from types import SimpleNamespace as NS
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from transcript_search_contract import match_spans,normalized_with_spans,transcript_results,title_score,contract_search
from semantic_passage_evidence import BoundedRetrieval,ValidationUnavailable

def row(i,text,**extra):
    return {'id':str(i),'video_id':1,'text':text,'speaker':'Unknown Speaker','transcript_speaker':'A',
        'start_time':8+(i-1)*10,'end_time':18+(i-1)*10,'timestamp_unit':'seconds',**extra}

def request(**extra):
    return NS(query='IPP',title=None,words=[],filter_type='text',transcript_contract='normalized',
        video_id=None,language=None,filter_date=None,filter_year=None,filter_month=None,max_scanned=10000,top_k=1000,**extra)

class Models:
    @staticmethod
    def FieldCondition(**kw):return kw
    MatchValue=FieldCondition;Filter=FieldCondition;MatchAny=FieldCondition

def judge(query,batch,timeout,facets):
    return {'required_facets':['topic'],'passages':[{'index':i,'score':.9,'complete':'elite' in r['text'],
        'evidence':{'topic':r['text']}} for i,r in enumerate(batch)]}

class Reader:
    def __init__(self,pages,candidates=()):self.pages=iter(pages);self.candidates=candidates;self.calls=[]
    def scroll(self,**kw):self.calls.append(kw);return next(self.pages)
    def query_points(self,**kw):self.calls.append(kw);return NS(points=[NS(id=r['id'],payload=r,score=.9) for r in self.candidates])

def points(rows):return [NS(id=r['id'],payload=r) for r in rows]

class StageThreeContractTests(unittest.TestCase):
    def test_literal_normalized_and_explicit_alias_are_distinct(self):
        text='IPP I.P.P. I P P آئی پی پی ipp'
        self.assertEqual(1,len(match_spans(text,'IPP','literal')))
        self.assertEqual(4,len(match_spans(text,'IPP','normalized')))
        self.assertEqual(5,len(match_spans(text,'IPP','alias')))
        self.assertEqual(1,len(match_spans(text,'آئی پی پی','normalized')))
        self.assertEqual(5,len(match_spans(text,'آئی پی پی','alias')))

    def test_all_acronym_variants_have_same_normalized_occurrences(self):
        text='IPP and I.P.P. and I P P'
        for query in ['IPP','I.P.P.','I P P']:
            spans=match_spans(text,query)
            self.assertEqual(['IPP','I.P.P','I P P'],[s['text'] for s in spans])
            for span in spans:self.assertEqual(span['text'],text[span['start']:span['end']])

    def test_boundaries_case_unicode_punctuation_whitespace_and_original_spans(self):
        self.assertEqual([],match_spans('shipping skipper IPProvider','IPP'))
        text='😀 ＩＰＰ،   تعلیم کی “فیس”'
        self.assertEqual('ＩＰＰ',match_spans(text,'ipp')[0]['text'])
        self.assertTrue(match_spans(text,'تعليم، کی فيس'))
        self.assertEqual([],match_spans(text,'ipp','literal'))
        self.assertTrue(match_spans('cafe\u0301','café'))

    def test_occurrences_and_source_identity_survive_consolidation_and_overlap(self):
        rows=[row(1,'IPP and I.P.P.'),row(2,'I P P again'),row(3,'IPP',video_id=2,start_time=9,end_time=11)]
        results=transcript_results(rows,'IPP','normalized')
        self.assertEqual(4,sum(r['occurrence_count'] for r in results))
        merged=next(r for r in results if len(r['segment_ids'])==2)
        self.assertEqual(['1','2'],[s['id'] for s in merged['source_utterances']])
        self.assertEqual(8,merged['start_time']);self.assertEqual(28,merged['end_time'])
        for hit in merged['match_spans']:self.assertEqual(hit['text'],merged['text'][hit['start']:hit['end']])

    def test_invalid_intervals_and_metadata_only_never_create_transcript_hits(self):
        self.assertEqual([],transcript_results([row(1,'Nothing here',video_title='IPP',summary_en='IPP'),
            row(2,'IPP',end_time=0),row(3,'IPP',timestamp_unit='milliseconds')],'IPP','normalized'))

    def test_title_all_terms_exact_rank_and_unicode(self):
        self.assertEqual(1,title_score('Hafiz Naeem Addressing Fuuast Graduation Ceremony','HAFIZ NAEEM ADDRESSING FUUAST GRADUATION CEREMONY!'))
        self.assertEqual(.9,title_score('Hafiz Naeem Addressing Fuuast Graduation Ceremony','Fuuast'))
        self.assertEqual(0,title_score('00089','graduation ceremony'))
        self.assertEqual(1,title_score('تعلیم، اور نوجوان','تعلیم اور نوجوان'))

    def test_title_exact_after_many_partials_is_not_hidden(self):
        partials=[row(i,'',video_title='IPP policy additional') for i in range(1,501)]
        exact=row(501,'',video_id=999,video_title='IPP policy')
        reader=Reader([(points(partials),1),(points([exact]),None)])
        req=request();req.filter_type='title';req.query='IPP policy';req.top_k=1
        result=contract_search(req,reader,'isolated',Models,None,None)
        self.assertEqual(999,result['results'][0]['video_id'])
        self.assertFalse(result['search_contract']['exhaustive'])
        self.assertEqual(0,result['search_contract']['counts']['displayed_passages'])

    def test_reader_counts_and_date_filter(self):
        reader=Reader([(points([row(1,'IPP IPP',video_created_at='2026-10-09'),row(2,'IPP',video_id=2,video_created_at='2025-01-01')]),None)])
        req=request();req.filter_year=2026
        result=contract_search(req,reader,'isolated',Models,None,None)
        self.assertEqual({'matching_videos':1,'displayed_passages':1,'matched_transcript_occurrences':2},result['search_contract']['counts'])
        self.assertNotIn('summary_en',reader.calls[0]['with_payload'])

    def test_summary_each_available_field_requires_own_transcript_evidence(self):
        for field in ['video_summary','video_summary_english','video_summary_urdu','summary_en','summary_ur']:
            reader=Reader([(points([{**row(1,''),field:'elite privileges'}]),None)],candidates=[row(10,'The elite receive unfair privileges.')])
            req=request();req.query='elite privileges';req.filter_type='summary'
            result=contract_search(req,reader,'isolated',Models,lambda q,t:[.1],judge)
            self.assertEqual(1,len(result['results']))
            self.assertEqual([field],result['results'][0]['matched_summary_fields'])
            self.assertTrue(result['results'][0]['passage_evidence'])
            self.assertIsNone(result['search_contract']['counts']['matched_transcript_occurrences'])

    def test_summary_without_support_does_not_invent_a_timestamp(self):
        reader=Reader([(points([{**row(1,''),'video_summary':'elite privileges'}]),None)],candidates=[row(10,'Unrelated statement')])
        req=request();req.query='elite privileges';req.filter_type='summary'
        result=contract_search(req,reader,'isolated',Models,lambda q,t:[.1],judge)
        self.assertEqual([],result['results'])
        self.assertEqual('no_matches',result['search_contract']['status'])

    def test_summary_provider_failure_is_not_zero_matches(self):
        reader=Reader([],candidates=[row(10,'elite receive privileges')]);req=request();req.query='elite privileges';req.filter_type='summary'
        req.summary_candidates=[{'video_id':1,'matched_summary_fields':['summary_english']}]
        def fail(*args):raise TimeoutError()
        with self.assertRaises(ValidationUnavailable):contract_search(req,reader,'isolated',Models,lambda q,t:[.1],fail)
        self.assertEqual(1,len(reader.calls))

    def test_scan_budget_returns_partial_not_false_exhaustive_counts(self):
        req=request();req.max_scanned=1
        result=contract_search(req,Reader([(points([row(1,'IPP')]),123)]),'isolated',Models,None,None)
        self.assertEqual('partial',result['search_contract']['status'])
        self.assertFalse(result['search_contract']['exhaustive'])

    def test_session_binding_cannot_reuse_literal_results_for_other_case_or_contract(self):
        from transcript_search_contract import search_session_binding
        req=request();req.transcript_contract='literal'
        original=search_session_binding(req)
        req.query='ipp'
        self.assertNotEqual(original,search_session_binding(req))
        req.query='IPP';req.transcript_contract='alias'
        self.assertNotEqual(original,search_session_binding(req))

    def test_literal_punctuation_is_preserved_and_word_edges_prevent_substrings(self):
        self.assertEqual(1,len(match_spans('X IPP X',' IPP ','literal')))
        self.assertEqual(1,len(match_spans('X.','.','literal')))
        self.assertEqual([],match_spans('shipping','ipp','literal'))
