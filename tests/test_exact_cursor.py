"""Synthetic stable index; no real transcript/audio claims or provider calls."""
import contextlib,copy,json,pathlib,sys,time,unittest,tracemalloc
from types import SimpleNamespace as NS
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from exact_search_cursor import exact_page,CursorError,RevisionedClient
from transcript_search_contract import CompiledMatcher,title_score
class Redis:
 def __init__(self):self.data={}
 def get(self,k):return self.data.get(k)
 def set(self,k,v,**kw):self.data[k]=v;return self
 def exists(self,k):return k in self.data
 def delete(self,k):self.data.pop(k,None)
 def lock(self,*a,**kw):return contextlib.nullcontext()
 def pipeline(self,**kw):return self
 def execute(self):pass
class Models:
 FieldCondition=staticmethod(lambda **kw:kw)
 MatchValue=FieldCondition;MatchAny=FieldCondition;Filter=FieldCondition
class Reader:
 def __init__(self,rows):self.rows=rows;self.calls=0
 def count(self,**kw):return NS(count=len(self.filtered(kw['count_filter'])))
 def filtered(self,f):
  rows=self.rows
  for c in f['must']:
   key=c['key'];m=c['match'];rows=[r for r in rows if (r.payload.get(key) in m['any'] if 'any'in m else r.payload.get(key)==m['value'])]
  return rows
 def scroll(self,**kw):
  self.calls+=1;rows=self.filtered(kw['scroll_filter']);i=kw.get('offset') or 0;page=rows[i:i+kw['limit']];n=i+len(page)
  return page,n if n<len(rows) else None
 def upsert(self,**kw):return True

def row(i,text='other',video=1):return NS(id=i,payload={'video_id':video,'text':text,'start_time':i,'end_time':i+1,'timestamp_unit':'seconds','processing_status':'completed','approval_status':'approved','is_archived':False})
def request(**kw):return {'query':'IPP','transcript_contract':'normalized','allowed_video_ids':[1],'authorization_scope':'fixture','scope_revision':'fixture',**kw}
def page(data,reader,redis,**kw):return exact_page(data,reader,redis,'fixture',Models,**kw)
class ExactTests(unittest.TestCase):
 def test_invalid_matching_timestamps_never_become_no_matches(self):
  broken=row(1,'IPP');broken.payload['end_time']=0
  with self.assertRaises(CursorError) as error:page(request(),Reader([broken]),Redis())
  self.assertEqual('search_source_timing_invalid',error.exception.code)
 def test_beyond_2400_continuation_empty_partial_and_final_counts(self):
  reader=Reader([row(i,'IPP IPP' if i in [2400,2401,4999] else 'other') for i in range(5000)]);redis=Redis();q=request();ids=[];responses=[]
  while True:
   r=page(q,reader,redis);responses.append(r);ids += [x['id'] for x in r['results']]
   if not r['cursor']['has_more']:break
   q['exact_cursor']=r['cursor']['next']
  self.assertEqual(['2400','2401','4999'],ids);self.assertEqual(6,r['search_contract']['counts']['matched_transcript_occurrences']);self.assertTrue(r['search_contract']['counts_final']);self.assertEqual('partial',responses[0]['search_contract']['status']);self.assertFalse(responses[0]['search_contract']['counts_final']);self.assertEqual([],responses[0]['results'])
 def test_page_boundaries_retry_and_opaque_query_bound_cursor(self):
  reader=Reader([row(i,'IPP') for i in range(251)]);redis=Redis();first=page(request(),reader,redis);token=first['cursor']['next'];q=request(exact_cursor=token);second=page(q,reader,redis);calls=reader.calls;retry=page(q,reader,redis)
  self.assertEqual(second['results'],retry['results']);self.assertEqual(calls,reader.calls);self.assertEqual(second['cursor'],retry['cursor']);self.assertNotIn('offset',token);self.assertEqual(100,len(second['results']))
  for change in [{'query':'ipp'},{'transcript_contract':'literal'},{'authorization_scope':'other'},{'scope_revision':'new'},{'allowed_video_ids':[2]}]:
   with self.assertRaises(CursorError):page({**q,**change},reader,redis)
  last=page(request(exact_cursor=second['cursor']['next']),reader,redis);self.assertEqual(251,last['search_contract']['counts']['displayed_passages']);self.assertEqual(251,len({r['id'] for r in first['results']+second['results']+last['results']}))
 def test_eligibility_permissions_true_empty_and_expiry(self):
  r=Reader([row(1,'IPP',2),row(2,'IPP',1)]);r.rows[-1].payload['is_archived']=True;redis=Redis();out=page(request(),r,redis);self.assertEqual('no_matches',out['search_contract']['status']);self.assertEqual(0,out['search_contract']['counts']['matched_transcript_occurrences'])
  r=Reader([row(i,'IPP') for i in range(101)]);first=page(request(),r,redis,clock=lambda:1)
  with self.assertRaises(CursorError):page(request(exact_cursor=first['cursor']['next']),r,redis,clock=lambda:1802)
 def test_index_mutation_even_same_count_invalidates_cursor(self):
  redis=Redis();reader=Reader([row(i,'IPP') for i in range(101)]);first=page(request(),reader,redis);RevisionedClient(reader,redis).upsert(points=[])
  with self.assertRaises(CursorError):page(request(exact_cursor=first['cursor']['next']),reader,redis)
 def test_multilingual_literal_normalized_alias_and_original_spans(self):
  cases=[('IPP','I.P.P. IPPs آئی پی پی'),('falasteen','Palestine فلسطین palesteen'),('naujawan','نوجوان'),('bijli','بجلی'),('mehngai','مہنگائی'),('sood','سود'),('IMF','I.M.F. آئی ایم ایف'),('K-Electric','کے الیکٹرک')]
  for query,text in cases:
   spans=CompiledMatcher(query,'alias').spans(text);self.assertTrue(spans,(query,text))
   for span in spans:self.assertEqual(span['text'],text[span['start']:span['end']])
  self.assertEqual([],CompiledMatcher('Gaza','alias').spans('Palestine فلسطین'));self.assertEqual([],CompiledMatcher('bijli','literal').spans('بجلی'));self.assertEqual([],CompiledMatcher('IPP','normalized').spans('shipping IPProvider'));self.assertEqual(.9,title_score('Press Conference_Hafiz Naeem (25-10-22)','Hafiz Naeem'))
 def test_failure_does_not_advance_cursor(self):
  redis=Redis();reader=Reader([row(i,'IPP') for i in range(101)]);first=page(request(),reader,redis);q=request(exact_cursor=first['cursor']['next']);before=copy.deepcopy(redis.data);old=reader.scroll
  def fail(**kw):raise TimeoutError()
  reader.scroll=fail
  with self.assertRaises(TimeoutError):page(q,reader,redis)
  self.assertEqual(before,redis.data);reader.scroll=old;self.assertEqual(1,len(page(q,reader,redis)['results']))
if __name__=='__main__':unittest.main()
