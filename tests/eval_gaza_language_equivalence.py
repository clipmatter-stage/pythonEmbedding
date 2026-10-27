"""Opt-in live-model language/direction regression controls; no source/index writes."""
import json,os
from openai import OpenAI
from grounded_validation import provider_judge,validate_grounded
QUERIES=['غزہ اور ہتھیار پہنچانا','Gaza aur hathyar pohanchana','Gaza and weapons being supplied']
CASES=[
 ('linked_supply_to_israel','امریکہ اسرائیل کو اسلحہ فراہم کر رہا ہے۔ غزہ میں جو کچھ ہو رہا ہے اس میں امریکہ پوری طرح شریک ہے۔',True),
 ('supply_to_gaza','غزہ میں مزاحمت کاروں کو ہتھیار پہنچائے جا رہے ہیں۔',True),
 ('gaza_without_supply','غزہ میں بچوں کے لیے کھانا اور دوائیں بھیجی جا رہی ہیں۔',False),
 ('unrelated_supply','ترکی نے اپنی فوج کے لیے ہتھیار خریدے۔ غزہ میں بارش ہوئی۔',False),
]
def evaluate():
 judge=provider_judge(OpenAI(),os.getenv('SEARCH_JUDGE_MODEL','gpt-6-luna'),'span_id')
 rows=[{'id':name,'video_id':i+1,'text':text,'start_time':0.,'end_time':10.,'score':1.}for i,(name,text,_)in enumerate(CASES)]
 outcomes=[]
 for query in QUERIES:
  accepted,_=validate_grounded(query,rows,judge,deadline_seconds=45)
  ids={r['id']for r in accepted}
  outcomes.extend({'query':query,'case':name,'passed':(name in ids)==expected}for name,_,expected in CASES)
 for query in ['Gaza ko hathyar pohanchana','غزہ میں ہتھیار پہنچانا','weapons supplied to Gaza']:
  accepted,_=validate_grounded(query,rows[:2],judge,deadline_seconds=45)
  ids={r['id']for r in accepted}
  outcomes.append({'query':query,'case':'explicit_destination','passed':ids=={'supply_to_gaza'}})
 print(json.dumps({'controls':outcomes,'all_passed':all(r['passed']for r in outcomes)},ensure_ascii=False),flush=True)
 return all(r['passed']for r in outcomes)
if __name__=='__main__':raise SystemExit(0 if evaluate()else 1)
