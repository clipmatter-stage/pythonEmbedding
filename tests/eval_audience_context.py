"""Explicit live-model controls for contextual audience interpretation; no index writes."""
from grounded_validation import provider_judge,validate_grounded
from openai import OpenAI
import os,json
CASES=[
 ('youth_context_after_advice','آپ نماز سے مدد لیں اور قرآن پڑھیں۔ مجھے خوشی ہے کہ آپ ہمارے بچوں نے، نوجوان نسل کے نمائندوں نے، دین کو اپنانے کا عزم کیا ہے۔',True),
 ('explicit_youth','ہم نوجوانوں کو نماز پڑھنے کی تلقین کرتے ہیں اور مسجد آنے کی دعوت دیتے ہیں۔',True),
 ('literal_seven_year_olds','ہم نے سات سال کے بچوں کو نماز کی مہم میں شامل کیا اور انہیں انعامات دیے۔',False),
 ('praying_for_youth','ہم نے نوجوانوں کی کامیابی کے لیے نماز پڑھی اور دعا کی۔',False),
 ('unlinked_youth','نوجوانوں کو روزگار ملنا چاہیے۔ اب والدین اپنے سات سال کے بچوں کو نماز کی تلقین کریں۔',False),
]
def evaluate():
 judge=provider_judge(OpenAI(),os.getenv('SEARCH_JUDGE_MODEL','gpt-6-luna'),'span_id')
 rows=[{'id':name,'video_id':i+1,'text':text,'start_time':0.,'end_time':10.,'score':1.}for i,(name,text,_)in enumerate(CASES)]
 accepted,meta=validate_grounded('nojawano ko namaz ki talqeen',rows,judge,deadline_seconds=45)
 ids={r['id']for r in accepted}
 outcomes=[{'case':name,'expected':expected,'accepted':name in ids,'passed':(name in ids)==expected}for name,_,expected in CASES]
 print(json.dumps({'controls':outcomes,'all_passed':all(r['passed']for r in outcomes),'provider_calls':meta['provider_calls']}),flush=True)
 return all(r['passed']for r in outcomes)
if __name__=='__main__':raise SystemExit(0 if evaluate()else 1)
