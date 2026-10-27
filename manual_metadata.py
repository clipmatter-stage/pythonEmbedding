"""Separate title/tag vectors: never fabricate transcript evidence or timestamps."""
import hashlib,json,uuid
from qdrant_client import models
from grounded_validation import validate_grounded
from semantic_passage_evidence import ValidationUnavailable
COLLECTION='manual_video_metadata_v1'
VERSION='manual_metadata_v1'

def point_id(video_id,fingerprint):
    return str(uuid.uuid5(uuid.NAMESPACE_URL,f'clipmatters:metadata:{video_id}:{fingerprint}'))

class ManualMetadata:
    def __init__(self,client,embed,dimensions,model,judge):
        self.client=client;self.embed=embed;self.dimensions=dimensions;self.model=model;self.judge=judge
    def ensure(self):
        if not self.client.collection_exists(COLLECTION):
            try:self.client.create_collection(COLLECTION,vectors_config=models.VectorParams(size=self.dimensions,distance=models.Distance.COSINE))
            except Exception:
                if not self.client.collection_exists(COLLECTION):raise
    def index(self,document):
        self.ensure();identity=point_id(document['video_id'],document['fingerprint'])
        text='Video title: '+document['title']+'\nManual tags: '+', '.join(document['tags'])
        old=self.client.retrieve(COLLECTION,ids=[identity],with_payload=True)
        if not old or old[0].payload.get('model')!=self.model:
            self.client.upsert(COLLECTION,points=[models.PointStruct(id=identity,vector=self.embed(text),payload={**document,'text':text,'model':self.model})],wait=True,timeout=30)
        check=self.client.retrieve(COLLECTION,ids=[identity],with_payload=True)
        if not check or check[0].payload.get('fingerprint')!=document['fingerprint']:raise RuntimeError('Metadata index verification failed')
        return document['fingerprint']
    def search(self,request):
        registry=getattr(request,'metadata_registry',[])or[]
        # Metadata cannot establish who spoke or when a passage occurred.
        if not registry or request.speaker or request.time_range:return []
        allowed=set(request.allowed_video_ids or [])
        entries=[r for r in registry if r['video_id']in allowed and(not request.video_id or r['video_id']==request.video_id)]
        if not entries:return []
        self.ensure();ids=[point_id(r['video_id'],r['fingerprint'])for r in entries]
        if len(self.client.retrieve(COLLECTION,ids=ids,with_payload=False))!=len(ids):
            raise ValidationUnavailable('Manual metadata index requires synchronization',reason='metadata_index_missing')
        hits=self.client.query_points(COLLECTION,query=self.embed(request.query),query_filter=models.Filter(must=[models.HasIdCondition(has_id=ids)]),limit=min(24,len(ids)),with_payload=True,timeout=10).points
        candidates=[]
        for hit in hits:
            p=hit.payload or{}
            if p.get('model')!=self.model:raise ValidationUnavailable('Manual metadata needs reindexing for the configured model',reason='metadata_model_changed')
            candidates.append({'id':str(hit.id),'video_id':p['video_id'],'text':p['text'],'score':float(hit.score),'start_time':0,'end_time':1,'speaker':'Unknown Speaker','document':p})
        rows,meta=validate_grounded(request.query,candidates,self.judge,batch_size=8,max_calls=6,deadline_seconds=40)
        if meta['status']=='partial':raise ValidationUnavailable('Title/tag evaluation is incomplete',reason='metadata_validation_incomplete')
        output=[]
        for row in rows:
            if row['score']<request.min_score:continue
            document=row['document']
            output.append({'id':'metadata:'+str(row['id']),'video_id':row['video_id'],'video_title':document['title'],
                'text':'','score':row['score'],'is_video_only':True,'transcript_verified':False,'evidence_verified':False,
                'evidence_source':'manual_metadata','matched_fields':[],
                'match_types':['manual_metadata_match'],'metadata_evidence':{'title':document['title'],'tags':document['tags'],'quotes':[q['quote']for q in row.get('evidence_spans',[])]},
                'score_type':'verified_metadata_relevance'})
        return output

def merge_metadata(result,rows,limit):
    output=list(result['results']);seen={r['video_id']for r in output}
    output.extend(row for row in rows if row['video_id']not in seen)
    result.update(results=output[:limit],returned=len(output[:limit]),unique_videos=len({r['video_id']for r in output[:limit]}))
    result['metadata']['manual_metadata']={'matched_videos':len(rows),'source':'manual_titles_and_tags','transcript_verified':False}
    validation=result['metadata'].get('passage_validation',{})
    if rows and validation.get('status')=='no_matches':validation['status']='completed'
    ranking=result.setdefault('ranking',{})
    if rows:
        ranking['candidate_sources']=list(dict.fromkeys([*ranking.get('candidate_sources',[]),'manual_titles_and_tags']))
        ranking['score_type']='mixed_source_relevance'
        ranking['evidence_verified']=False
    return result
