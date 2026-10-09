"""Explicit-scope payload-only backfill. Dry-run by default; never embeds/deletes."""
import argparse,json,os
from transcript_search_contract import indexed_text_fields


def backfill_video(client,collection,video_id,*,apply=False,max_points=5000):
    from qdrant_client import models
    points=[];offset=None
    while True:
        batch,offset=client.scroll(collection_name=collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(key='video_id',match=models.MatchValue(value=video_id))]),
            limit=min(200,max_points-len(points)),offset=offset,with_payload=['text','search_contract_version','transcript_normalized_v1'],
            with_vectors=False,timeout=10)
        points.extend(batch)
        if offset is None:break
        if len(points)>=max_points:raise RuntimeError('Explicit point limit exceeded; inspect and raise --max-points deliberately. No writes performed.')
    pending=[(p.id,indexed_text_fields((p.payload or {}).get('text'))) for p in points
             if any((p.payload or {}).get(k)!=v for k,v in indexed_text_fields((p.payload or {}).get('text')).items())]
    if apply:
        for point_id,payload in pending:
            client.set_payload(collection_name=collection,payload=payload,points=[point_id],wait=True,timeout=10)
    return {'video_id':video_id,'points':len(points),'payload_updates':len(pending),'applied':apply}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-id',type=int,action='append',required=True)
    parser.add_argument('--collection',required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--apply',action='store_true')
    parser.add_argument('--max-points',type=int,default=5000)
    parser.add_argument('--ensure-indexes',action='store_true',help='With --apply, create only the two new payload indexes.')
    args=parser.parse_args()
    if args.max_points<1 or any(i<1 for i in args.video_id):parser.error('IDs and point limit must be positive')
    if args.ensure_indexes and not args.apply:parser.error('--ensure-indexes requires --apply; dry-run never creates indexes')
    from qdrant_client import QdrantClient,models
    client=QdrantClient(url=os.environ['QDRANT_URL'],api_key=os.environ['QDRANT_API_KEY'],timeout=10)
    if args.ensure_indexes:
        client.create_payload_index(args.collection,'search_contract_version',models.PayloadSchemaType.KEYWORD,wait=True)
        client.create_payload_index(args.collection,'transcript_normalized_v1',models.TextIndexParams(type='text',tokenizer=models.TokenizerType.WORD,min_token_len=1,lowercase=True),wait=True)
    for video_id in dict.fromkeys(args.video_id):
        print(json.dumps(backfill_video(client,args.collection,video_id,apply=args.apply,max_points=args.max_points)))

if __name__=='__main__':main()
