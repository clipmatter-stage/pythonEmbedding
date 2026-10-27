"""Retrieval hints only. Original transcript text is always the evidence."""
import re
from transcript_search_contract import normalize
VERSION='asr_context_v1'
# Reviewed spelling/transliteration variants. These never establish relevance.
PLACE_VARIANTS=[(['makkah','mecca','مکہ','مکہ المکرمہ'],['مکہ','مکہ المکرمہ','مکت المکرمہ'])]
def retrieval_hints(query,terms):
    wanted=normalize(query);extra=[]
    for aliases,variants in PLACE_VARIANTS:
        if any(re.search(r'(?<!\w)'+re.escape(normalize(alias))+r'(?!\w)',wanted)for alias in aliases):extra.extend(variants)
    return list(dict.fromkeys([*extra,*terms]))[:4]

def reserve_candidates(query,rows,limit=4):
    """Give observed reviewed misspellings a bounded chance to be validated."""
    wanted=normalize(query);variants=[]
    for aliases,forms in PLACE_VARIANTS:
        if any(re.search(r'(?<!\w)'+re.escape(normalize(alias))+r'(?!\w)',wanted)for alias in aliases):
            canonical={normalize(alias)for alias in aliases}
            variants.extend(form for form in forms if normalize(form)not in canonical)
    found=[row for row in rows if any(normalize(form)in normalize(row.get('text',''))for form in variants)]
    found.sort(key=lambda row:(-row.get('retrieval_score',row.get('score',0)),str(row['video_id']),str(row['id'])))
    return found[:limit]

def canonical_query(query):
    """Canonicalize only reviewed whole-entity queries, never arbitrary claims."""
    value=normalize(query.strip())
    if value in {normalize(alias)for alias in PLACE_VARIANTS[0][0]}:return 'Makkah'
    return query
