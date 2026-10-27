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
