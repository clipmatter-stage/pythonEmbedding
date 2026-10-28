"""Retrieval hints only. Original transcript text is always the evidence.

Near-miss Roman spellings fold onto every reviewed alias family with one
phonetic key. Families are data. Nothing here maps a full question to keywords.
"""
import re
from functools import lru_cache
from intent_retrieval import fold_token
from transcript_search_contract import CONFIG, normalize
VERSION='asr_context_v2'
# Reviewed spelling families. Query aliases trigger retrieval; forms are what
# transcripts contain, including observed ASR spellings such as مکت / مکۃ.
_PLACE_ALIASES=['makkah','mecca','مکہ','مکہ المکرمہ']
_PLACE_FORMS=['مکہ','مکہ المکرمہ','مکت المکرمہ','مکۃ المکرمہ']
_AGREEMENT_ALIASES=['mohida','معاہدہ','معاہدے']
_AGREEMENT_FORMS=['معاہدہ','معاہدے','ایگریمنٹ']
PLACE_VARIANTS=[(_PLACE_ALIASES,_PLACE_FORMS)]
AGREEMENT_VARIANTS=[(_AGREEMENT_ALIASES,_AGREEMENT_FORMS)]
_PLACE_QUERY_KEYS={normalize(alias) for alias in _PLACE_ALIASES}
_LATIN=re.compile(r'^[a-z]+$')
# Trailing sentence marks are not a new request. They must not fork the
# judgment cache from the same words.
_TRAILING_MARKS=re.compile(r'(?:[\s.!?؟،,;:۔…])+$')
# Urdu topic heads in ـہ. The nominative is the retrieval and cache form.
# Oblique/plural ـے and ـوں fold onto it. Function words are not in this table.
_TOPIC_HEADS=(
    ('معاہدہ',('معاہدہ','معاہدے','معاہدوں')),
)
_INFLECTION_OF={}
for _canonical,_forms in _TOPIC_HEADS:
    for _form in _forms:_INFLECTION_OF[normalize(_form)]=_canonical

def roman_key(token):
    """Skeleton of a Roman token. Aspiration h and short a are spelling noise."""
    text=fold_token(token or '')
    if not _LATIN.fullmatch(text) or len(text)<4:return None
    text=text.replace('h','').replace('u','o')
    text=re.sub(r'a+','',text)
    text=re.sub(r'([a-z])\1+',r'\1',text)
    return text if len(text)>=2 else None

def _distance(left,right):
    if abs(len(left)-len(right))>2:return 3
    previous=list(range(len(right)+1))
    for index,char in enumerate(left,1):
        current=[index]
        for other_index,other in enumerate(right,1):
            current.append(min(previous[other_index]+1,current[other_index-1]+1,previous[other_index-1]+(char!=other)))
        previous=current
    return previous[-1]

def _latin_word(value):
    text=fold_token(value or '')
    return bool(_LATIN.fullmatch(text)) and ' ' not in value and '_' not in value

@lru_cache(maxsize=1)
def alias_families():
    """Canonical Roman form and single-token surfaces for every reviewed family."""
    named=[('makkah',_PLACE_ALIASES),('mohida',_AGREEMENT_ALIASES)]
    for key,aliases in CONFIG['aliases'].items():
        named.append((key,[key,*aliases]))
    families=[]
    for canonical,surfaces in named:
        singles=tuple(s for s in surfaces if ' ' not in s and '_' not in s)
        latin=[s for s in singles if _latin_word(s) and len(fold_token(s))>=4]
        if not latin:continue
        chosen=canonical if _latin_word(canonical) and len(fold_token(canonical))>=4 else latin[0]
        families.append((chosen,singles))
    return tuple(families)

def _shape(token):
    """Folded Roman form with u/o treated as the same vowel."""
    return fold_token(token).replace('u','o')

def nearest_alias(token):
    """Fold one near-miss token onto the single reviewed family it belongs to.

    An exact reviewed spelling is left alone. Ambiguous skeletons are left
    alone. This is not limited to one place or one agreement.
    """
    key=roman_key(token)
    if not key:return None
    shape=_shape(token)
    hits=[]
    for canonical,surfaces in alias_families():
        if any(normalize(surface)==normalize(token) for surface in surfaces):
            return None
        if any(roman_key(surface)==key and _distance(shape,_shape(surface))<=2 and shape[:2]==_shape(surface)[:2] for surface in surfaces):
            hits.append(canonical)
    hits=list(dict.fromkeys(hits))
    return hits[0] if len(hits)==1 else None

def _triggered(wanted,aliases):
    return any(re.search(r'(?<!\w)'+re.escape(normalize(alias))+r'(?!\w)',wanted) for alias in aliases)

def _matching_forms(query):
    wanted=normalize(query)
    return [variants for aliases,variants in [*PLACE_VARIANTS,*AGREEMENT_VARIANTS] if _triggered(wanted,aliases)]

def _rewrite_tokens(text):
    parts=text.split();changed=False;mapped=[]
    for token in parts:
        replacement=nearest_alias(token)
        if replacement:
            mapped.append(replacement);changed=True
        else:mapped.append(token)
    return ' '.join(mapped) if changed else text

def retrieval_hints(query,terms):
    """Native forms from each matched family, then extra spellings, then the plan."""
    triggered=_matching_forms(canonical_query(query) if query else query);extra=[]
    for variants in triggered:
        if variants:extra.append(variants[0])
    for variants in triggered:extra.extend(variants[1:])
    return list(dict.fromkeys([*extra,*terms]))[:4]

def reserve_candidates(query,rows,limit=4):
    """Give observed reviewed misspellings a bounded chance to be validated."""
    wanted=normalize(canonical_query(query) if query else query);variants=[]
    for aliases,forms in PLACE_VARIANTS:
        if _triggered(wanted,aliases):
            canonical={normalize(alias)for alias in aliases}
            variants.extend(form for form in forms if normalize(form)not in canonical)
    found=[row for row in rows if any(normalize(form)in normalize(row.get('text',''))for form in variants)]
    found.sort(key=lambda row:(-row.get('retrieval_score',row.get('score',0)),str(row['video_id']),str(row['id'])))
    return found[:limit]

def _fold_topic_heads(text):
    """Map a reviewed oblique/plural topic head onto its nominative.

    Whole tokens only. A longer word, a function word, or an unlisted ـے
    ending is left as written.
    """
    folded=[];changed=False
    for token in text.split():
        canonical=_INFLECTION_OF.get(normalize(token))
        if canonical and normalize(token)!=normalize(canonical):
            folded.append(canonical);changed=True
        else:folded.append(token)
    return ' '.join(folded) if changed else text

def canonical_query(query):
    """One cache and retrieval string for spelling, inflection, and punctuation.

    Near-miss Roman tokens fold onto reviewed alias families. Reviewed Urdu
    topic-head inflections fold onto the nominative. A trailing full stop,
    Urdu danda, or comma is not a distinct request. A whole place name still
    becomes Makkah. Relationships and negations stay in the request.
    """
    text=' '.join((query or '').split())
    text=_TRAILING_MARKS.sub('',text).strip()
    if not text:return query
    text=_fold_topic_heads(text)
    if normalize(text) in _PLACE_QUERY_KEYS:return 'Makkah'
    rewritten=_rewrite_tokens(text)
    if normalize(rewritten) in _PLACE_QUERY_KEYS:return 'Makkah'
    return rewritten
