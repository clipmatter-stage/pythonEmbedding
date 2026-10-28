"""Speaker constraints for grounded search.

Exact and alias transcript matches stay in the search contract. This module
does not map phrases to topic keywords. It only decides whether a query names
a known speaker, whether the rest is speech boilerplate or a topic, and
whether a stored speaker label belongs to that person's alias family.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from semantic_query_decomposition import extract_meaningful_query_terms
from speaker_identity import matches_identity
from transcript_search_contract import CONFIG, normalize

# Closed-class particles. They are not a topic.
_FUNCTION = {
    "a", "an", "the", "of", "to", "for", "about", "on", "in", "while", "during",
    "giving", "delivering", "and", "or", "me", "please",
    "ko", "ki", "ke", "ka", "kay", "ne", "mein", "main", "men", "par", "pe", "se",
    "huwe", "hue", "huye", "huway", "howe", "howein",
    "کو", "کی", "کے", "کا", "نے", "میں", "پر", "سے", "ہوئے", "ہوا", "ہوے", "ہیں",
}

# Speech-act boilerplate. Alone, these ask for the person's speech.
_SPEECH = {
    "bayan", "bayaan", "bayaan", "taqreer", "taqrir", "khitab", "khitaab",
    "speech", "speeches", "talk", "talks", "talking", "speak", "speaks",
    "speaking", "spoke", "remark", "remarks", "address", "addressing",
    "bolte", "bolta", "bolti", "bol", "karte", "karta", "karti", "kartay",
    "kartey", "rahe", "raha", "rahi", "rahay", "kerte", "kerta",
    "بیان", "تقریر", "خطاب", "بولتے", "بولتا", "کرتے", "کرتا",
}

_KEEP_SHORT = {"hnr", "h n r"}
_NAME_MARKERS = (
    "hafiz", "hafez", "hafis", "naeem", "naim", "naem", "neem", "rehman",
    "rahman", "حافظ", "نعیم", "نائیم", "رحمن", "رحمان",
)
_LEAD = re.compile(
    r"^(?:please\s+)?(?:find|show|get|search)(?:\s+me)?\s+"
    r"(?:(?:the|a)\s+)?(?:(?:clip|clips|moment|video|part|speech|bayan|taqreer)s?\s+)?"
    r"(?:(?:where|of|about|in\s+which)\s+)?",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_ERGATIVE = re.compile(r"^(?:ne|نے)\s+", re.IGNORECASE)
_ABOUT_LEAD = re.compile(r"^(?:ki|ke|ka|kay|کی|کے|کا)(?:\s|$)")
_CRITICISM = re.compile(r"(?:tanqeed|criticism|criticize|criticising|تنقید)")


def strip_search_lead(query: str) -> str:
    text = " ".join((query or "").split())
    updated = _LEAD.sub("", text, count=1).strip()
    return updated or text


def speech_wrapper_only(text: str) -> bool:
    """True when the remainder only asks to hear the person speak."""
    tokens = _tokens(text)
    if not tokens:
        return False
    topical = [token for token in tokens if token not in _FUNCTION and token not in _SPEECH]
    return not topical and any(token in _SPEECH for token in tokens)


def is_about_person(text: str) -> bool:
    """Possessive commentary about a person, not a request for their speech."""
    if speech_wrapper_only(text):
        return False
    folded = normalize(text or "")
    if _ABOUT_LEAD.match(folded):
        return True
    return bool(_CRITICISM.search(folded))


def token_close(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) < 4 or len(right) < 4 or abs(len(left) - len(right)) > 1:
        return False
    if left[0] != right[0]:
        return False
    return fuzz.ratio(left, right) >= 80


def same_name(label: str, alias: str) -> bool:
    """Whole-label equivalence across script, order-preserving particles, and spelling."""
    left = normalize(label or "").split()
    right = normalize(alias or "").split()
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) >= 2 and len(right) >= 2:
        short, long = (left, right) if len(left) <= len(right) else (right, left)
        if long[:len(short)] == short:
            return True
        if len(left) == len(right) and all(token_close(a, b) for a, b in zip(left, right)):
            return True
    return False


def _tokens(text: str) -> list[str]:
    return [normalize(token) for token in _TOKEN.findall(text or "") if normalize(token)]


def _is_name_form(value: str) -> bool:
    folded = normalize(value or "")
    if folded in _KEEP_SHORT:
        return True
    tokens = folded.split()
    if len(tokens) < 2:
        return False
    return any(token == marker or token_close(token, marker) for token in tokens for marker in _NAME_MARKERS)


def constraint_names(person_key: str, person_table: dict) -> list[str]:
    """Curated surfaces for one person: contract family, stored variants, name aliases.

    Ambiguous single tokens (hafiz, naeem, نعیم) stay out. Role titles that do
    not contain a name marker stay out.
    """
    data = person_table.get(person_key) or {}
    surfaces = [data.get("canonical", ""), *data.get("aliases", []), *data.get("speaker_variants", [])]
    normalized = {normalize(value) for value in surfaces if normalize(value)}
    names: list[str] = []
    seen = set()

    def add(value: str) -> None:
        text = " ".join(str(value or "").split())
        key = normalize(text)
        if text and key and key not in seen and _is_name_form(text):
            seen.add(key)
            names.append(text)

    for key, aliases in CONFIG.get("aliases", {}).items():
        group = [key, *aliases]
        if normalized.intersection(normalize(item) for item in group):
            for item in group:
                add(item)
    for value in surfaces:
        add(value)
    return names


def leading_name(text: str, names: list[str]):
    """Return (matched surface, remainder) when a curated name starts the query."""
    raw = " ".join((text or "").split())
    if not raw:
        return None
    for name in sorted({item.strip() for item in names if item and item.strip()}, key=len, reverse=True):
        if len(name) < 2:
            continue
        match = re.match(re.escape(name) + r"(?!\w)(?:\s+|$)(.*)$", raw, flags=re.IGNORECASE)
        if match:
            return name, match.group(1).strip()
    tokens = raw.split()
    for size in range(min(6, len(tokens)), 1, -1):
        head = " ".join(tokens[:size])
        if any(same_name(head, name) for name in names):
            return head, " ".join(tokens[size:]).strip()
    return None


def leading_person_key(query: str, person_table: dict):
    """Person whose curated name starts the query, including spelling noise."""
    stripped = strip_search_lead(query)
    for key in person_table:
        if leading_name(stripped, constraint_names(key, person_table)):
            return key
    return None


def label_matches(label: str, names: list[str]) -> bool:
    if not label or not names:
        return False
    if normalize(label) in {"unknown", "unknown speaker"}:
        return False
    return any(same_name(label, name) for name in names)


def _labels(row: dict) -> list[str]:
    labels = []
    for key in ("source_speaker_label", "speaker", "diarization_speaker", "transcript_speaker", "identity_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            labels.append(value)
    return labels


def passes_constraint(row: dict, speaker: str, names: list[str] | None) -> bool:
    """Verified identity, or a curated alias-family label.

    A one-value speaker filter still requires reviewed identity. An alias
    family (several curated surfaces) also accepts the stored source label,
    including Urdu and close spellings. That does not mark the row verified.
    """
    if not speaker:
        return True
    if matches_identity(speaker, row):
        return True
    curated = [name for name in (names or []) if name]
    if len(curated) <= 1:
        return False
    return any(label_matches(label, curated) for label in _labels(row))


def refine_speaker_query(query, decomposition, canonical, names):
    """Split a leading known speaker from a topic or a speech-only request.

    Existing structured decomposition wins. A bare name is left unchanged so
    title lookup can handle it. Possessive commentary ("ki talqeen par
    tanqeed") is not treated as that person speaking.
    """
    if not canonical or not names or (decomposition or {}).get("decomposed"):
        return decomposition
    found = leading_name(strip_search_lead(query), names)
    if not found:
        return decomposition
    _matched, tail = found
    tail = _ERGATIVE.sub("", tail).strip()
    if not tail:
        return decomposition
    if speech_wrapper_only(tail):
        updated = dict(decomposition)
        updated.update({
            "retrieval_query": " ".join((query or "").split()),
            "speaker": canonical,
            "topic": None,
            "topic_phrase": None,
            "meaningful_topic_terms": [],
            "intent": "speaker_only",
            "intent_confidence": 1.0,
            "relation": "spoken_by",
            "decomposed": True,
            "confidence": 1.0,
        })
        return updated
    if is_about_person(tail):
        return decomposition
    topic = re.sub(r"^(?:about|regarding)\s+", "", tail, count=1, flags=re.IGNORECASE).strip() or tail
    updated = dict(decomposition)
    updated.update({
        "retrieval_query": topic,
        "speaker": canonical,
        "topic": topic,
        "topic_phrase": topic,
        "meaningful_topic_terms": extract_meaningful_query_terms(topic),
        "intent": "speaker_topic_search",
        "intent_confidence": 1.0,
        "relation": "spoken_by",
        "decomposed": True,
        "confidence": 1.0,
    })
    return updated
