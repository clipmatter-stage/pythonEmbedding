"""Deterministic decomposition for speaker-plus-topic semantic queries.

This module intentionally has no service dependencies so its behavior can be
tested without Qdrant, OpenAI, or production credentials.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


_SPEAKER_TOPIC_PATTERNS = (
    re.compile(
        r"^\s*(?P<speaker>.+?)\s+(?:speech|talk|remarks|views|discussion)\s+"
        r"(?:on|about|regarding)\s+(?P<topic>.+?)\s*[?.!]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*what\s+did\s+(?P<speaker>.+?)\s+"
        r"(?:say|speak|talk)\s+about\s+(?P<topic>.+?)\s*[?.!]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?P<speaker>.+?)\s+(?:said|spoke|talked)\s+"
        r"(?:on|about)\s+(?P<topic>.+?)\s*[?.!]*\s*$",
        re.IGNORECASE,
    ),
)


def decompose_semantic_query(
    query: str,
    canonical_speaker: Optional[str] = None,
) -> Dict[str, object]:
    """Separate a known speaker constraint from the semantic topic.

    Decomposition is deliberately conservative: wrapper removal is performed
    only when the caller has already resolved a known speaker. Unknown names
    remain untouched so an out-of-domain query cannot accidentally become a
    broad in-domain topic search.
    """

    original = (query or "").strip()
    result: Dict[str, object] = {
        "original_query": original,
        "retrieval_query": original,
        "speaker": canonical_speaker,
        "topic": None,
        "relation": None,
        "decomposed": False,
        "confidence": 0.0,
    }

    if not original or not canonical_speaker:
        return result

    for pattern in _SPEAKER_TOPIC_PATTERNS:
        match = pattern.match(original)
        if not match:
            continue

        topic = re.sub(r"\s+", " ", match.group("topic")).strip(" .?!")
        speaker_text = re.sub(r"\s+", " ", match.group("speaker")).strip(" .?!")
        if len(topic) < 2 or len(speaker_text) < 2:
            continue

        result.update({
            "retrieval_query": topic,
            "speaker": canonical_speaker,
            "topic": topic,
            "relation": "spoken_by",
            "decomposed": True,
            "confidence": 1.0,
        })
        return result

    return result


def has_minimum_topic_evidence(text: str, min_characters: int = 24, min_words: int = 6) -> bool:
    """Reject fragments too small to support a semantic topic assertion."""

    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if len(normalized) < min_characters:
        return False

    return len([word for word in normalized.split(" ") if word]) >= min_words


def has_conceptual_topic_evidence(topic: str, text: str) -> bool:
    """Apply narrow deterministic guards for known ambiguous concepts.

    Most topics remain governed by semantic retrieval and LLM reranking. The
    Parliament guard exists because models can mistake the occupational title
    "parliamentarian" for a passage about the institution itself.
    """

    normalized_topic = re.sub(r"[^a-z0-9]+", " ", (topic or "").casefold()).strip()
    if normalized_topic not in {"parliament", "parliamentary"}:
        return True

    normalized_text = re.sub(r"\s+", " ", (text or "").casefold()).strip()
    if not normalized_text:
        return False

    institutional_evidence = (
        r"\bparliament\b",
        r"\blegislatur(?:e|es)\b",
        r"\blegislat(?:ion|ive|ing)\b",
        r"\blaw[ -]?making\b",
        r"\bnational assembly\b",
        r"\bprovincial assembly\b",
        r"\bsenate\b",
        r"\bmember(?:s)? of parliament\b",
        r"\bparliamentary (?:debate|authority|power|powers|committee|committees|session|sessions|vote|voting|accountability)\b",
        r"پارلیمنٹ",
        r"قومی اسمبلی",
        r"صوبائی اسمبلی",
        r"قانون سازی",
        r"مجلس شوری",
        r"سینیٹ",
        r"ایوان",
    )
    return any(re.search(pattern, normalized_text, re.IGNORECASE) for pattern in institutional_evidence)
