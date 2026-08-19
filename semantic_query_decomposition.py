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
    Guards exist only for concepts where production evidence showed that the
    reranker can join unrelated occurrences of the component words.
    """

    normalized_topic = re.sub(r"[^a-z0-9]+", " ", (topic or "").casefold()).strip()
    normalized_text = re.sub(r"\s+", " ", (text or "").casefold()).strip()
    if not normalized_text:
        return False

    def contains_any(patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, normalized_text, re.IGNORECASE) for pattern in patterns)

    def contains_near(
        left_patterns: tuple[str, ...],
        right_patterns: tuple[str, ...],
        maximum_gap: int,
    ) -> bool:
        left_matches = [
            match
            for pattern in left_patterns
            for match in re.finditer(pattern, normalized_text, re.IGNORECASE)
        ]
        right_matches = [
            match
            for pattern in right_patterns
            for match in re.finditer(pattern, normalized_text, re.IGNORECASE)
        ]
        return any(
            max(left.start(), right.start()) - min(left.end(), right.end()) <= maximum_gap
            for left in left_matches
            for right in right_matches
        )

    if normalized_topic in {"parliament", "parliamentary"}:
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
        return contains_any(institutional_evidence)

    if normalized_topic == "leadership development":
        explicit_leadership_development = (
            r"\bleadership[ -]?(?:development|training|skills?|capacity|program(?:me)?)\b",
            r"\bdevelopment of (?:new |young )?leaders(?:hip)?\b",
            r"\bdevelop(?:ing|ed)? (?:new |young )?leaders\b",
            r"\btrain(?:ing|ed)? (?:new |young )?leaders\b",
            r"\bmentor(?:ing|ed)? (?:new |young )?leaders\b",
            r"قیادت کی (?:تربیت|تیاری|صلاحیت)",
            r"لیڈر شپ (?:تربیت|تیاری|صلاحیت)",
        )
        if contains_any(explicit_leadership_development):
            return True

        leadership_evidence = (
            r"\bleader(?:s|ship)?\b",
            r"\bleadership\b",
            r"لیڈر",
            r"قیادت",
            r"رہنمائی",
        )
        development_evidence = (
            r"\btrain(?:ing|ed)?\b",
            r"\bmentor(?:ing|ship|ed)?\b",
            r"\bcapacity[ -]?build(?:ing)?\b",
            r"\bprepar(?:e|ing|ation)\b",
            r"\bskills?\b",
            r"تربیت",
            r"صلاحیت",
            r"تیار",
            r"نشوونما",
            r"کردار سازی",
        )
        return contains_near(leadership_evidence, development_evidence, maximum_gap=120)

    if normalized_topic in {"farmers rights", "farmer rights", "farmers right", "farmer right"}:
        farmer_evidence = (
            r"\bfarmers?\b",
            r"\bgrowers?\b",
            r"\bagricultur(?:e|al)\b",
            r"\bcrops?\b",
            r"کسان",
            r"زراعت",
            r"فصل",
            r"گندم",
            r"چاول",
            r"کپاس",
        )
        rights_or_welfare_evidence = (
            r"\brights?\b",
            r"\bentitle(?:ment|d)?\b",
            r"\bfair price\b",
            r"\bsubsid(?:y|ies)\b",
            r"\bcompensat(?:ion|e)\b",
            r"\blivelihood\b",
            r"\bexploit(?:ation|ed)?\b",
            r"حقوق?",
            r"مطالبہ",
            r"قیمت",
            r"سبسڈی",
            r"معاوضہ",
            r"آمدن",
            r"روزگار",
            r"استحصال",
            r"پریشان",
            r"کچھ بھی نہیں ملتا",
        )
        return contains_near(farmer_evidence, rights_or_welfare_evidence, maximum_gap=180)

    return True


def passes_structured_topic_validation(
    result: Dict[str, object],
    topic: str,
    minimum_score: float = 0.65,
) -> bool:
    """Require an explicit, complete, non-incidental LLM topic judgment.

    Structured speaker-plus-topic searches favor precision. Missing judgment
    fields therefore fail closed instead of allowing a keyword/title match to
    bypass validation when the model response is incomplete.
    """

    try:
        relevance_score = float(result.get("llm_relevance_score", 0) or 0)
    except (TypeError, ValueError):
        return False

    if relevance_score < minimum_score:
        return False
    if result.get("llm_complete_topic") is not True:
        return False
    if result.get("llm_incidental_match") is not False:
        return False

    return has_conceptual_topic_evidence(topic, str(result.get("text", "") or ""))


def has_complete_facet_coverage(
    required_facets: object,
    supported_facets: object,
) -> bool:
    """Return true only when every model-declared required facet is supported."""

    if not isinstance(required_facets, (list, tuple, set)):
        return False
    if not isinstance(supported_facets, (list, tuple, set)):
        return False

    required = {str(facet).strip() for facet in required_facets if str(facet).strip()}
    supported = {str(facet).strip() for facet in supported_facets if str(facet).strip()}
    return bool(required) and required.issubset(supported)
