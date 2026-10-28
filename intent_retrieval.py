"""Spoken-intent retrieval helpers for grounded search.

Exact/alias matching stays in the transcript contract. This module does not
map particular phrases to keywords. It strips search boilerplate, asks the
existing planner for short spoken forms, and stops a single shared audience
word from outranking a passage that covers the proposition.
"""
from __future__ import annotations

import re

_FOLD = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})

# Closed-class words and light verbs. They are not topical evidence.
_FUNCTION = {
    "a", "an", "the", "of", "to", "for", "about", "among", "where", "and", "or",
    "in", "on", "at", "by", "with", "from", "into", "that", "this", "these",
    "those", "me", "my",
    "ko", "ki", "ke", "ka", "kay", "ne", "mein", "main", "men", "par", "pe",
    "se", "aur", "hai", "hain", "tha", "thi", "ho", "woh", "yeh", "ye", "wo",
    "karna", "karne", "karo", "karein", "kiya", "kiye", "kar", "do", "does",
    "did", "baat", "taraf", "wala", "wali", "wale", "lekar",
    "کو", "کی", "کے", "کا", "نے", "میں", "پر", "سے", "اور", "ہے", "ہیں", "تھا",
    "تھی", "یہ", "وہ", "کیے", "کیا", "بات", "طرف", "کرنا", "کرنے",
}

# Audience and search-meta words. Alone they do not identify an action.
_GENERIC = {
    "children", "child", "kids", "people", "leader", "speech", "clip", "moment",
    "video", "bachon", "bachay", "bache", "bacha", "bachchon", "bacho", "log",
    "بچوں", "بچے", "بچو", "لوگوں", "لوگ", "لیڈر",
}

_EN_PREFIXES = (
    re.compile(
        r"^(?:please\s+)?(?:find|show|get|search)(?:\s+me)?\s+(?:the\s+)?"
        r"(?:clip|speech|moment|video|part)\s+(?:where|about|of|in\s+which)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?:find|show|get|search)(?:\s+me)?\s+(?:the\s+)?speech\s+about\s+",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:a|the)\s+speech\s+about\s+", re.IGNORECASE),
)
_ROMAN_REPORT = re.compile(
    r"^(?:the\s+)?leader\s+ne\s+(.+?)\s+ki\s+baat\s+ki\s*\.?$",
    re.IGNORECASE,
)
_URDU_REPORT = re.compile(r"^لیڈر\s+نے\s+(.+?)\s+کی\s+بات\s+کی\s*۔?$")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def proposition(query: str) -> str:
    """Drop search instructions and keep the spoken proposition.

    "Find the speech about X" and "leader ne X ki baat ki" become X. A query
    that is already a topic, name, or alias is unchanged.
    """
    text = " ".join((query or "").split())
    if not text:
        return text
    for _ in range(3):
        updated = text
        for pattern in _EN_PREFIXES:
            updated = pattern.sub("", updated, count=1)
        updated = updated.strip(" .?!۔")
        report = _ROMAN_REPORT.match(updated) or _URDU_REPORT.match(updated)
        if report:
            updated = report.group(1).strip(" .?!۔")
        updated = " ".join(updated.split())
        if updated == text or len(updated.split()) < 2 or len(updated) < 8:
            break
        text = updated
    return text


def compose_retrieval_queries(semantic: str, plan: dict | None, limit: int = 4) -> list[str]:
    """Order dense queries: proposition, spoken forms, faithful translations.

    The raw search question is last and is dropped once four embeddings are
    filled, so boilerplate does not crowd out the proposition.
    """
    plan = plan or {}
    spoken = _clean(plan.get("spoken_queries") or [], 400)
    faithful = _clean(plan.get("queries") or [], 2000)
    primary = proposition(semantic or "")
    ordered: list[str] = []
    for value in [primary, *spoken, *faithful, (semantic or "").strip()]:
        if value and len(value) <= 2000 and value not in ordered:
            ordered.append(value)
    return ordered[:limit] or ([semantic] if semantic else [])


def alignment_score(variants, text: str) -> float:
    """Best coverage of one retrieval variant, in 0..1.

    Variants are scored separately. A generic audience word submitted as its
    own variant cannot outscore a proposition that also names an action.
    """
    if isinstance(variants, str):
        variants = [variants]
    distinctive = []
    generic_only = []
    for variant in variants or []:
        if not isinstance(variant, str) or not variant.strip():
            continue
        weights = _weighted(variant)
        if not weights:
            continue
        covered = _coverage(weights, text)
        if any(weight >= 1 for weight in weights.values()):
            distinctive.append(covered)
        else:
            generic_only.append(covered)
    chosen = distinctive or generic_only
    return max(chosen) if chosen else 0.0


def keyword_vote_weight(text: str, queries, terms) -> float:
    """1 when a distinctive proposition token matches, else a weak audience vote.

    A query that is only an audience word still gets a full vote. A passage
    that shares only that audience word with a more specific query does not.
    """
    variants = [*(queries or []), *(terms or [])]
    weights: dict[str, float] = {}
    for variant in variants:
        if isinstance(variant, str):
            for token, weight in _weighted(variant).items():
                weights[token] = max(weights.get(token, 0.0), weight)
    if not weights:
        return 1.0
    text_tokens = _content_tokens(text)
    distinctive = [token for token, weight in weights.items() if weight >= 1]
    matched_distinctive = sum(1 for token in distinctive if _covers(token, text_tokens))
    matched_any = any(_covers(token, text_tokens) for token in weights)
    if matched_distinctive:
        return 1.0
    if not distinctive and matched_any:
        return 1.0
    if matched_any:
        return 0.2
    return 0.0


def _clean(values, limit: int) -> list[str]:
    cleaned = []
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if 0 < len(text) <= limit and text not in cleaned:
                cleaned.append(text)
    return cleaned


def fold_token(token: str) -> str:
    text = token.casefold().translate(_FOLD)
    if re.search(r"[a-z]", text):
        text = text.replace("aa", "a").replace("ee", "i").replace("ii", "i").replace("oo", "u")
        text = re.sub(r"([a-z])\1+", r"\1", text)
    return text


def _tokens(text: str):
    return _TOKEN.findall(text or "")


def _content_tokens(text: str) -> list[str]:
    tokens = []
    for raw in _tokens(text):
        token = fold_token(raw)
        if token in _FUNCTION:
            continue
        minimum = 3 if re.search(r"[a-z]", token) else 2
        if len(token) < minimum:
            continue
        tokens.append(token)
    return tokens


def _weighted(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for token in _content_tokens(text):
        weight = 0.15 if token in _GENERIC else 1.0
        weights[token] = max(weights.get(token, 0.0), weight)
    return weights


def _coverage(weights: dict[str, float], text: str) -> float:
    if not weights:
        return 0.0
    text_tokens = _content_tokens(text)
    total = sum(weights.values())
    covered = sum(weight for token, weight in weights.items() if _covers(token, text_tokens))
    return covered / total


def keys_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) < 4 or len(right) < 4:
        return False
    if left.startswith(right) or right.startswith(left):
        return True
    shared = 0
    for a, b in zip(left, right):
        if a != b:
            break
        shared += 1
    return shared >= 4 and len(left) - shared <= 3 and len(right) - shared <= 3


def _covers(token: str, text_tokens: list[str]) -> bool:
    return any(keys_match(token, other) for other in text_tokens)
