"""Bounded passage validation; retrieval metadata is never passage evidence.

No service imports or credentials. Judges are injected so behavior can be tested
separately from model accuracy. Source times are already seconds (Stage 1).
"""
from __future__ import annotations

import math
import logging
import uuid
import time
import unicodedata
from collections import OrderedDict


class ValidationUnavailable(RuntimeError):
    """Retryable provider or protocol failure, distinct from zero matches."""
    def __init__(self, message, *, diagnostic_id=None, reason=None):
        super().__init__(message)
        self.diagnostic_id = diagnostic_id
        self.reason = reason


logger = logging.getLogger(__name__)
_VALIDATION_REASONS = {
    'Invalid required facets': 'invalid_required_facets',
    'Facet plan changed between batches': 'facet_plan_changed',
    'Incomplete validation response': 'incomplete_judgments',
    'Invalid passage indices': 'invalid_indices',
    'Invalid or duplicate passage indices': 'duplicate_or_missing_indices',
    'Invalid boolean score': 'invalid_score',
    'Invalid passage judgment': 'invalid_judgment',
    'Missing facet evidence': 'missing_facet_evidence',
    'Evidence quote not present in passage': 'quote_not_in_passage',
    'Validation provider unavailable': 'provider_not_configured',
}


def validation_failure_reason(exc):
    # Never log exception messages from providers or arbitrary caller data.
    if isinstance(exc, ValidationUnavailable):
        return _VALIDATION_REASONS.get(str(exc), 'validation_failed')
    name = type(exc).__name__
    return {'JSONDecodeError': 'invalid_json', 'TimeoutError': 'provider_timeout',
            'APITimeoutError': 'provider_timeout', 'RateLimitError': 'provider_rate_limit',
            'AuthenticationError': 'provider_authentication',
            'PermissionDeniedError': 'provider_permission',
            'APIConnectionError': 'provider_connection',
            'AttributeError': 'invalid_response_type', 'KeyError': 'missing_response_field',
            'ValueError': 'invalid_response_value'}.get(name, 'provider_or_protocol_error')


def normalize_multilingual(value):
    # Preserve aspirated consonants and vowel distinctions. Only canonicalize
    # Arabic keyboard variants of Urdu yeh/kaf and punctuation.
    table = str.maketrans({'ي': 'ی', 'ى': 'ی', 'ك': 'ک', '’': "'", '‘': "'",
                           '“': '"', '”': '"', '،': ',', '؟': '?', '۔': '.',
                           '–': '-', '—': '-', '\u200c': ' ', '\u200d': ' '})
    return ' '.join(unicodedata.normalize('NFKC', str(value or '')).translate(table).casefold().split())


def consolidate_candidates(candidates, max_chars=2400, max_seconds=60):
    """Deduplicate IDs and combine only contiguous, same-speaker evidence.

    Distinct overlapping statements survive. Never merge across video/speaker
    boundaries or join separated excerpts into invented context.
    """
    unique = OrderedDict()
    for item in candidates:
        key = (str(item.get('video_id')), str(item.get('id')))
        if not item.get('id') or not item.get('text'):
            continue
        if key not in unique or item.get('score', 0) > unique[key].get('score', 0):
            unique[key] = dict(item)
    ordered = sorted(unique.values(), key=lambda r: (str(r.get('video_id')), float(r.get('start_time', 0)), str(r['id'])))
    output = []
    for item in ordered:
        item['segment_ids'] = list(item.get('segment_ids') or [item['id']])
        item['match_count'] = len(item['segment_ids'])
        if output:
            previous = output[-1]
            previous_label = previous.get('transcript_speaker') or previous.get('diarization_speaker')
            item_label = item.get('transcript_speaker') or item.get('diarization_speaker')
            identity = normalize_multilingual(item.get('speaker'))
            known_identity = identity not in {'', 'unknown', 'unknown speaker'}
            same_speaker = ((previous.get('speaker'), previous_label) == (item.get('speaker'), item_label)
                            and (bool(item_label) or known_identity))
            gap = float(item.get('start_time', 0)) - float(previous.get('end_time', 0))
            duration = float(item.get('end_time', 0)) - float(previous.get('start_time', 0))
            combined = previous['text'] + ' ' + item['text']
            if (same_speaker and previous.get('video_id') == item.get('video_id')
                    and 0 <= gap <= 1 and 0 <= duration <= max_seconds and len(combined) <= max_chars):
                previous['text'] = combined
                previous['segment_ids'] += item['segment_ids']
                previous['match_count'] = len(previous['segment_ids'])
                previous['end_time'] = item['end_time']
                previous['duration'] = duration
                previous['score'] = max(previous.get('score', 0), item.get('score', 0))
                continue
        output.append(item)
    return sorted(output, key=lambda r: (-float(r.get('score', 0)), str(r.get('video_id')), float(r.get('start_time', 0)), str(r['id'])))


def validate_passages(query, candidates, judge, *, speaker_names=(), max_candidates=60,
                      batch_size=20, max_calls=3, deadline_seconds=30, clock=time.monotonic):
    """Validate beyond the first 30, with bounded sequential expansion.

    Every returned passage is independently judged on its complete displayed
    text. A global, model-declared facet set must be covered by each passage;
    each facet must have an exact evidence quote found in that passage.
    """
    diagnostic_id = uuid.uuid4().hex
    started = clock()
    pool = consolidate_candidates(candidates)
    if speaker_names:
        allowed = {normalize_multilingual(name) for name in speaker_names}
        pool = [r for r in pool if any(normalize_multilingual(r.get(k)) in allowed for k in ('speaker', 'diarization_speaker'))]
    # One best candidate per video first; then keep the remaining stable order.
    seen = set(); first = []; tail = []
    for r in pool:
        if r.get('video_id') in seen:
            tail.append(r)
        else:
            first.append(r); seen.add(r.get('video_id'))
    pool = first + tail
    # These hard caps cannot be raised by callers accidentally.
    maximum = min(max_candidates, 60); batch_size = min(batch_size, 20)
    max_calls = min(max_calls, 3); deadline_seconds = min(deadline_seconds, 30)
    accepted = []; checked = 0; calls = 0; failure = None; input_chars = 0; required_facets = None
    eligible = [r for r in pool[:maximum] if len(r['text']) <= 2400]
    failure_reason = None
    logger.info('PASSAGE_VALIDATION_START diagnostic_id=%s candidates=%d eligible=%d', diagnostic_id, len(pool), len(eligible))
    for offset in range(0, len(eligible), batch_size):
        remaining = deadline_seconds - (clock() - started)
        if calls >= max_calls or remaining <= 0:
            break
        batch = eligible[offset:offset + batch_size]
        calls += 1
        input_chars += sum(len(r['text']) for r in batch)
        logger.info('PASSAGE_VALIDATION_BATCH diagnostic_id=%s batch=%d candidates=%d timeout_seconds=%.2f', diagnostic_id, calls, len(batch), min(10, remaining))
        try:
            response = judge(query, batch, min(10, remaining), required_facets)
            facets = response.get('required_facets')
            judgments = response.get('passages')
            if not isinstance(facets, list) or not facets or not all(isinstance(f, str) and f.strip() for f in facets):
                raise ValidationUnavailable('Invalid required facets')
            if required_facets is not None and facets != required_facets:
                raise ValidationUnavailable('Facet plan changed between batches')
            required_facets = facets
            if not isinstance(judgments, list) or len(judgments) != len(batch):
                raise ValidationUnavailable('Incomplete validation response')
            if not all(isinstance(j, dict) and type(j.get('index')) is int for j in judgments):
                raise ValidationUnavailable('Invalid passage indices')
            by_index = {j.get('index'): j for j in judgments}
            if set(by_index) != set(range(len(batch))):
                raise ValidationUnavailable('Invalid or duplicate passage indices')
            batch_accepted = []
            for index, r in enumerate(batch):
                judgment = by_index[index]
                if isinstance(judgment.get('score'), bool):
                    raise ValidationUnavailable('Invalid boolean score')
                score = float(judgment['score'])
                if not math.isfinite(score) or not 0 <= score <= 1 or type(judgment.get('complete')) is not bool:
                    raise ValidationUnavailable('Invalid passage judgment')
                if not judgment['complete'] or score < 0.65:
                    continue
                evidence = judgment.get('evidence')
                if not isinstance(evidence, dict) or not set(facets).issubset(evidence):
                    raise ValidationUnavailable('Missing facet evidence')
                text = normalize_multilingual(r['text'])
                if not all(isinstance(evidence[f], str) and len(normalize_multilingual(evidence[f])) >= 3
                           and normalize_multilingual(evidence[f]) in text for f in facets):
                    raise ValidationUnavailable('Evidence quote not present in passage')
                batch_accepted.append({**r, 'score': score, 'llm_relevance_score': score,
                    'llm_complete_topic': True, 'llm_incidental_match': False,
                    'llm_required_facets': facets, 'llm_supported_facets': facets,
                    'passage_evidence': evidence, 'intent_match': True,
                    'match_types': ['semantic', 'validated_passage']})
            accepted.extend(batch_accepted)
            checked += len(batch)
        except Exception as exc:
            failure = type(exc).__name__
            failure_reason = validation_failure_reason(exc)
            status = getattr(exc, 'status_code', None)
            status = status if type(status) is int else None
            logger.error('PASSAGE_VALIDATION_FAILED diagnostic_id=%s batch=%d reason=%s exception_type=%s provider_status=%s evaluated=%d accepted=%d',
                         diagnostic_id, calls, failure_reason, failure, status, checked, len(accepted))
            break
    incomplete = checked < len(pool)
    if failure and not accepted:
        raise ValidationUnavailable('Passage validation unavailable; retry the search', diagnostic_id=diagnostic_id, reason=failure_reason)
    status = 'partial' if incomplete else ('completed' if accepted else 'no_matches')
    accepted.sort(key=lambda r: (-r['score'], str(r.get('video_id')), float(r.get('start_time', 0)), str(r['id'])))
    # Keep each video contiguous so cursor pages never split its passages.
    groups = OrderedDict()
    for r in accepted:
        groups.setdefault(r.get('video_id'), []).append(r)
    accepted = [r for group in groups.values() for r in group]
    metadata = {'diagnostic_id': diagnostic_id, 'authority': 'python_passage_v1', 'status': status, 'retryable': bool(failure),
                'provider_failure': failure, 'candidate_count': len(pool), 'evaluated_count': checked,
                'provider_calls': calls, 'elapsed_seconds': round(clock() - started, 3),
                'input_passage_characters': input_chars,
                'limits': {'candidates': maximum, 'calls': max_calls, 'deadline_seconds': deadline_seconds,
                           'characters_per_passage': 2400, 'output_tokens_per_call': 2400},
                'counts': {'videos': len({r.get('video_id') for r in accepted}), 'passages': len(accepted),
                           'source_segments': sum(r['match_count'] for r in accepted), 'occurrences': None}}
    logger.info('PASSAGE_VALIDATION_COMPLETE diagnostic_id=%s status=%s evaluated=%d accepted=%d provider_calls=%d elapsed_seconds=%s', diagnostic_id, status, checked, len(accepted), calls, metadata['elapsed_seconds'])
    return accepted, metadata


def requested_speaker_names(query, aliases, explicit=None):
    """Resolve only a named speaker with an explicit speaking/action wrapper.

    Full names or configured acronyms are allowed; ambiguous standalone name
    components never create a speaker constraint for a topic search.
    """
    import re
    normalized = normalize_multilingual(query)
    for data in aliases.values():
        names = [data.get('canonical', '')] + list(data.get('aliases', []))
        for name in names:
            alias = normalize_multilingual(name)
            if not alias:
                continue
            if explicit and normalize_multilingual(explicit) == alias:
                return list(dict.fromkeys([data['canonical']] + list(data.get('speaker_variants', []))))
            if len(alias.split()) < 2 and alias not in {'hnr'}:
                continue
            match = re.match(r'^(?:(?:find|show)(?: me)?(?: clips?)?\s+)?(?:(?:where|what did)\s+)?' + re.escape(alias) + r'(?:\s+|$)(.*)$', normalized)
            if match and (not match.group(1) or re.search(
                r'^(?:talk|speak|discuss|speech|say|said|spoke|encourag|urging|ne\b|نے\b|نوجوان|طلبہ|students|youth|naujawan|ki speech|کا خطاب|کی تقریر|خطاب)', match.group(1))):
                return list(dict.fromkeys([data['canonical']] + list(data.get('speaker_variants', []))))
    return [explicit] if explicit else []


class RetrievalBudgetReached(RuntimeError):
    pass


class BoundedRetrieval:
    """Request-local read wrapper; never changes the indexing client."""
    def __init__(self, client, deadline_seconds=35, max_calls=12, clock=time.monotonic):
        self.client = client
        self.clock = clock
        self.deadline = clock() + deadline_seconds
        self.max_calls = max_calls
        self.calls = 0

    def remaining(self):
        remaining = self.deadline - self.clock()
        if remaining < 1 or self.calls >= self.max_calls:
            raise RetrievalBudgetReached('Candidate retrieval budget exhausted')
        return remaining

    def _read(self, method, **kwargs):
        # Qdrant REST timeout is also passed to HTTP transport by the client.
        kwargs['timeout'] = max(1, int(min(10, self.remaining())))
        self.calls += 1
        return getattr(self.client, method)(**kwargs)

    def scroll(self, **kwargs):
        return self._read('scroll', **kwargs)

    def query_points(self, **kwargs):
        return self._read('query_points', **kwargs)
