"""Bounded dense/sparse Qdrant retrieval without mandatory generative calls.

Sparse vectors contain normalized Unicode word frequencies with stable hashed
indices. Qdrant supplies IDF and rank fusion; no local model download is needed.
This is lexical TF-IDF, not a learned multilingual embedding or BM25 model.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from speaker_identity import view as identity_view, FIELDS as IDENTITY_FIELDS, matches_identity
from dataclasses import dataclass
import hashlib
import math
import os
import re
import time
import unicodedata

from qdrant_client import models
from contextual_search import SearchInputError, source_filter, timestamp_url
from intent_retrieval import alignment_score, compose_retrieval_queries, keyword_vote_weight
from transcript_search_contract import valid_interval, semantic_query_text

SPARSE_NAME = "lexical_tf_v1"


def lexical_tokens(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = text.translate(str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"}))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"[^\W_]+", text, re.UNICODE)


def lexical_vector(text, *, query=False):
    counts = Counter(lexical_tokens(text))
    features = defaultdict(float)
    for word, count in counts.items():
        index = int.from_bytes(hashlib.blake2s(word.encode(), digest_size=4).digest(), "big")
        features[index] += 1.0 if query else 1.0 + math.log(count)
    indices = sorted(features)
    norm = math.sqrt(sum(value * value for value in features.values())) or 1.0
    return models.SparseVector(indices=indices, values=[features[i] / norm for i in indices])


def with_sparse_vectors(points, enabled):
    """Preserve the unnamed dense vector when indexing the optional sparse field."""
    if not enabled:
        return points
    return [point.model_copy(update={"vector": {
        **(point.vector if isinstance(point.vector, dict) else {"": point.vector}),
        SPARSE_NAME: lexical_vector((point.payload or {}).get("text", "")),
    }}) for point in points]


@dataclass(frozen=True)
class FastSearchSettings:
    candidate_limit: int = 100
    minimum_dense_score: float = 0.0
    timeout: int = 8

    @classmethod
    def from_env(cls):
        result = cls(int(os.getenv("FAST_SEARCH_CANDIDATE_LIMIT", "100")),
                     float(os.getenv("SEARCH_MIN_DENSE_SCORE", "0")),
                     int(os.getenv("SEARCH_QDRANT_TIMEOUT", "8")))
        if not 1 <= result.candidate_limit <= 1000 or not -1 <= result.minimum_dense_score <= 1 or not 1 <= result.timeout <= 60:
            raise ValueError("Invalid fast search candidate, score or timeout settings")
        return result


class FastSemanticSearch:
    def __init__(self, client, collection, embed, understand, speaker_matches,
                 title_matches, *, speaker_values=None, speaker_index_types=None,
                 sparse_enabled=False, settings=None, reranker=None, validator=None,
                 retrieval_expander=None, diversify_candidates=False, lexical_indexed=False):
        self.client, self.collection = client, collection
        self.embed, self.understand = embed, understand
        self.speaker_matches, self.title_matches = speaker_matches, title_matches
        self.speaker_values = speaker_values
        self.speaker_index_types = speaker_index_types or {}
        self.sparse_enabled = sparse_enabled
        self.settings = settings or FastSearchSettings.from_env()
        self.reranker, self.validator = reranker, validator
        self.retrieval_expander = retrieval_expander
        self.diversify_candidates = diversify_candidates
        self.lexical_indexed = lexical_indexed

    def search(self, request, video_ids=None):
        started = time.perf_counter()
        query = semantic_query_text(request)
        if not query:
            raise SearchInputError("A non-empty query or words is required for semantic search")
        intent = self.understand(query)
        semantic = intent.get("semantic_query") or query
        speaker = request.speaker or (intent.get("extracted_speaker") if intent.get("query_type") == "speaker_search" else None)
        try:
            filters = source_filter(request, video_ids)
        except ValueError as error:
            raise SearchInputError(str(error)) from error
        # Pre-filter only where the existing index supports the corresponding
        # matcher. Always check speaker/title again on the returned payload.
        variants = self.speaker_values(speaker) if speaker and self.speaker_values else []
        conditions = []
        for field in ("speaker", "diarization_speaker"):
            kind = self.speaker_index_types.get(field)
            if variants and kind == "keyword":
                conditions.append(models.FieldCondition(key=field, match=models.MatchAny(any=variants)))
            elif variants and kind == "text":
                conditions.extend(models.FieldCondition(key=field, match=models.MatchText(text=name)) for name in variants)
        if conditions:
            filters.must.append(models.Filter(should=conditions))
        understanding_done = time.perf_counter()
        plan = self.retrieval_expander(semantic) if self.retrieval_expander else {}
        # Grounded search embeds the proposition and spoken forms. Fast mode
        # keeps the single raw query it already used.
        if self.retrieval_expander:
            queries = compose_retrieval_queries(semantic, plan)
        else:
            queries = [semantic] if semantic else []
        if len(queries) == 1:
            vectors = [self.embed(semantic)]
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(copy_context().run, self.embed, value) for value in queries]
                vectors = []; successful_queries = []
                for index, future in enumerate(futures):
                    try:
                        vectors.append(future.result()); successful_queries.append(queries[index])
                    except Exception:
                        if index == 0: raise
                        plan.setdefault('warnings', []).append('translated_embedding_unavailable')
                queries = successful_queries
        vector = vectors[0]
        embedding_done = time.perf_counter()
        limit = self.settings.candidate_limit
        sparse = lexical_vector(" ".join([*queries, query]), query=True)
        hybrid = self.sparse_enabled and bool(sparse.indices)
        arguments = dict(collection_name=self.collection, query_filter=filters,
                         limit=limit, with_payload=getattr(self,"payload_fields",True), with_vectors=False,
                         timeout=self.settings.timeout)
        provenance = defaultdict(list)
        keyword_terms = list(dict.fromkeys(t.strip() for t in (plan.get("retrieval_terms") or [query])
                        if isinstance(t, str) and 1 < len(t.strip()) <= 100))[:4]
        keyword_fallback = bool(not hybrid and self.lexical_indexed and keyword_terms)
        if hybrid or len(vectors) > 1 or keyword_fallback:
            arguments.update(prefetch=[
                models.Prefetch(query=v, filter=filters, limit=limit,
                                params=models.SearchParams(hnsw_ef=128),
                                score_threshold=self.settings.minimum_dense_score) for v in vectors
            ], query=models.FusionQuery(fusion=models.Fusion.RRF))
            if hybrid:
                arguments['prefetch'].append(models.Prefetch(query=sparse, using=SPARSE_NAME, filter=filters, limit=limit))
            elif keyword_fallback:
                # Native indexed keyword filtering supplies an independent lane,
                # even when the additive sparse-vector backfill is unavailable.
                for term in keyword_terms:
                    lexical_filter = models.Filter(must=[filters, models.FieldCondition(
                        key='text', match=models.MatchText(text=term))])
                    # Rank native-script keyword hits with a native-script
                    # equivalent, rather than the possibly weaker Roman query.
                    native = bool(re.search(r'[\u0600-\u06ff]',term))
                    matching = [v for q,v in zip(queries,vectors)
                                if bool(re.search(r'[\u0600-\u06ff]',q)) == native]
                    arguments['prefetch'].append(models.Prefetch(query=matching[0] if matching else vector, filter=lexical_filter, limit=limit))
        else:
            arguments.update(query=vector, score_threshold=self.settings.minimum_dense_score,
                             search_params=models.SearchParams(hnsw_ef=128))
        fused = hybrid or len(vectors) > 1 or keyword_fallback
        if self.diversify_candidates:
            # Group inside each prefetch too: grouping only after a dense top-N
            # cannot recover videos already excluded by that top-N.
            if 'prefetch' in arguments:
                lanes = []
                for prefetch in arguments.pop('prefetch'):
                    lane = dict(collection_name=self.collection, query=prefetch.query,
                        query_filter=prefetch.filter, limit=limit, group_by='video_id', group_size=2,
                        with_payload=getattr(self,'payload_fields',True), with_vectors=False,
                        timeout=self.settings.timeout)
                    if prefetch.using: lane['using'] = prefetch.using
                    if prefetch.score_threshold is not None: lane['score_threshold'] = prefetch.score_threshold
                    if prefetch.params is not None: lane['search_params'] = prefetch.params
                    lanes.append(lane)
                # Bounded Qdrant grouped lanes; reciprocal-rank fusion below
                # uses ranks, never compares cosine with sparse scores.
                with ThreadPoolExecutor(max_workers=4) as executor:
                    groups_by_lane = list(executor.map(lambda lane:self.client.query_points_groups(**lane).groups, lanes))
                merged = {}; ranks = defaultdict(float); video_ranks = defaultdict(float)
                semantic_lanes = len(vectors)
                for lane_index, groups in enumerate(groups_by_lane):
                    semantic_lane = lane_index < semantic_lanes
                    for rank, group in enumerate(groups, 1):
                        hit_weights = []
                        for depth, hit in enumerate(group.hits):
                            identity = str(hit.id); merged.setdefault(identity, hit)
                            if semantic_lane:
                                weight = 1.0
                            else:
                                weight = keyword_vote_weight(
                                    str((hit.payload or {}).get('text') or ''), queries, keyword_terms)
                            hit_weights.append(weight)
                            ranks[identity] += weight / (60 + rank + depth)
                            provenance[identity].append({'lane':lane_index,
                                'kind':'semantic' if semantic_lane else 'keyword',
                                'rank':rank,'depth':depth,'weight':weight})
                        video_ranks[group.id] += max(hit_weights or [0.0]) / (60 + rank)
                # Fuse videos first: different languages often retrieve
                # different passages of the SAME relevant video. Point-only
                # fusion would discard those independent retrieval votes.
                # Coverage is scored per spoken/faithful variant so one shared
                # audience word cannot fill the grounded candidate cap.
                alignments = {}
                best_alignment = defaultdict(float)
                for identity, point in merged.items():
                    text = str((point.payload or {}).get('text') or '')
                    alignments[identity] = alignment_score([*queries, *keyword_terms], text)
                    video_id = (point.payload or {}).get('video_id')
                    best_alignment[video_id] = max(best_alignment[video_id], alignments[identity])
                for video_id, covered in best_alignment.items():
                    video_ranks[video_id] += semantic_lanes * covered / 61
                def strong_keyword(identity):
                    return any(item['kind']=='keyword' and item.get('weight', 0) >= 1
                               for item in provenance[identity])
                points = sorted(merged.values(), key=lambda p:(
                    -video_ranks[(p.payload or {}).get('video_id')],
                    -int(strong_keyword(str(p.id))),
                    -alignments.get(str(p.id), 0.0),
                    -ranks[str(p.id)],str(p.id)))
                # A video's multilingual votes establish its rank, not proof
                # that an arbitrary passage from it matches. Prefer a keyword
                # anchor that covers a distinctive term over a semantic-only
                # turn from another topic. A generic audience overlap does not.
                for point in points:
                    identity = str(point.id)
                    video_id = (point.payload or {}).get('video_id')
                    point.score = (video_ranks[video_id]
                                   + (1 / 61 if strong_keyword(identity) else 0.0)
                                   + (0.25 / 61) * alignments.get(identity, 0.0))
                # One best passage per video before admitting second passages.
                buckets = defaultdict(list)
                for point in points: buckets[(point.payload or {}).get('video_id')].append(point)
                points = [hits[depth] for depth in range(2) for hits in buckets.values() if len(hits)>depth][:limit]
            else:
                arguments.update(group_by='video_id', group_size=2)
                groups = self.client.query_points_groups(**arguments).groups
                points = [group.hits[depth] for depth in range(2) for group in groups if len(group.hits)>depth][:limit]
        else:
            points = self.client.query_points(**arguments).points
        retrieved = time.perf_counter()
        results = []
        for point in points:
            row = identity_view(point.payload or {})
            if not str(row.get("text") or "").strip() or not valid_interval(row):
                continue
            if speaker and not matches_identity(speaker,row):
                continue
            if request.title and not self.title_matches(request.title, row.get("video_title", "")):
                continue
            results.append({**row, "id": point.id, "segment_ids": [point.id], "match_count": 1,
                            "score": point.score, "retrieval_score": point.score,
                            "dense_score": None if fused else point.score,
                            "score_type": "rrf" if fused else "cosine_similarity",
                            "llm_relevance_score": None, "fuzzy_score": 0.0,
                            "retrieval_lanes": provenance.get(str(point.id),[]),
                            "matched_field": "text", "context_text": row["text"],
                            "match_types": ["qdrant_hybrid" if hybrid or keyword_fallback else "qdrant_multilingual" if fused else "qdrant_dense"],
                            "duration": row["end_time"] - row["start_time"],
                            "youtube_url_timestamped": timestamp_url(row.get("youtube_url", ""), row["start_time"])})
        candidate_count = len(results)
        if self.reranker is not None:
            results = self.reranker.rerank(semantic, results[:self.reranker.settings.candidate_limit])
        reranked = time.perf_counter()
        validation = None
        if self.validator is not None:
            results, validation = self.validator(query, results, speaker)
            results = [row for row in results if row["score"] >= request.min_score]
        # Group by the best passage's position, without changing Qdrant's order
        # within a video. Cursor pages must not repeat/split video groups.
        groups = defaultdict(list)
        for result in results:
            groups[result.get("video_id")].append(result)
        results = [row for group in groups.values() for row in group][:request.top_k]
        completed = time.perf_counter()
        warnings = [] if hybrid else ["sparse_index_unavailable_using_dense"]
        warnings.extend(plan.get('warnings', []))
        if keyword_fallback: warnings = [w for w in warnings if w != 'sparse_index_unavailable_using_dense']
        if self.retrieval_expander and not hybrid and not keyword_fallback:
            warnings.append('keyword_index_unavailable')
        if request.min_score and self.validator is None:
            warnings.append("legacy_min_score_not_applied_to_retrieval_scores")
        return {
            "query": query, "words": request.words, "speaker_filter": speaker,
            "query_decomposition": intent.get("query_decomposition"),
            "collection": self.collection, "search_mode": "semantic",
            "total_semantic_hits": candidate_count, "total_speaker_hits": 0,
            "total_keyword_hits": 0, "total_title_hits": 0, "total_exact_phrase_hits": 0,
            "returned": len(results), "unique_videos": len({row.get("video_id") for row in results}),
            "metadata": {"passage_validation": validation},
            "filters_applied": {"video_id": request.video_id, "speaker": speaker,
                                "title": request.title, "language": request.language,
                                "time_range": request.time_range,
                                "min_score": request.min_score if self.validator is not None else None},
            "ranking": {
                "engine": "qdrant_hybrid_rrf" if hybrid or keyword_fallback else "qdrant_multilingual_rrf" if fused else "qdrant_dense",
                "score_type": "validated_passage_relevance" if validation is not None else (
                    "bge_m3_mean_maxsim" if self.reranker else "rrf" if fused else "cosine_similarity"),
                "score_is_probability": False, "candidate_limit": limit,
                "candidates_retrieved": candidate_count,
                "candidates_reranked": min(candidate_count, self.reranker.settings.candidate_limit) if self.reranker else 0,
                "query_understanding": intent.get("understanding_status", "local"),
                "evidence_verified": validation is not None,
                "minimum_dense_score": self.settings.minimum_dense_score,
                "retrieval_warnings": warnings,
                "retrieval_queries": queries,
                "keyword_retrieval": "sparse" if hybrid else "indexed_text" if keyword_fallback else "unavailable",
                "video_diversity": "qdrant_grouped" if self.diversify_candidates else "after_retrieval",
                "timings": {"understanding_seconds": round(understanding_done - started, 3),
                            "embedding_seconds": round(embedding_done - understanding_done, 3),
                            "retrieval_seconds": round(retrieved - embedding_done, 3),
                            "reranking_seconds": round(reranked - retrieved, 3),
                            "validation_seconds": round(completed - reranked, 3),
                            "total_seconds": round(completed - started, 3)},
            }, "results": results,
        }
