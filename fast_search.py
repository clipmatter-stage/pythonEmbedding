"""Bounded dense/sparse Qdrant retrieval without mandatory generative calls.

Sparse vectors contain normalized Unicode word frequencies with stable hashed
indices. Qdrant supplies IDF and rank fusion; no local model download is needed.
This is lexical TF-IDF, not a learned multilingual embedding or BM25 model.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
import os
import re
import time
import unicodedata

from qdrant_client import models
from contextual_search import SearchInputError, source_filter, timestamp_url
from transcript_search_contract import valid_interval

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
                 sparse_enabled=False, settings=None, reranker=None, validator=None):
        self.client, self.collection = client, collection
        self.embed, self.understand = embed, understand
        self.speaker_matches, self.title_matches = speaker_matches, title_matches
        self.speaker_values = speaker_values
        self.speaker_index_types = speaker_index_types or {}
        self.sparse_enabled = sparse_enabled
        self.settings = settings or FastSearchSettings.from_env()
        self.reranker, self.validator = reranker, validator

    def search(self, request, video_ids=None):
        started = time.perf_counter()
        query = " ".join([request.query or "", request.word or "", *(request.words or [])]).strip()
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
        vector = self.embed(semantic)  # Exactly one query embedding, cached by the caller.
        embedding_done = time.perf_counter()
        limit = self.settings.candidate_limit
        sparse = lexical_vector(" ".join([semantic, query]), query=True)
        hybrid = self.sparse_enabled and bool(sparse.indices)
        arguments = dict(collection_name=self.collection, query_filter=filters,
                         limit=limit, with_payload=getattr(self,"payload_fields",True), with_vectors=False,
                         timeout=self.settings.timeout)
        if hybrid:
            arguments.update(prefetch=[
                models.Prefetch(query=vector, filter=filters, limit=limit,
                                params=models.SearchParams(hnsw_ef=128),
                                score_threshold=self.settings.minimum_dense_score),
                models.Prefetch(query=sparse, using=SPARSE_NAME, filter=filters, limit=limit),
            ], query=models.FusionQuery(fusion=models.Fusion.RRF))
        else:
            arguments.update(query=vector, score_threshold=self.settings.minimum_dense_score,
                             search_params=models.SearchParams(hnsw_ef=128))
        points = self.client.query_points(**arguments).points
        retrieved = time.perf_counter()
        results = []
        for point in points:
            row = point.payload or {}
            if not str(row.get("text") or "").strip() or not valid_interval(row):
                continue
            if speaker and not self.speaker_matches(speaker, f"{row.get('speaker', '')} {row.get('diarization_speaker', '')}"):
                continue
            if request.title and not self.title_matches(request.title, row.get("video_title", "")):
                continue
            results.append({**row, "id": point.id, "segment_ids": [point.id], "match_count": 1,
                            "score": point.score, "retrieval_score": point.score,
                            "dense_score": None if hybrid else point.score,
                            "score_type": "rrf" if hybrid else "cosine_similarity",
                            "llm_relevance_score": None, "fuzzy_score": 0.0,
                            "matched_field": "text", "context_text": row["text"],
                            "match_types": ["qdrant_hybrid" if hybrid else "qdrant_dense"],
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
                "engine": "qdrant_hybrid_rrf" if hybrid else "qdrant_dense",
                "score_type": "validated_passage_relevance" if validation is not None else (
                    "bge_m3_mean_maxsim" if self.reranker else "rrf" if hybrid else "cosine_similarity"),
                "score_is_probability": False, "candidate_limit": limit,
                "candidates_retrieved": candidate_count,
                "candidates_reranked": min(candidate_count, self.reranker.settings.candidate_limit) if self.reranker else 0,
                "query_understanding": intent.get("understanding_status", "local"),
                "evidence_verified": validation is not None,
                "minimum_dense_score": self.settings.minimum_dense_score,
                "retrieval_warnings": warnings,
                "timings": {"understanding_seconds": round(understanding_done - started, 3),
                            "embedding_seconds": round(embedding_done - understanding_done, 3),
                            "retrieval_seconds": round(retrieved - embedding_done, 3),
                            "reranking_seconds": round(reranked - retrieved, 3),
                            "validation_seconds": round(completed - reranked, 3),
                            "total_seconds": round(completed - started, 3)},
            }, "results": results,
        }
