"""Semantic retrieval with contextual passages and native Qdrant MaxSim ranking.

The source collection keeps its existing OpenAI dense vectors. A separate,
content-addressed collection caches token vectors; only current, eligible source
candidates are ever queried there. No LLM scores or keyword boosts are blended.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
from dataclasses import dataclass
from datetime import date
from speaker_constraint import passes_constraint
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Callable
import uuid
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from qdrant_client import models
from semantic_passage_evidence import BoundedRetrieval, RetrievalBudgetReached
from transcript_search_contract import valid_interval

logger = logging.getLogger(__name__)


class SearchInputError(ValueError):
    """Invalid caller filters, distinct from infrastructure/model errors."""


@dataclass(frozen=True)
class SearchSettings:
    rerank_collection: str = "video_transcript_rerank_bge_m3_v1"
    model: str = "BAAI/bge-m3"
    dimension: int = 1024
    max_tokens: int = 1024
    batch_size: int = 4
    candidate_limit: int = 60
    context_seconds: float = 20.0
    # Mean MaxSim, NOT a probability. Calibrate on labeled deployment queries.
    minimum_rerank_score: float | None = None

    @classmethod
    def from_env(cls):
        minimum = os.getenv("QDRANT_RERANK_MIN_SCORE", "").strip()
        return cls(
            rerank_collection=os.getenv("QDRANT_RERANK_COLLECTION", cls.rerank_collection),
            model=os.getenv("QDRANT_RERANK_MODEL", cls.model),
            dimension=int(os.getenv("QDRANT_RERANK_DIMENSION", "1024")),
            max_tokens=int(os.getenv("QDRANT_RERANK_MAX_TOKENS", "1024")),
            batch_size=int(os.getenv("QDRANT_RERANK_BATCH_SIZE", "4")),
            candidate_limit=int(os.getenv("SEMANTIC_CANDIDATE_LIMIT", "60")),
            context_seconds=float(os.getenv("SEARCH_CONTEXT_SECONDS", "20")),
            minimum_rerank_score=float(minimum) if minimum else None,
        )

    def __post_init__(self):
        if not 1 <= self.candidate_limit <= 1000:
            raise ValueError("SEMANTIC_CANDIDATE_LIMIT must be between 1 and 1000")
        if not 64 <= self.max_tokens <= 8192 or self.batch_size < 1:
            raise ValueError("Invalid reranking token or batch limits")
        if self.context_seconds < 0 or self.dimension < 1:
            raise ValueError("Invalid context window or vector dimension")


def cpu_thread_budget() -> int:
    """Respect container quotas instead of allocating threads for the host."""
    available = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            available = min(available, max(1, (int(quota) + int(period) - 1) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    requested = int(os.getenv("QDRANT_RERANK_CPU_THREADS", "8"))
    if requested < 1:
        raise ValueError("QDRANT_RERANK_CPU_THREADS must be positive")
    return max(1, min(requested, available))


class MultilingualTokenEncoder:
    """Lazy BGE-M3 inference. Qdrant, rather than Python, computes relevance."""

    def __init__(self, settings: SearchSettings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def __call__(self, texts: list[str]) -> list[list[list[float]]]:
        with self._lock:
            if self._model is None:
                device = os.getenv("QDRANT_RERANK_DEVICE", "cpu")
                if device.lower() == "cpu":
                    threads = cpu_thread_budget()
                    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
                    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
                    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                    import torch
                    torch.set_num_threads(threads)
                    try:
                        torch.set_num_interop_threads(1)
                    except RuntimeError:
                        pass  # Another Torch consumer may have initialized it.
                    logger.info("Loading BGE-M3 on CPU with %d inference threads", threads)
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(
                    self.settings.model,
                    use_fp16=False,
                    devices=device,
                )
            output = self._model.encode(
                texts,
                batch_size=self.settings.batch_size,
                max_length=self.settings.max_tokens,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )
            return [vectors.tolist() for vectors in output["colbert_vecs"]]


class QdrantReranker:
    def __init__(self, client, settings: SearchSettings, encoder: Callable | None = None):
        self.client = client
        self.settings = settings
        self.encoder = encoder or MultilingualTokenEncoder(settings)
        self._ready = False
        self._lock = threading.Lock()

    def _ensure_collection(self):
        with self._lock:
            if self._ready:
                return
            name = self.settings.rerank_collection
            if not self.client.collection_exists(name):
                try:
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config={
                            "late": models.VectorParams(
                                size=self.settings.dimension,
                                # BGE-M3 emits normalized token vectors. Dot preserves
                                # its scoring definition without renormalizing them.
                                distance=models.Distance.DOT,
                                multivector_config=models.MultiVectorConfig(
                                    comparator=models.MultiVectorComparator.MAX_SIM,
                                ),
                                hnsw_config=models.HnswConfigDiff(m=0),
                                on_disk=True,
                            ),
                        },
                    )
                except Exception:
                    # Another worker may have created it concurrently.
                    if not self.client.collection_exists(name):
                        raise
            vectors = self.client.get_collection(name).config.params.vectors
            params = vectors.get("late") if isinstance(vectors, dict) else None
            if (
                params is None
                or params.size != self.settings.dimension
                or params.distance != models.Distance.DOT
                or params.multivector_config is None
                or params.multivector_config.comparator != models.MultiVectorComparator.MAX_SIM
            ):
                raise ValueError("Reranking collection schema mismatch; configure a new collection")
            self._ready = True

    def _cache_id(self, text: str) -> str:
        key = json.dumps(
            [self.settings.model, self.settings.dimension, self.settings.max_tokens, text],
            ensure_ascii=False,
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, hashlib.sha256(key.encode()).hexdigest()))

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        self._ensure_collection()
        documents = {self._cache_id(r["context_text"]): r["context_text"] for r in candidates}
        cached = self.client.retrieve(
            collection_name=self.settings.rerank_collection,
            ids=list(documents), with_payload=False, with_vectors=False,
        )
        existing = {str(point.id) for point in cached}
        missing = [(key, text) for key, text in documents.items() if key not in existing]
        # Bound each encode/upsert independently: token vectors can be large.
        for start in range(0, len(missing), self.settings.batch_size):
            batch = missing[start:start + self.settings.batch_size]
            vectors = self.encoder([text for _, text in batch])
            if len(vectors) != len(batch):
                raise ValueError("Token encoder returned an incomplete batch")
            # One multivector per request keeps JSON below typical server body
            # limits even with 1024-dimensional token vectors.
            for (key, _), vector in zip(batch, vectors):
                self.client.upsert(
                    collection_name=self.settings.rerank_collection,
                    points=[models.PointStruct(id=key, vector={"late": vector})],
                    wait=True,
                )
        query_vectors = self.encoder([query])[0]
        if not query_vectors:
            raise ValueError("Token encoder returned an empty query")
        ranked = self.client.query_points(
            collection_name=self.settings.rerank_collection,
            query=query_vectors,
            using="late",
            query_filter=models.Filter(must=[models.HasIdCondition(has_id=list(documents))]),
            limit=len(documents),
            with_payload=False,
            with_vectors=False,
        ).points
        # Qdrant returns SUM(MaxSim); BGE-M3 reports the mean over query tokens.
        # Division is constant for this query and does not change ranking.
        scores = {str(point.id): point.score / len(query_vectors) for point in ranked}
        result = []
        for candidate in candidates:
            score = scores.get(self._cache_id(candidate["context_text"]))
            if score is None:
                continue
            if self.settings.minimum_rerank_score is not None and score < self.settings.minimum_rerank_score:
                continue
            result.append({
                **candidate,
                "score": round(score, 6),
                "rerank_score": round(score, 6),
                "match_types": ["semantic", "qdrant_maxsim_reranked"],
                "score_type": "bge_m3_mean_maxsim",
            })
        return sorted(result, key=lambda r: (-r["score"], str(r["id"])))


def eligibility_conditions() -> list:
    return [
        models.FieldCondition(key="processing_status", match=models.MatchValue(value="completed")),
        models.FieldCondition(key="approval_status", match=models.MatchValue(value="approved")),
        models.FieldCondition(key="is_archived", match=models.MatchValue(value=False)),
    ]


def source_filter(request, video_ids: list[int] | None = None):
    conditions = eligibility_conditions()
    allowed=getattr(request,"allowed_video_ids",None)
    if allowed is not None:
        conditions.append(models.FieldCondition(key="video_id",match=models.MatchAny(any=allowed or [-1])))
    if video_ids is not None:
        conditions.append(models.FieldCondition(key="video_id", match=models.MatchAny(any=video_ids)))
    if request.video_id is not None:
        conditions.append(models.FieldCondition(key="video_id", match=models.MatchValue(value=request.video_id)))
    if request.language:
        conditions.append(models.FieldCondition(key="language", match=models.MatchValue(value=request.language)))
    for boundary, operator, field in [("start", "gte", "start_time"), ("end", "lte", "end_time")]:
        value = (request.time_range or {}).get(boundary)
        if value is not None:
            conditions.append(models.FieldCondition(key=field, range=models.Range(**{operator: value})))
    if request.filter_date or request.filter_year or request.filter_month:
        if request.filter_date:
            begin = date.fromisoformat(request.filter_date)
            from datetime import timedelta
            end = begin + timedelta(days=1)
        else:
            if not request.filter_year:
                raise SearchInputError("filter_year is required with filter_month")
            year, month = request.filter_year, request.filter_month or 1
            begin = date(year, month, 1)
            end = date(year + 1, 1, 1) if not request.filter_month or month == 12 else date(year, month + 1, 1)
        conditions.append(models.FieldCondition(
            key="video_created_at",
            range=models.DatetimeRange(gte=f"{begin.isoformat()}T00:00:00Z", lt=f"{end.isoformat()}T00:00:00Z"),
        ))
    return models.Filter(must=conditions)


def nearby_passage(focus: dict, neighbors: list[dict], seconds: float = 20.0) -> str:
    """Preserve the focus passage first so token limits cannot discard it.

    Only immediately neighboring turns by the same known speaker are included;
    another person's statements must not become evidence for this speaker.
    Metadata such as video title is deliberately excluded from ranking evidence.
    """
    text = str(focus.get("text") or "").strip()
    speaker = focus.get("speaker")
    if seconds <= 0 or not speaker or speaker == "UNKNOWN":
        return text
    begin, end = float(focus.get("start_time", 0)), float(focus.get("end_time", 0))
    before, after = [], []
    for neighbor in neighbors:
        if neighbor is focus or neighbor.get("speaker") != speaker:
            continue
        if neighbor.get("video_id") != focus.get("video_id"):
            continue
        other_begin = float(neighbor.get("start_time", 0))
        other_end = float(neighbor.get("end_time", 0))
        if (other_begin, other_end) == (begin, end):
            continue
        if 0 <= begin - other_end <= seconds:
            before.append(neighbor)
        if 0 <= other_begin - end <= seconds:
            after.append(neighbor)
    preceding = max(before, key=lambda r: float(r.get("end_time", 0)), default={}).get("text", "")
    following = min(after, key=lambda r: float(r.get("start_time", 0)), default={}).get("text", "")
    parts = [text]
    if preceding:
        parts.append(f"Preceding context: {str(preceding)[-2000:]}")
    if following:
        parts.append(f"Following context: {str(following)[:2000]}")
    return "\n\n".join(parts)


def timestamp_url(url: str, seconds: float) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query["t"] = str(int(seconds))
    return urlunsplit(parts._replace(query=urlencode(query)))


def source_passage(focus: dict, neighbors: list[dict], seconds: float) -> dict:
    """Build displayed evidence from adjacent source turns, with real provenance."""
    if seconds <= 0:
        return {}
    label = focus.get("transcript_speaker") or focus.get("diarization_speaker")
    speaker = focus.get("speaker")
    if str(speaker or "").casefold() in {"", "unknown", "unknown speaker"} and not label:
        return {}
    same = [row for row in neighbors if row.get("id") is not None and valid_interval(row)
            and row.get("video_id") == focus.get("video_id") and row.get("speaker") == speaker
            and row.get("source_identity_group")==focus.get("source_identity_group")
            and (row.get("transcript_speaker") or row.get("diarization_speaker")) == label
            and str(row.get("text") or "").strip() and str(row["id"]) != str(focus["id"])]
    before = [row for row in same if 0 <= focus["start_time"] - row["end_time"] <= min(1, seconds)]
    after = [row for row in same if 0 <= row["start_time"] - focus["end_time"] <= min(1, seconds)]
    turns = [focus]
    for row in [max(before, key=lambda r: r["end_time"], default=None),
                min(after, key=lambda r: r["start_time"], default=None)]:
        if row is None:
            continue
        proposed = sorted(turns + [row], key=lambda r: r["start_time"])
        if (proposed[-1]["end_time"] - proposed[0]["start_time"] <= 60
                and sum(len(r["text"]) + 1 for r in proposed) <= 2400):
            turns = proposed
    if len(turns) == 1:
        return {}
    start, end = turns[0]["start_time"], turns[-1]["end_time"]
    return {"text": " ".join(r["text"].strip() for r in turns), "start_time": start,
            "end_time": end, "duration": end - start, "segment_ids": [r["id"] for r in turns],
            "match_count": len(turns), "source_utterances": [
                {key: row.get(key) for key in ("id", "text", "start_time", "end_time", "speaker")}
                for row in turns], "youtube_url_timestamped": timestamp_url(focus.get("youtube_url", ""), start)}


class ContextualSearch:
    def __init__(self, client, collection: str, embed: Callable, understand: Callable,
                 speaker_matches: Callable, title_matches: Callable, reranker: QdrantReranker,
                 speaker_values: Callable | None = None, validator: Callable | None = None,
                 speaker_index_types: dict[str, str] | None = None):
        self.client, self.collection = client, collection
        self.embed, self.understand = embed, understand
        self.speaker_matches, self.title_matches = speaker_matches, title_matches
        self.reranker = reranker
        self.speaker_values = speaker_values
        self.validator = validator
        self.speaker_index_types = speaker_index_types

    def _populate_context(self, pool: list[dict], base_filter, reader) -> list[str]:
        """Fetch each video's neighboring turns once within the source-read budget."""
        warnings = []
        groups = defaultdict(list)
        for candidate in pool:
            candidate["context_text"] = candidate["text"]
            if self.reranker.settings.context_seconds > 0:
                groups[candidate["video_id"]].append(candidate)
        seconds = self.reranker.settings.context_seconds
        for video_id, candidates in groups.items():
            windows = [models.FieldCondition(key="start_time", range=models.Range(
                gte=max(0, r["start_time"] - seconds), lte=r["end_time"] + seconds,
            )) for r in candidates]
            window = models.Filter(must=list(base_filter.must or []) + [
                models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id)),
                models.Filter(should=windows),
            ])
            try:
                points, offset = reader.scroll(
                    collection_name=self.collection, scroll_filter=window,
                    limit=min(512, 64 * len(candidates)),
                    with_payload=["video_id", "speaker", "transcript_speaker", "diarization_speaker",
                                  "start_time", "end_time", "timestamp_unit", "text"],
                    with_vectors=False,
                )
            except RetrievalBudgetReached:
                warnings.append("context_retrieval_budget_reached")
                break
            neighbors = [{**(p.payload or {}), "id": p.id} for p in points if valid_interval(p.payload or {})]
            for candidate in candidates:
                candidate["context_text"] = nearby_passage(candidate, neighbors, seconds)
                candidate["_source_passage"] = source_passage(candidate, neighbors, seconds)
            if offset is not None:
                warnings.append("context_window_truncated")
        return list(dict.fromkeys(warnings))

    def _context(self, candidate: dict, base_filter):
        settings = self.reranker.settings
        if settings.context_seconds <= 0 or not valid_interval(candidate):
            return candidate["text"]
        window = models.Filter(must=list(base_filter.must or []) + [
            models.FieldCondition(key="video_id", match=models.MatchValue(value=candidate["video_id"])),
            models.FieldCondition(key="start_time", range=models.Range(
                gte=max(0, candidate["start_time"] - settings.context_seconds),
                lte=candidate["end_time"] + settings.context_seconds,
            )),
        ])
        # Bounded per candidate; no unbounded whole-video scans.
        points, _ = self.client.scroll(
            collection_name=self.collection, scroll_filter=window, limit=64,
            with_payload=["video_id", "speaker", "start_time", "end_time", "timestamp_unit", "text"],
            with_vectors=False,
        )
        return nearby_passage(candidate, [p.payload or {} for p in points if valid_interval(p.payload or {})], settings.context_seconds)

    def search(self, request, video_ids: list[int] | None = None) -> dict:
        stage = time.perf_counter()
        timings = {}
        query = (request.query or "").strip()
        extras = list(request.words or []) + ([request.word] if request.word else [])
        query = " ".join([query] + [term.strip() for term in extras if term.strip()]).strip()
        if not query.strip():
            raise SearchInputError("A non-empty query or words is required for semantic search")
        intent = self.understand(query)
        timings["understanding_seconds"] = round(time.perf_counter() - stage, 3)
        stage = time.perf_counter()
        semantic_query = intent.get("semantic_query") or query
        if not isinstance(semantic_query, str):
            semantic_query = query
        speaker = request.speaker
        if not speaker and intent.get("query_type") == "speaker_search":
            speaker = intent.get("extracted_speaker")
        # No implicit language filter. English queries may retrieve Urdu passages.
        try:
            search_filter = source_filter(request, video_ids)
        except ValueError as error:
            raise SearchInputError(str(error)) from error
        # Known speaker aliases can be filtered before ANN retrieval. For an
        # arbitrary fuzzy name, use the bounded post-filter below instead.
        variants = self.speaker_values(speaker) if speaker and self.speaker_values else []
        if variants and len(variants) <= 1:
            conditions = []
            for field in ("speaker", "diarization_speaker"):
                index_type = (self.speaker_index_types or {}).get(field)
                if self.speaker_index_types is None or index_type == "keyword":
                    conditions.append(models.FieldCondition(key=field, match=models.MatchAny(any=variants)))
                elif index_type == "text":
                    conditions.extend(models.FieldCondition(key=field, match=models.MatchText(text=name))
                                      for name in variants)
            if conditions:
                search_filter.must.append(models.Filter(should=conditions))
        settings = self.reranker.settings
        limit = settings.candidate_limit
        reader = BoundedRetrieval(self.client)
        retrieval_warnings = []
        queries = [semantic_query]
        translated = intent.get("semantic_query_translated")
        if isinstance(translated, str) and translated.strip() and translated != semantic_query:
            queries.append(translated.strip())
        candidates = {}
        for variation in queries:
            points = reader.query_points(
                collection_name=self.collection, query=self.embed(variation),
                query_filter=search_filter, limit=limit,
                # Dense cosine is candidate-generation evidence, not the final
                # relevance score. Legacy UI cutoffs must not hide passages
                # before native reranking and evidence validation can see them.
                score_threshold=None,
                search_params=models.SearchParams(hnsw_ef=128),
                with_payload=True, with_vectors=False,
            ).points
            for point in points:
                payload = point.payload or {}
                if not str(payload.get("text") or "").strip() or not valid_interval(payload):
                    continue
                if speaker and not passes_constraint(payload, speaker, variants):
                    continue
                if request.title and not self.title_matches(request.title, payload.get("video_title", "")):
                    continue
                previous = candidates.get(str(point.id))
                if previous is not None and previous["dense_score"] >= point.score:
                    continue
                start, end = payload.get("start_time", 0), payload.get("end_time", 0)
                candidates[str(point.id)] = {
                    **payload, "id": point.id, "segment_ids": [point.id], "match_count": 1,
                    "start_time": start, "end_time": end, "duration": max(0, end - start),
                    "dense_score": point.score, "matched_field": "text",
                    "llm_relevance_score": None, "fuzzy_score": 0.0,
                    "youtube_url_timestamped": timestamp_url(payload.get("youtube_url", ""), start),
                }
        # Existing full-text indexes can recover literal topic passages that
        # fall below the dense ANN pool. They receive no relevance-score boost;
        # native MaxSim and the same evidence validator judge the combined pool.
        lexical_buckets = []
        terms = intent.get("retrieval_terms", [])
        if terms and self.speaker_index_types is not None and self.speaker_index_types.get("text") != "text":
            retrieval_warnings.append("topic_text_index_unavailable")
            terms = []
        if isinstance(terms, list):
            for term in dict.fromkeys(term.strip() for term in terms
                                      if isinstance(term, str) and 1 < len(term.strip()) <= 100):
                if len(lexical_buckets) >= 4:
                    break
                topic_filter = models.Filter(must=list(search_filter.must or []) + [
                    models.FieldCondition(key="text", match=models.MatchText(text=term))])
                try:
                    points, _ = reader.scroll(collection_name=self.collection, scroll_filter=topic_filter,
                                              limit=min(12, limit), with_payload=True, with_vectors=False)
                except RetrievalBudgetReached:
                    retrieval_warnings.append("lexical_retrieval_budget_reached")
                    break
                bucket = []
                for point in points:
                    payload = point.payload or {}
                    if not str(payload.get("text") or "").strip() or not valid_interval(payload):
                        continue
                    if speaker and not passes_constraint(payload, speaker, variants):
                        continue
                    if request.title and not self.title_matches(request.title, payload.get("video_title", "")):
                        continue
                    start, end = payload["start_time"], payload["end_time"]
                    bucket.append(candidates.get(str(point.id)) or {
                        **payload, "id": point.id, "segment_ids": [point.id], "match_count": 1,
                        "start_time": start, "end_time": end, "duration": end - start,
                        "dense_score": None, "matched_field": "text", "llm_relevance_score": None,
                        "youtube_url_timestamped": timestamp_url(payload.get("youtube_url", ""), start),
                    })
                lexical_buckets.append(bucket)
        lexical = {}
        for turn in zip_longest(*lexical_buckets):
            for row in turn:
                if row is not None:
                    lexical.setdefault(str(row["id"]), row)
        dense_pool = sorted(candidates.values(), key=lambda r: -r["dense_score"])
        reserved = min(len(lexical), max(1, limit // 2))
        selected_lexical = list(lexical.values())[:reserved]
        selected_ids = {str(row["id"]) for row in selected_lexical}
        pool = selected_lexical + [row for row in dense_pool if str(row["id"]) not in selected_ids][:limit - reserved]
        timings["retrieval_seconds"] = round(time.perf_counter() - stage, 3)
        stage = time.perf_counter()
        retrieval_warnings.extend(self._populate_context(pool, search_filter, reader))
        timings["context_seconds"] = round(time.perf_counter() - stage, 3)
        stage = time.perf_counter()
        ranked = self.reranker.rerank(semantic_query, pool)
        timings["reranking_seconds"] = round(time.perf_counter() - stage, 3)
        stage = time.perf_counter()
        passage_validation = None
        if self.validator is not None:
            # Validate displayed transcript text, never title or surrounding hints.
            native_scores = {str(r["id"]): r["rerank_score"] for r in ranked}
            passages = [{**{key: value for key, value in row.items() if key != "_source_passage"},
                         **row.get("_source_passage", {})} for row in ranked]
            ranked, passage_validation = self.validator(query, passages, speaker)
            for result in ranked:
                result["rerank_score"] = max(native_scores[str(identity)] for identity in result["segment_ids"]
                                             if str(identity) in native_scores)
                result["score_type"] = "validated_passage_relevance"
                result["match_types"].append("qdrant_maxsim_reranked")
            ranked.sort(key=lambda r: (-r["rerank_score"], str(r["id"])))
            passage_validation["qdrant_calls"] = reader.calls
            passage_validation["qdrant_calls_scope"] = "source_collection_reads"
            passage_validation.setdefault("evaluation_scope", "retrieved_candidates")
            passage_validation["retrieval_limits"] = {"seconds": 35, "qdrant_calls": 12, "query_variations": 2}
            if retrieval_warnings:
                passage_validation["status"] = "partial"
                passage_validation["retryable"] = True
                passage_validation["retrieval_warnings"] = retrieval_warnings
        ranked = [row for row in ranked if row["score"] >= request.min_score]
        timings["validation_seconds"] = round(time.perf_counter() - stage, 3)
        # Rank videos by their best passage, keeping each video's passages together
        # so existing cursor pagination never splits a video into different pages.
        groups = defaultdict(list)
        for result in ranked:
            result.pop("_source_passage", None)
            groups[result.get("video_id")].append(result)
        results = [r for group in groups.values() for r in group][:request.top_k]
        if passage_validation is not None:
            passage_validation["returned_counts"] = {
                "videos": len({r.get("video_id") for r in results}),
                "passages": len(results), "occurrences": None,
            }
        return {
            "query": query, "words": request.words, "speaker_filter": speaker,
            "query_decomposition": intent.get("query_decomposition"),
            "collection": self.collection, "search_mode": "semantic",
            "total_semantic_hits": len(pool), "total_speaker_hits": 0,
            "total_keyword_hits": 0, "total_title_hits": 0, "total_exact_phrase_hits": 0,
            "returned": len(results), "unique_videos": len({r.get("video_id") for r in results}),
            "metadata": {"passage_validation": passage_validation},
            "filters_applied": {"video_id": request.video_id, "speaker": speaker,
                                "title": request.title, "language": request.language,
                                "time_range": request.time_range, "min_score": request.min_score},
            "ranking": {"engine": "qdrant_maxsim", "model": settings.model,
                        "score_type": "validated_passage_relevance" if passage_validation is not None else "bge_m3_mean_maxsim",
                        "rerank_score_type": "bge_m3_mean_maxsim", "score_is_probability": False,
                        "minimum_rerank_score": settings.minimum_rerank_score,
                        "minimum_relevance_score": request.min_score,
                        "candidates_reranked": len(pool), "candidate_limit": limit,
                        "lexical_candidates": reserved,
                        "context_seconds": settings.context_seconds,
                        "retrieval_warnings": retrieval_warnings,
                        "query_understanding": intent.get("understanding_status", "enabled"),
                        "timings": timings},
            "results": results,
        }
