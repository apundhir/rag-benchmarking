from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import app.retrieval.service as retrieval_service
import app.llm.client as llm_client


router = APIRouter(prefix="/v1", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = Field(default=False)


class RetrievedChunk(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: List[RetrievedChunk]
    timings_ms: Dict[str, float] | None = None
    tokens: Dict[str, int] | None = None
    groundedness: float | None = None


@router.post("/query", response_model=QueryResponse)
def post_query(req: QueryRequest) -> QueryResponse:
    timings: Dict[str, float] = {}
    try:
        from app.utils.timing import timer

        with timer() as t_retr:
            chunks = retrieval_service.retrieve_top_chunks(req.query, top_k=max(req.top_k, 10))
        timings["retrieve"] = t_retr["elapsed_ms"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not chunks:
        return QueryResponse(answer="", citations=[])
    # Generate an answer from the top chunks
    system_prompt = (
        "You are a helpful assistant. Answer based only on the provided context. Cite sources."
    )
    # Optional reranking
    if req.rerank:
        try:
            from app.retrieval.reranker import CrossEncoderReranker
            with timer() as t_rr:
                reranker = CrossEncoderReranker()
                # Rerank all candidates, then trim to requested top_k
                chunks = reranker.rerank(req.query, chunks, top_k=len(chunks))
            timings["rerank"] = t_rr["elapsed_ms"]
        except Exception:
            # Fallback silently if reranker unavailable
            chunks = chunks[: req.top_k]
    else:
        chunks = chunks[: req.top_k]

    context_blocks = "\n\n".join([f"[source: {c.get('source_id','')}]\n{c.get('text','')}" for c in chunks])
    user_prompt = f"Context:\n{context_blocks}\n\nQuestion: {req.query}\nAnswer:"
    llm = llm_client.LLMClient()
    with timer() as t_gen:
        answer = llm.generate(system_prompt, user_prompt)
    timings["generate"] = t_gen["elapsed_ms"]

    # Self-check groundedness
    try:
        from app.quality.self_check import compute_groundedness
        with timer() as t_sc:
            groundedness = compute_groundedness(answer, [c.get("text", "") for c in chunks])
        timings["self_check"] = t_sc["elapsed_ms"]
    except Exception:
        groundedness = None

    # If low groundedness, optionally retry with more context once
    from app.config.settings import get_settings
    settings = get_settings()
    if groundedness is not None and groundedness < settings.self_check_min_groundedness and settings.self_check_retry:
        # Expand retrieval window and try synthesis again
        try:
            with timer() as t_retr2:
                more_chunks = retrieval_service.retrieve_top_chunks(req.query, top_k=20)
            timings["retrieve_retry"] = t_retr2["elapsed_ms"]
            # Rerank again if requested
            if req.rerank:
                try:
                    from app.retrieval.reranker import CrossEncoderReranker
                    with timer() as t_rr2:
                        reranker = CrossEncoderReranker()
                        more_chunks = reranker.rerank(req.query, more_chunks, top_k=len(more_chunks))
                    timings["rerank_retry"] = t_rr2["elapsed_ms"]
                except Exception:
                    pass
            more_chunks = more_chunks[: req.top_k]
            context_blocks2 = "\n\n".join(
                [f"[source: {c.get('source_id','')}]\n{c.get('text','')}" for c in more_chunks]
            )
            user_prompt2 = f"Context:\n{context_blocks2}\n\nQuestion: {req.query}\nAnswer:"
            with timer() as t_gen2:
                answer2 = llm.generate(system_prompt, user_prompt2)
            timings["generate_retry"] = t_gen2["elapsed_ms"]
            with timer() as t_sc2:
                groundedness2 = compute_groundedness(answer2, [c.get("text", "") for c in more_chunks])
            timings["self_check_retry"] = t_sc2["elapsed_ms"]
            # If improved, adopt
            if groundedness2 >= (groundedness or 0.0):
                answer = answer2
                groundedness = groundedness2
                chunks = more_chunks
                citations = [RetrievedChunk(**{
                    "text": c.get("text", ""),
                    "source_id": c.get("source_id", ""),
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "score": float(c.get("score", 0.0)),
                }) for c in chunks]
        except Exception:
            pass
    citations = [RetrievedChunk(**{
        "text": c.get("text", ""),
        "source_id": c.get("source_id", ""),
        "chunk_index": int(c.get("chunk_index", 0)),
        "score": float(c.get("score", 0.0)),
    }) for c in chunks]
    # Token usage is provider-specific; placeholder zeros for now
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return QueryResponse(answer=answer, citations=citations, timings_ms=timings, tokens=tokens, groundedness=groundedness)


