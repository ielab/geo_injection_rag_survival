"""
reranker.py — LLM-based reranking for the RAG Attack Pipeline
==============================================================
Provides the pipeline-level adapter that bridges our ESCI corpus structures
with the external `llmrankers` library.

LLMRankersReranker
    A pipeline-level adapter that wraps *any* ``llmrankers.LlmRanker``
    and exposes the outer pipeline interface::

        rerank(candidate_pool, run, top_k) -> Dict[str, RankedList]

    This mirrors how ``run.py`` in llm-rankers drives each ranker per query.

Two-stage flow (mirrors llm-rankers/run.py)
-------------------------------------------
  Stage 1 – retrieval.py  → BM25 first-stage run (TREC format / run dict)
  Stage 2 – LLMRankersReranker loops over queries, builds SearchResult lists,
             calls ranker.rerank(query, results) exactly as run.py does, then
             converts back to (pid, score) pairs.

Format contract (used throughout the pipeline)
----------------------------------------------
  candidate_pool : Dict[str, QueryEntry]          from corpus.py
  run            : Dict[str, List[(pid, score)]]  sorted best-first
"""

from __future__ import annotations

import json

from tqdm import tqdm
from typing import Dict, List, Tuple

from RAG_attack_pipeline.corpus import QueryEntry

# llmrankers is installed editable from llm-rankers/
from llmrankers.rankers import LlmRanker, SearchResult


RankedList = List[Tuple[str, float]]


# ===========================================================================
# LLMRankersReranker — adapter for the llm-rankers library
# ===========================================================================

class LLMRankersReranker:
    """
    Pipeline-level adapter that wraps any ``llmrankers.LlmRanker`` subclass
    and exposes the pipeline's ``{qid: [(pid, score), ...]}`` interface.

    Mirrors the per-query loop in ``llm-rankers/run.py``:
    for each query it builds a ``List[SearchResult]`` from the first-stage
    run, calls ``ranker.rerank(query, results)``, then converts the output
    back to ``(pid, score)`` pairs — exactly as run.py does.

    Parameters
    ----------
    ranker : LlmRanker
        An already-constructed ranker instance (any LlmRanker subclass).
    name : str, optional
        Human-readable label used in progress bars.
    top_k : int
        Default number of documents to re-rank per query.

    Example
    -------
    >>> # Listwise from llmrankers
    >>> from llmrankers.listwise import ListwiseLlmRanker
    >>> inner = ListwiseLlmRanker("castorini/rankllama-v1-7b-listwise-10",
    ...                           tokenizer_name_or_path=None,
    ...                           device="cuda", window_size=10, step_size=5)
    >>> reranker = LLMRankersReranker(inner, name="listwise/rankllama")
    >>> reranked = reranker.rerank(corpus.candidate_pool, bm25_run, top_k=10)
    """

    def __init__(
        self,
        ranker:   LlmRanker,
        name:     str = "llmrankers",
        top_k:    int = 10,
    ) -> None:
        if not isinstance(ranker, LlmRanker):
            raise TypeError(f"Expected an llmrankers.LlmRanker, got {type(ranker)}")
        self.ranker  = ranker
        self.name    = name
        self.top_k   = top_k

    # ------------------------------------------------------------------
    def rerank(
        self,
        candidate_pool: Dict[str, QueryEntry],
        run:            Dict[str, RankedList],
        top_k:          int  = None,
        show_progress:  bool = True,
        log_path:       str  = None,
    ) -> Dict[str, RankedList]:
        """
        Re-rank each query's top-k documents using the wrapped llmrankers ranker.

        Parameters
        ----------
        candidate_pool : dict  {qid: QueryEntry}
        run : dict  {qid: [(pid, score), ...]}  — sorted best-first
        top_k : int, optional
            Override the instance default.
        log_path : str, optional
            If provided, write a JSONL file (one record per query) containing:
            - ``qid`` / ``query``  — query id and text
            - ``candidates``       — input docs with pid, bm25_score, and full text
            - ``reranked``         — output docs with pid and llm_score

        Returns
        -------
        dict  {qid: [(pid, score), ...]}  — sorted by llmrankers score desc
        """
        k = top_k if top_k is not None else self.top_k

        pid_to_text = {
            c.pid: c.text
            for entry in candidate_pool.values()
            for c in entry.candidates
        }

        reranked:  Dict[str, RankedList] = {}
        log_file = open(log_path, "w", encoding="utf-8") if log_path else None
        items = tqdm(run.items(), desc=f"llmrankers reranking ({self.name})") \
                if show_progress else run.items()

        for qid, ranked in items:
            entry = candidate_pool.get(qid)
            if entry is None:
                continue

            # Build SearchResult list from top-k of this query's run
            search_results = [
                SearchResult(docid=pid, score=score, text=pid_to_text.get(pid, ""))
                for pid, score in ranked[:k]
            ]

            # Delegate to the llmrankers ranker
            reranked_results: List[SearchResult] = self.ranker.rerank(
                entry.query, search_results
            )

            # Convert back to our (pid, score) format.
            reranked_top = [(r.docid, r.score) for r in reranked_results]
            top_pids     = {pid for pid, _ in reranked_top}
            tail_start   = min(s for _, s in reranked_top) - 1 if reranked_top else -k - 1
            tail = [
                (pid, tail_start - i)
                for i, (pid, _) in enumerate(ranked[k:])
                if pid not in top_pids
            ]
            reranked[qid] = reranked_top + tail

            # Write one JSONL record per query
            if log_file is not None:
                llm_trace = getattr(self.ranker, "last_compare_trace", None)
                log_file.write(json.dumps({
                    "qid":        qid,
                    "query":      entry.query,
                    "candidates": [
                        {"rank": i + 1, "pid": r.docid, "bm25_score": r.score, "text": r.text}
                        for i, r in enumerate(search_results)
                    ],
                    "reranked":   [
                        {"rank": i + 1, "pid": r.docid, "llm_score": r.score}
                        for i, r in enumerate(reranked_results)
                    ],
                    "llm_trace": llm_trace,
                }, ensure_ascii=False) + "\n")

        if log_file is not None:
            log_file.close()
        return reranked
