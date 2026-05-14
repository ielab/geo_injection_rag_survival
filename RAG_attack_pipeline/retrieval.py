"""
retrieval.py — BM25 retrieval over ESCI candidate pools
=========================================================
Runs BM25Okapi **in-memory** over each query's own judged candidate set.

This is the correct framing for ESCI Task-1: the benchmark already defines a
fixed candidate pool per query (~20–40 products).  Running a global index
would introduce unjudged documents that cannot be scored by any NDCG
evaluator.

Example
-------
>>> from RAG_attack_pipeline.corpus import ESCICorpus
>>> from RAG_attack_pipeline.retrieval import BM25Retriever
>>> corpus = ESCICorpus("task_1_test_sample_20_min20.jsonl")
>>> retriever = BM25Retriever()
>>> run = retriever.retrieve(corpus.candidate_pool, top_k=20)
>>> # run: {qid: [(pid, score), ...]}  — already sorted by BM25 score desc
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from RAG_attack_pipeline.corpus import QueryEntry


# Type alias: a ranked list of (product_id, score) pairs
RankedList = List[Tuple[str, float]]


def _tokenize(text: str) -> List[str]:
    """Whitespace tokeniser (lower-cased)."""
    return text.lower().split()


class BM25Retriever:
    """
    Per-query in-memory BM25 retriever.

    No index file is created — BM25Okapi is built on-the-fly for each query
    over its own candidate pool.  This is fast enough for pools of ~20–200
    documents and keeps the code dependency-free (no Lucene / Pyserini).

    Parameters
    ----------
    tokenizer : callable, optional
        Function ``str → List[str]``.  Defaults to whitespace split lower-case.
    """

    def __init__(self, tokenizer=None) -> None:
        self._tokenize = tokenizer or _tokenize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        candidate_pool: Dict[str, QueryEntry],
        top_k: int = 20,
        show_progress: bool = True,
    ) -> Dict[str, RankedList]:
        """
        Score every query's candidates with BM25 and return ranked results.

        Parameters
        ----------
        candidate_pool : dict  {qid: QueryEntry}
            Built by ``ESCICorpus.candidate_pool``.
        top_k : int
            Maximum number of documents to keep per query.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        dict
            ``{qid: [(pid, bm25_score), ...]}`` — sorted by score descending,
            truncated to ``top_k``.
        """
        run: Dict[str, RankedList] = {}

        items = tqdm(candidate_pool.items(), desc="BM25 retrieval") if show_progress \
            else candidate_pool.items()

        for qid, entry in items:
            ranked = self._score_query(entry, top_k)
            run[qid] = ranked

        return run

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _score_query(self, entry: QueryEntry, top_k: int) -> RankedList:
        """BM25-rank one query over its own candidate pool."""
        pids          = [c.pid  for c in entry.candidates]
        tokenized_docs = [self._tokenize(c.text) for c in entry.candidates]
        tokenized_q    = self._tokenize(entry.query)

        bm25   = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_q).tolist()

        ranked = sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
