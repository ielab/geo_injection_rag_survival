"""
corpus.py — ESCI Task-1 corpus loader
======================================
Reads the JSONL judgement file and builds two structures:

  1. CandidatePool  — per-query candidate pool used by the retriever and reranker
  2. Qrel           — graded relevance judgements used by the evaluator

ESCI gain values (per the original ESCI paper):
    E (Exact)      → 1.0
    S (Substitute) → 0.1
    C (Complement) → 0.01
    I (Irrelevant) → 0.0

Example
-------
>>> from RAG_attack_pipeline.corpus import ESCICorpus
>>> corpus = ESCICorpus("path/to/task_1_test.jsonl")
>>> pool = corpus.candidate_pool      # {qid: QueryEntry}
>>> qrel = corpus.qrel                # {qid: {pid: float}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# ESCI gain mapping
# ---------------------------------------------------------------------------
ESCI_GAINS: Dict[str, float] = {
    "E": 1.0,
    "S": 0.1,
    "C": 0.01,
    "I": 0.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A single (query, product) judgement."""
    pid:        str
    text:       str    # product_title + product_bullet_point
    esci_label: str    # E / S / C / I
    gain:       float  # float relevance grade


@dataclass
class QueryEntry:
    """All judged candidates for one query."""
    qid:        str
    query:      str
    candidates: List[Candidate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class ESCICorpus:
    """
    Load an ESCI Task-1 JSONL file into a ready-to-use candidate pool and qrel.

    Parameters
    ----------
    jsonl_path : str
        Path to the JSONL judgement file (e.g. ``task_1_test.jsonl``).
    """

    def __init__(self, jsonl_path: str) -> None:
        self.jsonl_path = jsonl_path
        self.candidate_pool: Dict[str, QueryEntry] = {}
        self.qrel:           Dict[str, Dict[str, float]] = {}
        self._load()

    @staticmethod
    def _build_text(ex: dict) -> str:
        """Concatenate product_title and product_bullet_point."""
        title   = (ex.get("product_title")        or "").strip()
        bullets = (ex.get("product_bullet_point") or "").strip()
        parts   = [p for p in [title, bullets] if p]
        return "\n".join(parts) if parts else ex["product_id"]

    def _load(self) -> None:
        with open(self.jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ex    = json.loads(line)
                qid   = str(ex["query_id"])
                pid   = ex["product_id"]
                label = ex["esci_label"]
                gain  = ESCI_GAINS[label]
                text  = self._build_text(ex)

                # candidate pool
                if qid not in self.candidate_pool:
                    self.candidate_pool[qid] = QueryEntry(qid=qid, query=ex["query"])
                self.candidate_pool[qid].candidates.append(
                    Candidate(pid=pid, text=text, esci_label=label, gain=gain)
                )

                # qrel (all four labels, I=0.0 kept for correct pooled NDCG)
                if qid not in self.qrel:
                    self.qrel[qid] = {}
                self.qrel[qid][pid] = gain

    def __len__(self) -> int:
        return len(self.candidate_pool)

    def __repr__(self) -> str:
        n_judged = sum(len(e.candidates) for e in self.candidate_pool.values())
        return (
            f"ESCICorpus(queries={len(self.candidate_pool):,}, "
            f"judgements={n_judged:,}, "
            f"path='{self.jsonl_path}')"
        )
