"""
attack_dataset.py — Attack dataset builder
==========================================
Converts the reranking pipeline output into a structured JSONL dataset
saved under the ``Attack_dataset/`` folder.

Each record captures one (query, ranked_document) pair and includes all
information needed for downstream adversarial / SEO attack research:

  query_id   — integer query identifier from the ESCI dataset
  query      — natural-language query string
  doc_id     — Amazon product ASIN / product_id
  doc_rank   — 1-based rank after reranking (lower = higher relevance)
  doc_title  — product title (first line of the document text)
  doc_content — full document text (title + bullet points concatenated)
  esci_label — ESCI grade: E (Exact) / S (Substitute) / C (Complement) / I (Irrelevant)
  qrel_score — float relevance gain: E=1.0, S=0.1, C=0.01, I=0.0

The format mirrors the ``books.jsonl`` sample so downstream code can
treat both files uniformly.

Example output record
---------------------
{
    "query_id":    "622",
    "query":       "1 1/2 sink drain without overflow",
    "doc_id":      "B07Q6V1DFR",
    "doc_rank":    1,
    "doc_title":   "REGALMIX Pop Up Drain ...",
    "doc_content": "REGALMIX Pop Up Drain ...\nBUILD-IN STRAINER ...",
    "esci_label":  "E",
    "qrel_score":  1.0
}

Usage
-----
  from RAG_attack_pipeline.attack_dataset import AttackDatasetBuilder
  builder = AttackDatasetBuilder()          # saves to Attack_dataset/
  path = builder.build(corpus.candidate_pool, reranked_run, corpus.qrel)
  print(f"Dataset saved to {path}")

  # From the pipeline CLI
  python run_pipeline.py --jsonl ... --reranker Qwen/Qwen3-8B \\
      --ranker_type pairwise --pairwise_method heapsort \\
      --attack_dataset
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from RAG_attack_pipeline.corpus import Candidate, QueryEntry

RankedList = List[Tuple[str, float]]


# ===========================================================================
# AttackDatasetBuilder
# ===========================================================================
class AttackDatasetBuilder:
    """
    Serialize the reranked pipeline results as a flat JSONL attack dataset.

    Parameters
    ----------
    output_dir : str, optional
        Directory where the JSONL file is written.  Defaults to
        ``<project_root>/Attack_dataset/``.
    """

    def __init__(self, output_dir: str) -> None:
        """Parameters
        ----------
        output_dir : str
            Directory where the JSONL file is written.  Always required;
            callers should pass ``run_dir`` so all artefacts stay together.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_title(text: str) -> str:
        """First non-empty line of the concatenated product text = product title."""
        for line in text.split("\n"):
            line = line.strip()
            if line:
                return line
        return text[:80]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        candidate_pool: Dict[str, QueryEntry],
        reranked_run:   Dict[str, RankedList],
        qrel:           Dict[str, Dict[str, float]],
        filename:       str           = "attack_dataset.jsonl",
        top_k:          Optional[int] = None,
    ) -> str:
        """
        Build and persist the attack dataset.

        Parameters
        ----------
        candidate_pool : dict  {qid: QueryEntry}
            Built by ``ESCICorpus``.
        reranked_run : dict  {qid: [(pid, score), ...]}
            Reranked (or BM25) run — sorted best-first per query.
        qrel : dict  {qid: {pid: float}}
            Relevance judgements from ``ESCICorpus.qrel``.
        filename : str
            Output filename inside *output_dir*
            (default: ``"attack_dataset.jsonl"``).
        top_k : int, optional
            If set, only include the top-k ranked documents per query.
            ``None`` means include all ranked documents.

        Returns
        -------
        str  Absolute path of the written JSONL file.
        """
        out_path      = os.path.join(self.output_dir, filename)
        total_records = 0

        with open(out_path, "w", encoding="utf-8") as fh:
            for qid, ranked in reranked_run.items():
                entry = candidate_pool.get(qid)
                if entry is None:
                    continue

                # Per-query pid → Candidate (preserves correct esci_label)
                pid_to_cand: Dict[str, Candidate] = {
                    c.pid: c for c in entry.candidates
                }
                qrel_qid   = qrel.get(qid, {})
                docs       = ranked if top_k is None else ranked[:top_k]

                for rank, (pid, _rerank_score) in enumerate(docs, start=1):
                    cand = pid_to_cand.get(pid)
                    if cand is None:
                        # Document was retrieved but has no judgement — skip
                        continue

                    title  = self._extract_title(cand.text)
                    record = {
                        "query_id":    qid,
                        "query":       entry.query,
                        "doc_id":      pid,
                        "doc_rank":    rank,
                        "doc_title":   title,
                        "doc_content": cand.text,
                        "esci_label":  cand.esci_label,
                        "qrel_score":  qrel_qid.get(pid, cand.gain),
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_records += 1

        print(
            f"[AttackDataset] {total_records:,} records written"
            f" ({len(reranked_run):,} queries) → {out_path}"
        )
        return out_path
