"""
evaluate.py — NDCG@k and Recall@k evaluation with float ESCI gain values
=========================================================================
Implements NDCG and Recall directly (no pytrec_eval) so that the exact
ESCI float gain values (1.0 / 0.1 / 0.01 / 0.0) are used in the DCG
formula — matching the original ESCI paper's evaluation protocol.

pytrec_eval requires integer grades, which would round away the subtle
S/C distinction.  Our DCG uses the standard log₂(i+2) discount.

Example
-------
>>> from RAG_attack_pipeline.evaluate import Evaluator
>>> ev = Evaluator(qrel, k_values=[5, 10, 20])
>>> metrics = ev.evaluate(run)
>>> ev.print_summary(metrics)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


RankedList = List[Tuple[str, float]]


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def _dcg(gains: List[float], k: int) -> float:
    """DCG@k  using  Σ gain_i / log₂(i+2)  (i is 0-based rank)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(
    ranked_pids: List[str],
    qrel_qid:    Dict[str, float],
    k:           int,
) -> float:
    """NDCG@k for a single query using float gains."""
    gains       = [qrel_qid.get(pid, 0.0) for pid in ranked_pids[:k]]
    ideal_gains = sorted(qrel_qid.values(), reverse=True)[:k]
    actual      = _dcg(gains, k)
    ideal       = _dcg(ideal_gains, k)
    return actual / ideal if ideal > 0.0 else 0.0


def recall_at_k(
    ranked_pids: List[str],
    qrel_qid:    Dict[str, float],
    k:           int,
) -> float:
    """Recall@k — fraction of relevant (gain > 0) docs retrieved in top-k."""
    relevant  = {pid for pid, g in qrel_qid.items() if g > 0.0}
    if not relevant:
        return 0.0
    top_k     = set(ranked_pids[:k])
    retrieved = len(top_k & relevant)
    return retrieved / len(relevant)


# ---------------------------------------------------------------------------
# Batch evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Compute NDCG@k and Recall@k over a full ranked run.

    Parameters
    ----------
    qrel : dict  {qid: {pid: float_gain}}
    k_values : list of int
        Cut-off values, e.g. ``[5, 10, 20]``.
    """

    def __init__(
        self,
        qrel:     Dict[str, Dict[str, float]],
        k_values: List[int] = None,
    ) -> None:
        self.qrel     = qrel
        self.k_values = k_values or [5, 10, 20]

    def evaluate(
        self,
        run: Dict[str, RankedList],
    ) -> Dict[str, dict]:
        """
        Evaluate a ranked run.

        Parameters
        ----------
        run : dict  {qid: [(pid, score), ...]}
            Pre-sorted (highest score first).

        Returns
        -------
        dict
            ``{"ndcg@10": {"mean": float, "per_query": {qid: float}}, ...}``
        """
        results: Dict[str, dict] = {}

        for k in self.k_values:
            ndcg_scores:   Dict[str, float] = {}
            recall_scores: Dict[str, float] = {}

            for qid, ranked in run.items():
                pids     = [pid for pid, _ in ranked]
                qrel_qid = self.qrel.get(qid, {})
                ndcg_scores[qid]   = ndcg_at_k(pids, qrel_qid, k)
                recall_scores[qid] = recall_at_k(pids, qrel_qid, k)

            results[f"ndcg@{k}"] = {
                "mean":      sum(ndcg_scores.values())   / len(ndcg_scores)   if ndcg_scores   else 0.0,
                "per_query": ndcg_scores,
            }
            results[f"recall@{k}"] = {
                "mean":      sum(recall_scores.values()) / len(recall_scores) if recall_scores else 0.0,
                "per_query": recall_scores,
            }

        return results

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    @staticmethod
    def print_summary(metrics: Dict[str, dict], label: str = "") -> None:
        """Pretty-print aggregate metric table."""
        if label:
            print(f"\n  {label}")
        k_vals = sorted({int(m.split("@")[1]) for m in metrics})
        print(f"  {'Metric':<14} {'Mean':>10}")
        print("  " + "-" * 26)
        for k in k_vals:
            print(f"  {'NDCG@'+str(k):<14} {metrics.get(f'ndcg@{k}',   {}).get('mean', float('nan')):>10.4f}")
            print(f"  {'Recall@'+str(k):<14} {metrics.get(f'recall@{k}', {}).get('mean', float('nan')):>10.4f}")
        print()

    @staticmethod
    def save_metrics(metrics_by_label: Dict[str, Dict[str, dict]], path: str) -> None:
        """
        Save a mapping of {label: evaluate() output} to a JSON file.

        Parameters
        ----------
        metrics_by_label : dict
            Keys are human-readable stage labels (e.g. ``"bm25"``,
            ``"reranked_setwise"``); values are the dicts returned by
            :meth:`evaluate`.  Per-query detail is stripped to keep the
            file compact — use ``save_metrics_verbose`` for the full dump.
        path : str
            Output file path.
        """
        import copy, json, os
        slim: Dict[str, dict] = {}
        for stage_label, metrics in metrics_by_label.items():
            slim[stage_label] = {}
            for metric_key, val in metrics.items():
                slim[stage_label][metric_key] = val.get("mean", 0.0)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(slim, fh, indent=2, ensure_ascii=False)
        print(f"[Evaluator] Ranking metrics saved → {path}")

    @staticmethod
    def save_metrics_verbose(metrics_by_label: Dict[str, Dict[str, dict]], path: str) -> None:
        """
        Save a mapping of {label: evaluate() output} including per-query detail.

        Parameters
        ----------
        metrics_by_label : dict
            Same structure as :meth:`save_metrics` but the full nested dict
            (including ``per_query``) is serialised.
        path : str
            Output file path.
        """
        import json, os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics_by_label, fh, indent=2, ensure_ascii=False)
        print(f"[Evaluator] Verbose ranking metrics saved → {path}")
