"""
utils.py — TREC and qrel I/O helpers
======================================
Utilities for reading/writing the standard TREC run format and Bergen-style
qrel JSON, used across the pipeline.

TREC run format (tab-separated):
    qid  q0  doc_id  rank  score  run_tag

Qrel JSON format (Bergen convention):
    {"qid": {"doc_id": float_gain, ...}, ...}
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple


RankedList = List[Tuple[str, float]]


# ---------------------------------------------------------------------------
# TREC run I/O
# ---------------------------------------------------------------------------

def write_trec(
    path:    str,
    run:     Dict[str, RankedList],
    run_tag: str = "run",
) -> None:
    """
    Write a ranked run to a TREC-format file.

    Parameters
    ----------
    path : str
        Output file path.
    run : dict  {qid: [(pid, score), ...]}
        Sorted highest-score first.
    run_tag : str
        Run identifier written in column 6 (default ``"run"``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for qid, ranked in run.items():
            for rank, (pid, score) in enumerate(ranked, start=1):
                fh.write(f"{qid}\tq0\t{pid}\t{rank}\t{score}\t{run_tag}\n")


def load_trec(path: str) -> Dict[str, RankedList]:
    """
    Load a TREC-format run file.

    Returns
    -------
    dict  {qid: [(pid, score), ...]}
        Sorted by score descending within each query.
    """
    raw: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                parts = line.split()
            qid, _, pid, _, score = parts[0], parts[1], parts[2], parts[3], parts[4]
            # keep highest score if a pid appears more than once
            if pid not in raw[qid] or float(score) > raw[qid][pid]:
                raw[qid][pid] = float(score)

    return {
        qid: sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)
        for qid, pid_scores in raw.items()
    }


# ---------------------------------------------------------------------------
# Qrel JSON I/O
# ---------------------------------------------------------------------------

def load_qrel(path: str) -> Dict[str, Dict[str, float]]:
    """Load Bergen-format qrel JSON → ``{qid: {pid: float_gain}}``."""
    raw = json.load(open(path, encoding="utf-8"))
    return {
        str(qid): {str(pid): float(grade) for pid, grade in docs.items()}
        for qid, docs in raw.items()
    }


def save_qrel(qrel: Dict[str, Dict[str, float]], path: str) -> None:
    """Save a qrel dict to Bergen-format JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(qrel, fh, indent=2)
