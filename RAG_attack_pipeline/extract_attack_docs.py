"""
extract_attack_docs.py — Position-based attack document extraction
=================================================================
Selects documents at fixed rank positions after LLM reranking.

Default positions are **6** and **10** — just inside and at the edge
of the reranker's top-10 window.  The ``doc_rank`` field in the ranked
attack-dataset JSONL mirrors the stage-2 TREC ordering, so rank 6 here
is the 6th row of ``stage2_setwise.trec`` for that query.

Output: ``position_attack_docs.jsonl``

Usage
-----
  # CLI
  python RAG_attack_pipeline/extract_attack_docs.py \\
      --input runs/<run_dir>/attack_dataset.jsonl --positions 6 10

  # As a library
  from RAG_attack_pipeline.extract_attack_docs import (
      extract_position_attack_docs, write_position_dataset,
  )
  result = extract_position_attack_docs("runs/.../attack_dataset.jsonl",
                                        positions=[6, 10])
  # result["records"]     — List[dict], two records per query
  # result["by_position"] — {6: [...], 10: [...]}
  # result["label_dist"]  — {6: {"E":n,"S":n,...}, 10: {...}}
  write_position_dataset(result, out_dir, positions=[6, 10])
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

# Labels present in ESCI (E is the gold label; S / C / I are non-exact)
ATTACK_LABELS = ("S", "C", "I")

# Output filename for position-based mode
_POSITION_FNAME = "position_attack_docs.jsonl"


# ===========================================================================
# Core logic
# ===========================================================================


def extract_position_attack_docs(
    input_path: str,
    positions: List[int] = None,
) -> Dict[str, object]:
    """
    Select attack documents at fixed rank positions after LLM reranking.

    Default positions are **6** and **10** — just inside and at the boundary
    of the reranker's top-10 window.  The ``doc_rank`` field mirrors the
    rank in the stage-2 TREC file.

    Parameters
    ----------
    input_path : str
        Path to the ranked JSONL dataset (e.g. ``attack_dataset.jsonl``).
    positions : list of int, optional
        Rank positions to extract.  Defaults to ``[6, 10]``.

    Returns
    -------
    dict with three keys:

    ``"records"``
        ``List[dict]`` — all extracted records (up to
        ``len(queries) × len(positions)``), sorted by query_id then rank.
        Each record carries an extra ``"selected_position"`` field.

    ``"by_position"``
        ``Dict[int, List[dict]]`` — records partitioned by rank position.

    ``"label_dist"``
        ``Dict[int, Dict[str, int]]`` — for each position, a count of
        ESCI labels (E / S / C / I) appearing there across all queries.
    """
    if positions is None:
        positions = [6, 10]

    # Collect all docs per query, keyed by doc_rank
    # Structure: {query_id: {doc_rank: record}}
    per_query: Dict[str, Dict[int, dict]] = {}

    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid  = rec["query_id"]
            rank = int(rec["doc_rank"])
            if qid not in per_query:
                per_query[qid] = {}
            per_query[qid][rank] = rec

    records: List[dict] = []
    by_position: Dict[int, List[dict]] = {pos: [] for pos in positions}
    label_dist:  Dict[int, Dict[str, int]] = {
        pos: {"E": 0, "S": 0, "C": 0, "I": 0, "MISSING": 0}
        for pos in positions
    }

    for qid in sorted(per_query.keys()):
        rank_map = per_query[qid]
        for pos in positions:
            if pos in rank_map:
                rec = dict(rank_map[pos])
                rec["selected_position"] = pos
                records.append(rec)
                by_position[pos].append(rec)
                lbl = rec.get("esci_label", "").upper()
                if lbl in label_dist[pos]:
                    label_dist[pos][lbl] += 1
                else:
                    label_dist[pos]["MISSING"] += 1
            else:
                label_dist[pos]["MISSING"] += 1
                print(
                    f"  [extract_position_attack_docs][WARN] "
                    f"query {qid} has no document at rank {pos} — skipped"
                )

    return {
        "records":     records,
        "by_position": by_position,
        "label_dist":  label_dist,
    }


def write_position_dataset(
    result:  Dict[str, object],
    out_dir: str,
    positions: List[int] = None,
) -> str:
    """
    Write the position-based attack dataset to a single JSONL file and
    print a label-distribution table for each extracted position.

    Parameters
    ----------
    result    : dict   Return value of :func:`extract_position_attack_docs`.
    out_dir   : str    Output directory (created if absent).
    positions : list of int, optional  Rank positions (default [6, 10]).

    Returns
    -------
    str  Absolute path of the written file.
    """
    if positions is None:
        positions = [6, 10]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, _POSITION_FNAME)
    records  = result["records"]

    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    label_dist  = result["label_dist"]
    by_position = result["by_position"]

    print(f"\n  {len(records)} records ({len(positions)} positions × queries) → {out_path}")
    for pos in positions:
        dist   = label_dist[pos]
        n_docs = len(by_position[pos])
        total  = sum(dist.values())
        print(f"\n  Rank-{pos} label distribution  ({n_docs} queries with a doc at this position)")
        print(f"  {'Label':<8}  {'Count':>5}  {'%':>6}")
        print(f"  {'─'*8}  {'─'*5}  {'─'*6}")
        _label_order = ["E", "S", "C", "I", "MISSING"]
        for lbl in _label_order:
            n   = dist.get(lbl, 0)
            if n == 0:
                continue
            pct = n / total * 100 if total else 0.0
            print(f"  {lbl:<8}  {n:>5}  {pct:>5.1f}%")

    return out_path


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract position-based attack documents from a ranked attack-dataset JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", required=True,
        help="Path to the ranked JSONL dataset (e.g. runs/<run>/attack_dataset.jsonl)",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Output directory. Defaults to the same directory as --input.",
    )
    p.add_argument(
        "--positions", nargs="+", type=int, default=[6, 10],
        metavar="RANK",
        help="Rank positions to extract (default: 6 10).",
    )
    return p


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = build_parser().parse_args()

    input_path = os.path.abspath(args.input)
    out_dir    = (
        os.path.abspath(args.out_dir)
        if args.out_dir
        else os.path.dirname(input_path)
    )

    print(f"\n[extract_attack_docs] Input     : {input_path}")
    print(f"[extract_attack_docs] Output    : {out_dir}")
    print(f"[extract_attack_docs] Positions : {args.positions}\n")

    result = extract_position_attack_docs(input_path, positions=args.positions)
    write_position_dataset(result, out_dir, positions=args.positions)

    print("\n[extract_attack_docs] Done.")
    print(f"\nSample records (first query, positions {args.positions}):")
    shown_pos: set = set()
    for rec in result["records"]:
        pos = rec["selected_position"]
        if pos in shown_pos:
            continue
        shown_pos.add(pos)
        print(
            f"\n  Rank-{pos} → query_id={rec['query_id']} | "
            f"esci_label={rec['esci_label']} | "
            f"qrel_score={rec['qrel_score']} | "
            f"doc_id={rec['doc_id']}\n"
            f"    query : {rec['query']}\n"
            f"    title : {rec['doc_title'][:80]}"
            f"{'...' if len(rec['doc_title']) > 80 else ''}"
        )


if __name__ == "__main__":
    main()
