#!/usr/bin/env python3
"""
run_pipeline.py — Two-stage ESCI retrieval + LLM reranking pipeline
====================================================================

Stage 1  BM25 first-stage retrieval (or skip with --trec to load an existing run).
Stage 2  Optional LLM reranking via the llm-rankers library.

Outputs (all saved to runs/<timestamp>/ unless overridden)
----------------------------------------------------------
  qrel.json              Relevance judgements derived from the JSONL file.
  stage1_{type}.trec     TREC run after Stage-1 retrieval (bm25 or dense).
  stage2_<type>.trec     TREC run after LLM reranking.
  ranking_log.jsonl      Per-query log: query text, candidate docs, and LLM scores.

Typical usage
-------------

  # Full pipeline with chatbot + attack dataset export
  python RAG_attack_pipeline/run_pipeline.py --jsonl ../esci-data/shopping_queries_dataset/task_1_test_sample_20_min20.jsonl --reranker Qwen/Qwen3-8B --ranker_type listwise --chatbot --chatbot_top_k 5 --attack_dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

# ── MKL / OpenMP threading fix ────────────────────────────────────────────────
# vLLM spawns a subprocess to inspect the model architecture.  That subprocess
# crashes when Intel MKL tries to use the INTEL threading layer while libgomp
# (GNU OpenMP, pulled in by PyTorch CUDA kernels) is already loaded.
# Setting MKL_THREADING_LAYER=GNU *before* any numpy/torch import fixes it, and
# the value is automatically inherited by every subprocess we spawn.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

# ── vLLM multi-GPU worker start method ───────────────────────────────────────
# vLLM 0.16.0 V1 engine: EngineCore forks GPU workers by default.  If CUDA
# was already initialised inside EngineCore before the fork, PyTorch raises
# "Cannot re-initialize CUDA in forked subprocess".  Forcing 'spawn' gives
# each GPU worker a clean process with no inherited CUDA state.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG_attack_pipeline.corpus         import ESCICorpus
from RAG_attack_pipeline.retrieval      import BM25Retriever
from RAG_attack_pipeline.evaluate       import Evaluator
from RAG_attack_pipeline.utils          import write_trec, load_trec, save_qrel
from RAG_attack_pipeline.attack_dataset import AttackDatasetBuilder


# ---------------------------------------------------------------------------
# Reranker factory
# ---------------------------------------------------------------------------
def _build_reranker(args):
    """Instantiate the requested reranker based on --ranker_type."""
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    rt       = args.ranker_type.lower()
    use_vllm = getattr(args, "use_vllm", False)
    tp_size  = getattr(args, "tensor_parallel_size", 1)

    from RAG_attack_pipeline.reranker import LLMRankersReranker
    from llmrankers.listwise import ListwiseLlmRanker
    inner = ListwiseLlmRanker(
        model_name_or_path=args.reranker,
        tokenizer_name_or_path=args.reranker,
        device=device,
        window_size=args.window_size,
        step_size=args.step_size,
        scoring=args.scoring,
        num_repeat=args.num_repeat,
        use_vllm=use_vllm,
        tensor_parallel_size=tp_size,
    )
    return LLMRankersReranker(inner, name=f"listwise/{args.reranker.split('/')[-1]}")


def _maybe_log_listwise_debug(
    args: argparse.Namespace,
    reranker_obj: Any,
    candidate_pool: dict,
    run: dict,
    run_dir: str,
    stage_tag: str,
) -> None:
    """
    Dump listwise prompt/context debug records before actual reranking.

    This is only active when:
      --ranker_type listwise --listwise_debug

    Output:
      runs/.../listwise_debug_<stage_tag>.jsonl
    """
    if getattr(args, "ranker_type", "").lower() != "listwise":
        return
    if not getattr(args, "listwise_debug", False):
        return

    inner = getattr(reranker_obj, "ranker", None)
    if inner is None:
        print("      [Listwise debug] skip: no inner ranker found")
        return

    from llmrankers.listwise import create_permutation_instruction_chat
    from llmrankers.rankers import SearchResult

    q_limit = max(1, int(getattr(args, "listwise_debug_queries", 2)))
    d_limit_arg = getattr(args, "listwise_debug_docs", None)
    if d_limit_arg is None:
        # Default debug width follows listwise window_size so the dumped prompt
        # matches the first compare() call in the ranker.
        d_limit = max(1, int(getattr(args, "window_size", 10)))
    else:
        d_limit = max(1, int(d_limit_arg))
    print_chars = max(200, int(getattr(args, "listwise_debug_print_chars", 1200)))
    prompt_k = min(int(getattr(args, "rerank_top_k", d_limit)), d_limit)

    out_path = os.path.join(run_dir, f"listwise_debug_{stage_tag}.jsonl")
    dumped = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for qid, ranked in run.items():
            if dumped >= q_limit:
                break
            entry = candidate_pool.get(qid)
            if entry is None:
                continue

            docs = [
                SearchResult(docid=pid, score=score, text=next(
                    (c.text for c in entry.candidates if c.pid == pid), ""
                ))
                for pid, score in ranked[:prompt_k]
            ]
            if not docs:
                continue

            messages = create_permutation_instruction_chat(
                entry.query, docs, model_name=None
            )

            rendered_prompt = None
            if hasattr(inner, "tokenizer") and hasattr(inner.tokenizer, "apply_chat_template"):
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                mt = getattr(getattr(inner, "config", None), "model_type", "")
                if mt in ["qwen3", "qwen3_moe"]:
                    kwargs["enable_thinking"] = False
                try:
                    rendered_prompt = inner.tokenizer.apply_chat_template(messages, **kwargs)
                except TypeError:
                    kwargs.pop("enable_thinking", None)
                    rendered_prompt = inner.tokenizer.apply_chat_template(messages, **kwargs)

            record = {
                "stage": stage_tag,
                "qid": qid,
                "query": entry.query,
                "ranker_type": "listwise",
                "reranker_model": getattr(args, "reranker", None),
                "window_size": getattr(args, "window_size", None),
                "step_size": getattr(args, "step_size", None),
                "num_repeat": getattr(args, "num_repeat", None),
                "scoring": getattr(args, "scoring", None),
                "debug_prompt_docs": prompt_k,
                "docs": [
                    {
                        "rank": i + 1,
                        "pid": d.docid,
                        "stage1_score": d.score,
                        "text": d.text,
                    }
                    for i, d in enumerate(docs)
                ],
                "chat_messages": messages,
                "rendered_prompt": rendered_prompt,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            pids = ", ".join(d.docid for d in docs)
            print(f"      [Listwise debug] qid={qid} docs={len(docs)} pids=[{pids}]")
            if rendered_prompt:
                print("      [Listwise debug] prompt preview:")
                print(rendered_prompt[:print_chars])
                if len(rendered_prompt) > print_chars:
                    print("      ... (truncated)")
            else:
                print("      [Listwise debug] chat_messages preview:")
                print(json.dumps(messages[:4], ensure_ascii=False)[:print_chars])
                print("      ...")
            dumped += 1

    print(f"      [Listwise debug] saved → {out_path}")


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------
def _build_retriever(args):
    """Instantiate the requested Stage-1 retriever based on --retriever_type."""
    rt = getattr(args, "retriever_type", "bm25").lower()
    if rt == "bm25":
        return BM25Retriever()
    if rt == "dense":
        from RAG_attack_pipeline.dense_retriever import DenseRetriever
        model      = getattr(args, "retriever_model", None) or "BAAI/bge-large-en-v1.5"
        batch_size = getattr(args, "retriever_batch_size", 32)
        return DenseRetriever(model, batch_size=batch_size)
    raise ValueError(
        f"Unknown --retriever_type '{rt}'. Choose from: bm25, dense"
    )


# ---------------------------------------------------------------------------
# Retriever-type auto-detection for validate modes
# ---------------------------------------------------------------------------
def _auto_detect_rtype(baseline_run_dir: str, preferred: str = "bm25") -> str:
    """
    Return the Stage-1 retriever type appropriate for *baseline_run_dir*.

    If ``stage1_{preferred}.trec`` already exists in the directory the
    *preferred* type is returned unchanged.  Otherwise the first matching
    ``stage1_{candidate}.trec`` file is used and a notice is printed.

    This lets ``--mode validate`` and ``--mode validate_e2e`` work correctly
    even when ``--retriever_type`` is not explicitly supplied on the CLI —
    e.g. a dense baseline run contains ``stage1_dense.trec``, not
    ``stage1_bm25.trec``, so the default of ``bm25`` would fail.
    """
    if os.path.exists(os.path.join(baseline_run_dir, f"stage1_{preferred}.trec")):
        return preferred
    for candidate in ("bm25", "dense"):
        candidate_path = os.path.join(baseline_run_dir, f"stage1_{candidate}.trec")
        if os.path.exists(candidate_path):
            print(
                f"    [Auto-detect] stage1_{preferred}.trec not found in "
                f"{os.path.basename(baseline_run_dir)}; "
                f"switching to '{candidate}' (stage1_{candidate}.trec found)"
            )
            return candidate
    # Nothing found — return preferred so the subsequent open() raises a
    # clear FileNotFoundError rather than a cryptic downstream failure.
    return preferred


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace) -> None:
    timestamp = getattr(args, "run_name", None) or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[0/4] Initialized run directory: {run_dir}")

    # 1. Load corpus
    print(f"\n[1/4] Loading corpus from: {args.jsonl}")
    corpus = ESCICorpus(args.jsonl)
    print(f"      {corpus}")

    # Optional: restrict corpus to a specific set of query IDs
    filter_ids_path = getattr(args, "filter_query_ids", None)
    if filter_ids_path:
        with open(filter_ids_path, encoding="utf-8") as _f:
            _filter_qids = {ln.strip() for ln in _f if ln.strip()}
        before = len(corpus.candidate_pool)
        corpus.candidate_pool = {k: v for k, v in corpus.candidate_pool.items() if k in _filter_qids}
        corpus.qrel           = {k: v for k, v in corpus.qrel.items()           if k in _filter_qids}
        print(f"      [filter_query_ids] {before} → {len(corpus.candidate_pool)} queries "
              f"(kept {len(_filter_qids)} requested IDs from {filter_ids_path})")

    if args.save_qrel:
        save_qrel(corpus.qrel, args.save_qrel)
        print(f"      Qrel saved → {args.save_qrel}")
    else:
        # Default save qrel to runs folder for debugging
        qrel_out = os.path.join(run_dir, "qrel.json")
        save_qrel(corpus.qrel, qrel_out)
        print(f"      Qrel auto-saved → {qrel_out}")

    # 2. Retrieve (Stage 1)
    if args.trec:
        print(f"\n[2/4] Loading existing TREC run from: {args.trec}")
        run_stage_1 = load_trec(args.trec)
        print(f"      Loaded {len(run_stage_1)} queries")
    else:
        rtype = getattr(args, "retriever_type", "bm25")
        print(f"\n[2/4] {rtype.upper()} retrieval (Stage 1)  (top_k={args.retrieve_top_k})")
        retriever   = _build_retriever(args)
        run_stage_1 = retriever.retrieve(corpus.candidate_pool, top_k=args.retrieve_top_k)

        stage1_out = args.out if args.out else os.path.join(run_dir, f"stage1_{rtype}.trec")
        write_trec(stage1_out, run_stage_1, run_tag=rtype)
        print(f"      {rtype.upper()} run saved \u2192 {stage1_out}")

        # Also print debug stats
        if args.verbose:
            total_retrieved = sum(len(docs) for docs in run_stage_1.values())
            print(f"      [Debug] Total queries retrieved: {len(run_stage_1)}")
            print(f"      [Debug] Total docs retrieved: {total_retrieved} (Avg: {total_retrieved/len(run_stage_1):.1f}/query)")

        # Release dense retriever GPU memory before loading the reranker.
        # SentenceTransformer keeps the encoder on GPU; without explicit release
        # the subsequent LLM reranker load may trigger an OOM error.
        if rtype == "dense" and hasattr(retriever, "release"):
            retriever.release()

    # 3. Reranking (Stage 2)
    reranked_run = run_stage_1
    if args.reranked_trec:
        # Fast-path: load a pre-computed Stage-2 run and skip model inference entirely
        print(f"\n[3/4] Loading existing reranked TREC run from: {args.reranked_trec}")
        reranked_run = load_trec(args.reranked_trec)
        print(f"      Loaded {len(reranked_run)} queries")
    elif args.reranker:
        print(f"\n[3/4] Reranking (Stage 2) type={args.ranker_type} model={args.reranker} top_k={args.rerank_top_k}")
        if args.verbose:
            print(f"      [Debug] Input to reranker: top-{args.rerank_top_k} documents per query")
        reranker  = _build_reranker(args)
        log_path  = os.path.join(run_dir, "ranking_log.jsonl")
        _maybe_log_listwise_debug(
            args=args,
            reranker_obj=reranker,
            candidate_pool=corpus.candidate_pool,
            run=run_stage_1,
            run_dir=run_dir,
            stage_tag="baseline",
        )
        reranked_run = reranker.rerank(
            corpus.candidate_pool, run_stage_1,
            top_k=args.rerank_top_k, log_path=log_path,
        )

        reranked_out = args.reranked_out if args.reranked_out else os.path.join(run_dir, f"stage2_{args.ranker_type}.trec")
        write_trec(reranked_out, reranked_run, run_tag=args.ranker_type)
        print(f"      Reranked run saved → {reranked_out}")
        print(f"      Ranking log saved  → {log_path}")

        # Also print debug stats
        if args.verbose:
            total_reranked = sum(len(docs) for docs in reranked_run.values())
            print(f"      [Debug] Total queries reranked: {len(reranked_run)}")
            print(f"      [Debug] Total docs output: {total_reranked} (Avg: {total_reranked/len(reranked_run):.1f}/query)")
    else:
        print(f"\n[3/4] Skipping reranking  (pass --reranker <model_id> or --reranked_trec <path> to enable)")

    # 4. Evaluate
    print(f"\n[4/4] Evaluation  k={args.k}")
    evaluator = Evaluator(corpus.qrel, k_values=args.k)

    rtype = getattr(args, "retriever_type", "bm25")
    print()
    stage1_metrics = evaluator.evaluate(run_stage_1)
    Evaluator.print_summary(stage1_metrics, label=f"{rtype.upper()} (Stage 1)")

    ranking_metrics: dict = {rtype: stage1_metrics}
    if args.reranker or args.reranked_trec:
        llm_metrics = evaluator.evaluate(reranked_run)
        if args.reranker:
            stage2_label = f"{args.ranker_type} ({args.reranker.split('/')[-1]}) (Stage 2)"
        else:
            stage2_label = f"loaded ({os.path.basename(args.reranked_trec)}) (Stage 2)"
        Evaluator.print_summary(llm_metrics, label=stage2_label)
        ranking_metrics[f"reranked_{args.ranker_type}"] = llm_metrics

    Evaluator.save_metrics(ranking_metrics, os.path.join(run_dir, "ranking_metrics.json"))
    Evaluator.save_metrics_verbose(ranking_metrics, os.path.join(run_dir, "ranking_metrics_verbose.json"))

    # 5. Attack dataset export (optional — also auto-triggered by --attack_eval)
    ds_path  = None
    chat_log = None
    if args.attack_dataset or args.attack_eval:
        print(f"\n[5] Building attack dataset (top_k={args.attack_top_k or 'all'})")
        # Save the full ranked JSONL into run_dir, not a separate Attack_dataset/ folder.
        # Everything for this run lives together in run_dir.
        builder = AttackDatasetBuilder(output_dir=args.attack_out_dir or run_dir)
        ds_path = builder.build(
            corpus.candidate_pool,
            reranked_run,
            corpus.qrel,
            top_k=args.attack_top_k,
        )
        print(f"      Attack dataset → {ds_path}")

        from RAG_attack_pipeline.extract_attack_docs import (
            extract_position_attack_docs, write_position_dataset,
        )

        # Extract docs at positions 6 / 10 (just inside / at the edge of rerank top-10).
        # This produces position_attack_docs.jsonl in run_dir — the file
        # that run_attack.py reads to generate ioa_position_attack_docs.jsonl, etc.
        _pos_result = extract_position_attack_docs(ds_path, positions=[6, 10])
        write_position_dataset(_pos_result, run_dir, positions=[6, 10])

    # ── Release the reranker's vLLM engine before starting the chatbot engine ──
    # vLLM v1 (EngineCore architecture) cannot have two LLM instances alive in
    # the same process simultaneously.  The SetwiseLlmRanker / VLLMPointwise-
    # Reranker already holds an LLM() with its background EngineCore process and
    # NCCL process group.  Creating a second LLM() for the chatbot without first
    # destroying the first one causes:
    #   RuntimeError: Engine core initialization failed. Failed core proc(s): {}
    # Fix: explicitly delete the reranker object, force GC, and flush GPU cache
    # so the EngineCore background processes have time to terminate cleanly.
    if args.chatbot and args.reranker and getattr(args, "use_vllm", False):
        import gc
        import torch as _torch
        print("\n[Pre-6] Releasing reranker vLLM engine before loading chatbot...")
        del reranker
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
        print("        vLLM engine released.")

    # 6. RAG Chatbot — aggregate top-5 reranked products and generate answer (optional)
    if args.chatbot:
        from RAG_attack_pipeline.chatbot import RagChatbot
        chatbot_model = args.chatbot_model or args.reranker
        if not chatbot_model:
            print("\n[6] Chatbot skipped — no model specified. "
                  "Pass --chatbot_model <hf_id> or --reranker <hf_id>.")
        else:
            print(f"\n[6] RAG Chatbot  model={chatbot_model}  top_k={args.chatbot_top_k}")
            bot      = RagChatbot(
                model_name_or_path=chatbot_model,
                max_new_tokens=512,
                top_k_context=args.chatbot_top_k,
                use_vllm=getattr(args, "use_vllm", False),
                tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
            )
            chat_log = os.path.join(run_dir, "chatbot_responses.jsonl")
            responses = bot.generate_all(
                corpus.candidate_pool,
                reranked_run,
                top_k=args.chatbot_top_k,
                log_path=chat_log,
            )
            print(f"      Chatbot responses saved → {chat_log}")

    # 7. Attack evaluation (optional)
    if args.attack_eval:
        from RAG_attack_pipeline.attack_eval import AttackEvaluator, save_report, save_report_verbose, save_report_markdown

        print(f"\n[7] Attack Evaluation  (nDCG@{max(args.k)} | Rank Promotion | Success@3)")

        # _pos_result was built in Step 5 (attack_dataset / attack_eval both trigger it).
        # It contains all docs at positions 6 / 11 / 20 with their original (pre-attack)
        # text.  This is our baseline: no adversarial injection yet, so Exposure@5
        # and Success@3 should be near 0%.
        attack_docs_by_label = _records_by_label(_pos_result["records"])
        print(f"      Using position attack docs (pre-attack baseline): "
              f"E={len(attack_docs_by_label['E'])}  "
              f"S={len(attack_docs_by_label['S'])}  "
              f"C={len(attack_docs_by_label['C'])}  "
              f"I={len(attack_docs_by_label['I'])}")

        evaluator_atk = AttackEvaluator(
            qrel=corpus.qrel,
            bm25_run=run_stage_1,
            reranked_run=reranked_run,
            k=max(args.k),
        )
        atk_report = evaluator_atk.evaluate_all(
            attack_docs_by_label,
            chatbot_log_path=chat_log,
            chatbot_top_k=args.chatbot_top_k,
        )
        ref_label = f"stage1_{rtype}.trec"
        if args.reranked_trec:
            atk_label = os.path.basename(args.reranked_trec)
        elif args.reranker:
            atk_label = f"stage2_{args.ranker_type}.trec"
        else:
            atk_label = ref_label
        AttackEvaluator.print_report(
            atk_report,
            k=max(args.k),
            chatbot_top_k=args.chatbot_top_k,
            reference_label=ref_label,
            attacked_label=atk_label,
            verbose=args.verbose,
        )

        role_runs = {
            "ref_retriever": run_stage_1,
            "ref_reranker": run_stage_1,
            "attack_retriever": run_stage_1,
            "attack_reranker": reranked_run,
        }
        role_labels = {
            "ref_retriever": ref_label,
            "ref_reranker": ref_label,
            "attack_retriever": ref_label,
            "attack_reranker": atk_label,
        }
        role_stats = AttackEvaluator.compute_rank_role_stats(
            attack_docs_by_label,
            role_runs,
        )
        AttackEvaluator.print_rank_role_stats(role_stats, role_labels)

        # Save reports
        save_report(
            atk_report,
            os.path.join(run_dir, "attack_eval_report.json"),
        )
        save_report_verbose(
            atk_report,
            os.path.join(run_dir, "attack_eval_report_verbose.json"),
        )
        save_report_markdown(
            atk_report,
            os.path.join(run_dir, "attack_eval_report.md"),
            k=max(args.k),
            chatbot_top_k=args.chatbot_top_k,
            reference_label=ref_label,
            attacked_label=atk_label,
            mode="baseline",
            extra_info={
                "Retriever": getattr(args, "retriever_type", "bm25").upper(),
                "Ranker": args.ranker_type if args.reranker else "none",
                "Run dir": run_dir,
            },
        )
        with open(os.path.join(run_dir, "attack_eval_rank_roles.json"), "w", encoding="utf-8") as fh:
            json.dump({"role_labels": role_labels, "role_stats": role_stats}, fh, indent=2, ensure_ascii=False)

    # ── Final Experiment Summary ──────────────────────────────────────────────
    _W = 64
    print(f"\n{'═' * _W}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'═' * _W}")
    rtype_display  = getattr(args, "retriever_type", "bm25").upper()
    ranker_display = args.ranker_type if (args.reranker or getattr(args, "reranked_trec", None)) else "none"
    print(f"  Retriever : {rtype_display}")
    print(f"  Ranker    : {ranker_display}")
    print(f"  Run dir   : {run_dir}")

    # Ranking quality
    print(f"\n  {'─' * (_W - 2)}")
    print(f"  Ranking Quality")
    for stage_key, stage_metrics in ranking_metrics.items():
        for kv in sorted(args.k):
            ndcg_val   = stage_metrics.get(f"ndcg@{kv}",   {}).get("mean", float("nan"))
            recall_val = stage_metrics.get(f"recall@{kv}", {}).get("mean", float("nan"))
            print(f"    {stage_key:<32} nDCG@{kv:<3} {ndcg_val:.4f}  Recall@{kv:<3} {recall_val:.4f}")

    # Attack effectiveness (only when attack_eval was run)
    if args.attack_eval:
        s3_ov = atk_report.get("success_at_3", {}).get("overall", {})
        n_atk  = s3_ov.get("n_attack_docs", 0)
        exp5   = s3_ov.get(f"exposure@{args.chatbot_top_k}", 0.0)
        exp3   = s3_ov.get("exposure@3", 0.0)
        suc3   = s3_ov.get("success@3", 0.0)
        s_promo = atk_report.get("rank_promotion", {}).get("S", {})
        norm_p  = s_promo.get("avg_norm_promotion", float("nan"))

        print(f"\n  {'─' * (_W - 2)}")
        print(f"  Attack Effectiveness  (N={n_atk} attack docs)")
        print(f"    Exposure@{args.chatbot_top_k}            {exp5:>6.1f}%")
        print(f"    Exposure@3             {exp3:>6.1f}%")
        print(f"    Success@3              {suc3:>6.1f}%")
        print(f"    S NormPromo            {norm_p:>+7.3f}")

        if "position_stats" in atk_report:
            pos_stats = atk_report["position_stats"]
            print(f"\n  {'─' * (_W - 2)}")
            print(f"  {'By Position':<12}  {'Exp@'+str(args.chatbot_top_k):>7}  {'Exp@3':>7}  {'Suc@3':>7}")
            for pos in sorted(int(k0) for k0 in pos_stats.keys()):
                pdata = pos_stats.get(pos) or pos_stats.get(str(pos), {})
                ps3   = pdata.get("success_at_3", {}).get("overall", {})
                p_e5  = ps3.get(f"exposure@{args.chatbot_top_k}", 0.0)
                p_e3  = ps3.get("exposure@3", 0.0)
                p_s3  = ps3.get("success@3", 0.0)
                diff  = "easy" if pos == 6 else "hard" if pos == 10 else "    "
                print(f"    rank-{pos} ({diff:<4})        {p_e5:>6.1f}%  {p_e3:>6.1f}%  {p_s3:>6.1f}%")

    print(f"{'═' * _W}\n")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _records_by_label(attack_records: list) -> dict:
    """Partition a flat list of attack records into {label: [records]} dict.

    Records come from ``position_attack_docs.jsonl`` and carry an
    ``esci_label`` field (E / S / C / I).  E-label documents are included
    because the AttackEvaluator._group_by_position() method uses this dict
    to build per-position × per-label breakdowns.
    """
    by_label: dict = {"E": [], "S": [], "C": [], "I": []}
    for rec in attack_records:
        # Prefer explicit attack_label; fall back to esci_label for
        # position-based attack files that don't carry attack_label.
        lbl = rec.get("attack_label") or rec.get("esci_label", "")
        if lbl in by_label:
            by_label[lbl].append(rec)
    return by_label


def _patch_pool(candidate_pool: dict, attacked_docs_path: str):
    """
    Read ``attacked_docs.jsonl`` (output of ``run_attack.py``) and return a
    deep copy of *candidate_pool* with each attacked document's text replaced
    by the adversarially modified ``doc_content``.

    Parameters
    ----------
    candidate_pool    : dict  {qid: QueryEntry}
    attacked_docs_path: str   Path to ``attacked_docs.jsonl``.

    Returns
    -------
    (attacked_records, patched_pool)
        attacked_records  list of dicts from attacked_docs.jsonl
        patched_pool      deep copy of candidate_pool with modified texts
    """
    import copy

    records = []
    with open(attacked_docs_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    patched   = copy.deepcopy(candidate_pool)
    n_patched = 0
    for rec in records:
        qid      = rec["query_id"]
        pid      = rec["doc_id"]
        new_text = rec["doc_content"]
        if qid in patched:
            for cand in patched[qid].candidates:
                if cand.pid == pid:
                    cand.text = new_text
                    n_patched += 1
                    break

    print(f"    [Patch] {n_patched}/{len(records)} docs patched in corpus pool")
    return records, patched


def _print_rerank_input_debug(
    candidate_pool: dict,
    run: dict,
    attacked_records: list,
    rerank_top_k: int,
) -> None:
    """
    Print the ranked doc list that will be fed to the reranker for every query,
    with clear markers on attacked documents.

    For each query shows:
      rank | doc_id | esci_label | title (truncated) | ** ATTACKED pos=X method=Y **

    Call this just before reranker.rerank() to verify attacked docs are present.
    """
    # Build lookup: (query_id, doc_id) -> attack info string
    attack_info: dict[tuple, str] = {}
    for rec in attacked_records:
        key = (str(rec["query_id"]), str(rec["doc_id"]))
        pos    = rec.get("selected_position", "?")
        method = rec.get("attack_method", "?")
        label  = rec.get("esci_label", "?")
        attack_info[key] = f"pos={pos} method={method} esci={label}"

    print("\n" + "─" * 72)
    print(f"  [DEBUG] Reranker input list  (top_k={rerank_top_k})")
    print("─" * 72)

    n_queries_with_both = 0
    for qid, ranked in sorted(run.items()):
        top_docs = ranked[:rerank_top_k]
        attacked_in_window = [
            (i + 1, pid) for i, (pid, _) in enumerate(top_docs)
            if (str(qid), str(pid)) in attack_info
        ]

        entry     = candidate_pool.get(qid)
        query_txt = entry.query if entry else "?"
        print(f"\n  Query {qid}: \"{query_txt}\"")
        print(f"  {'Rank':<5} {'DocID':<15} {'Title (truncated)':<45} {'Note'}")
        print(f"  {'----':<5} {'-----':<15} {'------------------':<45} {'----'}")

        for rank, (pid, score) in enumerate(top_docs, start=1):
            cand  = next((c for c in entry.candidates if c.pid == pid), None) if entry else None
            title = (cand.text.replace("\n", " ")[:45] if cand else "")
            key   = (str(qid), str(pid))
            note  = f"** ATTACKED {attack_info[key]} **" if key in attack_info else ""
            print(f"  {rank:<5} {pid:<15} {title:<45} {note}")

        n_attacked = len(attacked_in_window)
        if n_attacked == 0:
            print(f"  !! WARNING: no attacked docs in top-{rerank_top_k} for this query")
        elif n_attacked == 1:
            r, pid = attacked_in_window[0]
            print(f"  -> 1 attacked doc in window (rank {r})")
        else:
            ranks_str = ", ".join(f"rank {r}" for r, _ in attacked_in_window)
            print(f"  -> {n_attacked} attacked docs in window ({ranks_str})")
            n_queries_with_both += 1

    print("\n" + "─" * 72)
    print(f"  [DEBUG] Queries with ≥2 attacked docs in reranker window: "
          f"{n_queries_with_both}/{len(run)}")
    print("─" * 72 + "\n")


def _rank_in_run(run: dict, qid: str, pid: str) -> int | None:
    """Return 1-based rank of (qid, pid) in a TREC run dict, else None."""
    for i, (p, _) in enumerate(run.get(qid, []), start=1):
        if p == pid:
            return i
    return None


def _summarize_rank_flow(
    records: list[dict],
    clean_run: dict | None,
    attacked_run: dict,
    reranked_run: dict,
    top_k: int = 10,
) -> dict:
    """Summarize rank flow: clean retrieval -> attacked retrieval -> attacked rerank."""
    n = len(records)
    if n == 0:
        return {"n_total": 0}

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    rows = []
    for rec in records:
        qid = rec["query_id"]
        pid = rec["doc_id"]
        clean_rank = _rank_in_run(clean_run, qid, pid) if clean_run else None
        attacked_rank = _rank_in_run(attacked_run, qid, pid)
        rerank_rank = _rank_in_run(reranked_run, qid, pid)
        rows.append({
            "query_id": qid,
            "doc_id": pid,
            "esci_label": rec.get("esci_label"),
            "selected_position": rec.get("selected_position"),
            "clean_rank": clean_rank,
            "attacked_retrieval_rank": attacked_rank,
            "attacked_rerank_rank": rerank_rank,
        })

    clean_found = [r for r in rows if r["clean_rank"] is not None]
    attacked_found = [r for r in rows if r["attacked_retrieval_rank"] is not None]
    rerank_found = [r for r in rows if r["attacked_rerank_rank"] is not None]

    n_clean_top10 = sum(1 for r in rows if (r["clean_rank"] or 10**9) <= top_k)
    n_attacked_top10 = sum(1 for r in rows if (r["attacked_retrieval_rank"] or 10**9) <= top_k)
    n_rerank_top10 = sum(1 for r in rows if (r["attacked_rerank_rank"] or 10**9) <= top_k)

    n_drop_clean_to_attacked = sum(
        1 for r in rows
        if (r["clean_rank"] is not None and r["clean_rank"] <= top_k)
        and (r["attacked_retrieval_rank"] is None or r["attacked_retrieval_rank"] > top_k)
    )
    n_drop_attacked_to_rerank = sum(
        1 for r in rows
        if (r["attacked_retrieval_rank"] is not None and r["attacked_retrieval_rank"] <= top_k)
        and (r["attacked_rerank_rank"] is None or r["attacked_rerank_rank"] > top_k)
    )

    attacked_retrieval_ranks = [r["attacked_retrieval_rank"] for r in attacked_found]
    attacked_rerank_ranks = [r["attacked_rerank_rank"] for r in rerank_found]

    out = {
        "n_total": n,
        "top_k": top_k,
        "n_found_clean": len(clean_found),
        "n_found_attacked_retrieval": len(attacked_found),
        "n_found_attacked_rerank": len(rerank_found),
        "n_clean_top_k": n_clean_top10,
        "n_attacked_retrieval_top_k": n_attacked_top10,
        "n_attacked_rerank_top_k": n_rerank_top10,
        "clean_top_k_pct": n_clean_top10 / n * 100,
        "attacked_retrieval_top_k_pct": n_attacked_top10 / n * 100,
        "attacked_rerank_top_k_pct": n_rerank_top10 / n * 100,
        "n_drop_clean_to_attacked_retrieval": n_drop_clean_to_attacked,
        "n_drop_attacked_retrieval_to_rerank": n_drop_attacked_to_rerank,
        "avg_attacked_retrieval_rank": _mean([float(v) for v in attacked_retrieval_ranks]),
        "avg_attacked_rerank_rank": _mean([float(v) for v in attacked_rerank_ranks]),
        "ranks": {
            "attacked_retrieval": attacked_retrieval_ranks,
            "attacked_rerank": attacked_rerank_ranks,
        },
        "per_doc": rows,
    }
    if clean_run:
        clean_ranks = [r["clean_rank"] for r in clean_found]
        out["avg_clean_rank"] = _mean([float(v) for v in clean_ranks])
        out["ranks"]["clean"] = clean_ranks
    return out


def _compute_validate_e2e_rank_flow_stats(
    attacked_records: list[dict],
    baseline_bm25_run: dict | None,
    attacked_bm25_run: dict,
    reranked_run: dict,
    top_k: int = 10,
) -> dict:
    """Build overall + per-position + per-label rank-flow stats for validate_e2e."""
    by_pos: dict[int, list[dict]] = {}
    by_label: dict[str, list[dict]] = {"E": [], "S": [], "C": [], "I": []}
    for rec in attacked_records:
        pos = rec.get("selected_position")
        if isinstance(pos, int):
            by_pos.setdefault(pos, []).append(rec)
        lbl = rec.get("esci_label", "")
        if lbl in by_label:
            by_label[lbl].append(rec)

    report = {
        "overall": _summarize_rank_flow(
            attacked_records, baseline_bm25_run, attacked_bm25_run, reranked_run, top_k=top_k
        ),
        "by_position": {
            str(pos): _summarize_rank_flow(
                recs, baseline_bm25_run, attacked_bm25_run, reranked_run, top_k=top_k
            )
            for pos, recs in sorted(by_pos.items())
        },
        "by_label": {
            lbl: _summarize_rank_flow(
                recs, baseline_bm25_run, attacked_bm25_run, reranked_run, top_k=top_k
            )
            for lbl, recs in by_label.items() if recs
        },
    }
    return report


def _print_validate_e2e_rank_flow_stats(rank_flow: dict) -> None:
    """Stage-0 Retrieval Gate summary for validate_e2e — compact table format."""
    ov = rank_flow.get("overall", {})
    if ov.get("n_total", 0) == 0:
        print("\n[6.5] Rank-flow stats skipped (no attacked docs)")
        return

    k   = ov.get("top_k", 10)
    W   = 64
    SEP = "─" * W
    print(f"\n{SEP}")
    print(f"  Stage 0 │ Retrieval Gate  (Gap 1 — 'Guaranteed Retrieval' Assumption)")
    print(f"          Clean@{k} = doc in baseline BM25 top-{k};  "
          f"Survived@{k} = doc in attacked BM25 top-{k}")
    print(SEP)

    # per-label rows (S / C / I only — attack labels)
    by_label = rank_flow.get("by_label", {})
    print(f"  {'Label':<8} {'N':>4}  {'Clean@'+str(k):>10}  "
          f"{'Survived@'+str(k):>12}  {'Survival%':>10}  {'Dropped':>7}")
    print(f"  {'─'*8} {'─'*4}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*7}")
    for lbl in ("S", "C", "I"):
        s = by_label.get(lbl, {})
        n = s.get("n_total", 0)
        if n == 0:
            continue
        n_clean    = s.get("n_clean_top_k", 0)
        n_survived = s.get("n_attacked_retrieval_top_k", 0)
        n_dropped  = s.get("n_drop_clean_to_attacked_retrieval", 0)
        surv_pct   = s.get("attacked_retrieval_top_k_pct", 0.0)
        print(
            f"  {lbl:<8} {n:>4}  "
            f"{n_clean:>4}/{n:<5}  "
            f"{n_survived:>4}/{n:<7}  "
            f"{surv_pct:>9.1f}%  "
            f"{n_dropped:>7}"
        )
    # overall row
    n_ov       = ov["n_total"]
    n_cl_ov    = ov.get("n_clean_top_k", 0)
    n_sv_ov    = ov.get("n_attacked_retrieval_top_k", 0)
    n_dr_ov    = ov.get("n_drop_clean_to_attacked_retrieval", 0)
    surv_ov    = ov.get("attacked_retrieval_top_k_pct", 0.0)
    print(f"  {'─'*8} {'─'*4}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*7}")
    print(
        f"  {'overall':<8} {n_ov:>4}  "
        f"{n_cl_ov:>4}/{n_ov:<5}  "
        f"{n_sv_ov:>4}/{n_ov:<7}  "
        f"{surv_ov:>9.1f}%  "
        f"{n_dr_ov:>7}"
    )

    # position breakdown
    by_pos = rank_flow.get("by_position", {})
    if by_pos:
        print(f"\n  {'Pos':<10}  {'Survived@'+str(k):>12}  {'Reranked@'+str(k):>13}")
        print(f"  {'─'*10}  {'─'*12}  {'─'*13}")
        for pos_key in sorted(by_pos.keys(), key=lambda x: int(x)):
            s = by_pos[pos_key]
            n = s.get("n_total", 0)
            if n == 0:
                continue
            n_sv  = s.get("n_attacked_retrieval_top_k", 0)
            n_rer = s.get("n_attacked_rerank_top_k", 0)
            diff  = "easy" if int(pos_key) == 6 else "hard" if int(pos_key) == 10 else "    "
            print(
                f"  rank-{pos_key} ({diff:<4})  "
                f"{n_sv:>4}/{n} ({s.get('attacked_retrieval_top_k_pct', 0.0):>5.1f}%)  "
                f"{n_rer:>4}/{n} ({s.get('attacked_rerank_top_k_pct', 0.0):>5.1f}%)"
            )


# ---------------------------------------------------------------------------
# Validate mode  (uses pre-computed attacked_docs.jsonl from run_attack.py)
# ---------------------------------------------------------------------------

def run_validate(args: argparse.Namespace) -> None:
    """
    Validation — measure attack effectiveness on pre-computed attacked docs.

    Intended workflow
    -----------------
    1. Run baseline mode to produce a simulation run folder, e.g.
         runs/20260311_093030_query_w_40doc/
       This folder contains ``position_attack_docs.jsonl`` — the attack index
       listing docs at positions 6 / 11 / 20 with their *original* text.

    2. Apply an attack method to that file, e.g.
         python prompt_injection_attack/run_attack.py \\
             --input  runs/.../position_attack_docs.jsonl \\
             --output runs/.../ioa_position_attack_docs.jsonl
       The output file has the same schema but with ``doc_content`` replaced
       by adversarially modified text (e.g. injected ignore-other-advice prompt).

    3. Run this validate mode:
         python run_pipeline.py --mode validate \\
             --baseline_run_dir runs/20260311_093030_query_w_40doc/ \\
             --attacked_docs    runs/20260311_093030_query_w_40doc/ioa_position_attack_docs.jsonl \\
             --reranker Qwen/Qwen3-8B --ranker_type setwise --rerank_top_k 20

    What this mode does
    -------------------
    A. Load the original corpus and the baseline BM25 run (from --baseline_run_dir).
       The BM25 run already contains the attack target docs in its candidate pool —
       no re-retrieval needed.
    B. Patch the corpus pool: replace ``doc_content`` of each attack target with the
       adversarial text from ``--attacked_docs``.
    C. Rerank using the patched pool — the LLM now sees the injected text.
    D. Optional chatbot step — the chatbot sees the adversarial product in its top-k.
    E. Three-tier evaluation: nDCG degradation, rank promotion, Exposure@5 / Success@3.

    ``--rerank_top_k`` must be ≥ the highest attack position (20), otherwise the
    attack targets at rank 20 never enter the reranker window.

    Outputs (saved to ``runs/validate_<timestamp>/``)
    -------------------------------------------------
        ``stage2_validate_<ranker_type>.trec``
        ``ranking_log.jsonl``
        ``chatbot_responses.jsonl``   (if --chatbot)
        ``attack_eval_report.json``
        ``attack_eval_report_verbose.json``
    """
    _name   = getattr(args, "run_name", None)
    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs",
        _name if _name else f"validate_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(run_dir, exist_ok=True)

    SEP = "\u2550" * 64
    print(f"\n{SEP}")
    print(f"  VALIDATION \u2014 Attack Effectiveness")
    print(SEP)
    print(f"  Run dir       : {run_dir}")
    print(f"  Baseline dir  : {args.baseline_run_dir}")
    print(f"  Attacked docs : {args.attacked_docs}")
    if args.reranked_trec:
        print(f"  Reranked TREC : {args.reranked_trec}  (skipping reranking)")
    else:
        print(f"  Rerank top-k  : {args.rerank_top_k}  "
              f"(must be >= highest attack position, i.e. >= 20)")

    # 1. Load corpus
    print(f"\n[1] Loading corpus: {args.jsonl}")
    corpus = ESCICorpus(args.jsonl)
    print(f"    {corpus}")

    # Optional: restrict to the attacked 200 query IDs only
    _filter_qids: set | None = None
    filter_ids_path = getattr(args, "filter_query_ids", None)
    if filter_ids_path:
        with open(filter_ids_path, encoding="utf-8") as _f:
            _filter_qids = {ln.strip() for ln in _f if ln.strip()}
        before = len(corpus.candidate_pool)
        corpus.candidate_pool = {k: v for k, v in corpus.candidate_pool.items() if k in _filter_qids}
        corpus.qrel           = {k: v for k, v in corpus.qrel.items()           if k in _filter_qids}
        print(f"    [filter_query_ids] corpus {before} → {len(corpus.candidate_pool)} queries")

    # 2. Load baseline qrel + Stage-1 retrieval run
    # Auto-detect the retriever type from files actually present in the
    # baseline directory so that validate mode works for both BM25 and
    # dense baselines without requiring --retriever_type on the CLI.
    rtype       = _auto_detect_rtype(
        args.baseline_run_dir, getattr(args, "retriever_type", "bm25")
    )
    qrel_path   = os.path.join(args.baseline_run_dir, "qrel.json")
    stage1_trec = os.path.join(args.baseline_run_dir, f"stage1_{rtype}.trec")
    print(f"\n[2] Baseline artifacts from: {args.baseline_run_dir}")
    with open(qrel_path, encoding="utf-8") as fh:
        qrel = json.load(fh)
    baseline_bm25_run = load_trec(stage1_trec)
    print(f"    {len(baseline_bm25_run)} queries in {rtype.upper()} run")

    # Filter TREC runs and qrel to the attacked queries only
    if _filter_qids:
        qrel              = {k: v for k, v in qrel.items()              if k in _filter_qids}
        baseline_bm25_run = {k: v for k, v in baseline_bm25_run.items() if k in _filter_qids}
        print(f"    [filter_query_ids] TREC + qrel filtered → {len(baseline_bm25_run)} queries")

    # Validate rerank seed run:
    # - Prefer baseline Stage-2 run (stage2_<ranker_type>.trec) so that
    #   attacked docs selected by Stage-2 positions (e.g., rank 6/10) are
    #   injected at the same positions in the validate prompt context.
    # - Fall back to Stage-1 run when Stage-2 file is unavailable.
    rerank_seed_run = baseline_bm25_run
    rerank_seed_label = f"stage1_{rtype}.trec"
    if args.ranker_type:
        stage2_seed_path = os.path.join(
            args.baseline_run_dir, f"stage2_{args.ranker_type}.trec"
        )
        if os.path.exists(stage2_seed_path):
            rerank_seed_run = load_trec(stage2_seed_path)
            rerank_seed_label = os.path.basename(stage2_seed_path)
            # Filter stage-2 seed run as well
            if _filter_qids:
                rerank_seed_run = {k: v for k, v in rerank_seed_run.items() if k in _filter_qids}
            print(
                f"    [Validate seed] Using Stage-2 baseline run for rerank input: "
                f"{rerank_seed_label}"
            )
        else:
            print(
                f"    [Validate seed] {os.path.basename(stage2_seed_path)} not found; "
                f"falling back to {rerank_seed_label}"
            )

    # 3. Load attacked docs + patch corpus pool
    print(f"\n[3] Patching corpus with attacked docs: {args.attacked_docs}")
    attacked_records, patched_pool = _patch_pool(
        corpus.candidate_pool, args.attacked_docs
    )
    n_mod = sum(
        1 for r in attacked_records
        if r.get("doc_content") != r.get("original_doc_content", r.get("doc_content"))
    )
    print(f"    {len(attacked_records)} attack targets  ({n_mod} with modified content)")
    attack_docs_by_label = _records_by_label(attacked_records)

    # 3.5 Debug: print the full reranker input list so we can verify attacked docs are present
    _print_rerank_input_debug(patched_pool, rerank_seed_run, attacked_records, args.rerank_top_k)

    # 4. Rerank  (or load a pre-computed run to skip this expensive step)
    reranker = None
    if args.reranked_trec:
        print(f"\n[4] Loading pre-computed reranked run: {args.reranked_trec}")
        reranked_run = load_trec(args.reranked_trec)
        print(f"    {len(reranked_run)} queries loaded  (reranking skipped)")
        reranked_out = args.reranked_trec   # point back at the source file
    elif args.reranker:
        print(f"\n[4] LLM Reranking  type={args.ranker_type}  "
              f"model={args.reranker}  top_k={args.rerank_top_k}")
        reranker     = _build_reranker(args)
        log_path     = os.path.join(run_dir, "ranking_log.jsonl")
        _maybe_log_listwise_debug(
            args=args,
            reranker_obj=reranker,
            candidate_pool=patched_pool,
            run=rerank_seed_run,
            run_dir=run_dir,
            stage_tag="validate",
        )
        reranked_run = reranker.rerank(
            patched_pool, rerank_seed_run,
            top_k=args.rerank_top_k, log_path=log_path,
        )
        reranked_out = os.path.join(run_dir, f"stage2_validate_{args.ranker_type}.trec")
        write_trec(reranked_out, reranked_run, run_tag=f"validate_{args.ranker_type}")
        print(f"    Reranked run \u2192 {reranked_out}")
        print(f"    Ranking log  \u2192 {log_path}")
    else:
        print("\n[ERROR] Provide either --reranker (to rerank) or --reranked_trec (to load).")
        return

    # 5. Ranking quality
    print(f"\n[5] Ranking Quality  k={args.k}")
    ev = Evaluator(qrel, k_values=args.k)
    rtype = getattr(args, "retriever_type", "bm25")
    baseline_bm25_metrics = ev.evaluate(baseline_bm25_run)
    baseline_seed_metrics = ev.evaluate(rerank_seed_run)
    validate_rerank_metrics = ev.evaluate(reranked_run)
    Evaluator.print_summary(baseline_bm25_metrics, label=f"Baseline {rtype.upper()}")
    if rerank_seed_label != f"stage1_{rtype}.trec":
        Evaluator.print_summary(
            baseline_seed_metrics,
            label=f"Baseline seed ({rerank_seed_label})",
        )
    Evaluator.print_summary(validate_rerank_metrics, label=f"Validate ({args.ranker_type})")
    ranking_metrics = {
        f"baseline_{rtype}": baseline_bm25_metrics,
        "baseline_seed": baseline_seed_metrics,
        f"validate_{args.ranker_type}": validate_rerank_metrics,
    }
    Evaluator.save_metrics(ranking_metrics, os.path.join(run_dir, "ranking_metrics.json"))
    Evaluator.save_metrics_verbose(ranking_metrics, os.path.join(run_dir, "ranking_metrics_verbose.json"))

    # 6. Chatbot (optional)
    # ── Release reranker vLLM engine first (vLLM cannot have two LLM() alive) ──
    if args.chatbot and reranker is not None and getattr(args, "use_vllm", False):
        import gc
        import torch as _torch
        print("\n[Pre-6] Releasing reranker vLLM engine before loading chatbot...")
        del reranker
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
        print("        vLLM engine released.")

    chat_log = None
    if args.chatbot:
        from RAG_attack_pipeline.chatbot import RagChatbot
        chatbot_model = args.chatbot_model or args.reranker
        print(f"\n[6] RAG Chatbot  model={chatbot_model}  top_k={args.chatbot_top_k}")
        bot = RagChatbot(
            model_name_or_path=chatbot_model,
            max_new_tokens=512,
            top_k_context=args.chatbot_top_k,
            use_vllm=getattr(args, "use_vllm", False),
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        )
        chat_log = os.path.join(run_dir, "chatbot_responses.jsonl")
        bot.generate_all(patched_pool, reranked_run,
                         top_k=args.chatbot_top_k, log_path=chat_log)
        print(f"    Saved \u2192 {chat_log}")

    # 7. Three-tier attack evaluation
    print(f"\n[7] Attack Evaluation  "
          f"(nDCG@{max(args.k)} | Rank Promotion | Success@3)")
    print(f"    ℹ  Retriever stage BYPASSED — attack doc is guaranteed in candidate pool.")
    print(f"    ℹ  Tier 2 measures: clean_reranked_rank → attacked_reranked_rank")
    print(f"       reference = {rerank_seed_label}  |  attacked = {os.path.basename(reranked_out)}")
    from RAG_attack_pipeline.attack_eval import (
        AttackEvaluator, save_report, save_report_verbose, save_report_markdown,
    )
    evaluator_atk = AttackEvaluator(
        qrel=qrel,
        bm25_run=rerank_seed_run,
        reranked_run=reranked_run,
        k=max(args.k),
    )
    atk_report = evaluator_atk.evaluate_all(
        attack_docs_by_label,
        chatbot_log_path=chat_log,
        chatbot_top_k=args.chatbot_top_k,
    )
    AttackEvaluator.print_report(
        atk_report,
        k=max(args.k),
        chatbot_top_k=args.chatbot_top_k,
        reference_label=rerank_seed_label,
        attacked_label=os.path.basename(reranked_out),
        verbose=args.verbose,
    )

    role_runs = {
        "ref_retriever": baseline_bm25_run,
        "ref_reranker": rerank_seed_run,
        "attack_retriever": baseline_bm25_run,  # validate: retriever stage unchanged
        "attack_reranker": reranked_run,
    }
    role_labels = {
        "ref_retriever": f"stage1_{rtype}.trec",
        "ref_reranker": rerank_seed_label,
        "attack_retriever": f"stage1_{rtype}.trec",
        "attack_reranker": os.path.basename(reranked_out),
    }
    role_stats = AttackEvaluator.compute_rank_role_stats(
        attack_docs_by_label,
        role_runs,
    )
    AttackEvaluator.print_rank_role_stats(role_stats, role_labels)

    save_report(atk_report,
                os.path.join(run_dir, "attack_eval_report.json"))
    save_report_verbose(atk_report,
                        os.path.join(run_dir, "attack_eval_report_verbose.json"))
    save_report_markdown(
        atk_report,
        os.path.join(run_dir, "attack_eval_report.md"),
        k=max(args.k),
        chatbot_top_k=args.chatbot_top_k,
        reference_label=rerank_seed_label,
        attacked_label=os.path.basename(reranked_out),
        mode="validate",
        extra_info={
            "Ranker": args.ranker_type,
            "Run dir": run_dir,
        },
    )
    with open(os.path.join(run_dir, "attack_eval_rank_roles.json"), "w", encoding="utf-8") as fh:
        json.dump({"role_labels": role_labels, "role_stats": role_stats}, fh, indent=2, ensure_ascii=False)

    # ── Validate Final Summary ────────────────────────────────────────────────
    _W = 64
    print(f"\n{'═' * _W}")
    print(f"  VALIDATE SUMMARY  (reranker-only attack, retriever bypassed)")
    print(f"{'═' * _W}")
    print(f"  Ranker    : {args.ranker_type}  ← upper bound (retriever stage skipped)")
    print(f"  Run dir   : {run_dir}")

    print(f"\n  {'─' * (_W - 2)}")
    print(f"  Ranking Quality")
    for stage_key, stage_metrics in ranking_metrics.items():
        for kv in sorted(args.k):
            ndcg_val   = stage_metrics.get(f"ndcg@{kv}",   {}).get("mean", float("nan"))
            recall_val = stage_metrics.get(f"recall@{kv}", {}).get("mean", float("nan"))
            print(f"    {stage_key:<32} nDCG@{kv:<3} {ndcg_val:.4f}  Recall@{kv:<3} {recall_val:.4f}")

    s3_ov  = atk_report.get("success_at_3", {}).get("overall", {})
    n_atk  = s3_ov.get("n_attack_docs", 0)
    exp5   = s3_ov.get(f"exposure@{args.chatbot_top_k}", 0.0)
    exp3   = s3_ov.get("exposure@3", 0.0)
    suc3   = s3_ov.get("success@3", 0.0)
    s_promo = atk_report.get("rank_promotion", {}).get("S", {})
    norm_p  = s_promo.get("avg_norm_promotion", float("nan"))
    print(f"\n  {'─' * (_W - 2)}")
    print(f"  Attack Effectiveness  (N={n_atk} docs, retriever bypassed)")
    print(f"    Exposure@{args.chatbot_top_k}            {exp5:>6.1f}%")
    print(f"    Exposure@3             {exp3:>6.1f}%")
    print(f"    Success@3              {suc3:>6.1f}%")
    print(f"    S NormPromo            {norm_p:>+7.3f}")

    if "position_stats" in atk_report:
        pos_stats = atk_report["position_stats"]
        print(f"\n  {'─' * (_W - 2)}")
        print(f"  {'By Position':<12}  {'Exp@'+str(args.chatbot_top_k):>7}  {'Exp@3':>7}  {'Suc@3':>7}")
        for pos in sorted(int(k0) for k0 in pos_stats.keys()):
            pdata = pos_stats.get(pos) or pos_stats.get(str(pos), {})
            ps3   = pdata.get("success_at_3", {}).get("overall", {})
            p_e5  = ps3.get(f"exposure@{args.chatbot_top_k}", 0.0)
            p_e3  = ps3.get("exposure@3", 0.0)
            p_s3  = ps3.get("success@3", 0.0)
            diff  = "easy" if pos == 6 else "hard" if pos == 10 else "    "
            print(f"    rank-{pos} ({diff:<4})        {p_e5:>6.1f}%  {p_e3:>6.1f}%  {p_s3:>6.1f}%")
    print(f"{'═' * _W}\n")

    print(f"[VALIDATE] Complete. Results → {run_dir}")


# ---------------------------------------------------------------------------
# E2E Validate mode  (patch corpus → fresh BM25 → rerank → chatbot → eval)
# ---------------------------------------------------------------------------

def run_validate_e2e(args: argparse.Namespace) -> None:
    """
    End-to-end validation — patch the corpus first, then run BM25 from scratch.

    Difference from ``--mode validate``
    ------------------------------------
    ``validate``     : reuses the pre-computed baseline BM25 run so the attack
                       doc is guaranteed to be in the candidate pool.  Tests
                       only the reranker / chatbot stages.

    ``validate_e2e`` : patches the corpus FIRST, then runs BM25 fresh on the
                       modified pool.  This is the realistic end-to-end threat
                       model: the adversarially modified document is already in
                       the index when a user submits a query.

    Step-by-step
    ------------
    1. Load corpus (original ``--jsonl``).
    2. Load qrel — from ``--baseline_run_dir/qrel.json`` if provided, else
       derived directly from the corpus.
    3. Patch corpus pool with attacked docs.
    4. Run BM25 on the **patched** corpus (``stage1_bm25_attacked.trec``).
    5. Print retrieval quality before and after attack.
    6. LLM reranking (requires ``--reranker``).
    7. Optional RAG chatbot.
    8. Three-tier attack evaluation.

    Outputs (saved to ``runs/e2e_<timestamp>/``)
    -------------------------------------------
        ``qrel.json``
        ``stage1_bm25_attacked.trec``      BM25 run on patched corpus
        ``stage2_e2e_<ranker_type>.trec``  reranked run
        ``ranking_log.jsonl``
        ``chatbot_responses.jsonl``        (if --chatbot)
        ``attack_eval_report.json``
        ``attack_eval_report_verbose.json``
    """
    _name   = getattr(args, "run_name", None)
    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs",
        _name if _name else f"e2e_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(run_dir, exist_ok=True)

    SEP = "\u2550" * 64
    print(f"\n{SEP}")
    print(f"  E2E VALIDATION \u2014 Full BM25 + Rerank on Patched Corpus")
    print(SEP)
    print(f"  Run dir       : {run_dir}")
    print(f"  Attacked docs : {args.attacked_docs}")
    print(f"  Retrieve top-k: {args.retrieve_top_k}")
    print(f"  Rerank   top-k: {args.rerank_top_k}")
    if args.baseline_run_dir:
        print(f"  Baseline dir  : {args.baseline_run_dir}  (qrel + clean BM25 reference)")
    print(SEP)

    # 1. Load corpus
    print(f"\n[1] Loading corpus: {args.jsonl}")
    corpus = ESCICorpus(args.jsonl)
    print(f"    {corpus}")

    # Optional: restrict to the attacked 200 query IDs only
    _filter_qids: set | None = None
    filter_ids_path = getattr(args, "filter_query_ids", None)
    if filter_ids_path:
        with open(filter_ids_path, encoding="utf-8") as _f:
            _filter_qids = {ln.strip() for ln in _f if ln.strip()}
        before = len(corpus.candidate_pool)
        corpus.candidate_pool = {k: v for k, v in corpus.candidate_pool.items() if k in _filter_qids}
        corpus.qrel           = {k: v for k, v in corpus.qrel.items()           if k in _filter_qids}
        print(f"    [filter_query_ids] corpus {before} → {len(corpus.candidate_pool)} queries")

    # 2. Load / derive qrel; optionally load clean run for comparison.
    # Auto-detect the retriever type from files in baseline_run_dir so that
    # the correct stage1_*.trec file is loaded even when --retriever_type is
    # not explicitly passed (e.g. dense baseline but no --retriever_type dense).
    rtype = getattr(args, "retriever_type", "bm25")
    if args.baseline_run_dir:
        rtype = _auto_detect_rtype(args.baseline_run_dir, rtype)
        args.retriever_type = rtype  # propagate so _build_retriever and later getattr() calls are consistent
        qrel_path   = os.path.join(args.baseline_run_dir, "qrel.json")
        stage1_trec = os.path.join(args.baseline_run_dir, f"stage1_{rtype}.trec")
        print(f"\n[2] Loading qrel and clean {rtype.upper()} from: {args.baseline_run_dir}")
        with open(qrel_path, encoding="utf-8") as fh:
            qrel = json.load(fh)
        baseline_bm25_run = load_trec(stage1_trec)
        print(f"    Baseline {rtype.upper()} run: {len(baseline_bm25_run)} queries")
        # Filter both to attacked queries only
        if _filter_qids:
            qrel              = {k: v for k, v in qrel.items()              if k in _filter_qids}
            baseline_bm25_run = {k: v for k, v in baseline_bm25_run.items() if k in _filter_qids}
            print(f"    [filter_query_ids] TREC + qrel filtered → {len(baseline_bm25_run)} queries")
    else:
        qrel = corpus.qrel
        baseline_bm25_run = None
        print(f"\n[2] Using corpus-derived qrel ({len(qrel)} queries); no clean {rtype.upper()} reference")
    save_qrel(qrel, os.path.join(run_dir, "qrel.json"))

    # 3. Patch corpus with attacked docs
    print(f"\n[3] Patching corpus with attacked docs: {args.attacked_docs}")
    attacked_records, patched_pool = _patch_pool(
        corpus.candidate_pool, args.attacked_docs
    )
    n_mod = sum(
        1 for r in attacked_records
        if r.get("doc_content") != r.get("original_doc_content", r.get("doc_content"))
    )
    print(f"    {len(attacked_records)} attack targets  ({n_mod} with modified content)")
    attack_docs_by_label = _records_by_label(attacked_records)

    # 4. Fresh Stage-1 retrieval on patched corpus  ← KEY DIFFERENCE from --mode validate
    rtype = getattr(args, "retriever_type", "bm25")
    print(f"\n[4] {rtype.upper()} retrieval on PATCHED corpus  (top_k={args.retrieve_top_k})")
    retriever = _build_retriever(args)
    if rtype == "dense":
        # Incremental patch retrieval: load the original embedding cache,
        # re-encode only the ~4 k modified documents, and patch the embedding
        # matrix in-place.  This avoids a full re-encode of the entire corpus
        # and saves ~5–10 min of GPU time on typical attack datasets.
        # Falls back to a full retrieve() if the original cache is missing.
        print("      [Dense] Incremental patch: only modified docs will be re-encoded")
        attacked_bm25_run = retriever.retrieve_patched(
            patched_pool, corpus.candidate_pool, top_k=args.retrieve_top_k
        )
    else:
        attacked_bm25_run = retriever.retrieve(patched_pool, top_k=args.retrieve_top_k)
    stage1_attacked_out = os.path.join(run_dir, f"stage1_{rtype}_attacked.trec")
    write_trec(stage1_attacked_out, attacked_bm25_run, run_tag=f"{rtype}_attacked")
    print(f"    Attacked {rtype.upper()} run \u2192 {stage1_attacked_out}")
    total = sum(len(v) for v in attacked_bm25_run.values())
    print(f"    {total} total (query, doc) pairs retrieved")

    # Release dense retriever before loading the reranker (avoid OOM).
    if rtype == "dense" and hasattr(retriever, "release"):
        retriever.release()

    # 5. Retrieval quality comparison
    print(f"\n[5] Retrieval Quality  k={args.k}")
    ev = Evaluator(qrel, k_values=args.k)
    ranking_metrics: dict = {}
    rtype = getattr(args, "retriever_type", "bm25")
    if baseline_bm25_run:
        baseline_bm25_metrics = ev.evaluate(baseline_bm25_run)
        Evaluator.print_summary(baseline_bm25_metrics, label=f"Baseline {rtype.upper()} (clean corpus)")
        ranking_metrics[f"baseline_{rtype}"] = baseline_bm25_metrics
    attacked_bm25_metrics = ev.evaluate(attacked_bm25_run)
    Evaluator.print_summary(attacked_bm25_metrics, label=f"Attacked {rtype.upper()} (patched corpus)")
    ranking_metrics[f"attacked_{rtype}"] = attacked_bm25_metrics

    # Check what fraction of attack docs the retriever actually found in patched run
    n_retrieved = sum(
        1 for rec in attacked_records
        if rec["doc_id"] in attacked_bm25_run.get(rec["query_id"], {})
    )
    print(f"    Attack docs retrieved by {rtype.upper()}: {n_retrieved}/{len(attacked_records)}")

    # 5.5 Debug: print the full reranker input list so we can verify attacked docs are present
    _print_rerank_input_debug(patched_pool, attacked_bm25_run, attacked_records, args.rerank_top_k)

    # 6. LLM reranking on attacked BM25 candidates
    if not args.reranker:
        print("\n[WARN] No --reranker supplied; skipping reranking and downstream steps.")
        Evaluator.save_metrics(ranking_metrics, os.path.join(run_dir, "ranking_metrics.json"))
        Evaluator.save_metrics_verbose(ranking_metrics, os.path.join(run_dir, "ranking_metrics_verbose.json"))
        print(f"\n[E2E] Partial results \u2192 {run_dir}")
        return

    print(f"\n[6] LLM Reranking  type={args.ranker_type}  "
          f"model={args.reranker}  top_k={args.rerank_top_k}")
    reranker     = _build_reranker(args)
    log_path     = os.path.join(run_dir, "ranking_log.jsonl")
    _maybe_log_listwise_debug(
        args=args,
        reranker_obj=reranker,
        candidate_pool=patched_pool,
        run=attacked_bm25_run,
        run_dir=run_dir,
        stage_tag="e2e",
    )
    reranked_run = reranker.rerank(
        patched_pool, attacked_bm25_run,
        top_k=args.rerank_top_k, log_path=log_path,
    )
    reranked_out = os.path.join(run_dir, f"stage2_e2e_{args.ranker_type}.trec")
    write_trec(reranked_out, reranked_run, run_tag=f"e2e_{args.ranker_type}")
    print(f"    Reranked run \u2192 {reranked_out}")
    print(f"    Ranking log  \u2192 {log_path}")

    # 6.5 validate_e2e-specific rank-flow statistics for attacked docs
    rank_flow_report = _compute_validate_e2e_rank_flow_stats(
        attacked_records=attacked_records,
        baseline_bm25_run=baseline_bm25_run,
        attacked_bm25_run=attacked_bm25_run,
        reranked_run=reranked_run,
        top_k=10,
    )
    _print_validate_e2e_rank_flow_stats(rank_flow_report)
    rank_flow_path = os.path.join(run_dir, "attack_rank_flow_stats.json")
    with open(rank_flow_path, "w", encoding="utf-8") as fh:
        json.dump(rank_flow_report, fh, indent=2, ensure_ascii=False)
    print(f"    Rank-flow stats \u2192 {rank_flow_path}")

    e2e_rerank_metrics = ev.evaluate(reranked_run)
    Evaluator.print_summary(e2e_rerank_metrics, label=f"Attacked + Reranked ({args.ranker_type})")
    ranking_metrics[f"e2e_{args.ranker_type}"] = e2e_rerank_metrics
    Evaluator.save_metrics(ranking_metrics, os.path.join(run_dir, "ranking_metrics.json"))
    Evaluator.save_metrics_verbose(ranking_metrics, os.path.join(run_dir, "ranking_metrics_verbose.json"))

    # 7. RAG Chatbot (optional)
    # ── Release reranker vLLM engine first (vLLM cannot have two LLM() alive) ──
    if args.chatbot and getattr(args, "use_vllm", False):
        import gc
        import torch as _torch
        print("\n[Pre-7] Releasing reranker vLLM engine before loading chatbot...")
        del reranker
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
        print("        vLLM engine released.")

    chat_log = None
    if args.chatbot:
        from RAG_attack_pipeline.chatbot import RagChatbot
        chatbot_model = args.chatbot_model or args.reranker
        print(f"\n[7] RAG Chatbot  model={chatbot_model}  top_k={args.chatbot_top_k}")
        bot = RagChatbot(
            model_name_or_path=chatbot_model,
            max_new_tokens=512,
            top_k_context=args.chatbot_top_k,
            use_vllm=getattr(args, "use_vllm", False),
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        )
        chat_log = os.path.join(run_dir, "chatbot_responses.jsonl")
        bot.generate_all(patched_pool, reranked_run,
                         top_k=args.chatbot_top_k, log_path=chat_log)
        print(f"    Saved \u2192 {chat_log}")

    # 8. Three-tier attack evaluation
    #
    # Reference selection (mirrors --mode validate for an apples-to-apples
    # comparison): use the CLEAN stage-2 reranker run as both the rank
    # promotion baseline and the nDCG baseline.  This makes both modes
    # report the same quantity:
    #   rank_promotion = clean_stage2_rank → attacked_stage2_rank
    #   ndcg_delta     = attacked_stage2_ndcg − clean_stage2_ndcg
    # so e2e cleanly isolates the attack's pipeline-level impact instead of
    # mixing it with the reranker's natural gain over BM25.
    #
    # The orthogonal "did the doc survive the retrieval gate?" view is still
    # available via the role_stats block below (attack_retriever vs attack_reranker).
    # validate_e2e MUST use the clean baseline stage-2 as the eval reference so
    # that nDCG / rank-promotion deltas are directly comparable with --mode validate.
    # The previous "soft fallback" (use attacked stage-1 if no baseline available)
    # silently produced wrong references and inflated deltas — see the
    # fix_e2e_ndcg_reference.py post-hoc patcher and statistics/recompute_attack_eval.py.
    # We now hard-fail if the clean stage-2 baseline is missing.
    if not args.baseline_run_dir:
        raise SystemExit(
            "[run_pipeline][validate_e2e] --baseline_run_dir is REQUIRED. "
            "It must point at a baseline run directory containing "
            f"stage2_{args.ranker_type}.trec (clean reranker output) so the "
            "attack-eval reference matches --mode validate semantics. Pass "
            "the canonical 200-baseline (e.g. .../bm25_baseline_200_Qwen3-8B_listwise)."
        )
    stage2_ref_path = os.path.join(args.baseline_run_dir, f"stage2_{args.ranker_type}.trec")
    if not os.path.exists(stage2_ref_path):
        raise SystemExit(
            f"[run_pipeline][validate_e2e] Clean stage-2 reference not found at "
            f"{stage2_ref_path}. The baseline_run_dir must contain "
            f"stage2_{args.ranker_type}.trec produced by the clean baseline run."
        )
    ref_reranker_run = load_trec(stage2_ref_path)
    ref_reranker_label = os.path.basename(stage2_ref_path)
    print(f"    ✓  Using clean stage-2 reference: {ref_reranker_label}")

    eval_ref_run   = ref_reranker_run
    eval_ref_label = ref_reranker_label
    tier2_desc     = "clean_stage2_rank → attacked_stage2_rank  (matches --mode validate)"

    print(f"\n[8] Attack Evaluation  "
          f"(nDCG@{max(args.k)} | Rank Promotion | Success@3)")
    print(f"    ℹ  Full pipeline — retriever ran on patched corpus.")
    print(f"    ℹ  Tier 1 (nDCG) reference = {eval_ref_label}")
    print(f"    ℹ  Tier 2 (Rank Promotion) measures: {tier2_desc}")
    print(f"       reference = {eval_ref_label}  |  attacked = {os.path.basename(reranked_out)}")
    from RAG_attack_pipeline.attack_eval import (
        AttackEvaluator, save_report, save_report_verbose, save_report_markdown,
    )
    evaluator_atk = AttackEvaluator(
        qrel=qrel,
        bm25_run=eval_ref_run,            # rank-promotion reference (clean stage2 when available)
        reranked_run=reranked_run,
        k=max(args.k),
        ndcg_ref_run=eval_ref_run,        # nDCG reference (matches the comment in attack_eval.py:102-105)
    )
    atk_report = evaluator_atk.evaluate_all(
        attack_docs_by_label,
        chatbot_log_path=chat_log,
        chatbot_top_k=args.chatbot_top_k,
    )
    AttackEvaluator.print_report(
        atk_report,
        k=max(args.k),
        chatbot_top_k=args.chatbot_top_k,
        reference_label=eval_ref_label,
        attacked_label=os.path.basename(reranked_out),
        verbose=args.verbose,
    )

    # Role-aware rank stats for clearer semantics:
    # - ref_retriever   : clean baseline stage1 (if available)
    # - ref_reranker    : clean baseline stage2 (if available), else ref_retriever
    # - attack_retriever: attacked stage1
    # - attack_reranker : attacked stage2
    ref_retriever_run = baseline_bm25_run if baseline_bm25_run is not None else attacked_bm25_run
    ref_retriever_label = (
        f"stage1_{rtype}.trec"
        if baseline_bm25_run is not None else os.path.basename(stage1_attacked_out)
    )

    # ref_reranker_run is guaranteed non-None above (we hard-fail otherwise).

    role_runs = {
        "ref_retriever": ref_retriever_run,
        "ref_reranker": ref_reranker_run,
        "attack_retriever": attacked_bm25_run,
        "attack_reranker": reranked_run,
    }
    role_labels = {
        "ref_retriever": ref_retriever_label,
        "ref_reranker": ref_reranker_label,
        "attack_retriever": os.path.basename(stage1_attacked_out),
        "attack_reranker": os.path.basename(reranked_out),
    }
    role_stats = AttackEvaluator.compute_rank_role_stats(
        attack_docs_by_label,
        role_runs,
    )
    AttackEvaluator.print_rank_role_stats(role_stats, role_labels)

    save_report(atk_report,
                os.path.join(run_dir, "attack_eval_report.json"))
    save_report_verbose(atk_report,
                        os.path.join(run_dir, "attack_eval_report_verbose.json"))
    save_report_markdown(
        atk_report,
        os.path.join(run_dir, "attack_eval_report.md"),
        k=max(args.k),
        chatbot_top_k=args.chatbot_top_k,
        reference_label=eval_ref_label,
        attacked_label=os.path.basename(reranked_out),
        mode="validate_e2e",
        extra_info={
            "Retriever": getattr(args, "retriever_type", "bm25").upper(),
            "Ranker": args.ranker_type,
            "Run dir": run_dir,
        },
    )
    with open(os.path.join(run_dir, "attack_eval_rank_roles.json"), "w", encoding="utf-8") as fh:
        json.dump({"role_labels": role_labels, "role_stats": role_stats}, fh, indent=2, ensure_ascii=False)

    # ── E2E Final Summary ─────────────────────────────────────────────────────
    _W = 64
    print(f"\n{'═' * _W}")
    print(f"  E2E SUMMARY  (full pipeline: {rtype.upper()} → Reranker → Chatbot)")
    print(f"{'═' * _W}")
    print(f"  Retriever : {rtype.upper()}")
    print(f"  Ranker    : {args.ranker_type}")
    print(f"  Run dir   : {run_dir}")

    # Stage 0: Retrieval Gate
    ov_rf   = rank_flow_report.get("overall", {})
    k_rf    = ov_rf.get("top_k", 10)
    n_rf    = ov_rf.get("n_total", 0)
    n_cl    = ov_rf.get("n_clean_top_k", 0)
    n_sv    = ov_rf.get("n_attacked_retrieval_top_k", 0)
    surv_pct = ov_rf.get("attacked_retrieval_top_k_pct", 0.0)
    n_dr    = ov_rf.get("n_drop_clean_to_attacked_retrieval", 0)
    print(f"\n  {'─' * (_W - 2)}")
    print(f"  Retrieval Gate  (Gap 1)   clean top-{k_rf} → survived after attack injection")
    print(f"  {'Label':<8} {'N':>4}  {'Clean@'+str(k_rf):>10}  {'Survived@'+str(k_rf):>12}  {'Survival%':>10}")
    by_lbl_rf = rank_flow_report.get("by_label", {})
    for lbl in ("S", "C", "I"):
        s = by_lbl_rf.get(lbl, {})
        n_l = s.get("n_total", 0)
        if n_l == 0:
            continue
        n_cl_l  = s.get("n_clean_top_k", 0)
        n_sv_l  = s.get("n_attacked_retrieval_top_k", 0)
        sv_l    = s.get("attacked_retrieval_top_k_pct", 0.0)
        print(f"    {lbl:<8} {n_l:>4}  {n_cl_l:>4}/{n_l:<5}  {n_sv_l:>4}/{n_l:<7}  {sv_l:>9.1f}%")
    print(f"    {'overall':<8} {n_rf:>4}  {n_cl:>4}/{n_rf:<5}  {n_sv:>4}/{n_rf:<7}  {surv_pct:>9.1f}%"
          f"  ({n_dr} dropped)")

    # Ranking quality
    print(f"\n  {'─' * (_W - 2)}")
    print(f"  Ranking Quality")
    for stage_key, stage_metrics in ranking_metrics.items():
        for kv in sorted(args.k):
            ndcg_val   = stage_metrics.get(f"ndcg@{kv}",   {}).get("mean", float("nan"))
            recall_val = stage_metrics.get(f"recall@{kv}", {}).get("mean", float("nan"))
            print(f"    {stage_key:<32} nDCG@{kv:<3} {ndcg_val:.4f}  Recall@{kv:<3} {recall_val:.4f}")

    # Attack effectiveness
    s3_ov   = atk_report.get("success_at_3", {}).get("overall", {})
    n_atk   = s3_ov.get("n_attack_docs", 0)
    exp5    = s3_ov.get(f"exposure@{args.chatbot_top_k}", 0.0)
    exp3    = s3_ov.get("exposure@3", 0.0)
    suc3    = s3_ov.get("success@3", 0.0)
    s_promo = atk_report.get("rank_promotion", {}).get("S", {})
    norm_p  = s_promo.get("avg_norm_promotion", float("nan"))
    print(f"\n  {'─' * (_W - 2)}")
    print(f"  Attack Effectiveness  (N={n_atk} docs, full pipeline)")
    print(f"    Retrieval Survival     {surv_pct:>6.1f}%  ({n_sv}/{n_rf} survived BM25 top-{k_rf})")
    print(f"    Exposure@{args.chatbot_top_k}            {exp5:>6.1f}%")
    print(f"    Exposure@3             {exp3:>6.1f}%")
    print(f"    Success@3              {suc3:>6.1f}%")
    print(f"    S NormPromo            {norm_p:>+7.3f}")

    if "position_stats" in atk_report:
        pos_stats = atk_report["position_stats"]
        print(f"\n  {'─' * (_W - 2)}")
        print(f"  {'By Position':<12}  {'Survival%':>9}  {'Exp@'+str(args.chatbot_top_k):>7}  "
              f"{'Exp@3':>7}  {'Suc@3':>7}")
        for pos in sorted(int(k0) for k0 in pos_stats.keys()):
            pdata  = pos_stats.get(pos) or pos_stats.get(str(pos), {})
            ps3    = pdata.get("success_at_3", {}).get("overall", {})
            p_e5   = ps3.get(f"exposure@{args.chatbot_top_k}", 0.0)
            p_e3   = ps3.get("exposure@3", 0.0)
            p_s3   = ps3.get("success@3", 0.0)
            diff   = "easy" if pos == 6 else "hard" if pos == 10 else "    "
            # Retrieval survival for this position
            pos_rf = rank_flow_report.get("by_position", {}).get(str(pos), {})
            sv_pos = pos_rf.get("attacked_retrieval_top_k_pct", float("nan"))
            print(f"    rank-{pos} ({diff:<4})  {sv_pos:>8.1f}%  {p_e5:>6.1f}%  "
                  f"{p_e3:>6.1f}%  {p_s3:>6.1f}%")
    print(f"{'═' * _W}\n")

    print(f"[E2E] Complete. Results → {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
    description="ESCI retrieval + LLM reranking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Experiment mode ----
    p.add_argument(
        "--run_name", default=None,
        help=(
            "Override the auto-generated timestamp in the run directory name.  "
            "baseline → runs/<run_name>/  "
            "validate → runs/validate_<run_name>/  "
            "validate_e2e → runs/e2e_<run_name>/  "
            "Useful when submitting many jobs in parallel to avoid timestamp collisions."
        ),
    )
    p.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "validate", "validate_e2e"],
        help=(
            "baseline (default): full simulation — BM25 \u2192 reranking \u2192 chatbot \u2192 "
            "attack dataset export \u2192 evaluation.  "
            "validate: attack effectiveness using pre-computed BM25 candidates from "
            "--baseline_run_dir; only reranker + chatbot stages see the attacked docs.  "
            "validate_e2e: full end-to-end threat model — patches the corpus first, "
            "then re-runs BM25 from scratch, reranks, and evaluates; "
            "tests whether the attack also affects BM25 retrieval."
        ),
    )
    p.add_argument(
        "--baseline_run_dir",
        default=None,
        help=(
            "Path to a baseline run directory (e.g. runs/20260305_120000/). "
            "Required for --mode validate. "
            "Must contain: stage1_bm25.trec, qrel.json."
        ),
    )

    # ---- Validation input (--mode validate) ----
    p.add_argument(
        "--attacked_docs",
        default=None,
        help=(
            "Path to attacked position_attack_docs.jsonl produced by "
            "prompt_injection_attack/run_attack.py. "
            "Required for --mode validate. "
            "Contains the same schema as position_attack_docs.jsonl with "
            "doc_content replaced by adversarial text."
        ),
    )

    # ---- Data ----
    p.add_argument("--jsonl", required=True,
                   help="Path to ESCI Task-1 JSONL judgement file")
    p.add_argument("--filter_query_ids", default=None,
                   help="Path to a text file with one query_id per line. "
                        "When set, the corpus and qrel are restricted to only those query IDs. "
                        "Use this to evaluate only the 200 sampled/attacked queries instead of "
                        "the full dataset.")
    p.add_argument("--save_qrel", default=None,
                   help="Save the derived qrel to this JSON path (optional)")

    # ---- Retrieval ----
    p.add_argument("--trec", default=None,
                   help="Load an existing TREC run instead of running Stage-1 retrieval")
    p.add_argument("--retriever_type", default="bm25",
                   choices=["bm25", "dense"],
                   help="Stage-1 retriever: 'bm25' (default) or 'dense' (BAAI/bge)")
    p.add_argument("--retriever_model", default="BAAI/bge-large-en-v1.5",
                   help="HuggingFace model ID for dense retrieval "
                        "(default: BAAI/bge-large-en-v1.5)")
    p.add_argument("--retriever_batch_size", type=int, default=128,
                   help="Batch size for dense retrieval encoding (default 32)")
    p.add_argument("--retrieve_top_k", type=int, default=40,
                   help="Number of candidate documents to retrieve per query in Stage 1")
    p.add_argument("--out", default=None,
                   help="Override output path for the Stage-1 TREC run "
                        "(default: <run_dir>/stage1_{retriever_type}.trec)")

    # ---- Reranking ----
    p.add_argument("--reranker", default=None,
                   help="HuggingFace model ID or path for the reranker")
    p.add_argument("--ranker_type", default="listwise",
                   choices=["listwise"],
                   help="Reranker strategy")
    p.add_argument("--rerank_top_k", type=int, default=10,
                   help="Number of Stage-1 documents to pass to the reranker per query")
    p.add_argument("--reranked_trec", default=None,
                   help="Load an existing Stage-2 reranked TREC run and skip reranking "
                        "(e.g. runs/20260310_174611/stage2_setwise.trec). "
                        "Mutually exclusive with --reranker.")
    p.add_argument("--reranked_out", default=None,
                   help="Override output path for the Stage-2 reranked TREC run "
                        "(default: <run_dir>/stage2_<ranker_type>.trec)")
    # ---- llmrankers listwise options ----
    p.add_argument("--window_size", type=int, default=10,
                   help="Sliding-window size (listwise)")
    p.add_argument("--step_size",   type=int, default=5,
                   help="Sliding-window step size (listwise)")
    p.add_argument("--num_repeat",  type=int, default=1,
                   help="Number of sliding-window passes (listwise)")
    p.add_argument("--scoring", default="generation",
                   choices=["generation", "likelihood"],
                   help="Scoring method for listwise/setwise: 'generation' ranks by "
                        "generated token order, 'likelihood' uses output log-likelihood")
    p.add_argument("--listwise_debug", action="store_true",
                   help="Listwise-only debug: dump actual prompt/context inputs before reranking")
    p.add_argument("--listwise_debug_queries", type=int, default=2,
                   help="Listwise debug: number of queries to dump")
    p.add_argument("--listwise_debug_docs", type=int, default=None,
                   help="Listwise debug: number of top docs per dumped query (default: window_size)")
    p.add_argument("--listwise_debug_print_chars", type=int, default=1200,
                   help="Listwise debug: max prompt characters printed to stdout")

    # ---- Evaluation ----
    p.add_argument("--k", nargs="+", type=int, default=[5, 10],
                   help="NDCG/Recall cut-off values")

    # ---- RAG Chatbot (Stage 3) ----
    p.add_argument("--chatbot", action="store_true",
                   help="Enable RAG chatbot step: generates best-3 product recommendations "
                        "from top-k reranked results")
    p.add_argument("--chatbot_model", default=None,
                   help="HuggingFace model ID for the chatbot. "
                        "Defaults to --reranker if not set.")
    p.add_argument("--chatbot_top_k", type=int, default=5,
                   help="Number of top-reranked products to feed into the chatbot prompt")

    # ---- Attack Dataset Export ----
    p.add_argument("--attack_dataset", action="store_true",
                   help="Export the reranked results as a structured attack dataset JSONL")
    p.add_argument("--attack_top_k", type=int, default=None,
                   help="Number of top-ranked docs per query to include in the attack "
                        "dataset (default: all ranked docs)")
    p.add_argument("--attack_out_dir", default=None,
                   help="Output directory for the attack dataset "
                        "(default: <project_root>/Attack_dataset/)")

    # ---- Attack Evaluation (Tier 1–3) ----
    p.add_argument("--attack_eval", action="store_true",
                   help="Run three-tier attack evaluation: nDCG@k, rank promotion, "
                        "and Success@3 (automatically builds attack dataset; "
                        "Tier-3 requires --chatbot to also be set)")

    # ---- Output verbosity ----
    p.add_argument(
        "--verbose", action="store_true",
        help=(
            "Enable verbose console output: show retrieval/rerank debug stats "
            "and full per-label position analysis in the attack evaluation report. "
            "By default only essential metrics are shown."
        ),
    )

    # ---- Acceleration ----
    p.add_argument(
        "--use_vllm", action="store_true",
        help=(
            "Use vLLM for LLM inference instead of HuggingFace Transformers. "
            "For 'pointwise': batches ALL (query, doc) pairs across ALL queries in "
            "a single vLLM call (5–10× speed-up). "
            "For 'setwise'/'listwise': each comparison call is routed through vLLM "
            "(enables tensor parallelism). "
            "Chatbot generation is also fully batched via vLLM."
        ),
    )
    p.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help=(
            "Number of GPUs for vLLM tensor parallelism (only used with --use_vllm). "
            "Set to 2 to split the model across both A6000s. "
            "NOTE: if both --reranker and --chatbot are active, "
            "both vLLM instances must fit in GPU memory simultaneously. "
            "If memory is tight use tensor_parallel_size=1 (default)."
        ),
    )

    return p


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.mode == "baseline":
        run_pipeline(args)
    elif args.mode == "validate":
        if not args.baseline_run_dir:
            parser.error("--baseline_run_dir is required for --mode validate")
        if not args.attacked_docs:
            parser.error("--attacked_docs is required for --mode validate")
        run_validate(args)
    elif args.mode == "validate_e2e":
        if not args.attacked_docs:
            parser.error("--attacked_docs is required for --mode validate_e2e")
        run_validate_e2e(args)
    else:
        parser.error(f"Unknown --mode: {args.mode}")

