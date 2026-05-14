# GEO Injection RAG Survival — End-to-End Adversarial Evaluation

This is the official repo for the paper "Can It Reach the Generator? Investigating the Survival of GEO Prompt-Injection Attacks in Realistic RAG Settings".

---

## Table of Contents

1. [Three Research Stages](#1-three-research-stages)
2. [Dataset](#2-dataset)
3. [Environment Setup](#3-environment-setup)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [Directory Structure](#5-directory-structure)

---

## 1. Three Research Stages

![Pipeline Overview](images/surge_pipeline.svg)

The pipeline is structured around three sequential stages. Each stage depends on the outputs of the previous one.

```
Stage 1: Simulation  ──►  Stage 2: Attack  ──►  Stage 3: Validation
   run_pipeline.py            (attack data)          run_pipeline.py
   --mode baseline                                   --mode validate
                                                     --mode validate_e2e
```

### Stage 1 — Simulation (`run_pipeline.py --mode baseline`)

Runs the clean RAG pipeline over unmodified ESCI data: BM25 retrieval → listwise reranking → RAG chatbot. Establishes a ranking baseline and identifies one attack-target document per query.

| Output file | Description |
|-------------|-------------|
| `qrel.json` | Relevance judgements derived from ESCI labels |
| `stage1_bm25.trec` | BM25 ranked lists |
| `stage2_listwise.trec` | Reranked lists |
| `ranking_metrics.json` | nDCG / Recall per stage |
| `chatbot_responses.jsonl` | RAG chatbot recommendations |
| `position_attack_docs.jsonl` | One attack-target document per query |
| `attack_eval_report.json` | Pre-attack baseline evaluation |

All outputs are saved to `RAG_attack_pipeline/runs/<timestamp>/`.

**Attack target extraction (`--attack_dataset`):** `attack_dataset.py` serializes the full reranked run as a flat JSONL, then `extract_attack_docs.py` selects one attack-target document per query using priority I → C → S (most irrelevant first; within a label, the document ranked lowest by the reranker). This produces `position_attack_docs.jsonl` — the handoff file for Stage 2.

**Attack evaluation (`--attack_eval`):** `attack_eval.py` measures attack effectiveness across three tiers: nDCG@k delta between BM25 and reranked runs (Tier 1), rank promotion of the attack-target document (Tier 2), and Success@3 — whether the attack target appears in the chatbot's best-3 recommendation (Tier 3, requires `--chatbot`). In Stage 1 this establishes the pre-attack baseline; the same evaluation re-runs in Stage 3 to measure post-attack effectiveness.

---

### Stage 2 — Attack

This stage is independent of `run_pipeline.py`. It takes `position_attack_docs.jsonl` as input and produces `attacked_docs.jsonl`.

Attack data is released on HuggingFace: [Euanyu/geo-injection-rag-attack-data](https://huggingface.co/datasets/Euanyu/geo-injection-rag-attack-data). Download it with:

```bash
huggingface-cli download Euanyu/geo-injection-rag-attack-data \
    --repo-type dataset --local-dir ./attack_data
```

Files are organised as `<retriever>/<method>_<position>_attacked.jsonl`:

| Field | Values |
|-------|--------|
| retriever | `bm25`, `dense` |
| method | `ioa`, `raf`, `srp`, `sts`, `tap`, `core_reasoning`, `core_review` |
| position | `pos6`, `pos10` |

**File schema** — `attacked_docs.jsonl`:
```json
{
  "query_id": "42", "query": "waterproof hiking boots",
  "doc_id": "B099", "doc_rank": 18,
  "doc_title": "...", "esci_label": "I", "qrel_score": 0.0,
  "attack_method": "ioa",
  "original_doc_content": "<original text>",
  "doc_content": "<adversarially modified text>"
}
```

---

### Stage 3 — Validation (`run_pipeline.py --mode validate` / `validate_e2e`)

Re-runs the pipeline with attacked documents and measures attack effectiveness via three-tier evaluation (nDCG delta, rank promotion, Success@3).

Two sub-modes:

- `--mode validate` — Frozen-context (FC): retriever candidates are reused; only document text is patched. Tests whether the reranker promotes the attacked document.
- `--mode validate_e2e` — full end-to-end (E2E): corpus is patched before retriever, modelling the realistic threat where a product listing has been modified in the index.

| Output file | Description |
|-------------|-------------|
| `stage2_validate_listwise.trec` | Reranked run on attacked docs (`validate` mode) |
| `stage2_e2e_listwise.trec` | Reranked run on attacked docs (`validate_e2e` mode) |
| `stage1_{retriever}_attacked.trec` | Retrieval run on patched corpus (`validate_e2e` only) |
| `attack_eval_report.json` | Three-tier attack evaluation (same tiers as Stage 1 baseline) |

---

## 2. Dataset

The pipeline uses the **Amazon ESCI dataset** (Task 1 — product ranking), available at [amazon-science/esci-data](https://github.com/amazon-science/esci-data).

A sample filtered to queries with at least 40 annotated candidates is included at `RAG_attack_pipeline/dataset/task_1_test_filtered_k_40.jsonl`.

Document text is assembled as `product_title + "\n" + product_bullet_point`.

---

## 3. Environment Setup

```bash
conda env create -f environment.yml
conda activate geo-injection-rag
pip install -e RAG_attack_pipeline/llm-rankers/
```

---

## 4. Running the Pipeline

All commands are run from the **repo root**.

### Stage 1 — Simulation

```bash
PYTHONPATH=. python RAG_attack_pipeline/run_pipeline.py \
    --mode baseline \
    --jsonl RAG_attack_pipeline/dataset/task_1_test_filtered_k_40.jsonl \
    --retriever_type bm25 \
    --retrieve_top_k 40 \
    --reranker Qwen/Qwen3-8B \
    --ranker_type listwise \
    --window_size 10 \
    --step_size 1 \
    --rerank_top_k 10 \
    --chatbot \
    --chatbot_top_k 5 \
    --attack_dataset \
    --attack_eval \
    --use_vllm \
    --tensor_parallel_size 1
```

### Stage 2 — Attack

Download the attack data from HuggingFace (see [Stage 2 — Attack](#stage-2--attack) above for the download command and file layout), then proceed to Stage 3.

### Stage 3 — Validation

Pre-computed baseline runs are provided in `RAG_attack_pipeline/runs/`. To reproduce all paper results at once, use the provided script (after downloading attack data from HuggingFace):

```bash
bash run_validate_all_sampled200_local.sh ./attack_data
```

Or set `--baseline_run_dir` manually to run a single configuration.

#### Frozen-context (`--mode validate`)

```bash
PYTHONPATH=. python RAG_attack_pipeline/run_pipeline.py \
    --mode validate \
    --jsonl RAG_attack_pipeline/dataset/task_1_test_filtered_k_40.jsonl \
    --baseline_run_dir <BASELINE_RUN_DIR> \
    --attacked_docs    <BASELINE_RUN_DIR>/attacked_docs.jsonl \
    --reranker Qwen/Qwen3-8B \
    --ranker_type listwise \
    --window_size 10 \
    --step_size 1 \
    --rerank_top_k 10 \
    --chatbot \
    --chatbot_top_k 5 \
    --attack_eval \
    --use_vllm \
    --tensor_parallel_size 1
```

#### End-to-end (`--mode validate_e2e`)

```bash
PYTHONPATH=. python RAG_attack_pipeline/run_pipeline.py \
    --mode validate_e2e \
    --jsonl RAG_attack_pipeline/dataset/task_1_test_filtered_k_40.jsonl \
    --attacked_docs    <BASELINE_RUN_DIR>/attacked_docs.jsonl \
    --baseline_run_dir <BASELINE_RUN_DIR> \
    --retriever_type bm25 \
    --retrieve_top_k 40 \
    --reranker Qwen/Qwen3-8B \
    --ranker_type listwise \
    --window_size 10 \
    --step_size 1 \
    --rerank_top_k 10 \
    --chatbot \
    --chatbot_top_k 5 \
    --attack_eval \
    --use_vllm \
    --tensor_parallel_size 1
```

For a full parameter reference, see [`RAG_attack_pipeline/README.md`](RAG_attack_pipeline/README.md).

---

## 5. Directory Structure

```
geo_injection_rag_survival/
├── README.md
├── environment.yml
├── run_validate_all_sampled200_local.sh  # Reproduces all Stage 3 results
├── images/
│   └── surge_pipeline.svg
└── RAG_attack_pipeline/
    ├── __init__.py
    ├── run_pipeline.py               # Main orchestrator
    ├── corpus.py                     # Data loading
    ├── retrieval.py                  # BM25 retrieval
    ├── dense_retriever.py            # Dense retrieval
    ├── reranker.py                   # LLM reranker adapter
    ├── evaluate.py                   # nDCG / Recall metrics
    ├── utils.py                      # TREC / qrel I/O
    ├── chatbot.py                    # RAG chatbot
    ├── attack_dataset.py             # Ranked JSONL exporter
    ├── extract_attack_docs.py        # Attack target extraction
    ├── attack_eval.py                # Three-tier attack evaluation
    ├── README.md                     # Parameter reference
    ├── dataset/
    │   └── task_1_test_filtered_k_40.jsonl
    ├── runs/
    │   ├── bm25_baseline_200_Qwen3-8B_listwise/   # Pre-computed BM25 baseline
    │   └── dense_baseline_200_Qwen3-8B_listwise/  # Pre-computed dense baseline
    └── llm-rankers/
        ├── setup.py
        └── llmrankers/
            ├── rankers.py
            └── listwise.py
```
