# RAG Attack Pipeline — Parameter Reference

## `run_pipeline.py` Parameters

### Shared across all modes

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--jsonl` | *(required)* | Path to the ESCI Task-1 JSONL file |
| `--mode` | `baseline` | Pipeline mode: `baseline`, `validate`, `validate_e2e` |
| `--k` | `5 10` | nDCG / Recall cutoff values (space-separated list) |

### Stage 1 — Retrieval

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--retriever_type` | `bm25` | `bm25` or `dense` |
| `--retrieve_top_k` | `40` | Number of candidates per query |
| `--retriever_batch_size` | `128` | Batch size for dense embedding (dense only) |
| `--trec` | None | Load an existing TREC run instead of running retrieval |

### Stage 2 — LLM Reranking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--reranker` | None | HuggingFace model ID or local path (enables Stage 2) |
| `--ranker_type` | `listwise` | `listwise` ranking schema |
| `--rerank_top_k` | `10` | Candidates passed to the reranker per query |
| `--use_vllm` | off | Enable vLLM inference |
| `--tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |

#### Listwise-specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window_size` | `10` | Sliding-window size |
| `--step_size` | `5` | Sliding-window step |
| `--num_repeat` | `1` | Number of sliding-window passes |

### RAG Chatbot

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chatbot` | *(flag)* | Enable the RAG chatbot step |
| `--chatbot_model` | *(same as `--reranker`)* | HuggingFace model for the chatbot |
| `--chatbot_top_k` | `5` | Top-k reranked docs fed to the chatbot |

### Attack Flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--attack_dataset` | *(flag)* | Export reranked results as a flat JSONL file |
| `--attack_eval` | *(flag)* | Run three-tier attack evaluation |

### Stage 3 — Validation

| Parameter | Required? | Description |
|-----------|-----------|-------------|
| `--attacked_docs` | ✅ | Path to `attacked_docs.jsonl` from the attack stage |
| `--baseline_run_dir` | ✅ for `validate` | Baseline run directory (must contain `stage1_bm25.trec` and `qrel.json`) |
| `--rerank_top_k` | `10` | The number of context the LLM reranker would see and thus rerank|
| `--retrieve_top_k` | `40` | (`validate_e2e` only) BM25 top-k for the fresh run on the patched corpus |
