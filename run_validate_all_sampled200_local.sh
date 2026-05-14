#!/usr/bin/env bash
# Reproduce Stage 3 validation results for all attack configurations.
#
# Attack data: https://huggingface.co/datasets/Euanyu/geo-injection-rag-attack-data
# Download with:
#   huggingface-cli download Euanyu/geo-injection-rag-attack-data \
#       --repo-type dataset --local-dir ./attack_data
#
# Usage:
#   bash run_validate_all_sampled200_local.sh [ATTACK_DATA_DIR]
#
# ATTACK_DATA_DIR defaults to ./attack_data if not provided.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ATTACK_DATA_DIR="${1:-./attack_data}"
RERANKER="${RERANKER:-Qwen/Qwen3-8B}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
JSONL="RAG_attack_pipeline/dataset/task_1_test_filtered_k_40.jsonl"
BM25_BASELINE="RAG_attack_pipeline/runs/bm25_baseline_200_Qwen3-8B_listwise"
DENSE_BASELINE="RAG_attack_pipeline/runs/dense_baseline_200_Qwen3-8B_listwise"

METHODS=(core_reasoning core_review ioa raf srp sts tap)
POSITIONS=(pos6 pos10)
# ──────────────────────────────────────────────────────────────────────────────

if [ ! -d "$ATTACK_DATA_DIR" ]; then
    echo "Error: attack data directory '$ATTACK_DATA_DIR' not found."
    echo ""
    echo "Download it with:"
    echo "  huggingface-cli download Euanyu/geo-injection-rag-attack-data \\"
    echo "      --repo-type dataset --local-dir ./attack_data"
    exit 1
fi

run_pair() {
    local retriever="$1"
    local baseline="$2"
    local attacked_docs="$3"
    local retriever_type="$4"

    if [ ! -f "$attacked_docs" ]; then
        echo "  [skip] $(basename $attacked_docs) not found"
        return
    fi

    local label="$retriever | $(basename $attacked_docs)"

    echo ""
    echo ">>> validate     | $label"
    PYTHONPATH=. python RAG_attack_pipeline/run_pipeline.py \
        --mode validate \
        --jsonl "$JSONL" \
        --baseline_run_dir "$baseline" \
        --attacked_docs "$attacked_docs" \
        --reranker "$RERANKER" \
        --ranker_type listwise \
        --window_size 10 --step_size 1 \
        --rerank_top_k 10 \
        --chatbot --chatbot_top_k 5 \
        --attack_eval \
        --use_vllm \
        --tensor_parallel_size "$TENSOR_PARALLEL"

    echo ""
    echo ">>> validate_e2e | $label"
    PYTHONPATH=. python RAG_attack_pipeline/run_pipeline.py \
        --mode validate_e2e \
        --jsonl "$JSONL" \
        --baseline_run_dir "$baseline" \
        --attacked_docs "$attacked_docs" \
        --retriever_type "$retriever_type" \
        --retrieve_top_k 40 \
        --reranker "$RERANKER" \
        --ranker_type listwise \
        --window_size 10 --step_size 1 \
        --rerank_top_k 10 \
        --chatbot --chatbot_top_k 5 \
        --attack_eval \
        --use_vllm \
        --tensor_parallel_size "$TENSOR_PARALLEL"
}

for METHOD in "${METHODS[@]}"; do
    for POS in "${POSITIONS[@]}"; do
        run_pair bm25  "$BM25_BASELINE"  "$ATTACK_DATA_DIR/bm25/${METHOD}_${POS}_attacked.jsonl"  bm25
        run_pair dense "$DENSE_BASELINE" "$ATTACK_DATA_DIR/dense/${METHOD}_${POS}_attacked.jsonl" dense
    done
done

echo ""
echo "All validation runs complete."
