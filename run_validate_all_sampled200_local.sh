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

RETRIEVERS=(bm25 dense)
METHODS=(core_reasoning core_review ioa raf srp sts tap)
POSITIONS=(pos6 pos10)

# Derive a short model name for directory labels, e.g. "Qwen/Qwen3-8B" → "Qwen3-8B"
MODEL_NAME="${RERANKER##*/}"
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
    local method="$4"
    local pos="$5"

    if [ ! -f "$attacked_docs" ]; then
        echo "  [skip] $attacked_docs not found"
        return
    fi

    local fc_name="${retriever}_FC_${MODEL_NAME}_${method}_${pos}"
    local e2e_name="${retriever}_E2E_${MODEL_NAME}_${method}_${pos}"

    echo ""
    echo ">>> [FC ] retriever=${retriever} method=${method} position=${pos}"
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
        --tensor_parallel_size "$TENSOR_PARALLEL" \
        --run_name "$fc_name"

    echo ""
    echo ">>> [E2E] retriever=${retriever} method=${method} position=${pos}"
    PYTHONPATH=. python RAG_attack_pipeline/run_pipeline.py \
        --mode validate_e2e \
        --jsonl "$JSONL" \
        --baseline_run_dir "$baseline" \
        --attacked_docs "$attacked_docs" \
        --retriever_type "$retriever" \
        --retrieve_top_k 40 \
        --reranker "$RERANKER" \
        --ranker_type listwise \
        --window_size 10 --step_size 1 \
        --rerank_top_k 10 \
        --chatbot --chatbot_top_k 5 \
        --attack_eval \
        --use_vllm \
        --tensor_parallel_size "$TENSOR_PARALLEL" \
        --run_name "$e2e_name"
}

for RETRIEVER in "${RETRIEVERS[@]}"; do
    if [ "$RETRIEVER" = "bm25" ]; then
        BASELINE="$BM25_BASELINE"
    else
        BASELINE="$DENSE_BASELINE"
    fi

    for METHOD in "${METHODS[@]}"; do
        for POS in "${POSITIONS[@]}"; do
            run_pair "$RETRIEVER" "$BASELINE" \
                "$ATTACK_DATA_DIR/${RETRIEVER}/${METHOD}_${POS}_attacked.jsonl" \
                "$METHOD" "$POS"
        done
    done
done

echo ""
echo "All validation runs complete."
