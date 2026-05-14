"""
dense_retriever.py — Dense (sentence-transformer) retrieval over ESCI candidate pools
======================================================================================
Uses a SentenceTransformer-compatible model (default: BAAI/bge-large-en-v1.5) for
per-query dense retrieval over the ESCI candidate pool.

Like BM25Retriever, this operates *within* each query's fixed candidate pool — it
scores all candidates for a query and ranks them by cosine similarity.  The BGE query
instruction prefix is applied automatically for BAAI/bge-* models.

Two-pass encoding strategy
--------------------------
1. Collect all unique document PIDs across every query → encode in one batched pass.
   Documents shared between query pools are encoded only once.
2. Batch-encode all queries → look up each query's candidate embeddings and compute
   cosine similarity (dot product of L2-normalised embeddings).

Requirements
------------
  conda install -c conda-forge sentence-transformers
  # or
  pip install sentence-transformers

Example (library)
-----------------
>>> from RAG_attack_pipeline.dense_retriever import DenseRetriever
>>> retriever = DenseRetriever("BAAI/bge-large-en-v1.5")
>>> run = retriever.retrieve(corpus.candidate_pool, top_k=20)
>>> # run: {qid: [(pid, cosine_score), ...]}  sorted by similarity descending

Usage (pipeline CLI)
--------------------
  python run_pipeline.py --jsonl ... \\
      --retriever_type dense --retriever_model BAAI/bge-large-en-v1.5

  # validate_e2e with dense retrieval on patched corpus
  python run_pipeline.py --mode validate_e2e --jsonl ... \\
      --retriever_type dense --retriever_model BAAI/bge-large-en-v1.5 \\
      --attacked_docs path/to/attacked_docs.jsonl --reranker ...
"""
from __future__ import annotations

import gc
import hashlib
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from RAG_attack_pipeline.corpus import QueryEntry

# Type alias — matches BM25Retriever
RankedList = List[Tuple[str, float]]

# BGE v1.5 query instruction (recommended by BAAI for retrieval tasks)
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Default cache directory — pre-encoded embeddings are persisted here so
# that subsequent runs on the same corpus skip the expensive encode step.
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache", "embeddings"
)


class DenseRetriever:
    """
    Dense retriever backed by a SentenceTransformer-compatible model.

    Encodes queries and candidate documents, then ranks candidates by cosine
    similarity (dot product of L2-normalised embeddings).  For BAAI/bge-* models
    the recommended query instruction prefix is applied automatically.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path (e.g. ``"BAAI/bge-large-en-v1.5"``).
    batch_size : int
        Batch size for sentence-transformer encoding (default 32).
    device : str, optional
        ``"cuda"`` / ``"cpu"`` — auto-detected when ``None``.
    use_query_instruction : bool
        Prepend the BGE query instruction prefix for ``bge-*`` models.
        Set ``False`` to disable for non-BGE models (default ``True``).
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size:            int           = 32,
        device:                Optional[str] = None,
        use_query_instruction: bool          = True,
        cache_dir:             Optional[str] = _DEFAULT_CACHE_DIR,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for DenseRetriever.\n"
                "Install with:  conda install -c conda-forge sentence-transformers\n"
                "           or:  pip install sentence-transformers"
            )

        self.model_name = model_name_or_path
        self.batch_size = batch_size
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir  = cache_dir
        if cache_dir:
            pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Apply BGE query instruction for BAAI/bge-* model family
        model_lower = model_name_or_path.lower()
        self.query_instruction = (
            _BGE_QUERY_INSTRUCTION
            if (use_query_instruction and "bge" in model_lower)
            else ""
        )

        print(f"      [DenseRetriever] Loading '{model_name_or_path}' on {self.device} ...")
        self._st = SentenceTransformer(model_name_or_path, device=self.device)
        if self.query_instruction:
            print(f"      [DenseRetriever] Query prefix  : '{self.query_instruction}'")
        if cache_dir:
            print(f"      [DenseRetriever] Embedding cache: {cache_dir}")
        print(f"      [DenseRetriever] Ready.")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(items: List[Tuple[str, str]]) -> str:
        """Stable SHA-256 fingerprint over (id, text) pairs, order-independent."""
        h = hashlib.sha256()
        for item_id, text in sorted(items, key=lambda x: x[0]):
            h.update(item_id.encode("utf-8", errors="replace"))
            h.update(b"\x00")
            h.update(text.encode("utf-8", errors="replace"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    def _doc_cache_path(
        self, pid_list: List[str], doc_texts: Dict[str, str]
    ) -> Optional[str]:
        """Return the cache file path for the given document set (or None if disabled)."""
        if not self.cache_dir:
            return None
        model_slug = self.model_name.replace("/", "_").replace("\\", "_")
        doc_hash   = self._compute_hash([(p, doc_texts[p]) for p in pid_list])
        return os.path.join(self.cache_dir, f"{model_slug}_docs_{doc_hash}.pt")

    def _query_cache_path(
        self, qids: List[str], query_texts: List[str]
    ) -> Optional[str]:
        """Return the cache file path for the given query set (or None if disabled)."""
        if not self.cache_dir:
            return None
        model_slug = self.model_name.replace("/", "_").replace("\\", "_")
        q_hash     = self._compute_hash(list(zip(qids, query_texts)))
        return os.path.join(self.cache_dir, f"{model_slug}_queries_{q_hash}.pt")

    # ------------------------------------------------------------------
    # GPU release
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Delete the SentenceTransformer model and flush GPU memory.

        Call this immediately after ``retrieve()`` when a large reranker or
        chatbot model will be loaded next, to avoid OOM errors.
        """
        if getattr(self, "_st", None) is not None:
            print("      [DenseRetriever] Releasing model from GPU memory...")
            del self._st
            self._st = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            print("      [DenseRetriever] GPU memory released.")

    # ------------------------------------------------------------------
    # Public API  (mirrors BM25Retriever.retrieve signature)
    # ------------------------------------------------------------------
    def retrieve(
        self,
        candidate_pool: Dict[str, QueryEntry],
        top_k:          int  = 20,
        show_progress:  bool = True,
    ) -> Dict[str, RankedList]:
        """
        Rank each query's candidate pool by dense cosine similarity.

        Parameters
        ----------
        candidate_pool : dict  {qid: QueryEntry}
            Built by ``ESCICorpus.candidate_pool``.
        top_k : int
            Maximum documents to return per query.
        show_progress : bool
            Show tqdm progress bars during encoding.

        Returns
        -------
        dict
            ``{qid: [(pid, cosine_score), ...]}``  sorted descending,
            truncated to *top_k*.
        """
        # --- Pass 1: encode all unique documents in one batched pass -------
        doc_texts: Dict[str, str] = {}   # pid → concatenated product text
        for entry in candidate_pool.values():
            for cand in entry.candidates:
                if cand.pid not in doc_texts:
                    doc_texts[cand.pid] = cand.text

        pid_list    = list(doc_texts.keys())
        doc_strings = [doc_texts[p] for p in pid_list]

        doc_cache = self._doc_cache_path(pid_list, doc_texts)
        if doc_cache and os.path.exists(doc_cache):
            print(f"      [DenseRetriever] Loading doc embeddings from cache "
                  f"({len(doc_strings):,} docs)  {os.path.basename(doc_cache)}")
            payload  = torch.load(doc_cache, map_location="cpu", weights_only=False)  # noqa: S614
            pid_list = payload["pid_list"]
            doc_embs = payload["embeddings"].to(self.device)
        else:
            print(f"      [DenseRetriever] Encoding {len(doc_strings):,} unique documents ...")
            doc_embs = self._st.encode(
                doc_strings,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )  # Tensor  (n_docs, dim)
            if doc_cache:
                torch.save(
                    {"pid_list": pid_list, "embeddings": doc_embs.cpu()},
                    doc_cache,
                )
                print(f"      [DenseRetriever] Doc embeddings cached → {doc_cache}")

        pid_to_idx = {pid: i for i, pid in enumerate(pid_list)}

        # --- Pass 2: encode all queries in one batched pass ----------------
        qids        = list(candidate_pool.keys())
        query_texts = [
            self.query_instruction + candidate_pool[q].query for q in qids
        ]

        q_cache = self._query_cache_path(qids, query_texts)
        if q_cache and os.path.exists(q_cache):
            print(f"      [DenseRetriever] Loading query embeddings from cache "
                  f"({len(query_texts):,} queries)  {os.path.basename(q_cache)}")
            # Saved as {qid: cpu_tensor}; reconstruct ordered tensor for this run
            q_cache_data = torch.load(q_cache, map_location="cpu", weights_only=False)  # noqa: S614
            q_embs = torch.stack([q_cache_data[qid] for qid in qids]).to(self.device)
        else:
            print(f"      [DenseRetriever] Encoding {len(query_texts):,} queries ...")
            q_embs = self._st.encode(
                query_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )  # Tensor  (n_queries, dim)
            if q_cache:
                # Store as {qid: tensor} — order-independent, safe across runs
                torch.save(
                    {qid: q_embs[i].cpu() for i, qid in enumerate(qids)},
                    q_cache,
                )
                print(f"      [DenseRetriever] Query embeddings cached → {q_cache}")

        # --- Per-query ranking ---------------------------------------------
        run: Dict[str, RankedList] = {}
        iter_items = (
            tqdm(enumerate(qids), total=len(qids), desc="Dense ranking")
            if show_progress else enumerate(qids)
        )
        for i, qid in iter_items:
            entry     = candidate_pool[qid]
            cand_pids = [c.pid for c in entry.candidates]

            # Keep index tensor on the same device as doc_embs to avoid a
            # host-device round-trip on every iteration.
            idx_t     = torch.tensor(
                [pid_to_idx[p] for p in cand_pids],
                dtype=torch.long, device=doc_embs.device,
            )
            cand_embs = doc_embs[idx_t]          # (n_cands, dim)

            # 1-D × 2-D matmul: q_embs[i] has shape (dim,), cand_embs.T is
            # (dim, n_cands) → result is (n_cands,) with no unsqueeze/squeeze
            scores = q_embs[i] @ cand_embs.T     # (n_cands,)

            # GPU-side top-k avoids sorting all n_cands scores on the CPU.
            k_actual          = min(top_k, len(cand_pids))
            top_scores, top_i = torch.topk(scores, k_actual)
            top_scores_list   = top_scores.cpu().tolist()
            top_i_list        = top_i.cpu().tolist()

            run[qid] = [(cand_pids[j], s)
                        for j, s in zip(top_i_list, top_scores_list)]

        return run

    # ------------------------------------------------------------------
    # Incremental-patch retrieval (validate_e2e with dense)
    # ------------------------------------------------------------------

    def retrieve_patched(
        self,
        candidate_pool: Dict[str, QueryEntry],
        original_pool:  Dict[str, QueryEntry],
        top_k:          int  = 20,
        show_progress:  bool = True,
    ) -> Dict[str, RankedList]:
        """
        Retrieve from a patched corpus efficiently by reusing the original
        embedding cache and only re-encoding documents whose text changed.

        Typical use-case: ``validate_e2e`` with a dense retriever, where an
        adversarial attack modifies ~4 k documents out of tens of thousands.
        Re-encoding the full corpus would be wasteful.  This method:

        1. Locates the existing doc-embedding cache for *original_pool*.
        2. Identifies which PIDs have different text in *candidate_pool*.
        3. Re-encodes only those documents.
        4. Patches the embedding matrix in-place.
        5. Runs per-query ranking over the patched embeddings.

        The patched embedding matrix is **not** saved to cache because the
        modified corpus is transient and must not overwrite the clean cache.

        Falls back to a full ``retrieve()`` call if the original cache cannot
        be found.

        Parameters
        ----------
        candidate_pool : dict  {qid: QueryEntry}
            The **patched** candidate pool (adversarially modified texts).
        original_pool : dict  {qid: QueryEntry}
            The **original** (unmodified) candidate pool — used only for
            locating the existing embedding cache.
        top_k : int
            Maximum documents to return per query.
        show_progress : bool
            Show tqdm progress bars during encoding.

        Returns
        -------
        dict  {qid: [(pid, cosine_score), ...]}  sorted descending, top_k.
        """
        # ── A: locate the original embedding cache ─────────────────────
        original_doc_texts: Dict[str, str] = {}
        for entry in original_pool.values():
            for cand in entry.candidates:
                if cand.pid not in original_doc_texts:
                    original_doc_texts[cand.pid] = cand.text

        original_pid_list = list(original_doc_texts.keys())
        doc_cache = self._doc_cache_path(original_pid_list, original_doc_texts)

        if not (doc_cache and os.path.exists(doc_cache)):
            print(
                "      [DenseRetriever] Original doc cache not found; "
                "falling back to full re-encode of patched corpus ..."
            )
            return self.retrieve(
                candidate_pool, top_k=top_k, show_progress=show_progress
            )

        print(
            f"      [DenseRetriever] Loading original doc cache "
            f"({len(original_pid_list):,} docs)  {os.path.basename(doc_cache)}"
        )
        payload    = torch.load(doc_cache, map_location="cpu", weights_only=False)  # noqa: S614
        pid_list   = payload["pid_list"]
        doc_embs   = payload["embeddings"].to(self.device)   # mutable copy on device
        pid_to_idx = {pid: i for i, pid in enumerate(pid_list)}

        # ── B: find documents with modified text ───────────────────────
        patched_doc_texts: Dict[str, str] = {}
        for entry in candidate_pool.values():
            for cand in entry.candidates:
                if cand.pid not in patched_doc_texts:
                    patched_doc_texts[cand.pid] = cand.text

        modified_pids = [
            pid for pid in patched_doc_texts
            if (
                patched_doc_texts[pid] != original_doc_texts.get(pid)
                and pid in pid_to_idx      # must exist in original cache
            )
        ]
        print(
            f"      [DenseRetriever] Modified docs: {len(modified_pids):,} "
            f"(out of {len(patched_doc_texts):,} total in pool)"
        )

        # ── C: re-encode modified docs and patch the embedding matrix ──
        if modified_pids:
            mod_texts = [patched_doc_texts[p] for p in modified_pids]
            print(
                f"      [DenseRetriever] Re-encoding {len(modified_pids):,} "
                "modified docs ..."
            )
            mod_embs = self._st.encode(
                mod_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )  # Tensor (n_modified, dim) on self.device

            # Scatter updated embeddings back into the matrix in-place.
            mod_idx = torch.tensor(
                [pid_to_idx[p] for p in modified_pids],
                dtype=torch.long,
                device=self.device,
            )
            doc_embs[mod_idx] = mod_embs
            print("      [DenseRetriever] Embedding matrix patched.")

        # ── D: encode queries — reuse cache if available ───────────────
        qids        = list(candidate_pool.keys())
        query_texts = [
            self.query_instruction + candidate_pool[q].query for q in qids
        ]
        q_cache = self._query_cache_path(qids, query_texts)
        if q_cache and os.path.exists(q_cache):
            print(
                f"      [DenseRetriever] Loading query embeddings from cache "
                f"({len(query_texts):,} queries)  {os.path.basename(q_cache)}"
            )
            q_cache_data = torch.load(q_cache, map_location="cpu", weights_only=False)  # noqa: S614
            q_embs = torch.stack(
                [q_cache_data[qid] for qid in qids]
            ).to(self.device)
        else:
            print(f"      [DenseRetriever] Encoding {len(query_texts):,} queries ...")
            q_embs = self._st.encode(
                query_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )  # Tensor (n_queries, dim) on self.device
            if q_cache:
                torch.save(
                    {qid: q_embs[i].cpu() for i, qid in enumerate(qids)},
                    q_cache,
                )
                print(f"      [DenseRetriever] Query embeddings cached → {q_cache}")

        # ── E: per-query ranking (identical to retrieve()) ─────────────
        run: Dict[str, RankedList] = {}
        iter_items = (
            tqdm(enumerate(qids), total=len(qids), desc="Dense ranking (patched)")
            if show_progress else enumerate(qids)
        )
        for i, qid in iter_items:
            entry     = candidate_pool[qid]
            cand_pids = [c.pid for c in entry.candidates]

            idx_t     = torch.tensor(
                [pid_to_idx[p] for p in cand_pids],
                dtype=torch.long, device=doc_embs.device,
            )
            cand_embs = doc_embs[idx_t]           # (n_cands, dim)
            scores    = q_embs[i] @ cand_embs.T   # (n_cands,)

            k_actual          = min(top_k, len(cand_pids))
            top_scores, top_i = torch.topk(scores, k_actual)
            run[qid] = [
                (cand_pids[j], s)
                for j, s in zip(top_i.cpu().tolist(), top_scores.cpu().tolist())
            ]

        return run

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(model={self.model_name}, "
            f"device={self.device}, batch_size={self.batch_size})"
        )
