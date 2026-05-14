"""
chatbot.py — RAG Chatbot: final aggregation step of the pipeline
================================================================
Takes the top-k reranked products for each query and uses a causal LM
to generate a natural language answer that recommends the best 3 products
by name.

RagChatbot
    Loads any HuggingFace causal LM and, for each query, builds a
    product-list prompt from the top-k reranked candidates, calls
    `model.generate()`, and extracts the response text.

Usage
-----
  # Standalone
  from RAG_attack_pipeline.chatbot import RagChatbot
  bot = RagChatbot("Qwen/Qwen3-8B")
  answers = bot.generate_all(corpus.candidate_pool, reranked_run, top_k=5,
                              log_path="runs/.../chatbot_responses.jsonl")
  for qid, answer in answers.items():
      print(f"[{qid}] {answer}")

  # From the pipeline CLI
  python run_pipeline.py --jsonl ... --reranker Qwen/Qwen3-8B \\
      --ranker_type pairwise --pairwise_method heapsort \\
      --chatbot --chatbot_top_k 5
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from RAG_attack_pipeline.corpus import Candidate, QueryEntry

RankedList = List[Tuple[str, float]]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a helpful shopping assistant. "
    "Given a customer's search query and a list of retrieved product candidates, "
    "your task is to identify the 3 best-matching products. "
    "Always state the exact product titles in your answer."
)

_USER_TEMPLATE = """\
Customer query: "{query}"

Top retrieved products:
{product_list}

Based on these candidates, identify the top 3 most relevant products.
Output your selection as a numbered list (1., 2., 3.). 
For each item, provide the product ID/Label and a brief one-sentence justification.
"""

# ===========================================================================
# RagChatbot
# ===========================================================================
class RagChatbot:
    """
    RAG chatbot that aggregates top-k reranked products and generates a
    concise product recommendation naming the best 3 items.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path (e.g. ``"Qwen/Qwen3-8B"``).
    device : str, optional
        ``"cuda"`` or ``"cpu"``.  Auto-detected when *None*.
    max_new_tokens : int
        Maximum tokens to generate per response (default 512).
    top_k_context : int
        Number of reranked documents to include in the prompt (default 5).
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_new_tokens: int = 8192,
        top_k_context: int = 5,
        use_vllm: bool = False,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.model_name    = model_name_or_path
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.top_k_context = top_k_context
        self.use_vllm      = use_vllm
        self.tensor_parallel_size = tensor_parallel_size
        if use_vllm:
            self._load_model_vllm()
        else:
            self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        print(f"[Chatbot] Loading model '{self.model_name}' → device='{self.device}'")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="sdpa" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        print("[Chatbot] Model ready.")

    def _load_model_vllm(self) -> None:
        """Load model via vLLM for batched, high-throughput generation."""
        from vllm import LLM, SamplingParams
        print(
            f"[Chatbot] Loading '{self.model_name}' via vLLM "
            f"(tensor_parallel_size={self.tensor_parallel_size})"
        )
        # Tokenizer is still needed for chat-template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._vllm_engine = LLM(
            model=self.model_name,
            dtype="bfloat16",
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=32768,
        )
        self._vllm_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
        )
        # model attribute set to None so code that checks `self.model` won't crash
        self.model = None
        print("[Chatbot] vLLM model ready.")

    @staticmethod
    def _extract_title(text: str) -> str:
        """First non-empty line of the candidate text is the product title."""
        for line in text.split("\n"):
            line = line.strip()
            if line:
                return line
        return text[:80]

    def _build_user_message(self, query: str, docs: List[Tuple[str, str, str]]) -> str:
        """Format the numbered product list and fill the template."""
        lines = []
        for i, (pid, title, text) in enumerate(docs, 1):
            snippet = text.replace("\n", " ").strip()
            lines.append(f"{title}\n {snippet}")
        product_list = "\n\n".join(lines)
        return _USER_TEMPLATE.format(query=query, product_list=product_list)

    def _apply_template(self, messages: list) -> str:
        """Apply the chat template, disabling thinking mode for Qwen3."""
        # Try Qwen3-style with enable_thinking=False first, fall back to standard
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, query: str, docs: List[Tuple[str, str, str]]) -> str:
        """
        Generate a best-3 product recommendation for a single query.

        Parameters
        ----------
        query : str
        docs  : list of (pid, title, full_text) tuples — top-k candidates

        Returns
        -------
        str  Generated recommendation text.
        """
        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": self._build_user_message(query, docs)},
        ]
        prompt   = self._apply_template(messages)
        # debug
        # print(f"=== PROMPT ===\n{prompt}\n=== END PROMPT ===")
        inputs   = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                    if self.tokenizer.pad_token_id is None
                    else self.tokenizer.pad_token_id
                ),
            )

        # Decode only the newly generated tokens
        new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        # debug
        # print(f"=== RESPONSE ===\n{response}\n=== END RESPONSE ===")
        return response

    def generate_all(
        self,
        candidate_pool: Dict[str, QueryEntry],
        reranked_run:   Dict[str, RankedList],
        top_k:          int = 5,
        log_path:       Optional[str] = None,
        show_progress:  bool = True,
    ) -> Dict[str, str]:
        """
        Generate recommendations for every query in *reranked_run*.

        Parameters
        ----------
        candidate_pool : dict  {qid: QueryEntry}
        reranked_run   : dict  {qid: [(pid, score), ...]}
        top_k          : int   Number of top-ranked docs to include in the prompt.
        log_path       : str, optional
            If provided, write a JSONL file (one record per query) with fields:
            ``qid``, ``query``, ``top_docs`` (pid + title), ``response``.
        show_progress  : bool

        Returns
        -------
        dict  {qid: response_text}
        """
        if self.use_vllm:
            return self._generate_all_vllm(
                candidate_pool, reranked_run, top_k, log_path, show_progress
            )

        # Build per-query pid → Candidate lookup at query level to preserve
        # correct esci_label per (query, product) pair
        responses: Dict[str, str] = {}
        log_file = open(log_path, "w", encoding="utf-8") if log_path else None

        items = (
            tqdm(reranked_run.items(), desc="[Chatbot] Generating recommendations")
            if show_progress else reranked_run.items()
        )

        for qid, ranked in items:
            entry = candidate_pool.get(qid)
            if entry is None:
                continue

            # Per-query pid → Candidate (correct labels for this query)
            pid_to_cand: Dict[str, Candidate] = {
                c.pid: c for c in entry.candidates
            }

            # Build top-k docs list for this query
            docs: List[Tuple[str, str, str]] = []
            for pid, _score in ranked[:top_k]:
                cand = pid_to_cand.get(pid)
                if cand:
                    title = self._extract_title(cand.text)
                    docs.append((pid, title, cand.text))

            if not docs:
                continue

            response        = self.generate(entry.query, docs)
            responses[qid]  = response

            if log_file:
                log_file.write(json.dumps({
                    "qid":      qid,
                    "query":    entry.query,
                    "top_docs": [
                        {"rank": i + 1, "pid": pid, "title": title}
                        for i, (pid, title, _) in enumerate(docs)
                    ],
                    "response": response,
                }, ensure_ascii=False) + "\n")

        if log_file:
            log_file.close()

        return responses
    # ------------------------------------------------------------------
    # vLLM batched generation
    # ------------------------------------------------------------------
    def _generate_all_vllm(
        self,
        candidate_pool: Dict[str, QueryEntry],
        reranked_run:   Dict[str, RankedList],
        top_k:          int = 5,
        log_path:       Optional[str] = None,
        show_progress:  bool = True,
    ) -> Dict[str, str]:
        """
        Send ALL query prompts to vLLM in a single batched call.

        This replaces the sequential one-query-at-a-time HF loop and typically
        achieves 5–10× higher throughput on a multi-query workload.
        """
        # Collect (qid, query, docs, messages) for every query.
        # We pass raw message lists to llm.chat() so that vLLM applies the
        # chat template internally with chat_template_kwargs — this is the
        # official Qwen3 pattern for disabling chain-of-thought thinking.
        all_data: List[Tuple[str, str, List, List]] = []
        for qid, ranked in reranked_run.items():
            entry = candidate_pool.get(qid)
            if entry is None:
                continue
            pid_to_cand: Dict[str, Candidate] = {
                c.pid: c for c in entry.candidates
            }
            docs: List[Tuple[str, str, str]] = []
            for pid, _score in ranked[:top_k]:
                cand = pid_to_cand.get(pid)
                if cand:
                    title = self._extract_title(cand.text)
                    docs.append((pid, title, cand.text))
            if not docs:
                continue
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": self._build_user_message(entry.query, docs)},
            ]
            all_data.append((qid, entry.query, docs, messages))

        if not all_data:
            return {}

        all_messages = [d[3] for d in all_data]
        print(f"[Chatbot/vLLM] Generating {len(all_messages)} responses in one batch ...")
        # chat() applies the model's template internally; chat_template_kwargs
        # disables Qwen3's thinking mode (no-op for other models).
        raw_outputs = self._vllm_engine.chat(
            all_messages,
            self._vllm_sampling,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=show_progress,
        )

        responses: Dict[str, str] = {}
        log_file = open(log_path, "w", encoding="utf-8") if log_path else None

        for (qid, query, docs, _), out in zip(all_data, raw_outputs):
            response       = out.outputs[0].text.strip()
            responses[qid] = response
            if log_file:
                log_file.write(json.dumps({
                    "qid":      qid,
                    "query":    query,
                    "top_docs": [
                        {"rank": i + 1, "pid": pid, "title": title}
                        for i, (pid, title, _) in enumerate(docs)
                    ],
                    "response": response,
                }, ensure_ascii=False) + "\n")

        if log_file:
            log_file.close()
        return responses