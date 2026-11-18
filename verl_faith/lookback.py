"""Bbox-free Lookback-Lens feature extractor for Qwen3-VL-Thinking.

Adapted from the SLAKE reference (attn-probe-SLAKE/src/faithscan/models/lookback_lens.py),
specifically the Option D bbox-free variant: for each generated token, each (layer, head),
the feature is sum(attn -> vision_tokens) / sum(attn -> all_tokens). Averaged over
generated tokens this yields a 448-dim vector per rollout (28 layers x 16 heads).

Usage:
    from verl_faith.lookback import LookbackExtractor
    ext = LookbackExtractor(model_path=<HF id — REQUIRED, features are model-specific>)
    feat = ext.extract_efficient(image=PIL.Image, question=str, response=str)  # -> np.ndarray (feat_dim,)

Notes:
- Vision token range is detected from <|vision_start|> / <|vision_end|> special tokens.
- Uses model forward pass (not generate) on the full prompt+response sequence; we already
  have the response string from the rollout, so we just need attentions.
- Returns the feature averaged across response tokens only (skipping prompt + system).
"""
from __future__ import annotations
import os
import re
import threading
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

# Qwen3-VL rotary helper used by the shared _project_qk core (collection + serve).
# MUST be a module-level import: the refactor that hoisted Q/K projection into
# _project_qk dropped the old in-hook imports, which made every extraction raise
# NameError -> _compute_faith caught it and returned neutral 0.0 -> the faith arm
# silently collapsed to vanilla. Same import path as before, so no version drift.
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

# Qwen3-VL vision patch-embed Conv3d (kernel==stride==full spatial => 1x1x1 output
# => a per-patch matmul) HANGS on GPU at scale, under both cuDNN and native — see
# patches/patch_verl_qwen3vl_conv.py for the full root-cause. The extractor loads
# the model via transformers directly (AutoModelForImageTextToText), NOT verl's
# patched loader, so the conv->matmul fix must be installed HERE too or the
# extractor's vision forward hangs (the faith arm's reward loop would freeze).
import torch.nn.functional as _F_lb  # noqa: E402


def _patched_patchembed_forward_lb(self, hidden_states):
    target_dtype = self.proj.weight.dtype
    x = hidden_states.reshape(
        -1, self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    ).to(target_dtype)
    w = self.proj.weight.reshape(self.embed_dim, -1)
    return _F_lb.linear(x, w, self.proj.bias)


try:
    from transformers.models.qwen3_vl import modeling_qwen3_vl as _q3vl_lb

    _q3vl_lb.Qwen3VLVisionPatchEmbed.forward = _patched_patchembed_forward_lb
except Exception:  # noqa: BLE001 — model version without this class; nothing to patch
    pass


_VL_THINKING_SYSTEM = (
    "You are VL-Thinking\U0001F914, a helpful assistant with excellent reasoning ability. "
    "A user asks you a question, and you should try to solve it. You should first think "
    "about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>"
)


# ---------------------------------------------------------------------------
# Shared lookback-ratio core (single source of truth).
#
# Factored out of extract_efficient() and extract_batch() so the feature math
# CANNOT drift between them — and so the future verl ref-forward fold (compute
# faith inside the KL ref pass, zero extra forwards) can call the IDENTICAL code.
# A drift here = the probe sees features unlike its training data = garbage faith,
# so this must stay the one place the ratio is computed. Model-agnostic.
# NOTE (future, model-dependent): a ref-forward hook would call _project_qk then
# _lookback_ratio per layer on the ref/base model's forward; see
# memory project_14b_switch_plan ("DEFERRED optimization").
# ---------------------------------------------------------------------------

def _project_qk(module, hidden_states, position_embeddings):
    """Q/K projection + norm + rotary + GQA expand for ONE attention layer.
    Returns (q_rot, k_expanded, scaling). Shared by every extraction path."""
    head_dim = module.head_dim
    num_kv_groups = module.num_key_value_groups
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)
    q_full = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k_full = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    cos, sin = position_embeddings
    q_rot, k_rot = apply_rotary_pos_emb(q_full, k_full, cos, sin)
    k_expanded = k_rot.repeat_interleave(num_kv_groups, dim=1)  # GQA -> (B,H,L,D)
    scaling = head_dim ** -0.5
    return q_rot, k_expanded, scaling


def _lookback_ratio(q_rot, k_expanded, prompt_len, resp_end, actual_len, vis_s, vis_e, scaling):
    """Per-head vision-attention ratio over the response rows of ONE sequence.
    q_rot/k_expanded: (1, H, L, D). actual_len=None for unpadded input (single
    sequence); set actual_len (< L) to also mask right-padding (batched input).
    fp32 scores + softmax to match eager. Returns (H,) float32."""
    L = q_rot.shape[2]
    n_resp = resp_end - prompt_len
    q_resp = q_rot[:, :, prompt_len:resp_end, :]
    scores = torch.matmul(q_resp.float(), k_expanded.float().transpose(-1, -2)) * scaling
    row_idx = torch.arange(n_resp, device=scores.device) + prompt_len
    col_idx = torch.arange(L, device=scores.device)
    mask = col_idx[None, :] > row_idx[:, None]                      # causal
    if actual_len is not None and actual_len < L:
        mask = mask | (col_idx[None, :] >= actual_len).expand(n_resp, -1)  # padding
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    vision_sum = attn[:, :, :, vis_s:vis_e].sum(dim=-1)             # (1,H,n_resp)
    return vision_sum.mean(dim=-1).squeeze(0).float()              # (H,)


# Max tokens the boundary BPE-merge may legitimately eat. A benign merge at the
# prompt/response seam (prompt-only ends `...X Ċ`; the full sequence merges the
# trailing newline into the response's leading newline -> `...X ĊĊ ...`) shortens
# the true common prefix by ~1. A gap larger than this means the chat template
# actually injected tokens into the assistant turn -> real misalignment -> fail.
_PROMPT_BOUNDARY_SLACK = 4


def _resolve_prompt_len(full_ids: torch.Tensor, prompt_ids: torch.Tensor,
                        tokenizer, where: str) -> int:
    """Return the response-window start index = LONGEST COMMON PREFIX length of
    the prompt-only and full (prompt+response) token sequences.

    Why not just len(prompt-only): the two are tokenized separately, so a BPE
    merge at the seam (single `Ċ` in the prompt vs merged `ĊĊ` in the full
    sequence) makes prompt-only NOT an exact prefix — the old code asserted
    exact-prefix and hard-failed ~20% of iou rollouts. The common-prefix start
    is:
      - IDENTICAL to len(prompt-only) when they ARE an exact prefix (normal
        case — no behavior change for the vast majority of rollouts);
      - the seam position when a boundary token merged (the merged token then
        counts as response position 0 — deterministic, and identical at
        collect and serve since the same response text templates the same way).
    A common prefix shorter than len(prompt-only) by more than
    _PROMPT_BOUNDARY_SLACK is a genuine template break -> hard-fail (the guard's
    original protective intent, preserved). Fires in BOTH collect and serve."""
    f = full_ids[0].detach().cpu()
    p = prompt_ids[0].detach().cpu()
    plen, flen = int(p.shape[0]), int(f.shape[0])
    n = min(plen, flen)
    cpl = n
    for i in range(n):
        if int(f[i]) != int(p[i]):
            cpl = i
            break
    if plen - cpl > _PROMPT_BOUNDARY_SLACK:
        def _tok(ids, i):
            if i >= int(ids.shape[0]):
                return None, "<past-end>"
            tid = int(ids[i])
            return tid, tokenizer.convert_ids_to_tokens([tid])[0]
        p_id, p_tok = _tok(p, cpl)
        f_id, f_tok = _tok(f, cpl)
        raise RuntimeError(
            f"[{where}] prompt/full common prefix ({cpl}) is >{_PROMPT_BOUNDARY_SLACK} "
            f"shorter than the templated prompt ({plen}); seq_len={flen}; divergence at "
            f"{cpl}: prompt token {p_id}={p_tok!r} vs full token {f_id}={f_tok!r}. "
            f"Real chat-template break (tokens injected into the assistant turn) — "
            f"the response window would be misaligned; fix the templating.")
    return cpl


class LookbackExtractor:
    """Bbox-free Lookback-Lens feature extractor for Qwen3-VL.

    For (image, question, response), runs one forward pass with output_attentions=True
    and returns a (n_layers * n_heads,) feature vector: ratio of attention to vision
    tokens vs all tokens, averaged over response token positions.
    """

    def __init__(
        self,
        model_path: str,  # REQUIRED — probe features are model-specific; no silent default
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",  # sdpa: memory-efficient; use extract_efficient
    ):
        if not model_path:
            raise ValueError(
                "LookbackExtractor: model_path is required (got falsy). Probe features "
                "are model-specific — pass the exact HF id the probe was/will be "
                "collected on (e.g. Qwen/Qwen3-VL-8B-Thinking); there is no default.")
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device,
            attn_implementation=attn_implementation,
        )
        self._attn_implementation = attn_implementation
        self.model.eval()
        self.device = device
        # Serializes the shared-model hook section in extract_efficient:
        # verl computes rewards concurrently over one shared extractor.
        self._lock = threading.Lock()

        tok = self.processor.tokenizer
        # Resolve vision boundary tokens — names match the Qwen2/3-VL convention.
        self.vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        if self.vision_start_id == tok.unk_token_id or self.vision_end_id == tok.unk_token_id:
            raise RuntimeError(
                f"Could not resolve vision boundary tokens in {model_path}'s tokenizer; "
                f"got start={self.vision_start_id}, end={self.vision_end_id}"
            )

        self.n_layers = self.model.config.text_config.num_hidden_layers
        self.n_heads = self.model.config.text_config.num_attention_heads
        self.feat_dim = self.n_layers * self.n_heads

    def _vision_range(self, input_ids: torch.Tensor) -> tuple[int, int]:
        """Return (start_idx, end_idx) of the vision-token block in input_ids[0]."""
        ids = input_ids[0].tolist()
        try:
            s = ids.index(self.vision_start_id)
            e = ids.index(self.vision_end_id)
        except ValueError as exc:
            raise RuntimeError("vision boundary tokens not present in input_ids") from exc
        # Inclusive of pads between markers; exclusive of the markers themselves.
        return s + 1, e

    @torch.inference_mode()
    def extract(
        self,
        image: Image.Image,
        question: str,
        response: str,
        max_response_tokens: int = 4096,  # no-skew: == collect window == FAITH_SERVE_WINDOW
    ) -> np.ndarray:
        """Run one forward pass and return the 448-dim lookback feature vector."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Build the full chat: system + user(image+text) + assistant(response).
        # Qwen3-VL processor's apply_chat_template requires list-of-dicts content for ALL messages.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _VL_THINKING_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]
        # apply_chat_template with tokenize=True returns tensors directly.
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        vis_s, vis_e = self._vision_range(input_ids)

        # Locate the response token range so we only average attention from response positions.
        # Find the last assistant turn opener in the chat template.
        prompt_msgs = messages[:-1]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        seq_len = int(input_ids.shape[1])
        prompt_len = _resolve_prompt_len(input_ids, prompt_inputs["input_ids"], self.processor.tokenizer, "extract")
        if prompt_len >= seq_len:
            raise RuntimeError(f"prompt_len {prompt_len} >= seq_len {seq_len}; chat template mismatch")
        resp_end = min(seq_len, prompt_len + max_response_tokens)

        out = self.model(
            **inputs,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        # attentions: tuple of length n_layers, each (batch, n_heads, seq_len, seq_len)
        attentions = out.attentions
        assert len(attentions) == self.n_layers, f"got {len(attentions)} layers, want {self.n_layers}"

        # For each response position p in [prompt_len, resp_end), each layer l, head h:
        # ratio = sum(attn[l][0, h, p, vis_s:vis_e]) / sum(attn[l][0, h, p, :p+1])
        # Then average across positions.
        feats = torch.zeros(self.n_layers, self.n_heads, dtype=torch.float32, device=self.device)
        positions = torch.arange(prompt_len, resp_end, device=self.device)
        if positions.numel() == 0:
            return feats.flatten().cpu().numpy()
        for l, attn in enumerate(attentions):
            # attn: (1, n_heads, seq_len, seq_len)
            slice_p = attn[0, :, prompt_len:resp_end, :]  # (n_heads, n_resp, seq_len)
            attn_vision = slice_p[:, :, vis_s:vis_e].sum(dim=-1)  # (n_heads, n_resp)
            attn_total = slice_p.sum(dim=-1).clamp(min=1e-10)  # (n_heads, n_resp)
            ratio = attn_vision / attn_total  # (n_heads, n_resp)
            feats[l] = ratio.mean(dim=-1).to(torch.float32)  # (n_heads,)
        return feats.flatten().cpu().numpy()

    @torch.inference_mode()
    def extract_efficient(
        self,
        image: Image.Image,
        question: str,
        response: str,
        max_response_tokens: int = 4096,  # no-skew: == collect window == FAITH_SERVE_WINDOW
    ) -> np.ndarray:
        """Memory-efficient lookback extraction using forward hooks.

        Same 448-dim output as extract(), but uses SDPA attention forward (no full
        N×N attention matrix materialized) and computes the lookback ratio inline
        per-layer via hooks. Memory: ~400 MB extra per forward (vs ~22 GB for the
        full-attention path on a 5500-token sequence).

        Requires the model to be loaded with attn_implementation="sdpa".
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": _VL_THINKING_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=False,
            tokenize=True, return_tensors="pt", return_dict=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        vis_s, vis_e = self._vision_range(input_ids)

        prompt_msgs = messages[:-1]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_msgs, add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        )
        seq_len = int(input_ids.shape[1])
        prompt_len = _resolve_prompt_len(input_ids, prompt_inputs["input_ids"], self.processor.tokenizer, "extract_efficient")
        if prompt_len >= seq_len:
            raise RuntimeError(f"prompt_len {prompt_len} >= seq_len {seq_len}; chat template mismatch")
        resp_end = min(seq_len, prompt_len + max_response_tokens)

        feats = torch.zeros(self.n_layers, self.n_heads, dtype=torch.float32, device=self.device)


        def make_hook(layer_idx: int):
            def hook(module, args, kwargs, output):
                # Extract hidden_states and position_embeddings from forward call.
                hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                position_embeddings = args[1] if len(args) > 1 else kwargs.get("position_embeddings")
                if hidden_states is None or position_embeddings is None:
                    raise RuntimeError(f"hook layer {layer_idx}: missing hidden_states or position_embeddings")

                # Shared core (see _project_qk / _lookback_ratio above) — single
                # (unpadded) sequence, so actual_len=None.
                q_rot, k_expanded, scaling = _project_qk(module, hidden_states, position_embeddings)
                feats[layer_idx] = _lookback_ratio(
                    q_rot, k_expanded, prompt_len, resp_end, None, vis_s, vis_e, scaling)
                # side-effect only; don't modify the layer output.
            return hook

        # Register on each text-decoder attention layer. SERIALIZED: hooks live on
        # the SHARED model and capture per-call closure state (prompt_len, resp_end,
        # feats); concurrent calls cross-fire hooks onto each others forward and
        # corrupt shapes/features (b=0 / length-mismatch errors). concurrency fix.
        with self._lock:
            handles = []
            text_layers = self.model.model.language_model.layers
            for l_idx, layer in enumerate(text_layers):
                h = layer.self_attn.register_forward_hook(make_hook(l_idx), with_kwargs=True)
                handles.append(h)
            try:
                # Forward pass — SDPA used internally; no full attention matrix materialized.
                self.model(**inputs, use_cache=False, output_attentions=False)
            finally:
                for h in handles:
                    h.remove()

        out = feats.flatten().cpu().numpy()
        # Return this forward's peak activation memory to the CUDA driver. The
        # extractor is a SEPARATE process colocated on the training GPU; without
        # this the caching allocator keeps ~40GB reserved-but-idle from the
        # worst-case forward, invisible to the actor process, so the actor's
        # optimizer step (which peaks AFTER all rewards, when the extractor is
        # idle) OOMs. Freeing here drops the idle extractor from ~56GB -> ~18GB
        # (model only), giving the opt step ~40GB of headroom. Numerically inert
        # (post-forward cleanup only); costs a re-alloc on the next extract.
        del inputs, input_ids
        torch.cuda.empty_cache()
        return out

    @torch.inference_mode()
    def extract_batch(
        self,
        image: Image.Image,
        question: str,
        responses: list,
        max_response_tokens: int = 4096,  # no-skew: == collect window == FAITH_SERVE_WINDOW
    ) -> list:
        """Batched extraction: process N responses (same image+question) in ONE
        HF forward pass. Returns a list of N feature vectors.

        Big speedup over calling extract_efficient N times because:
        - Vision encoder runs once-per-batch (8 copies in parallel — same GPU work)
        - Text decoder forward is amortized across N sequences
        - GPU utilization improves for longer sequences where N>1 helps fill SMs

        Requires sdpa attention. Memory scales linearly with N — typical 8 rollouts
        of mcq @ 2048 tokens needs ~8-12 GB peak.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        N = len(responses)
        if N == 0:
            return []

        # Process each (image, question, response) through the chat template separately,
        # then pad to common length and stack into a batch.
        all_inputs = []
        for response in responses:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": _VL_THINKING_SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": response}]},
            ]
            inp = self.processor.apply_chat_template(
                messages, add_generation_prompt=False,
                tokenize=True, return_tensors="pt", return_dict=True,
            )
            all_inputs.append(inp)

        # Compute prompt_len once (same for all — image+question are identical).
        prompt_msgs = [
            {"role": "system", "content": [{"type": "text", "text": _VL_THINKING_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_msgs, add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        )
        prompt_len = _resolve_prompt_len(all_inputs[0]["input_ids"], prompt_inputs["input_ids"], self.processor.tokenizer, "extract_batch")
        # All items share the image+question prompt, so the prefix check on the
        # first sequence covers the shared prompt_len used for every element
        # (resolved above via the common-prefix helper).

        # Per-element actual seq len and response end (capped at max_response_tokens)
        actual_lens = [int(inp["input_ids"].shape[1]) for inp in all_inputs]
        max_len = max(actual_lens)
        resp_ends = [min(L, prompt_len + max_response_tokens) for L in actual_lens]

        # Right-pad input_ids and attention_mask to max_len
        tok = self.processor.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        padded_input_ids = []
        attention_masks = []
        for inp in all_inputs:
            ids = inp["input_ids"][0]  # (L,)
            L = ids.shape[0]
            pad_amount = max_len - L
            if pad_amount > 0:
                padded_ids = torch.cat([ids, torch.full((pad_amount,), pad_id, dtype=ids.dtype)])
                mask = torch.cat([torch.ones(L, dtype=torch.long), torch.zeros(pad_amount, dtype=torch.long)])
            else:
                padded_ids = ids
                mask = torch.ones(L, dtype=torch.long)
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
        input_ids = torch.stack(padded_input_ids, dim=0).to(self.device)  # (N, max_len)
        attention_mask = torch.stack(attention_masks, dim=0).to(self.device)  # (N, max_len)

        # Pixel_values / image_grid_thw: stack N copies (model needs per-batch-element images)
        pixel_values = torch.cat([inp["pixel_values"].to(self.device) for inp in all_inputs], dim=0)
        image_grid_thw = torch.cat([inp["image_grid_thw"].to(self.device) for inp in all_inputs], dim=0)

        # vision range — same for all elements since image+question is same
        vis_s, vis_e = self._vision_range(input_ids[0:1])  # use element 0

        feats = torch.zeros(N, self.n_layers, self.n_heads, dtype=torch.float32, device=self.device)


        def make_hook(layer_idx: int):
            def hook(module, args, kwargs, output):
                hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                position_embeddings = args[1] if len(args) > 1 else kwargs.get("position_embeddings")

                # Shared core (see _project_qk / _lookback_ratio above). Project once
                # for the whole batch, then per-element ratio with its own pad/causal mask.
                q_rot, k_expanded, scaling = _project_qk(module, hidden_states, position_embeddings)
                for b in range(N):
                    resp_end_b = resp_ends[b]
                    if resp_end_b <= prompt_len:
                        continue
                    feats[b, layer_idx] = _lookback_ratio(
                        q_rot[b:b + 1], k_expanded[b:b + 1], prompt_len, resp_end_b,
                        actual_lens[b], vis_s, vis_e, scaling)
            return hook

        # SERIALIZED like extract_efficient: the hooks live on the SHARED model
        # and close over per-call state (prompt_len, resp_ends, feats) — a
        # concurrent caller would cross-fire hooks onto the other's forward and
        # corrupt shapes/features. Collection is single-threaded today, but the
        # lock makes that safety a property of the extractor, not the caller.
        with self._lock:
            handles = []
            text_layers = self.model.model.language_model.layers
            for l_idx, layer in enumerate(text_layers):
                handles.append(layer.self_attn.register_forward_hook(make_hook(l_idx), with_kwargs=True))

            try:
                # Call the INNER Qwen3VLModel (not the full ConditionalGeneration). This skips
                # the lm_head vocab projection which would OOM at batch size > 1 (logits tensor
                # is N × seq_len × vocab_size). We only need attention via hooks; logits aren't
                # used, so skipping them is both safe and necessary for batched extraction.
                self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=False, output_attentions=False,
                )
            finally:
                for h in handles:
                    h.remove()

        return [feats[b].flatten().cpu().numpy() for b in range(N)]


class BatchingFaithQueue:
    """Micro-batching shim for the reward path.

    verl's reward loop dispatches compute_score per-rollout on a thread pool, so
    at any moment several rollouts of the SAME prompt (same image+question) are
    waiting on the extractor lock. This shim collects same-prompt requests that
    arrive within `window_s` and serves each group with ONE extract_batch
    forward (~2x fewer forwards at typical in-flight depths) instead of N
    serialized extract_efficient calls. Requests for different prompts are
    re-queued, never dropped. On a batch failure it falls back to per-item
    extract_efficient before failing any caller.

    Opt-in: scorers routes through this only when FAITH_BATCH_WINDOW_S > 0.
    """

    def __init__(self, extractor: "LookbackExtractor", window_s: float = 0.25, max_batch: int = 8):
        import queue
        import os
        self.ext = extractor
        self.window_s = float(window_s)
        self.max_batch = int(max_batch)
        # The padded extract_batch path FAILED the parity audit on long sequences
        # (mcq max|Δ|=0.044 vs 5e-3 tol). It is OFF unless explicitly allowed, so
        # even if micro-batching is enabled (window_s>0) every request still goes
        # through the exact single-request extract_efficient path.
        self.allow_batch = os.environ.get("FAITH_ALLOW_BATCH_EXTRACT", "0") == "1"
        if not self.allow_batch:
            print("[faith-queue] FAITH_ALLOW_BATCH_EXTRACT!=1 -> serving every request via "
                  "extract_efficient (batched path disabled; deterministic serve).", flush=True)
        self.q: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="faith-batch-shim")
        self._thread.start()

    @staticmethod
    def _key(image, question):
        """Group key: question text + image fingerprint (16x16 thumb CRC).
        Question alone is unsafe — a wrong-image grouping would silently
        compute features against the wrong image."""
        import zlib
        try:
            return (question, image.size, zlib.crc32(image.resize((16, 16)).tobytes()))
        except Exception:
            return (question, id(image), -1)

    def extract(self, image, question, response, max_response_tokens):
        from concurrent.futures import Future
        fut = Future()
        self.q.put((self._key(image, question), image, question, response,
                    int(max_response_tokens), fut))
        return fut.result()  # re-raises worker-side exceptions in the caller

    def _serve(self, group):
        # Single request, or batched path disabled -> serve each via the exact
        # single-request path (the parity-safe, deterministic route).
        if len(group) == 1 or not self.allow_batch:
            for k, img, q, resp, win, fut in group:
                try:
                    fut.set_result(self.ext.extract_efficient(
                        image=img, question=q, response=resp, max_response_tokens=win))
                except Exception as e:
                    if not fut.done():
                        fut.set_exception(e)
            return
        try:
            feats = self.ext.extract_batch(
                image=group[0][1], question=group[0][2],
                responses=[g[3] for g in group],
                max_response_tokens=group[0][4])
            for g, f in zip(group, feats):
                g[5].set_result(f)
        except Exception:
            # Batch failed (e.g. OOM on a long group) — per-item fallback so a
            # batch-level failure can't zero out 8 rollouts' faith at once.
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            for k, img, q, resp, win, fut in group:
                try:
                    fut.set_result(self.ext.extract_efficient(
                        image=img, question=q, response=resp, max_response_tokens=win))
                except Exception as e2:
                    if not fut.done():
                        fut.set_exception(e2)

    def _loop(self):
        import queue as _q
        import time as _t
        while True:
            first = self.q.get()
            group, pending = [first], []
            deadline = _t.monotonic() + self.window_s
            while len(group) < self.max_batch:
                timeout = deadline - _t.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = self.q.get(timeout=timeout)
                except _q.Empty:
                    break
                if item[0] == first[0] and item[4] == first[4]:
                    group.append(item)
                else:
                    pending.append(item)
            for p in pending:  # different prompt — back in line, served next rounds
                self.q.put(p)
            try:
                self._serve(group)
            except Exception as e:  # never let the shim thread die
                for g in group:
                    if not g[5].done():
                        g[5].set_exception(e)


def _self_test():
    """Quick smoke: load model, extract features for one example, print shape + stats."""
    import json
    import pyarrow.parquet as pq
    from io import BytesIO

    print("loading model...")
    ext = LookbackExtractor()
    print(f"feat_dim={ext.feat_dim} (layers={ext.n_layers}, heads={ext.n_heads})")

    print("loading one parquet row...")
    t = pq.read_table(__import__("os").environ.get("FTGRPO_SMOKE_PARQUET", "data/vlaa_smoke_digit/train.parquet"))
    df = t.to_pandas()
    r = df.iloc[0]
    img_dict = r["images"][0]
    img = Image.open(BytesIO(img_dict["bytes"]))

    # User question is the second message in our prompt; first is system.
    user_content = r["prompt"][1]["content"]
    question = user_content.replace("<image>\n", "")  # raw question
    response = "<think>Let me count.</think> <answer>4</answer>"

    print("extracting...")
    feat = ext.extract_efficient(image=img, question=question, response=response)
    print(f"feat shape: {feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}, min={feat.min():.4f}, max={feat.max():.4f}")


if __name__ == "__main__":
    _self_test()
