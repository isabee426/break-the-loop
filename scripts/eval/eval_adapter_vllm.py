"""vLLM-based eval. Drop-in replacement for eval_adapter.py with:

- Continuous-batching inference (all 1000 prompts in one llm.generate call) —
  10-30× faster than HF's batch=1 loop AND immune to the tail-mcq
  model.generate hang that wedged phase-1.
- Greedy decode (temperature=0) so it matches the baseline eval protocol.
- Default --max-examples=None evaluates the FULL test parquet
  (1000 prompts) — matches the README baseline numbers.
- Works with or without a LoRA adapter (--no-adapter for the base eval).

Output JSON schema matches eval_adapter.py so eval_aggregate.py picks up the
results without changes.

Usage:
    # Full 1000-prompt eval of a checkpoint (paper-style):
    python eval_adapter_vllm.py \\
        --checkpoint checkpoints/ft_faith/global_step_140 \\
        --test-parquet data/vlaa_subset_5400/test.parquet \\
        --output results/eval/<fam>__step140__test.json \\
        --max-new-tokens 4096 --gpu 0

    # Base-model baseline (no adapter):
    python eval_adapter_vllm.py --no-adapter \\
        --test-parquet data/vlaa_subset_5400/test.parquet \\
        --output results/eval/baseline_2048.json \\
        --max-new-tokens 4096 --gpu 0
"""
from __future__ import annotations
import os
import sys
# Evals are ALWAYS strict (training correctness definition: tag-required +
# payload-exact scoring). FORCED — not setdefault — so a stray exported
# STRICT_CORRECTNESS=0 can't silently flip paper numbers to the lenient
# scorer.
_PREV_STRICT = os.environ.get("STRICT_CORRECTNESS")
os.environ["STRICT_CORRECTNESS"] = "1"
if _PREV_STRICT is not None and _PREV_STRICT != "1":
    print(f"[eval-vllm] WARNING: environment had STRICT_CORRECTNESS={_PREV_STRICT!r} — "
          "OVERRIDDEN to '1'; eval scoring is always strict",
          file=sys.stderr, flush=True)
# Must be set BEFORE any import that touches CUDA / torch / vllm — vLLM v1's
# engine spawns a subprocess, and torch's CUDA can't be re-initialized after a
# fork. "spawn" forces a clean subprocess start (vLLM's recommended setting).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import io
import json
import time
from collections import defaultdict
from pathlib import Path

VL_THINKING_SYSTEM = (
    "You are VL-Thinking\U0001F914, a helpful assistant with excellent reasoning ability. "
    "A user asks you a question, and you should try to solve it. You should first think "
    "about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>"
)

# Import scorers from THIS repo checkout (resolved from __file__), never the
# stale installed copy — the scorers_sha1 in meta then proves provenance.
#
SCRIPTS_DIR = str(Path(__file__).resolve().parents[2])  # repo root
sys.path.insert(0, SCRIPTS_DIR)


def pin_gpu(gpu: str | None) -> None:
    """HARD RULE: never land on GPU 0 implicitly on a shared training box.
    Must be called before any torch/vLLM import."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"[gpu] pinned CUDA_VISIBLE_DEVICES={gpu}", flush=True)
        return
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        raise SystemExit(
            "--gpu not given and CUDA_VISIBLE_DEVICES is unset/empty — refusing to "
            "default to GPU 0 on a shared box (training may be running there). "
            "Pass --gpu N or export CUDA_VISIBLE_DEVICES deliberately.")
    print(f"[gpu] respecting pre-set CUDA_VISIBLE_DEVICES={cvd}", flush=True)


def eval_provenance(engine: str, model_path: str, adapter_path,
                    gen_params: dict) -> dict:
    """Traceability block recorded in every results JSON."""
    import hashlib
    import inspect
    import subprocess
    import verl_faith.scorers as _scorers_mod
    scorers_file = inspect.getsourcefile(_scorers_mod)
    scorers_sha1 = hashlib.sha1(Path(scorers_file).read_bytes()).hexdigest()
    try:
        git_commit = subprocess.run(
            ["git", "-C", SCRIPTS_DIR, "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        git_commit = None
    return {
        "strict_correctness": os.environ["STRICT_CORRECTNESS"],
        "scorers_sha1": scorers_sha1,
        "scorers_path": scorers_file,
        "engine": engine,
        "model_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "generation": gen_params,
        "eval_script_git": git_commit,
    }


def read_lora_rank(adapter_dir: Path) -> int:
    """LoRA rank from the adapter's own config — never a hardcoded guess.
    Returns max(r, 16) as vLLM's max_lora_rank."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise SystemExit(
            f"missing {cfg_path} — cannot determine LoRA rank for vLLM's "
            "max_lora_rank; is this a real PEFT adapter dir?")
    try:
        r = int(json.loads(cfg_path.read_text())["r"])
    except Exception as exc:
        raise SystemExit(f"cannot read LoRA rank `r` from {cfg_path}: {exc}")
    print(f"[eval-vllm] adapter LoRA rank r={r} -> max_lora_rank={max(r, 16)}", flush=True)
    return max(r, 16)


def build_llm(base_model: str, max_model_len: int, gpu_memory_util: float,
              enable_lora: bool, max_lora_rank: int = 16):
    from vllm import LLM
    kwargs = dict(
        model=base_model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_util,
        # CUDA graphs ON by default (single-process eval has no contention).
        # On small cards (e.g. RTX A5000 25GB) an 8B model + graph-capture
        # reservation starves the KV cache -> set VLLM_ENFORCE_EAGER=1 to skip
        # graph capture (faster startup, frees several GB for KV).
        enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
        # Multi-modal cap per prompt.
        limit_mm_per_prompt={"image": 1},
    )
    if enable_lora:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = max_lora_rank
    return LLM(**kwargs)


def resolve_adapter_dir(checkpoint: Path) -> Path:
    # verl wraps adapter under <ckpt>/actor/lora_adapter/; accept both layouts.
    for cand in (checkpoint / "actor" / "lora_adapter", checkpoint / "lora_adapter"):
        if (cand / "adapter_model.safetensors").is_file():
            return cand
    raise SystemExit(
        f"missing adapter dir under {checkpoint} "
        "(checked actor/lora_adapter and lora_adapter)"
    )


def _resolve_verl_merger() -> Path:
    """Locate verl's legacy_model_merger.py (merges FSDP shards -> a single HF dir).
    Prefers $VERL_DIR/scripts/, else the scripts/ dir alongside the installed verl
    package. No hardcoded box path."""
    cands = []
    vd = os.environ.get("VERL_DIR")
    if vd:
        cands.append(Path(vd) / "scripts" / "legacy_model_merger.py")
    try:
        import verl
        cands.append(Path(verl.__file__).resolve().parent.parent / "scripts" / "legacy_model_merger.py")
    except Exception:
        pass
    for c in cands:
        if c.is_file():
            return c
    raise SystemExit(
        "cannot locate verl's legacy_model_merger.py — set VERL_DIR to your verl "
        f"checkout root (tried: {', '.join(str(c) for c in cands) or 'none'}).")


def merge_fullft_checkpoint(checkpoint: Path, merged_dir: Path, reuse: bool = False) -> Path:
    """Merge a FULL fine-tune (no-LoRA) verl FSDP checkpoint's model shards under
    <checkpoint>/actor into one HF dir that vLLM loads directly as the base model.

    Full-FT saves the WHOLE model (model_world_size_*.pt) — there is no adapter, so
    eval runs with enable_lora=False on the merged weights. This is distinct from
    eval_checkpoint.py's merge (quarantined for the LoRA-drop save bug): that bug is
    N/A here — nothing is dropped, the full model is what was saved."""
    import subprocess
    actor_dir = checkpoint / "actor"
    if not actor_dir.is_dir():
        raise SystemExit(
            f"missing actor dir at {actor_dir} — expected a full-FT FSDP checkpoint "
            "(model_world_size_*.pt shards under <checkpoint>/actor).")
    if merged_dir.is_dir() and any(merged_dir.iterdir()):
        if reuse:
            print(f"[full-ft-merge] reusing populated {merged_dir} (--reuse-merged)", flush=True)
            return merged_dir
        raise SystemExit(
            f"merged dir {merged_dir} exists and is non-empty. Refusing to silently "
            "reuse it (it may belong to a different step/run). Pass --reuse-merged, "
            "or point --merged-dir elsewhere / delete it to re-merge.")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merger = _resolve_verl_merger()
    cmd = [sys.executable, str(merger), "merge", "--backend", "fsdp",
           "--local_dir", str(actor_dir), "--target_dir", str(merged_dir)]
    print(f"[full-ft-merge] {actor_dir} -> {merged_dir}\n  {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return merged_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="global_step_N dir containing lora_adapter/; omit with --no-adapter")
    ap.add_argument("--no-adapter", action="store_true",
                    help="evaluate the base model directly")
    ap.add_argument("--full-ft-checkpoint", default=None,
                    help="FULL fine-tune checkpoint (global_step_N with actor/model_world_size_*.pt "
                         "shards, no LoRA adapter). Merges the shards to --merged-dir and evals the "
                         "full model directly (no adapter). Mutually exclusive with --checkpoint.")
    ap.add_argument("--merged-dir", default=None,
                    help="output HF dir for the merged full-FT weights (required with --full-ft-checkpoint)")
    ap.add_argument("--reuse-merged", action="store_true",
                    help="reuse an already-populated --merged-dir instead of erroring (never reused "
                         "silently; it may be stale)")
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-model",
                    default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Thinking"),
                    help="defaults to $MODEL_PATH (set by configs/full_ft.env)")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="matched to the training response budget (4096 everywhere); older baseline protocol used 2048")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="if set, stratified-by-verifier subsample of N prompts (seed=42); "
                         "default = full test set")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="vLLM context budget — prompt(<=4096)+response(4096)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--gpu", default=None,
                    help="GPU index to pin (sets CUDA_VISIBLE_DEVICES). If omitted, an "
                         "already-exported CUDA_VISIBLE_DEVICES is respected; if neither "
                         "is set the script refuses to run.")
    ap.add_argument("--save-all-responses", action="store_true",
                    help="save every prompt's response (not just the per-verifier sample) — enables per-example diff analysis between checkpoints")
    args = ap.parse_args()

    if args.full_ft_checkpoint:
        if args.checkpoint:
            ap.error("--full-ft-checkpoint and --checkpoint are mutually exclusive")
        if not args.merged_dir:
            ap.error("--merged-dir is required with --full-ft-checkpoint")
        # Full-FT: whole model saved, no adapter -> eval the merged weights directly.
        args.no_adapter = True
    if not args.no_adapter and not args.checkpoint:
        ap.error("--checkpoint required unless --no-adapter is set")

    pin_gpu(args.gpu)  # before any torch/vLLM import
    os.environ.pop("FAITH_PROBE_PATH", None)

    # Full-FT: merge the FSDP shards to an HF dir and point vLLM at it as the base
    # model (enable_lora=False). Done after pin_gpu so the merge inherits the GPU env.
    if args.full_ft_checkpoint:
        merged = merge_fullft_checkpoint(
            Path(args.full_ft_checkpoint), Path(args.merged_dir), args.reuse_merged)
        args.base_model = str(merged)
    from verl_faith.scorers import compute_score  # noqa: WPS433

    from PIL import Image
    import pandas as pd
    import pyarrow.parquet as pq
    from transformers import AutoProcessor

    print(f"[eval-vllm] base = {args.base_model}", flush=True)
    print(f"[eval-vllm] test_parquet = {args.test_parquet}", flush=True)

    df = pq.read_table(args.test_parquet).to_pandas()
    total_n_full = len(df)
    if args.max_examples is not None and args.max_examples > 0 and args.max_examples < len(df):
        # Stratified subsample by verifier (same seed/logic as eval_adapter.py).
        # The test parquet is grouped by data_source, so df.head(N) would return
        # a single-verifier (digit-only) slice.
        counts = df["data_source"].value_counts()
        total = counts.sum()
        parts = []
        for ds in counts.index:
            grp = df[df["data_source"] == ds]
            n = max(1, min(len(grp), int(round(args.max_examples * len(grp) / total))))
            parts.append(grp.sample(n=n, random_state=42))
        df = pd.concat(parts, ignore_index=True)
        print(f"[eval-vllm] subsampled to {len(df)} prompts (target {args.max_examples}, "
              f"stratified by verifier)", flush=True)
    else:
        print(f"[eval-vllm] using full {len(df)} prompts", flush=True)

    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)

    # Resolve the adapter (and its true LoRA rank) BEFORE engine bring-up so a
    # bad adapter dir fails fast and max_lora_rank matches reality.
    adapter_dir = None
    max_lora_rank = 16
    if not args.no_adapter:
        adapter_dir = resolve_adapter_dir(Path(args.checkpoint))
        max_lora_rank = read_lora_rank(adapter_dir)

    # Build vLLM LLM + sampling params + (optional) LoRA request
    print("[eval-vllm] loading vLLM engine...", flush=True)
    t0 = time.time()
    llm = build_llm(
        base_model=args.base_model,
        max_model_len=args.max_model_len,
        gpu_memory_util=args.gpu_memory_utilization,
        enable_lora=not args.no_adapter,
        max_lora_rank=max_lora_rank,
    )

    lora_request = None
    if not args.no_adapter:
        from vllm.lora.request import LoRARequest
        print(f"[eval-vllm] loading adapter from {adapter_dir}", flush=True)
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=str(adapter_dir),
        )

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.0,           # greedy
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        n=1,
    )

    # Build vLLM-formatted inputs: (prompt_text, image) for each row.
    print(f"[eval-vllm] building {len(df)} prompts...", flush=True)
    inputs = []
    for r in df.itertuples():
        img = Image.open(io.BytesIO(r.images[0]["bytes"])).convert("RGB")
        user_text = r.prompt[1]["content"].replace("<image>\n", "")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": VL_THINKING_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ]},
        ]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": img},
        })

    print(f"[eval-vllm] generating (continuous batching)...", flush=True)
    gen_t0 = time.time()
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    gen_elapsed = time.time() - gen_t0
    print(f"[eval-vllm] generation done in {gen_elapsed:.1f}s "
          f"({gen_elapsed / len(df):.2f}s/prompt avg)", flush=True)

    # Score each response. Saved samples are stratified by verifier (4 per verifier
    # = 16 total) so behavioral analysis covers every task type, not just whatever
    # comes first in iteration order (which previously meant 16 digit-only samples).
    PER_VERIFIER_SAMPLES = 4
    correct_by_v = defaultdict(int)
    total_by_v = defaultdict(int)
    iou_continuous: list[float] = []
    iou_parsed = 0
    sample_responses: list[dict] = []
    samples_per_v: dict[str, int] = defaultdict(int)
    all_responses: list[dict] = []  # populated only when --save-all-responses

    # --- collapse/truncation metrics (collapse-metrics add) ------------------
    # Definitions are byte-identical to eval_ood_vllm.py
    # so these in-domain per-verifier numbers are directly comparable to the OOD
    # Tables 3 (seq-rep-4), 4 (truncation) and the collapse figures.
    import zlib as _zlib
    from collections import Counter as _Counter

    def _is_collapsed(raw: str) -> bool:
        """Content-based repetition-collapse flag: zlib compresses >=4x, OR a
        12-word window repeats >=5x. Identical to eval_ood_vllm.is_collapsed
        (the metric behind the collapse figures)."""
        t = raw or ""
        if not t:
            return False
        b = t.encode("utf-8", "ignore")
        if len(b) / max(1, len(_zlib.compress(b, 6))) >= 4.0:
            return True
        w = t.split()
        if len(w) < 24:
            return False
        g = [" ".join(w[i:i + 12]) for i in range(len(w) - 11)]
        return bool(g) and _Counter(g).most_common(1)[0][1] >= 5

    def _seq_rep_4(raw: str) -> float:
        """Welleck et al. (2020) seq-rep-4: 1 - |unique 4-grams|/|total 4-grams|,
        whitespace-tokenized. Standard-metric reimplementation (repo had none),
        comparable to Table 3 up to tokenization. 0.0 for <4-token responses."""
        w = (raw or "").split()
        if len(w) < 4:
            return 0.0
        grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        return 1.0 - len(set(grams)) / len(grams)

    trunc_len_by_v = defaultdict(int)     # finish_reason == "length"   (Table 4)
    trunc_think_by_v = defaultdict(int)   # no closing </think>         (paper prose / data_loader defn)
    collapsed_by_v = defaultdict(int)     # _is_collapsed               (collapse figures)
    seqrep_sum_by_v = defaultdict(float)  # sum of seq-rep-4            (Table 3)
    # -------------------------------------------------------------------------

    for i, (r, out) in enumerate(zip(df.itertuples(), outputs)):
        verifier = r.data_source
        gt = r.reward_model["ground_truth"]
        response = out.outputs[0].text  # greedy single completion
        # collapse/truncation flags for this response (collapse-metrics add)
        finish_reason = out.outputs[0].finish_reason
        _trunc_len = (finish_reason == "length")
        _trunc_think = ("</think>" not in (response or ""))
        _collapsed = _is_collapsed(response)
        _seqrep = _seq_rep_4(response)
        img = Image.open(io.BytesIO(r.images[0]["bytes"])).convert("RGB")

        result = compute_score(
            data_source=verifier,
            solution_str=response,
            ground_truth=gt,
            extra_info={"image": img},
        )
        if isinstance(result, dict):
            score = float(result.get("score", 0.0))
            # Count correctness off the scorer's honest `acc`, NOT score>=0.5.
            # `score` includes the +0.1 format bonus, which for iou pushes
            # 0.4<=IoU<0.5 over the 0.5 line and inflates iou acc ~1-3pp; `acc`
            # is binary pass@0.5 on raw IoU (and plain correctness for the binary
            # verifiers, identical to before). (eval fix 0602)
            acc = float(result.get("acc", 1.0 if score >= 0.5 else 0.0))
            iou_val = float(result.get("iou_continuous", 0.0))
        else:
            score = float(result)
            acc = 1.0 if score >= 0.5 else 0.0
            iou_val = 0.0

        total_by_v[verifier] += 1
        if acc >= 0.5:
            correct_by_v[verifier] += 1
        # collapse/truncation accumulation (over ALL responses, not just the
        # 4-per-verifier saved sample) (collapse-metrics add)
        trunc_len_by_v[verifier] += int(_trunc_len)
        trunc_think_by_v[verifier] += int(_trunc_think)
        collapsed_by_v[verifier] += int(_collapsed)
        seqrep_sum_by_v[verifier] += _seqrep
        if verifier == "iou":
            iou_continuous.append(iou_val)
            if iou_val > 0.0:
                iou_parsed += 1
        if samples_per_v[verifier] < PER_VERIFIER_SAMPLES:
            sample_responses.append({
                "verifier": verifier,
                "gt": gt,
                "response": response,           # NO truncation — vLLM eval keeps full responses
                "response_len_chars": len(response),
                "score": score,
                "acc": acc,                     # honest correctness (no +0.1 bonus)
                "finish_reason": finish_reason,           # (collapse-metrics add)
                "is_truncated": _trunc_len,               # finish_reason == "length" (Table 4)
                "is_truncated_nothink": _trunc_think,     # no closing </think>
                "is_collapsed": _collapsed,               # collapse figures
                "seq_rep_4": round(_seqrep, 4),           # Table 3
            })
            samples_per_v[verifier] += 1
        if args.save_all_responses:
            all_responses.append({
                "idx": i,
                "verifier": verifier,
                "gt": gt,
                "response": response,
                "response_len_chars": len(response),
                "score": score,
                "acc": acc,
                "finish_reason": finish_reason,           # (collapse-metrics add)
                "is_truncated": _trunc_len,
                "is_truncated_nothink": _trunc_think,
                "is_collapsed": _collapsed,
                "seq_rep_4": round(_seqrep, 4),
            })

    # Build result JSON
    result_json: dict = {}
    for v in ("digit", "mcq", "math", "iou"):
        n = total_by_v.get(v, 0)
        c = correct_by_v.get(v, 0)
        result_json[v] = {
            "correct": c, "total": n, "acc": round(c / n, 4) if n else None,
            # per-verifier collapse/truncation (rates in %, seq-rep-4 mean fraction) (collapse-metrics add)
            "trunc_rate": round(100 * trunc_len_by_v.get(v, 0) / n, 2) if n else None,
            "trunc_rate_nothink": round(100 * trunc_think_by_v.get(v, 0) / n, 2) if n else None,
            "collapse_rate": round(100 * collapsed_by_v.get(v, 0) / n, 2) if n else None,
            "seq_rep_4": round(seqrep_sum_by_v.get(v, 0.0) / n, 4) if n else None,
        }
    if iou_continuous:
        n_iou = len(iou_continuous)
        result_json["iou"]["mean_iou"] = round(sum(iou_continuous) / n_iou, 4)
        if iou_parsed:
            parsed_vals = [v for v in iou_continuous if v > 0.0]
            result_json["iou"]["mean_iou_parsed"] = round(sum(parsed_vals) / len(parsed_vals), 4)
            result_json["iou"]["parsed_count"] = iou_parsed

    total_n = sum(total_by_v.values())
    total_c = sum(correct_by_v.values())
    _tl = sum(trunc_len_by_v.values()); _tt = sum(trunc_think_by_v.values())
    _cl = sum(collapsed_by_v.values()); _sr = sum(seqrep_sum_by_v.values())
    result_json["overall"] = {
        "correct": total_c, "total": total_n,
        "acc": round(total_c / total_n, 4) if total_n else None,
        # aggregate collapse/truncation across all verifiers (collapse-metrics add)
        "trunc_rate": round(100 * _tl / total_n, 2) if total_n else None,
        "trunc_rate_nothink": round(100 * _tt / total_n, 2) if total_n else None,
        "collapse_rate": round(100 * _cl / total_n, 2) if total_n else None,
        "seq_rep_4": round(_sr / total_n, 4) if total_n else None,
    }
    result_json["samples"] = sample_responses
    if args.save_all_responses:
        result_json["all_responses"] = all_responses
    if args.full_ft_checkpoint:
        result_json["checkpoint"] = str(args.full_ft_checkpoint)
    elif args.checkpoint and not args.no_adapter:
        result_json["checkpoint"] = str(args.checkpoint)
    else:
        result_json["checkpoint"] = "BASE_NO_ADAPTER"
    result_json["meta"] = {
        "full_ft": bool(args.full_ft_checkpoint),
        "merged_dir": str(args.merged_dir) if args.full_ft_checkpoint else None,
        "max_new_tokens": args.max_new_tokens,
        "max_examples": args.max_examples,
        "max_model_len": args.max_model_len,
        "max_lora_rank": max_lora_rank,
        "generation_seconds": round(gen_elapsed, 1),
        "total_seconds": round(time.time() - t0, 1),
        # Subsampling comparability record.
        "sampling": {
            "method": ("stratified_by_verifier_proportional"
                       if (args.max_examples and 0 < args.max_examples < total_n_full)
                       else "full"),
            "n": args.max_examples,
            "seed": 42 if args.max_examples else None,
        },
        **eval_provenance(
            engine="vllm",
            model_path=args.base_model,
            adapter_path=adapter_dir,
            gen_params={"decoding": "greedy", "temperature": 0.0, "top_p": 1.0,
                        "max_new_tokens": args.max_new_tokens, "seed": None},
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result_json, indent=2))
    tmp.replace(out_path)
    print(f"[eval-vllm] wrote {out_path}", flush=True)

    print("=" * 60, flush=True)
    print("PER-VERIFIER ACCURACY + COLLAPSE:", flush=True)
    for v in ("digit", "mcq", "math", "iou"):
        d = result_json[v]
        print(f"  {v:6s}: acc {d['correct']:>3d}/{d['total']:>3d} = {d['acc']}  "
              f"trunc {d['trunc_rate']}%  collapse {d['collapse_rate']}%  "
              f"seq-rep-4 {d['seq_rep_4']}", flush=True)
    od = result_json["overall"]
    print(f"  OVERALL: acc {od['correct']}/{od['total']} = {od['acc']}  "
          f"trunc {od['trunc_rate']}%  collapse {od['collapse_rate']}%  "
          f"seq-rep-4 {od['seq_rep_4']}", flush=True)
    print(f"  generation time: {gen_elapsed:.1f}s ({gen_elapsed/total_n:.2f}s/prompt)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
