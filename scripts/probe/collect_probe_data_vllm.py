"""vLLM-batched version of collect_probe_data.py.

Two-phase pipeline to work around vLLM not exposing per-head attention:
  Phase A: vLLM batched generation of all (prompt × N rollouts) → responses in memory
  Phase B: cleanup vLLM, load HF LookbackExtractor, extract 448-dim attention features
  Phase C: save features.npz (X, y, prompt_id, verifier_type)

Output schema matches the original collect_probe_data.py so train_probe.py reads it unchanged.

Speedup vs the HF version: vLLM continuous batching processes all
(prompt × n_rollouts) inputs concurrently instead of one prompt at a time.
On a 250-prompt × 8-rollout math shard: HF takes ~24h, vLLM ~30-60 min for
phase A. Phase B (extraction) is still HF but is fast per-rollout (~1-2s each),
so total wall is ~1.5-2h per shard instead of ~24h.
"""
from __future__ import annotations
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Fragmentation-resistant allocator (replaces the old per-prompt empty_cache
# crutch). Must be set before torch first allocates; vLLM manages its own pool.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Probe labels must use the SAME correctness definition as training/eval.
# The run convention is strict (tag-required, no fallback) — default it here so
# a collection run can't silently label with lenient extraction while the
# trained-against scorer is strict. Override with STRICT_CORRECTNESS=0.
os.environ.setdefault("STRICT_CORRECTNESS", "1")

import argparse
import gc
import pickle
import sys
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verl_faith"))


def _gen_seed(base: int, prompt_id: int, temp_group_idx: int) -> int:
    """Deterministic per-request sampling seed. zlib.crc32 (stable across
    processes) — NOT Python hash(), which is salted per-process. rollout_index
    is subsumed: one seeded vLLM request draws its n samples from a single
    per-request generator stream, so (prompt_id, temp_group) pins all of them."""
    return zlib.crc32(f"{base}|{prompt_id}|{temp_group_idx}".encode()) & 0x7FFFFFFF


def _build_messages(image: Image.Image, question: str, vl_thinking_system: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": vl_thinking_system}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = no cap")
    ap.add_argument("--filter-verifier", default=None)
    ap.add_argument("--per-verifier", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    # NO-SKEW INVARIANT: must equal FAITH_SERVE_WINDOW (scorers.py) and the RL
    # data.max_response_length so the probe is collected and served on the SAME
    # response window (the feature is a mean over response tokens). All three = 4096.
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temperatures", default=None,
                    help="comma list of temps to sample across, len = n_rollouts. "
                         "Phase-1 default: '0.7,1.0,1.0,1.0,1.0,1.0,1.3,1.3' for n=8. "
                         "Overrides --temperature when set.")
    # REQUIRED (no silent 2B default): the probe feature dim is model-specific
    # (layers×heads), so a defaulted --model silently collects features for the
    # wrong model size. The 8B launcher passes it explicitly.
    ap.add_argument("--model", required=True,
                    help="HF model id/path, e.g. Qwen/Qwen3-VL-8B-Thinking (REQUIRED)")
    ap.add_argument("--gen-seed-base", type=int, default=42,
                    help="base for deterministic per-request sampling seeds "
                         "(crc32(base|prompt_id|temp_group_idx)); negative disables. "
                         "Forced OFF when resuming a legacy (v1-fingerprint) partial "
                         "so an in-flight run keeps the regime it started with.")
    ap.add_argument("--max-model-len", type=int, default=8192)  # prompt(<=4096)+resp(4096); matches RL max_model_len
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--no-batch", action="store_true",
                    help="disable batched phase-B extraction (extract_batch: one forward "
                         "per PROMPT) and use one forward per rollout instead")
    ap.add_argument("--adapter", default=None,
                    help="Optional verl checkpoint dir (e.g. .../global_step_240). When set, "
                         "Phase A (vLLM generation) loads the LoRA adapter from "
                         "<checkpoint>/actor/lora_adapter so rollouts come from the tuned "
                         "policy. Phase B (HF attention extraction) still uses BASE model — "
                         "matches training-time probe scoring, which is what we're auditing.")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # PROMPT/SCORER CONSISTENCY GUARD: under strict payload-exact scoring the
    # parquet must be the v3 exact-payload format — otherwise the probe is
    # collected on prompts that don't state the rule its labels enforce, and
    # serve-time prompts (v3) would differ from collection prompts (feature
    # skew). Same guard as the training launchers. Override for deliberate
    # legacy collection with COLLECT_ALLOW_FORMAT_MISMATCH=1.
    _strict_on = os.environ.get("STRICT_CORRECTNESS", "") not in ("", "0", "false", "False", "no", "NO")
    stamp = Path(args.parquet).resolve().parent / "parquet_format_version.json"
    if _strict_on and os.environ.get("COLLECT_ALLOW_FORMAT_MISMATCH", "0") == "0":
        if not (stamp.is_file() and "v3-exact-payload" in stamp.read_text()):
            raise SystemExit(
                f"FATAL: {stamp} missing or pre-v3 while STRICT_CORRECTNESS is on.\n"
                f"Regenerate the parquet (prep_vlaa_subset.py --stage parquet) so the\n"
                f"prompts state the exact-payload rule the strict labels enforce, or set\n"
                f"COLLECT_ALLOW_FORMAT_MISMATCH=1 to override deliberately.")

    # Load parquet, filter, shard — same logic as the HF script.
    import pyarrow.parquet as pq
    import pandas as pd
    df = pq.read_table(args.parquet).to_pandas()
    if args.filter_verifier:
        df = df[df["data_source"] == args.filter_verifier].reset_index(drop=True)
        print(f"filtered to verifier={args.filter_verifier}: {len(df)} rows", flush=True)
    if args.per_verifier > 0:
        rng = np.random.default_rng(args.seed)
        parts = []
        for v, sub in df.groupby("data_source"):
            n = min(args.per_verifier, len(sub))
            idx = rng.choice(sub.index.values, size=n, replace=False)
            parts.append(df.loc[idx])
        df = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    elif args.max_prompts > 0:
        df = df.iloc[: args.max_prompts].reset_index(drop=True)
    if args.num_shards > 1:
        df = df.iloc[args.shard_idx :: args.num_shards].reset_index(drop=True)
        print(f"shard {args.shard_idx}/{args.num_shards}: {len(df)} rows | "
              f"counts={df['data_source'].value_counts().to_dict()}", flush=True)

    # Build per-prompt inputs.
    from verl_faith.lookback import _VL_THINKING_SYSTEM
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)

    prompts_data = []  # list of (idx, verifier, gt, image, question, prompt_text)
    for idx, row in df.iterrows():
        verifier = row["data_source"]
        gt = row["reward_model"]["ground_truth"]
        img = Image.open(BytesIO(row["images"][0]["bytes"])).convert("RGB")
        user_content = row["prompt"][1]["content"]
        question = user_content[len("<image>\n"):] if user_content.startswith("<image>\n") else user_content
        messages = _build_messages(img, question, _VL_THINKING_SYSTEM)
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts_data.append((int(idx), str(verifier), gt, img, question, prompt_text))
    print(f"prepared {len(prompts_data)} prompts", flush=True)

    # ── Crash/hang resilience: incremental checkpoint + resume ───────────────
    # Phase B checkpoints every 25 prompts to <out>.partial.npz. On restart with
    # the SAME sampling config, already-done prompts are skipped in BOTH phases
    # (their rollouts come from a different generation pass than the original —
    # fine: every prompt's rollouts are still i.i.d. samples from the same
    # policy/temps). Motivated 2026-07-04: a math_verify hang at row ~896 cost
    # the full 16h math run because features lived only in memory.
    # v1 fingerprint (legacy): dataset-selection fields only. v2 additionally
    # pins everything that changes the sampled distribution — an in-flight run's
    # v1 partial is still resumable (with a loud warning about the unchecked
    # fields), while new runs get the full check.
    _fp_v1 = (f"parquet={args.parquet}|filter={args.filter_verifier}|"
              f"pv={args.per_verifier}|mp={args.max_prompts}|seed={args.seed}|"
              f"shard={args.shard_idx}/{args.num_shards}|n={len(prompts_data)}")
    _temps_fp = args.temperatures if args.temperatures else f"single:{args.temperature}"
    _fp_v2 = (f"v2|{_fp_v1}|model={args.model}|adapter={args.adapter}|"
              f"temps={_temps_fp}|nr={args.n_rollouts}|mnt={args.max_new_tokens}|"
              f"gsb={args.gen_seed_base}")
    _fingerprint = _fp_v2  # what checkpoints get stamped with (downgraded to v1 on legacy resume)
    seed_mode = args.gen_seed_base >= 0
    partial_path = Path(str(args.out) + ".partial.npz")
    n_done = 0
    _resume = {"X": [], "y": [], "y_lenient": [], "formatted": [],
               "prompt_id": [], "verifier_type": []}
    feature_path_flags: list[int] = []   # per-row: 1 = batched extraction, 0 = serve path
    kept_lens: list[int] = []            # response char-lengths, kept rows
    skipped_lens: list[int] = []         # response char-lengths, skipped rows
    n_skipped_extract = 0
    n_skipped_score = 0
    resumed_parity_failed = 0
    if partial_path.is_file():
        d = np.load(partial_path, allow_pickle=True)
        stored_fp = str(d.get("fingerprint"))
        fp_ok = False
        if stored_fp == _fp_v2:
            fp_ok = True
        elif stored_fp == _fp_v1:
            fp_ok = True
            _fingerprint = _fp_v1  # keep stamping v1 so later resumes of THIS run still match
            print("=" * 78, flush=True)
            print(f"[resume][WARN] {partial_path} carries a LEGACY v1 fingerprint. It matches\n"
                  f"on dataset-selection fields only — the following were NOT checked against\n"
                  f"the original launch and are YOUR responsibility to verify:\n"
                  f"  --model={args.model}  --adapter={args.adapter}\n"
                  f"  --temperatures={_temps_fp}  --n-rollouts={args.n_rollouts}\n"
                  f"  --max-new-tokens={args.max_new_tokens}", flush=True)
            print("=" * 78, flush=True)
            if seed_mode:
                seed_mode = False
                print("[resume] legacy partial detected — per-request sampling seeds OFF for "
                      "this run (keeps the sampling regime the run started with)", flush=True)
        else:
            print(f"[resume] {partial_path} fingerprint mismatch — ignoring it "
                  f"(collecting from scratch)", flush=True)
        if fp_ok:
            n_done = int(d["n_prompts_done"])
            for k in _resume:
                key = "X" if k == "X" else k
                _resume[k] = list(d[key])
            if "feature_path" in d.files:
                feature_path_flags = [int(v) for v in d["feature_path"]]
            else:
                fill = 0 if args.no_batch else 1
                feature_path_flags = [fill] * len(_resume["y"])
                print(f"[resume] old checkpoint lacks feature_path — assuming {fill} "
                      f"({'serve' if fill == 0 else 'batch'}) for the {len(_resume['y'])} "
                      f"resumed rows", flush=True)
            if "kept_lens" in d.files:
                kept_lens = [int(v) for v in d["kept_lens"]]
            if "skipped_lens" in d.files:
                skipped_lens = [int(v) for v in d["skipped_lens"]]
            if "n_skipped_extract" in d.files:
                n_skipped_extract = int(d["n_skipped_extract"])
            if "n_skipped_score" in d.files:
                n_skipped_score = int(d["n_skipped_score"])
            if "parity_failed" in d.files:
                resumed_parity_failed = int(d["parity_failed"])
            print(f"[resume] {partial_path}: {n_done}/{len(prompts_data)} prompts "
                  f"already done ({len(_resume['y'])} features) — skipping them",
                  flush=True)
    print(f"[gen-seed] per-request sampling seeds "
          f"{'ON (base=%d)' % args.gen_seed_base if seed_mode else 'OFF'}", flush=True)
    gen_prompts = prompts_data[n_done:]
    if not gen_prompts:
        print("[resume] nothing left to collect — writing final output", flush=True)

    # === Phase A: vLLM batched generation ===
    import time
    print(f"\n=== Phase A: vLLM generation ({len(prompts_data)} prompts × {args.n_rollouts} rollouts) ===", flush=True)
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        limit_mm_per_prompt={"image": 1},
    )
    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        ck = Path(args.adapter)
        adapter_dir = None
        for cand in (ck / "actor" / "lora_adapter", ck / "lora_adapter"):
            if (cand / "adapter_model.safetensors").is_file():
                adapter_dir = cand
                break
        if adapter_dir is None:
            raise SystemExit(f"adapter dir missing under {ck}")
        print(f"[adapter] loading from {adapter_dir}", flush=True)
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 16
        lora_request = LoRARequest(lora_name="adapter", lora_int_id=1,
                                   lora_path=str(adapter_dir))
    llm = LLM(**llm_kwargs)
    vllm_inputs = [{"prompt": pd[5], "multi_modal_data": {"image": pd[3]}}
                   for pd in gen_prompts]
    print(f"vllm load time: {time.time() - t0:.1f}s", flush=True)

    # Multi-temperature: group rollouts by unique temperature value, call llm.generate
    # once per temp group with n=count_at_that_temp. Concatenate responses per prompt.
    # Single-temperature: one call with n=n_rollouts at args.temperature.
    if args.temperatures:
        temps = [float(t) for t in args.temperatures.split(",")]
        if len(temps) != args.n_rollouts:
            print(f"[warn] --temperatures has {len(temps)} entries but --n-rollouts={args.n_rollouts}; "
                  f"using len(temperatures) as effective n_rollouts", flush=True)
        from collections import Counter
        temp_counts = Counter(temps)
        print(f"[multi-temp] groups: {dict(temp_counts)}", flush=True)

        # Per-prompt list of responses, built incrementally across temp groups.
        responses_per_prompt: list[list[str]] = [[] for _ in range(len(gen_prompts))]
        gen_t0 = time.time()
        for temp_idx, (temp_val, count) in enumerate(sorted(temp_counts.items())):
            if seed_mode:
                # One SamplingParams per prompt: deterministic per-request seed
                # keyed on the STABLE prompt id (pd[0], the parquet row index —
                # resume-invariant because dataset selection is fingerprinted).
                sp = [SamplingParams(
                          temperature=temp_val, top_p=1.0,
                          max_tokens=args.max_new_tokens, n=count,
                          seed=_gen_seed(args.gen_seed_base, pd[0], temp_idx))
                      for pd in gen_prompts]
            else:
                sp = SamplingParams(
                    temperature=temp_val, top_p=1.0,
                    max_tokens=args.max_new_tokens, n=count,
                )
            print(f"[multi-temp] generating temp={temp_val} (n={count}/prompt, "
                  f"seeded={seed_mode})", flush=True)
            outputs = llm.generate(vllm_inputs, sampling_params=sp, lora_request=lora_request)
            for prompt_idx, out in enumerate(outputs):
                for o in out.outputs:
                    responses_per_prompt[prompt_idx].append(o.text)
            del outputs
        gen_elapsed = time.time() - gen_t0
        n_total_rollouts = sum(len(r) for r in responses_per_prompt)
    else:
        if seed_mode:
            sampling_params = [SamplingParams(
                                   temperature=args.temperature, top_p=1.0,
                                   max_tokens=args.max_new_tokens, n=args.n_rollouts,
                                   seed=_gen_seed(args.gen_seed_base, pd[0], 0))
                               for pd in gen_prompts]
        else:
            sampling_params = SamplingParams(
                temperature=args.temperature, top_p=1.0,
                max_tokens=args.max_new_tokens, n=args.n_rollouts,
            )
        gen_t0 = time.time()
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params, lora_request=lora_request)
        gen_elapsed = time.time() - gen_t0
        responses_per_prompt = [[o.text for o in out.outputs] for out in outputs]
        n_total_rollouts = len(gen_prompts) * args.n_rollouts
        del outputs

    print(f"vllm gen done in {gen_elapsed:.1f}s "
          f"({gen_elapsed / max(n_total_rollouts,1):.2f}s/rollout, "
          f"{gen_elapsed / max(len(prompts_data),1):.2f}s/prompt)",
          flush=True)

    # Crash insurance: dump the phase-A generations (hours of GPU work) to disk
    # BEFORE the vLLM→HF handover, so a crash during extraction/scoring can't
    # lose them. Schema v2 (ALIGNED PAIRS): prompts_meta[i] ↔
    # responses_per_prompt[i] for exactly the prompts generated IN THIS PROCESS.
    # On a resume the first n_done prompts' responses were never persisted (only
    # their features, in the .partial.npz), so they CANNOT be included — the old
    # v1 bundle listed meta for ALL prompts against responses for only the
    # remainder, mispairing every row when phase_b consumed it.
    # collect_probe_phase_b.py verifies the alignment invariant and refuses
    # misaligned legacy bundles.
    out_path = args.out if args.out.endswith(".npz") else args.out + ".npz"
    resp_pkl = Path(out_path).with_suffix(".responses.pkl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    resp_meta = [(idx, verifier, gt, img, question)
                 for idx, verifier, gt, img, question, _ in gen_prompts]
    assert len(resp_meta) == len(responses_per_prompt), (
        f"internal alignment bug: {len(resp_meta)} meta vs "
        f"{len(responses_per_prompt)} response lists")
    _pkl_tmp = str(resp_pkl) + ".tmp"
    with open(_pkl_tmp, "wb") as f:
        pickle.dump({"schema_version": 2,
                     "prompts_meta": resp_meta,
                     "responses_per_prompt": responses_per_prompt,
                     "n_prompts_resumed_skip": n_done,
                     "n_prompts_total": len(prompts_data),
                     "model": args.model,
                     "max_new_tokens": args.max_new_tokens}, f)
    os.replace(_pkl_tmp, resp_pkl)
    print(f"saved phase-A generations to {resp_pkl} "
          f"(schema v2, {len(resp_meta)} aligned pairs, resumed-skip={n_done})", flush=True)

    # === Cleanup vLLM ===
    print("\n=== Cleaning up vLLM (freeing GPU mem before HF) ===", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # === Phase B: HF attention extraction + scoring ===
    print(f"\n=== Phase B: HF attention extraction ===", flush=True)
    ext_t0 = time.time()
    from verl_faith.lookback import LookbackExtractor
    from verl_faith.scorers import compute_score

    extractor = LookbackExtractor(model_path=args.model)
    print(f"hf load time: {time.time() - ext_t0:.1f}s", flush=True)

    def _score_both(verifier, response, gt, img):
        """Score once under the live (strict) mode and once lenient.

        y (training label) comes from the live mode. The lenient
        content-correctness column is saved alongside for ANALYSIS ONLY: the
        strict label conflates "knows the answer" with "complied with the
        format", so the strict-vs-lenient gap per verifier quantifies how much
        label mass is right-but-misformatted (probe-label noise on exactly the
        axis the probe measures). It does NOT affect training labels."""
        out = compute_score(data_source=verifier, solution_str=response,
                            ground_truth=gt, extra_info={"image": img})
        prev = os.environ.get("STRICT_CORRECTNESS")
        os.environ["STRICT_CORRECTNESS"] = "0"
        try:
            out_len = compute_score(data_source=verifier, solution_str=response,
                                    ground_truth=gt, extra_info={"image": img})
        finally:
            if prev is None:
                os.environ.pop("STRICT_CORRECTNESS", None)
            else:
                os.environ["STRICT_CORRECTNESS"] = prev
        return out, out_len

    feats: list[np.ndarray] = _resume["X"]
    labels: list[int] = [int(v) for v in _resume["y"]]
    labels_lenient: list[int] = [int(v) for v in _resume["y_lenient"]]
    formatted_flags: list[int] = [int(v) for v in _resume["formatted"]]
    prompt_ids: list[int] = [int(v) for v in _resume["prompt_id"]]
    verifier_types: list[str] = [str(v) for v in _resume["verifier_type"]]

    def _save_checkpoint(n_prompts_done):
        tmp = str(partial_path) + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f,
                     X=np.stack(feats, axis=0).astype(np.float32) if feats else
                         np.zeros((0, extractor.feat_dim), dtype=np.float32),
                     y=np.array(labels, dtype=np.int64),
                     y_lenient=np.array(labels_lenient, dtype=np.int64),
                     formatted=np.array(formatted_flags, dtype=np.int64),
                     prompt_id=np.array(prompt_ids, dtype=np.int64),
                     verifier_type=np.array(verifier_types),
                     feature_path=np.array(feature_path_flags, dtype=np.int64),
                     kept_lens=np.array(kept_lens, dtype=np.int64),
                     skipped_lens=np.array(skipped_lens, dtype=np.int64),
                     n_skipped_extract=n_skipped_extract,
                     n_skipped_score=n_skipped_score,
                     parity_failed=int(batch_state["parity_failed"]),
                     model=args.model,
                     n_prompts_done=n_prompts_done, fingerprint=_fingerprint)
        os.replace(tmp, partial_path)

    # Batched phase B with a PERIODIC parity audit against the serve path
    # (extract_efficient). The padded batch path can diverge on long sequences
    # (empirically: mcq failed at 0.044 while iou passed at 0.003) — so we
    # re-check every COLLECT_PARITY_EVERY prompts and, on the first failure,
    # PERMANENTLY switch this run to per-rollout extraction (the serve path,
    # exact by construction) using the reference features we just computed.
    # No hard exit: an unattended overnight chain degrades to slow-but-correct.
    # parity_failed persists across resume: once a run has recorded a parity
    # failure, resumed segments stay on the serve path and the flag stays set
    # (rows are flagged per-row in feature_path; _fit_probe_variants
    # --drop-batch-path filters them downstream).
    batch_state = {"use_batch": (not args.no_batch) and not resumed_parity_failed,
                   "checked_at": None,
                   "parity_failed": resumed_parity_failed}
    if resumed_parity_failed and not args.no_batch:
        print("[batch-parity] resumed run previously FAILED parity — staying on "
              "per-rollout (serve-path) extraction", flush=True)
    PARITY_EVERY = max(1, int(os.environ.get("COLLECT_PARITY_EVERY", "100")))
    PARITY_TOL = float(os.environ.get("COLLECT_PARITY_TOL", "5e-3"))

    def _per_rollout(img, question, rollouts, idx):
        out = []
        for r_idx, response in enumerate(rollouts):
            try:
                out.append(extractor.extract_efficient(
                    image=img, question=question, response=response,
                    max_response_tokens=args.max_new_tokens))
            except Exception as exc:
                print(f"[skip] row {idx} rollout {r_idx}: extract failed: {exc}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                out.append(None)
        return out

    def _extract_prompt_feats(img, question, rollouts, idx, prompt_i):
        """Returns (per-rollout feature list, path_flag): 1 = batched extraction,
        0 = per-rollout serve path. The flag is stored per-row in the npz so a
        later parity failure lets downstream (_fit_probe_variants
        --drop-batch-path) drop every batch-path row, not just the failing one."""
        if not batch_state["use_batch"]:
            return _per_rollout(img, question, rollouts, idx), 0
        try:
            feats_b = extractor.extract_batch(
                image=img, question=question, responses=list(rollouts),
                max_response_tokens=args.max_new_tokens)
        except Exception as exc:
            print(f"[batch-fallback] row {idx}: extract_batch failed ({exc}); "
                  f"per-rollout for this prompt", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _per_rollout(img, question, rollouts, idx), 0
        due = (batch_state["checked_at"] is None
               or prompt_i - batch_state["checked_at"] >= PARITY_EVERY)
        if due:
            ref = _per_rollout(img, question, rollouts, idx)
            pairs = [(a, b) for a, b in zip(feats_b, ref) if b is not None]
            dmax = max((float(np.abs(a - b).max()) for a, b in pairs), default=0.0)
            prev_checked = batch_state["checked_at"]
            batch_state["checked_at"] = prompt_i
            print(f"[batch-parity] prompt {prompt_i}: max|batch - efficient| = {dmax:.6f}",
                  flush=True)
            if dmax > PARITY_TOL:
                since = 0 if prev_checked is None else prev_checked + 1
                print(f"[batch-parity] FAIL (> {PARITY_TOL}) — switching to PER-ROLLOUT "
                      f"extraction for the rest of this run (serve-path exact); batch "
                      f"features for this prompt discarded in favor of the reference.\n"
                      f"[batch-parity] SUSPECT RANGE: batch-path rows from prompts "
                      f"{since}..{prompt_i - 1} were NOT re-checked since the last passing "
                      f"audit; ALL rows with feature_path=1 in the npz are flagged suspect "
                      f"(parity_failed=1) — _fit_probe_variants.py --drop-batch-path "
                      f"(default ON) drops them.", flush=True)
                batch_state["use_batch"] = False
                batch_state["parity_failed"] = 1
                return ref, 0  # already computed on the serve path
        return feats_b, 1

    phase_b_start = time.time()
    for prompt_i in range(n_done, len(prompts_data)):
        idx, verifier, gt, img, question, _ptext = prompts_data[prompt_i]
        rollouts = responses_per_prompt[prompt_i - n_done]
        prompt_feats, path_flag = _extract_prompt_feats(img, question, rollouts, idx, prompt_i)
        for r_idx, (response, feat) in enumerate(zip(rollouts, prompt_feats)):
            if feat is None:
                n_skipped_extract += 1
                skipped_lens.append(len(response))
                continue
            try:
                score_out, lenient_out = _score_both(verifier, response, gt, img)
            except Exception as exc:
                # Skip the whole row (feature+label+ids together) so arrays stay
                # aligned — a single bad ground_truth must not kill the verifier.
                print(f"[skip] row {idx} rollout {r_idx}: scoring failed: {exc}", flush=True)
                n_skipped_score += 1
                skipped_lens.append(len(response))
                continue
            # Label from `acc` (honest correctness), NOT `score` (the reward):
            # score includes the +0.1 format bonus and, for iou, is the
            # CONTINUOUS IoU — so `score >= 0.5` mislabeled formatted iou
            # rollouts with IoU in [0.4, 0.5) as correct. The probe predicts
            # correctness, so it must be trained on correctness.
            score = float(score_out["acc"]) if isinstance(score_out, dict) else float(score_out)
            lenient_acc = float(lenient_out["acc"]) if isinstance(lenient_out, dict) else float(lenient_out)
            fmt_flag = int(score_out.get("formatted", 0)) if isinstance(score_out, dict) else 0
            feats.append(feat)
            labels.append(int(score >= 0.5))
            labels_lenient.append(int(lenient_acc >= 0.5))
            formatted_flags.append(fmt_flag)
            prompt_ids.append(idx)
            verifier_types.append(str(verifier))
            feature_path_flags.append(path_flag)
            kept_lens.append(len(response))
        # empty_cache per PROMPT was a 2B/24GB-era OOM crutch: on an H200 with
        # ~120GB free in phase B it only costs allocator churn (~5-15% of phase
        # B). expandable_segments (set at import) handles fragmentation instead;
        # throttled to the checkpoint cadence as belt-and-suspenders.
        if (prompt_i + 1 - n_done) % 25 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _save_checkpoint(prompt_i + 1)

        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        elapsed = time.time() - phase_b_start
        rate = (prompt_i + 1 - n_done) / max(elapsed, 1e-6)
        eta_s = (len(prompts_data) - prompt_i - 1) / max(rate, 1e-6)
        print(f"row {prompt_i + 1}/{len(prompts_data)} idx={idx} | collected={len(labels)} | "
              f"pos={n_pos} neg={n_neg} | rate={rate * 60:.1f} rows/min | eta={eta_s/60:.1f}min",
              flush=True)

    if not feats:
        print(f"WARNING: zero usable rows (all extractions/scorings failed) — nothing to save; "
              f"generations are preserved in {resp_pkl}", flush=True)
        sys.exit(1)

    X = np.stack(feats, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    y_len = np.array(labels_lenient, dtype=np.int64)
    fmt_arr = np.array(formatted_flags, dtype=np.int64)
    pid = np.array(prompt_ids, dtype=np.int64)
    vtypes = np.array(verifier_types)
    # Atomic write: tmp + os.replace so a mid-write crash can't leave a truncated
    # npz that the launcher's skip-if-exists would then treat as done.
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez(f, X=X, y=y, prompt_id=pid, verifier_type=vtypes,
                 y_lenient=y_len, formatted=fmt_arr,
                 feature_path=np.array(feature_path_flags, dtype=np.int64),
                 parity_failed=int(batch_state["parity_failed"]),
                 model=args.model,
                 n_skipped_extract=n_skipped_extract,
                 n_skipped_score=n_skipped_score)
    os.replace(tmp_path, out_path)
    partial_path.unlink(missing_ok=True)
    print(f"\nsaved {out_path} | shape={X.shape} | "
          f"pos={int(y.sum())}/{len(y)} ({y.mean():.3f}) | unique_prompts={len(set(prompt_ids))}",
          flush=True)
    # Format-compliance gap per verifier: lenient-correct minus strict-correct
    # = rollouts that KNOW the answer but broke the payload format. A large gap
    # means the strict probe labels carry format noise on that verifier — and a
    # very low strict pos-rate starves the per-verifier balancer in train_probe.
    print("[label-audit] per-verifier: strict_pos / lenient_pos / formatted_rate (n)", flush=True)
    for v in sorted(set(verifier_types)):
        m = vtypes == v
        print(f"  {v:6s} {y[m].mean():.3f} / {y_len[m].mean():.3f} / {fmt_arr[m].mean():.3f} "
              f"(n={int(m.sum())})  gap={y_len[m].mean() - y[m].mean():+.3f}", flush=True)
    # Skip audit: skips are NOT missing-at-random (extraction/scoring failures
    # correlate with response length, and length correlates with the label), so
    # report length stats of skipped vs kept and hard-fail on a high skip rate.
    n_skipped = n_skipped_extract + n_skipped_score
    n_attempted = len(labels) + n_skipped
    skip_rate = n_skipped / max(n_attempted, 1)

    def _lenstats(v):
        if not v:
            return "n=0"
        a = np.asarray(v)
        return (f"n={len(a)} mean={a.mean():.0f} p50={np.median(a):.0f} "
                f"min={a.min()} max={a.max()}")
    print(f"[skip-audit] attempted={n_attempted} kept={len(labels)} "
          f"skipped_extract={n_skipped_extract} skipped_score={n_skipped_score} "
          f"skip_rate={skip_rate:.3f}", flush=True)
    print(f"[skip-audit] kept    resp chars: {_lenstats(kept_lens)}", flush=True)
    print(f"[skip-audit] skipped resp chars: {_lenstats(skipped_lens)}", flush=True)
    print(f"total time: phase A {gen_elapsed:.1f}s + phase B {time.time() - phase_b_start:.1f}s", flush=True)
    SKIP_RATE_MAX = float(os.environ.get("SKIP_RATE_MAX", "0.10"))
    if skip_rate > SKIP_RATE_MAX:
        print(f"FATAL: skip_rate {skip_rate:.3f} > SKIP_RATE_MAX {SKIP_RATE_MAX} — the "
              f"saved npz is written but length-correlated skips at this rate bias the "
              f"probe labels; investigate the [skip] log lines before using it. "
              f"(Override the gate with SKIP_RATE_MAX.)", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
