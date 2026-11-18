"""vLLM-batched OOD eval for phase-2 checkpoints against phase-1 medical datasets
(SLAKE / VQARAD / PathVQA), using phase-1's prompt template and token-F1 scorer.

Why this exists: phase-1's eval_test_set.py / eval_crossdataset.py use HF
model.generate() in a one-example-at-a-time loop (~12-14 s/example × 1061 SLAKE
= ~3.5 h). This script does the same eval with vLLM continuous batching for the
same numeric output (matching prompt + scorer), ~10-30× faster.

Numbers from this script are directly comparable to phase-1's paper_table
results (e.g. slake_tiebreak_bboxfree_nodrop.json) only when --prompt-style is
phase1_medical (the default).

Usage:
    python eval_ood_vllm.py \\
        --checkpoint checkpoints/ft_vanilla/global_step_150 \\
        --dataset slake \\
        --output results/eval/ood/vanilla_S150__slake.json \\
        --gpu 7

    # Quick directional read (matches phase-1 n=300 subset convention, seed=42):
    python eval_ood_vllm.py --checkpoint ... --dataset slake \\
        --max-examples 300 --output ... --gpu 7

    # Base model baseline:
    python eval_ood_vllm.py --no-adapter --dataset vqarad --output ... --gpu 7
"""
from __future__ import annotations

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import sys
import time
from pathlib import Path

# Reuse vLLM bring-up + LoRA adapter resolution + GPU pinning from the
# in-domain eval script (same repo checkout).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_adapter_vllm import (  # noqa: E402
    build_llm, pin_gpu, read_lora_rank, resolve_adapter_dir, merge_fullft_checkpoint,
)

# Phase-1 sources (token-F1 scorer + dataset loaders).
PHASE1_PYTHONPATHS = [p for p in os.environ.get("FAITHSCAN_PYTHONPATH", "").split(os.pathsep) if p]

VL_THINKING_SYSTEM = (
    "You are VL-Thinking\U0001F914, a helpful assistant with excellent reasoning ability. "
    "A user asks you a question, and you should try to solve it. You should first think "
    "about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>"
)

# Verbatim from phase-1 eval_test_set.py / eval_crossdataset.py.
PHASE1_USER_TEMPLATE = (
    "Answer this medical question. Think step by step, then give your "
    "final answer in <answer>...</answer> tags.\nQuestion: {question}"
)


def load_slake(slake_dir: str, organ_only: bool):
    """Returns list of dicts with keys: question, answer, image (PIL)."""
    from PIL import Image
    slake_path = Path(slake_dir)
    with open(slake_path / "test.json") as f:
        questions = json.load(f)
    questions = [q for q in questions if q.get("q_lang") == "en"]

    if organ_only:
        all_organs = set()
        for img_dir in (slake_path / "imgs").iterdir():
            det_path = img_dir / "detection.json"
            if det_path.exists():
                with open(det_path) as f:
                    dets = json.load(f)
                for det in dets:
                    for organ in det:
                        all_organs.add(organ.lower())
        questions = [
            q for q in questions
            if any(o in (q["question"].lower() + " " + q["answer"].lower()) for o in all_organs)
        ]

    examples = []
    for q in questions:
        img_path = slake_path / "imgs" / q["img_name"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        examples.append({
            "question": q["question"],
            "answer": q["answer"],
            "image": image,
        })
    return examples


def load_crossdataset(name: str):
    """Returns list of dicts (same schema as load_slake). Self-contained HF branches
    (pope/mathvista/mathvision) load directly; vqarad/pathvqa use faithscan. (Folded
    in from the former patch_* scripts so a fresh checkout is self-sufficient.)"""
    import os
    if name == "pope":
        os.environ["HF_HUB_OFFLINE"] = "0"
        from datasets import load_dataset
        raw = load_dataset("lmms-lab/POPE", split="test")
        return [{"question": r["question"], "answer": r["answer"], "image": r["image"]} for r in raw]
    if name == "mmstar":
        # Curated hard multiple-choice (Lin-Chen/MMStar, 1500 val). Options are
        # embedded in the question; answer is a bare letter -> strict letter-exact
        # scoring (mc_correct). A hard, OOD, STRICTLY-scorable probe (no synonym noise).
        os.environ["HF_HUB_OFFLINE"] = "0"
        from datasets import load_dataset
        raw = load_dataset("Lin-Chen/MMStar", split="val")
        instr = ("\nAnswer with ONLY the letter of the correct option inside the answer "
                 "tag, e.g. <answer>A</answer>.")
        return [{"question": r["question"] + instr, "answer": str(r["answer"]).strip(),
                 "image": r["image"]} for r in raw]
    if name == "mmmu_pro":
        # Hard, strict multiple-choice (MMMU/MMMU_Pro, "standard (10 options)", test).
        # 10-way MC, expert-level -> high collapse; letter-exact -> mc_correct scorer.
        # Most questions are single-image (image_1); the few multi-image ones use
        # image_1 only (approximation, symmetric across arms) — noted in the writeup.
        import ast
        os.environ["HF_HUB_OFFLINE"] = "0"
        from datasets import load_dataset
        raw = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
        out = []
        for r in raw:
            opts = r["options"]
            if isinstance(opts, str):
                try:
                    opts = ast.literal_eval(opts)
                except Exception:
                    opts = [opts]
            q = r["question"] + "\nOptions: " + " ".join(
                f"({chr(65 + i)}) {o}" for i, o in enumerate(opts))
            q += ("\nAnswer with ONLY the letter of the correct option inside the answer "
                  "tag, e.g. <answer>A</answer>.")
            img = r.get("image_1")
            if img is None:
                continue
            out.append({"question": q, "answer": str(r["answer"]).strip(), "image": img})
        return out
    if name in ("mathvista", "mathvision"):
        os.environ["HF_HUB_OFFLINE"] = "0"
        from datasets import load_dataset
        if name == "mathvista":
            raw = load_dataset("AI4Math/MathVista", split="testmini")
        else:
            raw = load_dataset("MathLLMs/MathVision", split="test")
        out = []
        for r in raw:
            q = r["question"]
            # multi-choice: append lettered options (MathVista uses `choices`,
            # MathVision uses `options`)
            choices = r.get("choices") if name == "mathvista" else r.get("options")
            if choices and (name == "mathvision" or r.get("question_type") == "multi_choice"):
                q = q + "\nChoices: " + " ".join(
                    f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
            img = r.get("decoded_image")
            if img is None:
                continue
            try:  # resize longer-side<=896 (huge diagrams overflow even 8192 ctx)
                W, H = img.size
                m = max(W, H)
                if m > 896:
                    s = 896.0 / m
                    img = img.resize((max(1, int(W * s)), max(1, int(H * s))))
            except Exception:
                pass
            out.append({"question": q, "answer": str(r["answer"]), "image": img})
        return out
    if name == "realworldqa":
        # RealWorldQA (lmms-lab, 765 test): real-world spatial/perception MC. Single image,
        # question already embeds options + asks for a letter -> strict letter-exact (mc_correct).
        # NON-MATH, low-collapse -> gain-vs-collapse control that breaks the math-heaviness critique.
        os.environ["HF_HUB_OFFLINE"] = "0"
        from datasets import load_dataset
        raw = load_dataset("lmms-lab/RealWorldQA", split="test")
        instr = ("\nAnswer with ONLY the letter of the correct option inside the answer "
                 "tag, e.g. <answer>A</answer>.")
        return [{"question": r["question"] + instr, "answer": str(r["answer"]).strip(),
                 "image": r["image"]} for r in raw]
    if name == "blink":
        # BLINK (BLINK-Benchmark, 14 val configs): visual PERCEPTION MC (depth, correspondence,
        # spatial, counting...). Many tasks are multi-image -> composite side-by-side into ONE
        # image (the pipeline is single-image); answer '(A)' -> 'A', letter-exact (mc_correct).
        # Hard, NON-MATH, truly OOD perception.
        os.environ["HF_HUB_OFFLINE"] = "0"
        from datasets import load_dataset, get_dataset_config_names
        from PIL import Image as _PImg
        instr = ("\nAnswer with ONLY the letter of the correct option inside the answer "
                 "tag, e.g. <answer>A</answer>.")
        out = []
        for cfg in get_dataset_config_names("BLINK-Benchmark/BLINK"):
            raw = load_dataset("BLINK-Benchmark/BLINK", cfg, split="val")
            for r in raw:
                imgs = [r.get(f"image_{k}") for k in (1, 2, 3, 4)]
                imgs = [im.convert("RGB") for im in imgs if im is not None]
                if not imgs:
                    continue
                if len(imgs) == 1:
                    img = imgs[0]
                else:
                    H = min(im.height for im in imgs)
                    imgs = [im.resize((max(1, int(im.width * H / im.height)), H)) for im in imgs]
                    img = _PImg.new("RGB", (sum(im.width for im in imgs), H), (255, 255, 255))
                    x = 0
                    for im in imgs:
                        img.paste(im, (x, 0)); x += im.width
                # cap image tokens: BLINK composites can be very wide -> exceed 8192 ctx.
                _W, _H = img.size; _m = max(_W, _H)
                if _m > 768:
                    _sc = 768.0 / _m
                    img = img.resize((max(1, int(_W * _sc)), max(1, int(_H * _sc))))
                ch = r.get("choices") or []
                q = r["question"]
                if ch:
                    q = q + "\nOptions: " + " ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(ch))
                out.append({"question": q + instr,
                            "answer": str(r["answer"]).strip().strip("()"), "image": img})
        return out
    if name == "vqarad":
        # Load
        # the canonical HF VQA-RAD instead (flaviagiammarino/vqa-rad, 451 test rows) —
        # same underlying dataset. Source is recorded in the eval meta for provenance.
        os.environ["HF_HUB_OFFLINE"] = "0"
        from datasets import load_dataset
        raw = load_dataset("flaviagiammarino/vqa-rad", split="test")
        return [{"question": r["question"], "answer": str(r["answer"]), "image": r["image"]}
                for r in raw]
    for p in PHASE1_PYTHONPATHS:
        if p not in sys.path:
            sys.path.insert(0, p)
    if name == "vqarad":
        from faithscan.data.dataset import load_vqarad, get_image
        raw = list(load_vqarad(splits=["test"], image_cache_dir="data/images"))
    elif name == "pathvqa":
        from faithscan.data.dataset import load_pathvqa, get_image
        raw = list(load_pathvqa(splits=["test"], image_cache_dir="data/images"))
    else:
        raise ValueError(f"unknown dataset {name}")

    examples = []
    for ex in raw:
        try:
            image = get_image(ex)
        except Exception:
            continue
        examples.append({
            "question": ex.question,
            "answer": ex.answer,
            "image": image,
        })
    return examples


def subsample(examples: list, n: int, seed: int = 42):
    """Match phase-1's RandomState(42).choice subsample so subset numbers align."""
    import numpy as np
    if n is None or n >= len(examples):
        return examples
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(examples), n, replace=False)
    return [examples[int(i)] for i in idx]


def build_prompt(processor, image, question: str, prompt_style: str) -> str:
    if prompt_style == "phase1_medical":
        # No system message; matches phase-1 evals exactly.
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PHASE1_USER_TEMPLATE.format(question=question)},
        ]}]
    elif prompt_style == "phase2_thinking":
        # VL-Thinking system prompt; matches phase-2 training. Add explicit
        # format reminder so the scorer can extract <answer>...</answer> reliably.
        user_text = (
            f"{question}\n\n"
            "Give your final answer in <answer>...</answer> tags."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": VL_THINKING_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
        ]
    else:
        raise ValueError(f"unknown prompt_style {prompt_style}")
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="global_step_N dir containing lora_adapter/; omit with --no-adapter")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--full-ft-checkpoint", default=None,
                    help="FULL fine-tune checkpoint (global_step_N with actor/model_world_size_*.pt "
                         "shards, no LoRA adapter). Merges to --merged-dir and evals the full model "
                         "directly. Mutually exclusive with --checkpoint.")
    ap.add_argument("--merged-dir", default=None,
                    help="output HF dir for the merged full-FT weights (required with --full-ft-checkpoint)")
    ap.add_argument("--reuse-merged", action="store_true",
                    help="reuse an already-populated --merged-dir instead of erroring")
    ap.add_argument("--dataset", choices=["slake", "vqarad", "pope", "mathvista", "mathvision", "mmstar", "mmmu_pro", "realworldqa", "blink"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-model",
                    default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Thinking"),
                    help="defaults to $MODEL_PATH (set by configs/full_ft.env)")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="matched to the training response budget (4096 everywhere); "
                         "older evals used 2048 (phase-2) / 512 (phase-1) — set explicitly for comparability")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="if set, evaluate first/sampled N examples (seed=42)")
    ap.add_argument("--max-model-len", type=int, default=8192)  # prompt(<=4096)+resp(4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--gpu", default=None,
                    help="GPU index to pin (sets CUDA_VISIBLE_DEVICES). If omitted, an "
                         "already-exported CUDA_VISIBLE_DEVICES is respected; if neither "
                         "is set the script refuses to run.")
    ap.add_argument("--arm-label", default=None,
                    help="summary key naming this eval arm (e.g. corr_only, "
                         "tiebreak_faith_only). Default: derived from the checkpoint "
                         "dir name (run__global_step_N); zero_shot for --no-adapter.")
    ap.add_argument("--ood-scorer-v1", action="store_true",
                    help="use the v1 math-set matching rules (boundary-free substring "
                         "containment + always-last-number compare) for back-comparison "
                         "with old 2B numbers. Extraction is strict last-tag either way.")
    ap.add_argument("--prompt-style", choices=["phase1_medical", "phase2_thinking"],
                    default="phase1_medical",
                    help="phase1_medical = no system + medical prompt (default; comparable "
                         "to phase-1 numbers); phase2_thinking = VL-Thinking system "
                         "(matches phase-2 training distribution)")
    ap.add_argument("--slake-dir", default=os.environ.get("SLAKE_DIR", "data/slake"))
    ap.add_argument("--organ-only", action="store_true",
                    help="SLAKE only: restrict to organ-bearing questions (matches phase-1 train filter)")
    ap.add_argument("--seed", type=int, default=0,
                    help="vLLM sampling seed; label OOD runs for the multi-eval-seed CI pass. "
                         "At temperature=0 it only perturbs kernel/batch nondeterminism.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="decoding temperature (0.0 = greedy headline regime; >0 for a "
                         "sampling-based noise-bounding pass).")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="vLLM repetition_penalty (>1 penalizes repeats; anti-collapse control)")
    args = ap.parse_args()

    if args.full_ft_checkpoint:
        if args.checkpoint:
            ap.error("--full-ft-checkpoint and --checkpoint are mutually exclusive")
        if not args.merged_dir:
            ap.error("--merged-dir is required with --full-ft-checkpoint")
        args.no_adapter = True   # full-FT: whole model saved, no adapter
    if not args.no_adapter and not args.checkpoint:
        ap.error("--checkpoint required unless --no-adapter")

    pin_gpu(args.gpu)  # before any torch/vLLM import

    # Full-FT: merge the FSDP shards to an HF dir and eval the full model directly
    # (enable_lora=False). After pin_gpu so the merge inherits the GPU env.
    if args.full_ft_checkpoint:
        merged = merge_fullft_checkpoint(
            Path(args.full_ft_checkpoint), Path(args.merged_dir), args.reuse_merged)
        args.base_model = str(merged)

    # OPTIONAL legacy phase-1 scorer (kept only as a secondary column if importable).
    # The HEADLINE OOD score is the vendored STRICT scorer below — self-contained, no
    # remote dependency, with these rules: scores ONLY the LAST closed <answer> tag (no
    # middle-grab), with yes/no-first-token and entity/number guards.
    try:
        for p in PHASE1_PYTHONPATHS:
            if p not in sys.path:
                sys.path.insert(0, p)
        from faithscan.reward import compute_correctness as _legacy_correctness  # type: ignore
        _HAVE_LEGACY = True
    except Exception:
        _HAVE_LEGACY = False
        _legacy_correctness = None

    import re as _re
    import zlib as _zlib
    from collections import Counter as _Counter

    def _strict_extract(text):
        m = _re.findall(r"<answer>(.*?)</answer>", text or "", _re.DOTALL | _re.IGNORECASE)
        return m[-1].strip() if m else None

    def _norm_toks(t):
        s = _re.sub(r"\s*-\s*", "", (t or "").strip().lower())
        s = _re.sub(r"[^\w\s.]", "", s)              # strip punctuation but keep dots
        s = _re.sub(r"(?<!\d)\.|\.(?!\d)", "", s)    # drop dots NOT between digits
        return s.split()

    def strict_correctness(raw, gt):
        a = _strict_extract(raw)
        if a is None:
            return 0.0
        at, gtt = _norm_toks(a), _norm_toks(gt)
        if not at or not gtt:
            return 0.0
        gtc = " ".join(gtt)
        if gtc in ("yes", "no"):
            decision = next((w for w in at if w in ("yes", "no")), None)
            return 1.0 if decision == gtc else 0.0
        if _re.fullmatch(r"\d+", gtc):
            return 1.0 if (gtc in at and len(at) <= 6) else 0.0
        if len(gtt) == 1:
            return 1.0 if gtt[0] in at else 0.0
        if set(gtt).issubset(set(at)):
            return 1.0
        ov = set(at) & set(gtt)
        if not ov:
            return 0.0
        p, r = len(ov) / len(set(at)), len(ov) / len(set(gtt))
        return 2 * p * r / (p + r)

    def is_collapsed(raw):
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

    # Math-dataset scorer: the medical strict_correctness mangles numbers
    # ("2.5"->"25", "-2"->"2", "1/2"->"12"). For mathvista/mathvision use symbolic
    # equivalence via math_verify, with MCQ-letter / numeric-tolerance / string
    # fallbacks. (Medical sets still use strict_correctness.)
    _MATH_DATASETS = {"mathvista", "mathvision"}
    try:
        from math_verify import parse as _mv_parse, verify as _mv_verify  # type: ignore
        _MV_OK = True
        # Reference-style extraction config (Open-R1 / TRL): parse the FULL latex
        # so the extractor recognizes \boxed{} and equations; ExprExtractionConfig
        # is the bare-scalar fallback. Best-effort across math_verify versions.
        _MV_CFG = None
        try:
            from math_verify import LatexExtractionConfig as _LatexCfg, ExprExtractionConfig as _ExprCfg
            try:
                from latex2sympy2_extended import NormalizationConfig as _NormCfg
            except Exception:
                from latex2sympy2_extended.math_normalization import NormalizationConfig as _NormCfg
            _MV_CFG = [_LatexCfg(normalization_config=_NormCfg(
                basic_latex=True, boxed="all", units=True)), _ExprCfg()]
        except Exception:
            _MV_CFG = None
    except Exception:
        _MV_OK = False
        _MV_CFG = None

    def _mv_parse_cfg(s):
        return _mv_parse(s, extraction_config=_MV_CFG) if _MV_CFG else _mv_parse(s)
    if args.dataset in _MATH_DATASETS and not _MV_OK:
        print("[ood-eval] WARNING: math_verify unavailable — math scoring falls back "
              "to numeric/string match (symbolic equivalence disabled)", flush=True)

    # v2 (default) tightens the two false-positive-prone MATCHING rules below;
    # extraction stays strict last-<answer>-tag in both versions. --ood-scorer-v1
    # reproduces the old behavior for back-comparison with 2B-era numbers.
    OOD_SCORER_VERSION = 1 if args.ood_scorer_v1 else 2

    def _nums(s):
        return _re.findall(r"-?\d+\.?\d*", (s or "").replace(",", ""))

    def _last_num(s):
        m = _nums(s)
        return float(m[-1]) if m else None

    def math_correct(raw, gt):
        # STRICT: no closed <answer> tag -> 0, same as the medical datasets.
        # Never salvage from raw text (hard rule).
        pred = _strict_extract(raw)
        if pred is None:
            return 0.0
        pred = pred.strip()
        gt = str(gt).strip()
        # MCQ letter answers (A-E): a bare-letter payload wins; otherwise take
        # the LAST standalone uppercase A-E in the payload ("(B) 45 degrees" ->
        # B) — never the first char of the uppercased string ("The..." -> 'E').
        if _re.fullmatch(r"[A-Ea-e]", gt):
            if _re.fullmatch(r"[A-Ea-e]", pred):
                return 1.0 if pred.upper() == gt.upper() else 0.0
            letters = _re.findall(r"\b([A-E])\b", pred)
            return 1.0 if (letters and letters[-1] == gt.upper()) else 0.0
        # symbolic equivalence (fractions, decimals, equivalent forms). Parse the
        # gt as-is and re-anchor a bare pred in \boxed{} so math_verify's extractor
        # can parse equations (a naked "y = ..." extracts to nothing otherwise).
        if _MV_OK:
            try:
                pred_latex = pred if "\\boxed" in pred else "\\boxed{%s}" % pred
                if _mv_verify(_mv_parse_cfg(gt), _mv_parse_cfg(pred_latex)):
                    return 1.0
            except Exception:
                pass
        # numeric with relative tolerance. v2: compare numerically ONLY when the
        # payload contains exactly one number — v1's last-number grab scored
        # multi-number payloads on an arbitrary trailing number.
        pred_nums = _nums(pred)
        b = _last_num(gt)
        if b is not None and (
            (OOD_SCORER_VERSION == 1 and pred_nums)
            or (OOD_SCORER_VERSION >= 2 and len(pred_nums) == 1)
        ):
            a = float(pred_nums[-1])
            return 1.0 if abs(a - b) <= 1e-4 * max(1.0, abs(b)) else 0.0
        # string containment fallback. v2: word-boundary containment so a short
        # gt ("no", "ray") can't match inside a longer word ("normal", "array").
        #
        if OOD_SCORER_VERSION == 1:
            return 1.0 if gt.lower() in (pred or "").strip().lower() else 0.0
        gt_norm = gt.strip().lower()
        if not gt_norm:
            return 0.0
        return 1.0 if _re.search(
            r"(?<!\w)" + _re.escape(gt_norm) + r"(?!\w)", pred.lower()) else 0.0

    def mc_correct(raw, gt):
        # Strict multiple-choice: exact-match the FIRST standalone option letter the
        # model commits to inside <answer>. No token-F1, no paraphrase credit.
        a = _strict_extract(raw)
        if a is None:
            return 0.0
        m = _re.search(r"(?<![A-Za-z])([A-Ha-h])(?![A-Za-z])", a.strip())
        letter = m.group(1).upper() if m else (a.strip()[:1].upper() if a.strip() else "")
        return 1.0 if letter == gt.strip().upper() else 0.0

    if args.dataset in _MATH_DATASETS:
        _score_fn = math_correct
    elif args.dataset in ("mmstar", "mmmu_pro"):
        _score_fn = mc_correct
    else:
        _score_fn = strict_correctness

    print(f"[ood-eval] dataset = {args.dataset}", flush=True)
    if args.dataset == "slake":
        examples = load_slake(args.slake_dir, args.organ_only)
    else:
        examples = load_crossdataset(args.dataset)

    examples = subsample(examples, args.max_examples)
    print(f"[ood-eval] {len(examples)} examples loaded", flush=True)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)

    # Resolve the adapter (and its true LoRA rank) BEFORE engine bring-up so a
    # bad adapter dir fails fast and max_lora_rank matches reality.
    adapter_dir = None
    max_lora_rank = 16
    if not args.no_adapter:
        adapter_dir = resolve_adapter_dir(Path(args.checkpoint))
        max_lora_rank = read_lora_rank(adapter_dir)

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
        print(f"[ood-eval] adapter = {adapter_dir}", flush=True)
        lora_request = LoRARequest(
            lora_name="adapter", lora_int_id=1, lora_path=str(adapter_dir),
        )

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=1.0, max_tokens=args.max_new_tokens,
        n=1, seed=args.seed, repetition_penalty=args.repetition_penalty,
    )
    print(f"[ood-eval] sampling: temperature={args.temperature} seed={args.seed} repetition_penalty={args.repetition_penalty}", flush=True)

    print(f"[ood-eval] building {len(examples)} prompts (style={args.prompt_style})...", flush=True)
    vllm_inputs = []
    for ex in examples:
        prompt_text = build_prompt(processor, ex["image"], ex["question"], args.prompt_style)
        vllm_inputs.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": ex["image"]},
        })

    print("[ood-eval] generating (continuous batching)...", flush=True)
    gen_t0 = time.time()
    outputs = llm.generate(
        vllm_inputs, sampling_params=sampling_params, lora_request=lora_request,
    )
    gen_elapsed = time.time() - gen_t0
    print(f"[ood-eval] gen done in {gen_elapsed:.1f}s "
          f"({gen_elapsed / max(len(examples), 1):.2f}s/prompt)", flush=True)

    scores: list[float] = []
    per_example = []
    for ex, out in zip(examples, outputs):
        raw = out.outputs[0].text
        s = _score_fn(raw, ex["answer"])   # math_correct for math sets, else strict
        scores.append(s)
        rec = {
            "question": ex["question"],
            "gt": ex["answer"],
            "raw": raw,
            "answer": _strict_extract(raw),
            "correct": s,                 # == strict_correct (kept for downstream consumers)
            "strict_correct": s,
            # True length-truncation: generation hit the max_new_tokens budget.
            # (Old definition — missing </think> — misflagged think-free
            # completions.) Field name kept for downstream consumers.
            "is_truncated": out.outputs[0].finish_reason == "length",
            "is_collapsed": is_collapsed(raw),
        }
        if _HAVE_LEGACY:                   # secondary, non-authoritative legacy column
            try:
                rec["legacy_correct"] = _legacy_correctness(raw, ex["answer"])
            except Exception:
                rec["legacy_correct"] = None
        per_example.append(rec)

    n = len(scores)
    mean_f1 = sum(scores) / n if n else 0.0
    exact = sum(1 for s in scores if s > 0.5)

    # Arm label for the summary key — no longer hardcoded "corr_only" for every
    # adapter. Default: <run_name>__global_step_N from the checkpoint path.
    #
    if args.arm_label:
        model_label = args.arm_label
    elif args.full_ft_checkpoint:
        ckpt_path = Path(args.full_ft_checkpoint)
        model_label = (f"{ckpt_path.parent.name}__{ckpt_path.name}"
                       if ckpt_path.name.startswith("global_step") else ckpt_path.name)
    elif args.no_adapter:
        model_label = "zero_shot"
    else:
        ckpt_path = Path(args.checkpoint)
        model_label = (f"{ckpt_path.parent.name}__{ckpt_path.name}"
                       if ckpt_path.name.startswith("global_step") else ckpt_path.name)

    # Provenance: this script vendors its own strict scorer, so
    # hash THIS file (not verl_faith/scorers.py, which it does not use).
    import hashlib as _hashlib
    import subprocess as _subprocess
    try:
        _git = _subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        _git = None

    result = {
        "summary": {
            model_label: {"f1": mean_f1, "exact": exact, "n": n},
        },
        "scorer_version": OOD_SCORER_VERSION,
        "examples": per_example,
        "checkpoint": (str(args.full_ft_checkpoint) if args.full_ft_checkpoint
                       else (str(args.checkpoint) if args.checkpoint and not args.no_adapter else "BASE")),
        "meta": {
            "dataset": args.dataset,
            "arm_label": model_label,
            "full_ft": bool(args.full_ft_checkpoint),
            "merged_dir": str(args.merged_dir) if args.full_ft_checkpoint else None,
            "scorer_version": OOD_SCORER_VERSION,
            "scorer": "vendored_strict_last_answer_tag"
                      + ("+math_v" + str(OOD_SCORER_VERSION) if args.dataset in _MATH_DATASETS else ""),
            "eval_script_sha1": _hashlib.sha1(Path(__file__).resolve().read_bytes()).hexdigest(),
            "eval_script_git": _git,
            # This script's scorer is ALWAYS strict (vendored last-<answer>-tag
            # extraction), independent of the STRICT_CORRECTNESS env var — record
            # that fact directly rather than an env value set by an import side-effect.
            "strict_correctness": "always (vendored strict scorer)",
            "engine": "vllm",
            "model_path": str(args.base_model),
            "adapter_path": str(adapter_dir) if adapter_dir is not None else None,
            "max_lora_rank": max_lora_rank,
            "max_examples": args.max_examples,
            "sampling": {
                "method": ("np_random_state_choice"
                           if args.max_examples is not None else "full"),
                "n": args.max_examples,
                "seed": 42 if args.max_examples is not None else None,
            },
            "max_new_tokens": args.max_new_tokens,
            "generation": {"temperature": args.temperature, "top_p": 1.0,
                           "max_new_tokens": args.max_new_tokens, "seed": args.seed},
            "prompt_style": args.prompt_style,
            "generation_seconds": round(gen_elapsed, 1),
            "total_seconds": round(time.time() - t0, 1),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str))
    tmp.replace(out_path)
    print(f"[ood-eval] wrote {out_path}", flush=True)

    print("=" * 60, flush=True)
    print(f"{args.dataset.upper()} RESULT ({model_label}):", flush=True)
    print(f"  F1     = {mean_f1:.4f}", flush=True)
    print(f"  Exact  = {exact}/{n}", flush=True)
    print(f"  gen time: {gen_elapsed:.1f}s ({gen_elapsed/max(n,1):.2f}s/prompt)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
