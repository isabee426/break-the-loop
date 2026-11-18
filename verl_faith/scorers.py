"""Per-verifier reward scorers for VLAA-Thinking-verifiable subset.

verl invokes `compute_score(data_source, solution_str, ground_truth, extra_info)` —
we dispatch on the data_source field, which we configure to be `verifier_type`
(set `data.reward_fn_key="verifier_type"` in the verl run).

Each scorer returns:
    {"score": float in [0, 1],
     "extra_info": {"formatted": bool, "raw_extracted": str, ...}}

`formatted` is True iff we could pull a clean answer from the response
(via <answer>...</answer>, post-</think>, or \\boxed{...}). Used by the
tiebreak adv estimator to drop unformatted rollouts from the gradient.
"""
from __future__ import annotations
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Answer extraction (shared)
# ---------------------------------------------------------------------------

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_THINK_END = re.compile(r"</think>\s*(.*)", re.DOTALL | re.IGNORECASE)


def _find_boxed(s: str) -> str | None:
    """Content of the first \\boxed{...} with FULL brace counting, or None.
    The old regex handled exactly one nesting level, so
    \\boxed{\\frac{\\sqrt{2}}{2}} truncated at the wrong brace. Unbalanced
    braces (truncated rollout) -> None, same as a regex miss."""
    start = s.find("\\boxed{")
    if start == -1:
        return None
    depth = 0
    for i in range(start + 6, len(s)):  # start+6 = index of the opening "{"
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 7:i]
    return None


def _strict_mode() -> bool:
    """STRICT_CORRECTNESS gates strict scoring at BOTH levels:
      1. extraction — only a closed, final <answer>...</answer> counts (no
         \\boxed/post-</think>/raw fallback); no tag -> score 0;
      2. payload — the tag must contain ONLY the answer (digit: just a number;
         mcq: just the UPPERCASE letter, case-sensitive; math: just
         \\boxed{...}; iou: just [x1, y1, x2, y2]). Anything else in the tag
         -> score 0 AND
         formatted=0 (no format bonus): wrong payload = wrong format. No
         within-payload salvage parsing — the model must learn the format.
    Default (unset/0) leaves the lenient behavior — and therefore every
    pre-strict run — completely unchanged."""
    return os.environ.get("STRICT_CORRECTNESS", "") not in ("", "0", "false", "False", "no", "NO")


# One-time banner so the run log records which extraction mode is live.
if _strict_mode():
    print("[scorers] STRICT_CORRECTNESS on: tag-required + payload-exact scoring "
          "(last closed <answer> only, payload must be ONLY the answer; "
          "no fallbacks)", flush=True)


def extract_answer(response: str) -> tuple[str, bool]:
    """Return (extracted_text, formatted_flag).

    Lenient (default): <answer> tag > \\boxed{} > post-</think> raw > raw response.
    Only the first two paths set formatted=True. Post-</think> raw text is
    returned as the extracted answer (so downstream verifiers can still try
    to score it) but with formatted=False — emitting </think> alone is not
    enough to qualify for the format reward; the model must close <answer>
    or use \\boxed{}.

    Strict (STRICT_CORRECTNESS=1): ONLY a clean, closed, final <answer>...</answer>
    counts — take the LAST tag (the model's concluded answer); no fallback to
    \\boxed / post-</think> / raw. No closed tag -> ("", False) -> correctness 0
    (a truncated rollout that never closes </answer> scores 0, as intended).
    Because every verifier scores ONLY this returned text, confining extraction
    to the tag is sufficient to make all four verifiers strict — their raw-text
    fallbacks (nums[-1], matches[-1], math-verify, bbox-search) then operate on
    the tag payload alone, never the whole response. Mirrors the OOD
    scorer's _strict_extract (last tag, no fallback) so train and eval agree.
    """
    if _strict_mode():
        ms = list(_ANSWER_TAG.finditer(response))
        if ms:
            return ms[-1].group(1).strip(), True
        return "", False
    if (ms := list(_ANSWER_TAG.finditer(response))):
        return ms[-1].group(1).strip(), True   # LAST closed tag = concluded answer (was .search = first)
    if (m := _THINK_END.search(response)):
        rest = m.group(1).strip()
        if rest:
            # \boxed{} after </think> still counts as formatted (R1 convention).
            if (bx := _find_boxed(rest)) is not None:
                return bx.strip(), True
            return rest, False
    if (bx := _find_boxed(response)) is not None:
        return bx.strip(), True
    return response.strip(), False


# ---------------------------------------------------------------------------
# digit: exact match on a stripped numeric string
# ---------------------------------------------------------------------------

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _num_eq(a: str, b: str) -> bool:
    """Numeric-tolerance equality: '5.0' == '5', '0.50' == '0.5'."""
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return False


def score_digit(response: str, gt: str) -> dict:
    text, formatted = extract_answer(response)
    gt_norm = gt.strip()
    pred = text.strip()
    if _strict_mode():
        # Payload must be ONLY a number — no salvage parsing from prose.
        if not _NUM.fullmatch(pred):
            return {"score": 0.0, "extra_info": {
                "formatted": False, "raw_extracted": text, "note": "payload_not_bare_number"}}
        ok = pred == gt_norm or _num_eq(pred, gt_norm)
        return {"score": 1.0 if ok else 0.0,
                "extra_info": {"formatted": formatted, "raw_extracted": pred}}
    # Lenient: plain string match first (most common: gt="5", model says "5"),
    # then numeric tolerance ("5.0" vs "5" scored 0 under string-only compare).
    if pred == gt_norm or _num_eq(pred, gt_norm):
        return {"score": 1.0, "extra_info": {"formatted": formatted, "raw_extracted": text}}
    # Numeric fallback: pull last number from extracted text and compare.
    nums = _NUM.findall(text)
    if nums and (nums[-1] == gt_norm or _num_eq(nums[-1], gt_norm)):
        return {"score": 1.0, "extra_info": {"formatted": formatted, "raw_extracted": nums[-1]}}
    return {"score": 0.0, "extra_info": {"formatted": formatted, "raw_extracted": text}}


# ---------------------------------------------------------------------------
# mcq: extract A/B/C/D/E (case-insensitive) and exact-match the gt letter
# ---------------------------------------------------------------------------

_MCQ_LETTER = re.compile(r"\b([A-E])\b")
# Phrasing patterns the model uses when explicitly stating its answer.
# Phrase words are case-insensitive (scoped (?i:...)); the LETTER is
# case-SENSITIVE uppercase and must not run into a word. Under a global
# IGNORECASE the English article in "the answer is a small dog" matched as
# choice A (confirmed empirically) and inflated unformatted-mcq reward
# whenever gt=A. Lowercase answers still score via the formatted path,
# which uppercases the tag payload before matching.
_MCQ_PHRASE = re.compile(
    r"(?i:answer(?:\s+is)?(?:\s*:)?|option|the\s+correct\s+(?:answer|option)\s+is)"
    r"\s*\(?([A-E])\)?(?![A-Za-z])"
)


def score_mcq(response: str, gt: str) -> dict:
    text, formatted = extract_answer(response)
    gt_letter = gt.strip().upper()
    if not (len(gt_letter) == 1 and gt_letter in "ABCDE"):
        # Defensive — treat malformed gt as graceful 0
        return {"score": 0.0, "extra_info": {"formatted": formatted, "raw_extracted": text, "error": "bad_gt"}}
    if _strict_mode():
        # Payload must be ONLY the UPPERCASE letter, case-SENSITIVE (the v3
        # prompt hint states this) — "b", "(B)", "B.", "B) 45 degrees" all
        # score 0 and lose the format bonus.
        p = text.strip()
        if not (len(p) == 1 and p in "ABCDE"):
            return {"score": 0.0, "extra_info": {
                "formatted": False, "raw_extracted": text, "note": "payload_not_bare_uppercase_letter"}}
        return {"score": 1.0 if p == gt_letter else 0.0,
                "extra_info": {"formatted": formatted, "raw_extracted": p}}
    # 1) When the response is well-formatted (<answer>X</answer> or post-</think>),
    #    `text` is already the clean answer payload — first letter is the convention.
    # 2) When unformatted, we're parsing raw thinking text. The FIRST A-E in raw
    #    text is unreliable (matches artifacts like "AB" in geometry problems).
    #    Prefer an explicit phrasing ("answer is X", "option X") if present;
    #    otherwise fall back to the LAST A-E (reasoning converges to conclusion).
    if formatted:
        # Uppercase ONLY a bare single-char payload (preserves "<answer>a</answer>"
        # -> A); longer payloads match case-SENSITIVELY so the English article in
        # "<answer>a small dog</answer>" can't score as choice A (same bug class
        # as the raw-text path below — it inflated lenient labels for gt=A rows).
        p = text.strip()
        matches = _MCQ_LETTER.findall(p.upper() if len(p) == 1 else text)
        if not matches:
            return {"score": 0.0, "extra_info": {"formatted": formatted, "raw_extracted": text}}
        pred = matches[0]
    else:
        phrase = _MCQ_PHRASE.findall(text)
        if phrase:
            pred = phrase[-1].upper()
        else:
            # Case-SENSITIVE on raw thinking text: uppercasing first would turn
            # every article "a" into a standalone "A" match (same bug as above).
            matches = _MCQ_LETTER.findall(text)
            if not matches:
                return {"score": 0.0, "extra_info": {"formatted": formatted, "raw_extracted": text}}
            pred = matches[-1]
    return {
        "score": 1.0 if pred == gt_letter else 0.0,
        "extra_info": {"formatted": formatted, "raw_extracted": pred},
    }


# ---------------------------------------------------------------------------
# math: MathVerify (HF) symbolic equivalence — VLAA-Thinker convention
# ---------------------------------------------------------------------------

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify  # type: ignore
    _MATH_VERIFY_OK = True
except Exception:
    _MATH_VERIFY_OK = False

# Reference-style extraction config (Open-R1 / TRL / lighteval): math_verify.parse
# is an EXTRACTOR — it pulls math out of \boxed{}/$...$/\[...\] and, with the
# latex config, parses equations (y = ...) and relations. Hand it the FULL latex
# string (gold as-is; pred WITH its \boxed{}), NEVER a pre-stripped bare inner —
# a naked "y = -2|x+3|+2" has no anchor and extracts to nothing, flooring math
# acc to bare-scalars-only (the bug this replaces). ExprExtractionConfig is the
# bare-number fallback. Best-effort: falls back to default parse if the config
# classes/NormalizationConfig aren't importable on this math_verify version.
_MV_CFG = None
if _MATH_VERIFY_OK:
    try:
        from math_verify import LatexExtractionConfig as _LatexCfg, ExprExtractionConfig as _ExprCfg
        try:
            from latex2sympy2_extended import NormalizationConfig as _NormCfg
        except Exception:
            from latex2sympy2_extended.math_normalization import NormalizationConfig as _NormCfg
        # NB: no equations=True — deprecated (the parser handles equations now).
        _MV_CFG = [_LatexCfg(normalization_config=_NormCfg(
            basic_latex=True, boxed="all", units=True)), _ExprCfg()]
    except Exception:
        _MV_CFG = None


def _mv_parse_cfg(s: str):
    """parse() with the reference extraction config when available, else default."""
    if _MV_CFG is not None:
        return _mv_parse(s, extraction_config=_MV_CFG)
    return _mv_parse(s)


# ── math_verify hard timeout ─────────────────────────────────────────────────
# sympy can spin FOREVER on pathological latex — observed 2026-07-04: one math
# rollout pegged a core for 22h and stalled the entire collection (and would
# identically freeze a training run's reward loop). Verification runs in a
# KILLABLE child process with a hard timeout; on timeout the child is
# terminated and the caller falls back to the cheap normalized string compare.
# MATH_VERIFY_TIMEOUT_S=0 restores the direct in-process call.
# SPAWN, not fork: in colocated-extractor mode this process has initialized
# CUDA before the first math reward, and a fork()ed child of a CUDA process can
# hang or corrupt state. spawn children are clean interpreters (pure sympy).
_MV_TIMEOUT_S = float(os.environ.get("MATH_VERIFY_TIMEOUT_S", "5.0"))


def _mv_child_check(gold: str, pred: str) -> bool:
    """Runs inside the timeout child. Pure sympy — must never touch CUDA.
    (The spawn child re-imports this module, so _mv_parse_cfg/_MV_CFG exist.)"""
    return bool(_mv_verify(_mv_parse_cfg(gold), _mv_parse_cfg(pred)))


_mv_pool_state: dict[str, object] = {}
# Running counts: total timeout-guarded math verifications and how many timed out
# (fell back to normalized string compare). The RATE (fallbacks/total) is what
# matters for reward bias — a raw count is uninterpretable without the
# denominator. Logged on each timeout occurrence and again at interpreter exit.
_mv_fallback_count = [0]
_mv_total_count = [0]


def _mv_report_fallbacks() -> None:
    if _mv_fallback_count[0]:
        tot = _mv_total_count[0] or 1
        rate = 100.0 * _mv_fallback_count[0] / tot
        print(f"[scorers] math_verify timeout fallbacks this run: "
              f"{_mv_fallback_count[0]}/{_mv_total_count[0]} ({rate:.2f}%) "
              f"(scored by normalized string compare)", flush=True)


import atexit as _atexit
_atexit.register(_mv_report_fallbacks)


def _mv_executor():
    """Lazily-created, persistent single-worker spawn ProcessPoolExecutor.
    Recreated if broken (killed worker / BrokenProcessPool). Caller holds the lock."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as _mp
    pool = _mv_pool_state.get("pool")
    if pool is None:
        pool = ProcessPoolExecutor(max_workers=1, mp_context=_mp.get_context("spawn"))
        _mv_pool_state["pool"] = pool
    return pool


def _mv_kill_executor() -> None:
    """Hard-kill the worker (it may be stuck in sympy) and drop the pool."""
    pool = _mv_pool_state.pop("pool", None)
    if pool is None:
        return
    try:
        for p in list(getattr(pool, "_processes", {}).values()):
            p.kill()
    except Exception:
        pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _mv_verify_timeout(gold: str, pred: str):
    """Return True/False from math_verify, or None on timeout (caller falls
    back to _norm_math string compare). Child crash (bad latex) -> False."""
    if _MV_TIMEOUT_S <= 0:
        try:
            return bool(_mv_verify(_mv_parse_cfg(gold), _mv_parse_cfg(pred)))
        except Exception:
            return False
    from concurrent.futures import TimeoutError as _FutTimeout
    from concurrent.futures.process import BrokenProcessPool as _Broken
    lock = _mv_pool_state.setdefault("lock", __import__("threading").Lock())
    with lock:
        _mv_total_count[0] += 1
        try:
            fut = _mv_executor().submit(_mv_child_check, gold, pred)
            return fut.result(timeout=_MV_TIMEOUT_S)
        except _FutTimeout:
            _mv_fallback_count[0] += 1
            _rate = 100.0 * _mv_fallback_count[0] / (_mv_total_count[0] or 1)
            print(f"[scorers] WARNING: math_verify TIMED OUT after {_MV_TIMEOUT_S}s "
                  f"(pathological latex; worker killed) — falling back to string "
                  f"compare (fallback {_mv_fallback_count[0]}/{_mv_total_count[0]} "
                  f"= {_rate:.2f}% this run)", flush=True)
            _mv_kill_executor()  # worker is stuck; recreate lazily on next call
            return None
        except _Broken:
            _mv_kill_executor()
            return False
        except Exception:
            # child crashed (bad latex etc.) — treat like the old except path
            return False

# Provenance banner: prove in the run log which math-scoring path was live —
# symbolic math_verify vs the string/numeric _norm_math fallback. (A silently
# absent math_verify on the training host means every math reward used the
# weaker fallback; this makes that visible instead of guessable.)
print(f"[scorers] math_verify available: {_MATH_VERIFY_OK} "
      f"({'symbolic path live' if _MATH_VERIFY_OK else 'STRING/NUMERIC FALLBACK live'})",
      flush=True)


def _norm_math(s: str) -> str:
    """Normalize a math answer for the no-math_verify fallback: pull \\boxed{}
    content, strip latex wrappers/braces/whitespace. So '\\boxed{42}' and '42'
    compare equal. (The old fallback didn't strip \\boxed -> every boxed answer
    scored 0; that's the bug that made math val acc a hard 0 for the whole run.)"""
    s = (s or "").strip()
    if (bx := _find_boxed(s)) is not None:
        s = bx
    s = re.sub(r"\\\[|\\\]|\\\(|\\\)|\$|\\,|\\!|\\text\{[^}]*\}", "", s)
    return s.replace("{", "").replace("}", "").replace(" ", "").strip()


def _exact_boxed(s: str) -> str | None:
    """Return the inner content iff the (stripped) string is EXACTLY one
    \\boxed{...} with balanced braces and nothing around it, else None.
    Brace-counted rather than regex so nested latex (\\boxed{\\sqrt{\\frac{1}{2}}})
    isn't wrongly rejected."""
    s = s.strip()
    prefix = "\\boxed{"
    if not s.startswith(prefix) or not s.endswith("}"):
        return None
    depth = 0
    for i in range(len(prefix) - 1, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[len(prefix):i] if i == len(s) - 1 else None
    return None


def score_math(response: str, gt: str) -> dict:
    text, formatted = extract_answer(response)
    # math_verify.parse needs the \boxed{}/latex anchor to extract+parse (esp.
    # equations like "y = ..."); hand it the WRAPPED payload, never a bare inner.
    # raw_extracted keeps the readable inner for logging.
    raw_extracted = text
    mv_payload = text
    if _strict_mode():
        # Payload must be ONLY \boxed{...} (the prompt hint asks for exactly
        # <answer>\boxed{...}</answer>) — prose around it, or a bare answer,
        # scores 0 and loses the format bonus.
        inner = _exact_boxed(text)
        if inner is None:
            return {"score": 0.0, "extra_info": {
                "formatted": False, "raw_extracted": text, "note": "payload_not_exact_boxed"}}
        raw_extracted = inner               # readable inner for logs / eval
        mv_payload = "\\boxed{%s}" % inner   # KEEP the anchor for math_verify
    elif "\\boxed" not in mv_payload:
        # Lenient: extract_answer may have returned a bare inner — re-anchor it so
        # math_verify extracts+parses equations identically.
        mv_payload = "\\boxed{%s}" % mv_payload
    if not _MATH_VERIFY_OK:
        # Fallback (math_verify unavailable): normalize \boxed{}/latex, then
        # string-equal OR numeric-equal (42 == 42.0).
        a, b = _norm_math(mv_payload), _norm_math(gt)
        ok = (a == b)
        if not ok:
            try:
                ok = abs(float(a) - float(b)) < 1e-6
            except (ValueError, TypeError):
                pass
        return {
            "score": 1.0 if ok else 0.0,
            "extra_info": {"formatted": formatted, "raw_extracted": raw_extracted, "fallback": True},
        }
    ok = _mv_verify_timeout(gt, mv_payload)
    if ok is None:
        # math_verify timed out — score with the cheap normalized compare so a
        # pathological payload gets a deterministic 0/1 instead of a hang.
        a, b = _norm_math(mv_payload), _norm_math(gt)
        ok = (a == b)
        if not ok:
            try:
                ok = abs(float(a) - float(b)) < 1e-6
            except (ValueError, TypeError):
                pass
        return {"score": 1.0 if ok else 0.0,
                "extra_info": {"formatted": formatted, "raw_extracted": raw_extracted,
                               "note": "math_verify_timeout_fallback"}}
    return {
        "score": 1.0 if ok else 0.0,
        "extra_info": {"formatted": formatted, "raw_extracted": raw_extracted},
    }


# ---------------------------------------------------------------------------
# iou: parse [x1, y1, x2, y2] (normalized [0,1]) and threshold IoU at 0.5
# ---------------------------------------------------------------------------

_BBOX = re.compile(
    r"\[?\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]?"
)
# Strict payload form: brackets REQUIRED (the hint shows "[x1, y1, x2, y2]"),
# nothing before/after. Used with fullmatch.
_BBOX_EXACT = re.compile(
    r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]"
)


def _parse_bbox(s: str) -> tuple[float, float, float, float] | None:
    m = _BBOX.search(s)
    if not m:
        return None
    x1, y1, x2, y2 = (float(g) for g in m.groups())
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# Qwen3-VL emits bbox coords in a fixed [0, 1000] grid regardless of the actual
# image dimensions. Empirically confirmed 2026-05-17 on mixed_tiebreak_step80:
# 20/20 predictions had max coord > image-pixel dims and rescaled cleanly via
# /1000 against the normalized [0,1] gt (mean IoU 0.558, pass@0.5 = 0.75).
_QWEN_BBOX_GRID = 1000.0
# Grid-vs-normalized decision boundary: max coord > 2.0 -> grid (/1000);
# all coords <= 2.0 -> normalized (clamped to [0,1]). The old `any(c > 1.0)`
# trigger rescaled a merely-sloppy normalized box like [0.9, 0.9, 1.05, 1.0]
# down to ~0 IoU. Predictions with max coord in (1.0, 2.0] are AMBIGUOUS under
# either rule — counted + periodically logged so 8B re-validation has data.
_IOU_GRID_MIN = 2.0
_iou_ambig_count = [0]


def score_iou(response: str, gt: str, threshold: float = 0.5,
              image_size: tuple[int, int] | None = None) -> dict:
    """Score a bbox response against a normalized [0,1] gt.

    Prompt convention: gt is always `[x1,y1,x2,y2]` normalized to [0,1].
    Qwen3-VL emits in its native 0-1000 grid even when asked for normalized
    coords, so when the max pred coord > 2 we divide by 1000 to bridge;
    max coord <= 2 is treated as normalized (clamped to [0,1]).
    `image_size` is accepted (verl passes it) but unused — Qwen's grid is
    image-size-independent.

    `extra_info["iou"]` is the continuous IoU value (always reported, regardless
    of threshold) so eval can aggregate mean IoU / AP curves alongside the
    binarized score used as the training reward.
    """
    del image_size
    text, formatted = extract_answer(response)
    if _strict_mode():
        # Payload must be ONLY "[x1, y1, x2, y2]" — no salvage-searching a bbox
        # out of surrounding prose. (Corner-swap normalization and the /1000
        # Qwen grid rescale below are NOT formatting leniency — they bridge
        # coordinate conventions — so they stay.)
        if not _BBOX_EXACT.fullmatch(text.strip()):
            return {"score": 0.0, "extra_info": {
                "formatted": False,
                "raw_extracted": text,
                "iou": 0.0,
                "pred_box": None,
                "rescaled_by_grid": False,
                "note": "payload_not_bare_bbox",
            }}
    pred_box = _parse_bbox(text)
    gt_box = _parse_bbox(gt)
    if pred_box is None or gt_box is None:
        return {"score": 0.0, "extra_info": {
            "formatted": formatted and pred_box is not None,
            "raw_extracted": text,
            "iou": 0.0,
            "pred_box": None,
            "rescaled_by_grid": False,
        }}
    mx = max(pred_box)
    rescaled = False
    if mx > _IOU_GRID_MIN:
        g = _QWEN_BBOX_GRID
        pred_norm = (pred_box[0] / g, pred_box[1] / g, pred_box[2] / g, pred_box[3] / g)
        rescaled = True
    else:
        # Normalized convention — clamp slop into [0,1].
        pred_norm = tuple(min(1.0, max(0.0, c)) for c in pred_box)
        if mx > 1.0:  # ambiguous band (1.0, 2.0]: could be a tiny grid box too
            _iou_ambig_count[0] += 1
            if _iou_ambig_count[0] == 1 or _iou_ambig_count[0] % 100 == 0:
                print(f"[scorers] iou grid-heuristic AMBIGUOUS band hit "
                      f"(max coord {mx:.3f} in (1, {_IOU_GRID_MIN}]) — treated as "
                      f"normalized+clamped; count={_iou_ambig_count[0]}", flush=True)
    iou = _iou(pred_norm, gt_box)
    return {
        "score": 1.0 if iou >= threshold else 0.0,
        "extra_info": {
            "formatted": formatted,
            "raw_extracted": text,
            "iou": iou,
            "pred_box": list(pred_norm),
            "rescaled_by_grid": rescaled,
        },
    }


# ---------------------------------------------------------------------------
# Dispatcher (the entry point verl calls)
# ---------------------------------------------------------------------------

_DISPATCH = {
    "digit": score_digit,
    "mcq": score_mcq,
    "math": score_math,
    "iou": score_iou,
}

# Diagnostic logging: write the first N scorer calls to a file so we can see
# what verl actually passes (data_source dispatch key, raw rollout text, gt).
# Path from SCORER_DEBUG_PATH (legacy alias VERL_FAITH_DEBUG_LOG); default is a
# box-independent ~/faith_logs (a hardcoded path + bare
# except silently swallowed every record on any other machine). "" disables.
import threading, json as _json
_DEBUG_PATH = os.environ.get("SCORER_DEBUG_PATH",
                             os.environ.get("VERL_FAITH_DEBUG_LOG",
                                            os.path.join(os.path.expanduser("~/faith_logs"),
                                                         "scorer_debug.jsonl")))
_DEBUG_MAX = int(os.environ.get("VERL_FAITH_DEBUG_MAX", "20"))
_debug_lock = threading.Lock()
_debug_count = [0]
_debug_warned = [False]


def _debug_log(rec: dict) -> None:
    if not _DEBUG_PATH:
        return
    with _debug_lock:
        if _debug_count[0] >= _DEBUG_MAX:
            return
        _debug_count[0] += 1
    try:
        os.makedirs(os.path.dirname(_DEBUG_PATH) or ".", exist_ok=True)
        with open(_DEBUG_PATH, "a") as f:
            f.write(_json.dumps(rec) + "\n")
    except OSError as exc:
        if not _debug_warned[0]:
            _debug_warned[0] = True
            print(f"[scorers] WARNING: cannot write scorer debug log {_DEBUG_PATH!r}: {exc} "
                  f"(set SCORER_DEBUG_PATH; further such warnings suppressed)", flush=True)


# ---------------------------------------------------------------------------
# Faith probe loading (lazy + thread-safe). Activated by env vars:
#   FAITH_PROBE_PATH  -> pickled sklearn LogisticRegression
#   FAITH_PROBE_MODEL -> HF id for the Lookback-Lens forward model
# When set, compute_score also returns a faith score in [0, 1].
# ---------------------------------------------------------------------------

_probe_lock = threading.Lock()
_probe_state: dict[str, object] = {}


def _maybe_load_probe():
    """Idempotent loader. Returns (extractor, probe) tuple or (None, None)."""
    if "loaded" in _probe_state:
        return _probe_state.get("ext"), _probe_state.get("probe")
    with _probe_lock:
        if "loaded" in _probe_state:
            return _probe_state.get("ext"), _probe_state.get("probe")
        probe_path = os.environ.get("FAITH_PROBE_PATH")
        if not probe_path:
            # Faith intentionally disabled (vanilla/length/random arms).
            _probe_state["loaded"] = True
            return None, None
        # FAITH_PROBE_PATH SET but pointing at a missing file is a CONFIG error
        # (e.g. a stale default path) — raise so it
        # PROPAGATES and hard-stops the run, same treatment as the feat_dim
        # mismatch below, instead of silently degrading the faith arm to vanilla.
        if not os.path.isfile(probe_path):
            raise RuntimeError(
                f"FAITH_PROBE_PATH is set but no file exists at {probe_path!r} — "
                f"the faith arm would silently run as vanilla. Point FAITH_PROBE_PATH "
                f"at a probe refit for THIS model size, or unset it to intentionally "
                f"disable faith.")
        # MODEL IDENTITY: probe features are model-specific (feat_dim AND head
        # semantics) — never default the extractor model. FAITH_PROBE_MODEL is
        # REQUIRED whenever the faith serve path is enabled.
        model_id = os.environ.get("FAITH_PROBE_MODEL")
        if not model_id:
            raise RuntimeError(
                "FAITH_PROBE_PATH is set but FAITH_PROBE_MODEL is not — the faith "
                "extractor model must be named explicitly (a silent 2B default here "
                "would score 8B rollouts with the wrong model). Set FAITH_PROBE_MODEL "
                "to the exact HF id the probe was collected on "
                "(e.g. Qwen/Qwen3-VL-8B-Thinking).")
        # Sidecar provenance check: train_probe.py writes <probe.pkl>.meta.json
        # with the collection model. If present, it must agree with FAITH_PROBE_MODEL.
        _meta_path = probe_path + ".meta.json"
        if os.path.isfile(_meta_path):
            try:
                with open(_meta_path) as _mf:
                    _meta_model = _json.load(_mf).get("model")
            except (OSError, ValueError) as _me:
                raise RuntimeError(f"cannot read probe sidecar {_meta_path}: {_me}")
            if _meta_model:
                if str(_meta_model).strip().lower() != model_id.strip().lower():
                    raise RuntimeError(
                        f"probe sidecar {_meta_path} records model={_meta_model!r} but "
                        f"FAITH_PROBE_MODEL={model_id!r} — model/probe mismatch; the "
                        f"probe MUST be scored with the model it was collected on.")
            else:
                print(f"[scorers] WARNING: probe sidecar {_meta_path} has no 'model' "
                      f"field — cannot verify probe/model identity; proceeding on "
                      f"FAITH_PROBE_MODEL={model_id} alone.", flush=True)
        else:
            print(f"[scorers] WARNING: no sidecar meta.json next to {probe_path} "
                  f"({_meta_path} missing) — probe/model identity UNVERIFIED; "
                  f"trusting FAITH_PROBE_MODEL={model_id}. Refit with train_probe.py "
                  f"to get a provenance sidecar.", flush=True)
        import pickle
        probe = ext = None
        # SINGLE-GPU COLOCATED MODE: this reward worker is a 0-GPU Ray actor
        # (Ray blanked its CUDA_VISIBLE_DEVICES). FAITH_FORCE_CVD re-pins the
        # process onto the training GPU before the first CUDA init so the
        # extractor colocates with vLLM+FSDP on one physical device — the only
        # faith layout that works without Fabric Manager (same-GPU contexts are
        # fine; a second GPU is not). Set by the faith launcher when PROBE_GPU="".
        _force = os.environ.get("FAITH_FORCE_CVD")
        if _force is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = _force
            print(f"[scorers] FAITH_FORCE_CVD={_force}: extractor colocating on the "
                  f"training GPU (0-GPU reward worker re-pinned)", flush=True)
        try:
            with open(probe_path, "rb") as f:
                probe = pickle.load(f)
            from verl_faith.lookback import LookbackExtractor
            ext = LookbackExtractor(
                model_path=model_id,
                device=os.environ.get("FAITH_PROBE_DEVICE", "cuda"),
            )
        except Exception as exc:
            # FAITH_PROBE_PATH is set, so a probe/extractor that can't load is a
            # config/environment error (bad pickle, OOM, broken import) — never
            # transient. Re-raise so the run dies loudly instead of the faith arm
            # silently degrading to vanilla.
            print(f"[scorers] FAILED to load faith probe: {exc}", flush=True)
            raise
        # A DIM MISMATCH is a CONFIG error (wrong-size probe for this model), not a
        # transient failure — raise OUTSIDE the try so it PROPAGATES and hard-stops
        # the run, instead of silently degrading a (multi-day, expensive) faith arm
        # into a vanilla one. Only reachable when FAITH_PROBE_PATH is set (faith arm).
        if probe is not None and ext is not None:
            n_in = getattr(probe, "n_features_in_", None)
            feat_dim = getattr(ext, "feat_dim", None)
            if n_in is not None and feat_dim is not None and n_in != feat_dim:
                raise RuntimeError(
                    f"faith probe expects {n_in}-dim features but the extractor for "
                    f"{model_id} "
                    f"produces {feat_dim}-dim — retrain the probe for this model size and "
                    f"point FAITH_PROBE_PATH at it.")
            print(f"[scorers] loaded faith probe from {probe_path} "
                  f"(feat_dim={feat_dim}, probe_n_in={n_in})", flush=True)
            # Opt-in micro-batching (FAITH_BATCH_WINDOW_S > 0): group same-prompt
            # extraction requests from the reward loop's concurrent threads into
            # one extract_batch forward. Off by default.
            win_s = float(os.environ.get("FAITH_BATCH_WINDOW_S", "0") or 0)
            if win_s > 0:
                from verl_faith.lookback import BatchingFaithQueue
                _probe_state["shim"] = BatchingFaithQueue(ext, window_s=win_s)
                print(f"[scorers] faith micro-batching ON (window={win_s}s)", flush=True)
        _probe_state["ext"] = ext
        _probe_state["probe"] = probe
        _probe_state["loaded"] = True
        return _probe_state.get("ext"), _probe_state.get("probe")


# Faith is only requested when FAITH_PROBE_PATH is set, so any unavailable faith
# is an anomaly (broken probe load or missing image/question plumbing) — warn,
# rate-limited per reason so a systematic failure can't flood the training log.
_faith_warn_counts: dict[str, int] = {}
_FAITH_WARN_MAX = int(os.environ.get("FAITH_WARN_MAX", "20"))


FAITH_FAIL_SENTINEL = -1.0  # faith is a probability in [0,1]; -1.0 = "unavailable"


def _faith_unavailable(reason: str) -> float:
    """Warn (rate-limited) that faith was expected but couldn't be computed, and
    return the -1.0 sentinel (the tiebreak estimator EXCLUDES faith<0 members
    from within-tie jitter — 0.0 is a legitimate probability and must not be
    conflated with failure). A steady stream of these means the probe pipeline
    or the image/question plumbing is broken — must NOT happen in a healthy run."""
    n = _faith_warn_counts.get(reason, 0)
    if n < _FAITH_WARN_MAX:
        _faith_warn_counts[reason] = n + 1
        tail = "  (further such warnings suppressed)" if n + 1 == _FAITH_WARN_MAX else ""
        print(f"[scorers] WARNING: faith unavailable ({reason}) -> sentinel "
              f"{FAITH_FAIL_SENTINEL}{tail}", flush=True)
    # Dead-man switch: trips on EITHER (a) a sustained consecutive run of
    # failures, or (b) a sustained failure RATE over the last _FAITH_DEAD_MAX
    # calls (catches a partial outage — e.g. only long responses failing — that
    # never produces 512 consecutive misses). Die loudly instead of silently
    # corrupting the A/B by running the faith arm as (partial) vanilla.
    _faith_dead_streak[0] += 1
    _faith_fail_window.append(1)
    if os.environ.get("FAITH_PROBE_PATH"):
        if _faith_dead_streak[0] >= _FAITH_DEAD_MAX:
            raise RuntimeError(
                f"faith signal dead for {_faith_dead_streak[0]} consecutive rollouts "
                f"(last reason: {reason}) — the faith arm would silently run as vanilla; "
                f"fix the probe/extractor pipeline before resuming.")
        if (len(_faith_fail_window) == _FAITH_DEAD_MAX
                and sum(_faith_fail_window) > _FAITH_FAIL_RATE_MAX * _FAITH_DEAD_MAX):
            raise RuntimeError(
                f"faith failure rate {sum(_faith_fail_window) / _FAITH_DEAD_MAX:.1%} over "
                f"the last {_FAITH_DEAD_MAX} rollouts exceeds "
                f"{_FAITH_FAIL_RATE_MAX:.0%} (last reason: {reason}) — sustained partial "
                f"faith outage; fix the probe/extractor pipeline before resuming.")
    return FAITH_FAIL_SENTINEL


def _coerce_pil_image(img):
    """Coerce a faith image to PIL.Image. DataProto may deliver it as a numpy
    array (PIL auto-converted in non_tensor_batch), a {bytes:...} dict, or a
    torch tensor. extract_efficient needs a PIL.Image (it calls image.mode)."""
    from PIL import Image as _PILImage
    if isinstance(img, _PILImage.Image):
        return img
    if isinstance(img, dict) and "bytes" in img:
        from io import BytesIO as _BIO
        return _PILImage.open(_BIO(img["bytes"])).convert("RGB")
    try:
        import numpy as _np
        arr = img
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = _np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = _np.transpose(arr, (1, 2, 0))
        if arr.dtype != _np.uint8:
            arr = arr * 255.0 if float(_np.nanmax(arr)) <= 1.0 + 1e-6 else arr
            arr = _np.clip(arr, 0, 255).astype(_np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return _PILImage.fromarray(arr).convert("RGB")
    except Exception:
        return None


def _compute_faith(extra_info: dict | None, solution_str: str,
                   correctness: float = None, data_source: str = None) -> float:
    """Return faith probability in [0,1], or the -1.0 sentinel if unavailable.

    Sentinel is -1.0, NOT 0.0: 0.0 is a legitimate probe output, so a failed
    extraction returning 0.0 would silently rank the rollout as genuinely
    low-faith (and, worse, an anti-length failure bias would masquerade as an
    anti-length faith signal). The tiebreak estimator excludes faith<0 members
    from within-tie jitter. Matches the FAITH_PROBE_PATH-unset path (also -1.0).
    Every unavailable path emits a rate-limited warning via _faith_unavailable
    — it should not fire.
    """
    if extra_info is None:
        return _faith_unavailable("extra_info is None")
    ext, probe = _maybe_load_probe()
    if ext is None or probe is None:
        return _faith_unavailable("probe/extractor not loaded")
    image = extra_info.get("image")
    question = extra_info.get("question")
    if image is None or question is None:
        return _faith_unavailable("image/question missing from extra_info")
    image = _coerce_pil_image(image)
    if image is None:
        return _faith_unavailable("image not coercible to PIL.Image")
    try:
        # extract_efficient is the sdpa-compatible path (the extractor is loaded
        # with attn_implementation="sdpa"). Plain extract() needs output_attentions,
        # which sdpa returns as None -> silent fallback on every call.
        # NO-SKEW INVARIANT: this serve window MUST equal both the probe's
        # collection window (collect_probe_data_vllm.py --max-new-tokens) AND
        # data.max_response_length. At 4096 (>= the 4096 response budget) neither
        # collection nor serve ever truncates, so both average the vision-attention
        # ratio over the FULL response -> identical statistic -> zero train/serve skew.
        win = int(os.environ.get("FAITH_SERVE_WINDOW", "4096"))
        shim = _probe_state.get("shim")
        if shim is not None:
            feat = shim.extract(image, question, solution_str, win)
        else:
            feat = ext.extract_efficient(
                image=image, question=question, response=solution_str,
                max_response_tokens=win,
            )
    except Exception as exc:
        return _faith_unavailable(f"extract failed: {exc}")
    import numpy as np
    # Guard: a dim mismatch (stale 2B 448-dim probe vs a 7B/14B extractor) or any
    # sklearn failure must degrade to neutral faith, NOT crash the reward worker.
    try:
        prob = float(probe.predict_proba(feat[None, :])[0, 1])
    except Exception as exc:
        return _faith_unavailable(f"predict_proba failed: {exc}")
    _faith_dead_streak[0] = 0  # genuine faith computed — reset the dead-man switch
    _faith_fail_window.append(0)
    # ONLINE PROBE REFRESH: buffer this genuine (feature, correctness) pair and,
    # every N steps, refit the probe on the policy's own rollouts (no-op unless
    # FAITH_ONLINE_REFIT_EVERY_STEPS is set — static arms are unaffected).
    _online_record(feat, correctness, data_source, question)
    return prob


# ---------------------------------------------------------------------------
# Composite reward encoding for the tiebreak advantage estimator
# ---------------------------------------------------------------------------

# ACTUAL encoding (single source of truth — keep comment/code in lockstep):
#   unformatted -> composite = 0.0 exactly
#   formatted   -> composite = 10.0 + correctness + clamp(faith, 0, 1) * 1e-3
# with correctness in {0, 1} and faith in [0, 1] (the -1.0 unavailable sentinel
# clamps to 0 here — the composite is a LEGACY consumer; the estimator reads
# faith raw from non_tensor_batch). Formatted values therefore live in
# [10.0, 11.001]; the formatted slot is 10.0 (an earlier comment said 100 while
# the code used 10.0 — the CODE was always right). Anything below 10.0 decodes
# as unformatted (0, 0, 0); the old `< 1.0` cutoff decoded garbage in [1, 10)
# to negative correctness.
#
# Decoder in the adv estimator reads (formatted, correctness, faith) back from
# this scalar; faith only affects ordering when correctness is tied.

def _encode_composite(correctness: float, faith: float, formatted: bool) -> float:
    if not formatted:
        return 0.0
    # formatted slot 10.0; +correctness in {0,1}; +clamp(faith) * 1e-3.
    return 10.0 + float(correctness) + max(0.0, min(1.0, float(faith))) * 1e-3


def decode_composite(composite: float) -> tuple[float, float, float]:
    """Decode (formatted, correctness, faith) from a composite scalar.
    Robust: any value below the 10.0 formatted slot -> unformatted (0, 0, 0);
    correctness/faith clamped to their valid ranges."""
    if composite < 10.0:
        return 0.0, 0.0, 0.0  # unformatted (only 0.0 is ever emitted here)
    formatted = 1.0
    after_fmt = composite - 10.0
    # faith*1e-3 <= 0.001 so a 0.5 threshold cleanly separates correctness 0/1.
    correctness = 1.0 if after_fmt >= 0.5 else 0.0
    faith = (after_fmt - correctness) * 1e3
    faith = max(0.0, min(1.0, faith))
    return formatted, correctness, faith


# ── Periodic faith-health heartbeat ───────────────────────────────────────────
# scorer_debug.jsonl caps at 20 rows/process, so it can't show whether faith stays
# genuine deep into a run. This emits a WINDOWED [faith-health] line every
# FAITH_HEALTH_EVERY rollouts (default 64 ≈ one training step) — each line covers
# only the last window, so a mid-run degradation (extract breaking → sentinel)
# shows up on the very next line. genuine = faith >= 0 (faith < 0 is the -1.0
# unavailable/failed sentinel). Each line also reports failure counts per
# response-length quartile (char lengths, quartiles within the window) — a
# length-skewed failure pattern (e.g. only long rollouts OOMing) is an
# anti-length bias on the jitter and must be visible in logs.
# Grep `[faith-health]` for the full trajectory.
import collections as _collections

_FAITH_HEALTH_EVERY = max(1, int(os.environ.get("FAITH_HEALTH_EVERY", "64")))
_faith_health = {"n": 0, "genuine": 0, "failed": 0, "sum": 0.0,
                 "per_v": _collections.defaultdict(lambda: [0, 0.0]),
                 "len_fail": []}  # (response_char_len, failed 0/1) per rollout
# Dead-man switch state (see _faith_unavailable): consecutive-failure streak
# (reset by _compute_faith on success) + rolling last-N fail/success window for
# the sustained-RATE trip (FAITH_FAIL_RATE_MAX, default 20%).
_FAITH_DEAD_MAX = int(os.environ.get("FAITH_DEAD_MAX", "512"))
_FAITH_FAIL_RATE_MAX = float(os.environ.get("FAITH_FAIL_RATE_MAX", "0.2"))
_faith_dead_streak = [0]
_faith_fail_window = _collections.deque(maxlen=_FAITH_DEAD_MAX)


def _faith_health_record(data_source: str, faith: float, resp_len: int) -> None:
    h = _faith_health
    h["n"] += 1
    h["len_fail"].append((int(resp_len), int(faith < 0.0)))
    if faith < 0.0:  # -1.0 sentinel
        h["failed"] += 1
    else:
        h["genuine"] += 1
        h["sum"] += faith
        pv = h["per_v"][str(data_source)]
        pv[0] += 1
        pv[1] += faith
    if h["n"] >= _FAITH_HEALTH_EVERY:
        mean = h["sum"] / h["genuine"] if h["genuine"] else 0.0
        per_v = {k: round(v[1] / v[0], 3) for k, v in h["per_v"].items() if v[0]}
        # Length-stratified failure counts: quartiles of response char-length
        # WITHIN this window; fail_by_len_q must stay flat in a healthy run.
        pairs = sorted(h["len_fail"])
        nq = len(pairs)
        qs = [pairs[i * nq // 4:(i + 1) * nq // 4] for i in range(4)]
        fail_q = [sum(f for _, f in q) for q in qs]
        n_q = [len(q) for q in qs]
        bounds = [q[-1][0] for q in qs if q]
        print(f"[faith-health] window={h['n']} genuine={h['genuine']} "
              f"failed={h['failed']} mean={mean:.3f} per_verifier={per_v} "
              f"fail_by_len_q={fail_q} n_by_len_q={n_q} len_q_upper={bounds}", flush=True)
        h["n"] = 0
        h["genuine"] = 0
        h["failed"] = 0
        h["sum"] = 0.0
        h["per_v"].clear()
        h["len_fail"] = []


# ── Faith example capture (high-faith vs low-faith response comparison) ───────
# Appends (verifier, correctness, formatted, faith, response) per genuine-faith
# rollout to a jsonl so high/low-faith responses can be compared offline (sort
# by faith). Path from FAITH_EXAMPLES_PATH; default is box-independent
# ~/faith_logs (a hardcoded path + bare except silently
# dropped every record elsewhere). Per-PID file to avoid concurrent-append
# corruption across the reward workers; merge with `cat <base>.* `. Capped to
# bound size; disable with FAITH_EXAMPLES_PATH="".
_FAITH_EXAMPLES_BASE = os.environ.get("FAITH_EXAMPLES_PATH",
                                      os.path.join(os.path.expanduser("~/faith_logs"),
                                                   "faith_examples.jsonl"))
_FAITH_EXAMPLES_MAX = int(os.environ.get("FAITH_EXAMPLES_MAX", "8000"))
_faith_examples_n = [0]
_faith_examples_warned = [False]


def _faith_example_record(data_source, faith, correctness, formatted, solution_str):
    if not _FAITH_EXAMPLES_BASE or _faith_examples_n[0] >= _FAITH_EXAMPLES_MAX:
        return
    try:
        import json as _json
        path = f"{_FAITH_EXAMPLES_BASE}.{os.getpid()}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as _f:
            _f.write(_json.dumps({
                "verifier": str(data_source),
                "faith": round(float(faith), 4),
                "correctness": float(correctness),
                "formatted": int(bool(formatted)),
                "response": solution_str[:2000],
            }) + "\n")
        _faith_examples_n[0] += 1
    except OSError as exc:
        if not _faith_examples_warned[0]:
            _faith_examples_warned[0] = True
            print(f"[scorers] WARNING: cannot write faith examples to "
                  f"{_FAITH_EXAMPLES_BASE!r}.<pid>: {exc} (set FAITH_EXAMPLES_PATH; "
                  f"further such warnings suppressed)", flush=True)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None):
    """Entry point for verl's custom reward function.

    Always returns a dict so verl's reward manager can plumb extras into
    `non_tensor_batch`:
      - "score": raw correctness 0/1 (this is the reward tensor → critic/score
        metrics & val-aux reward show correctness directly, comparable across
        vanilla and tiebreak runs).
      - "acc": raw correctness 0/1 (verl picks this up for val-core/{ds}/acc).
      - "faith": faith probe probability in [0,1]; -1.0 sentinel when
        unavailable (probe disabled/skipped OR extraction/scoring failed —
        the tiebreak estimator excludes faith<0 from within-tie jitter).
      - "formatted": 0/1 (did the response have a usable answer format).
      - "composite": legacy composite scalar (kept for downstream consumers).
    The tiebreak adv estimator reads faith/correctness/formatted from
    non_tensor_batch instead of decoding the composite scalar.
    """
    fn = _DISPATCH.get(str(data_source))
    if fn is None:
        _debug_log({"data_source": str(data_source), "solution": solution_str[:500], "gt": ground_truth, "score": 0.0, "note": "unknown_verifier"})
        # Same key set as the normal return — verl stacks these dicts across
        # rollouts and a missing key here would break the batch collation.
        return {"score": 0.0, "acc": 0.0, "faith": FAITH_FAIL_SENTINEL, "formatted": 0,
                "composite": 0.0, "iou_continuous": 0.0}

    if str(data_source) == "iou":
        image_size = None
        if extra_info is not None:
            img = extra_info.get("image")
            if img is not None:
                try:
                    image_size = (int(img.width), int(img.height))
                except Exception:
                    image_size = None
        out = score_iou(solution_str, ground_truth, image_size=image_size)
    else:
        out = fn(solution_str, ground_truth)

    correctness = float(out.get("score", 0.0)) if isinstance(out, dict) else float(out)
    formatted = True
    extra = out.get("extra_info") if isinstance(out, dict) else None
    if extra and "formatted" in extra:
        formatted = bool(extra["formatted"])

    # Phase-2: compute faith for ALL rollouts (formatted + unformatted).
    # The previous `if formatted else 0.0` guard was an optimization for the
    # pre-phase-2 drop_unformatted era where unformatted rollouts had
    # advantage=0 anyway. With no-drop tiebreak, faith on unformatted provides
    # within-bottom-tier ordering (rollouts that attended to the image rank
    # above those that hallucinated), matching the tiebreak philosophy:
    # faith differentiates when correctness signal is degenerate, and the
    # all-correctness=0 unformatted tier is the degenerate case.
    # Probe is slightly less calibrated on raw unformatted text but the small
    # noise cost is preferred over zero signal.
    # FAITH_SKIP_VERIFIERS: comma list of verifiers to skip faith for (speed
    # knob; skipped rollouts get the -1.0 sentinel -> excluded from within-tie
    # jitter -> exactly vanilla there). CAUTION under strict scoring: iou DOES
    # tie at exactly 0.0 (unformatted band), so skipping iou removes real
    # tiebreak signal — measure before using. Default: skip nothing.
    _faith_skip = {v.strip() for v in os.environ.get("FAITH_SKIP_VERIFIERS", "").split(",") if v.strip()}
    if os.environ.get("FAITH_PROBE_PATH") and str(data_source) not in _faith_skip:
        faith = _compute_faith(extra_info, solution_str,
                               correctness=correctness, data_source=data_source)
        _faith_health_record(data_source, faith, len(solution_str))
        if faith >= 0.0:  # only genuine-faith rollouts; -1.0 is the unavailable sentinel
            _faith_example_record(data_source, faith, correctness, formatted, solution_str)
    else:
        # Faith intentionally not computed (vanilla arm / skipped verifier) —
        # same unambiguous sentinel; 0.0 is a legitimate probability.
        faith = FAITH_FAIL_SENTINEL

    composite = _encode_composite(correctness, faith, formatted)

    # Continuous IoU for the iou verifier — used as the dense REWARD signal so
    # GRPO gets nonzero advantage even when no rollout crosses the 0.5 threshold.
    # `acc` (binary pass@0.5) is the metric verl reports for val-core/iou/acc.
    # The tiebreak adv estimator reads the `score` tensor + `faith` from
    # non_tensor_batch (NOT `acc`), so making the reward continuous does NOT
    # affect the tiebreaker's lex-sort behavior. 0.0 for non-iou verifiers
    # (no meaning, just a stable shape).
    iou_continuous = 0.0
    if str(data_source) == "iou" and extra is not None:
        iou_continuous = float(extra.get("iou", 0.0))

    # For iou, the "signal" used as the reward is the continuous IoU value.
    # For digit/mcq/math, the natural reward is binary correctness (no partial
    # credit possible — exact-match scorers).
    signal = iou_continuous if str(data_source) == "iou" else correctness

    # FAITH_REWARD_ALPHA controls the "composite" reward family (Liu et al.-style
    # weighted scalar). When set, score = alpha*signal + (1-alpha)*faith.
    # When unset, score is just `signal` (used by vanilla and tiebreak — tiebreak
    # reads faith separately from non_tensor_batch).
    alpha_env = os.environ.get("FAITH_REWARD_ALPHA")
    if alpha_env is not None:
        try:
            alpha = float(alpha_env)
        except ValueError:
            alpha = 1.0
        alpha = max(0.0, min(1.0, alpha))
        # Mirror tiebreak's "drop_unformatted" behavior: unformatted rollouts
        # get a reward of 0 (so they don't get positive credit from faith alone).
        # max(0, faith): the -1.0 unavailable sentinel must contribute 0 to the
        # weighted reward, never a negative penalty.
        if formatted:
            score_out = alpha * signal + (1.0 - alpha) * max(0.0, float(faith))
        else:
            score_out = 0.0
    else:
        score_out = signal

    # Literature-standard format reward (R1, VLAA-Thinker, Open-R1): small bonus
    # for proper <think>/<answer> tag structure. Gives vanilla GRPO a format
    # signal without needing drop_unformatted gradient masking. Tiebreak's
    # lex-sort sorts by (correctness, faith) — the bonus is constant within
    # formatted rollouts (the only ones that survive tiebreak's drop), so the
    # rank order is preserved. Default 0.1 per literature; set FORMAT_BONUS=0
    # to disable for ablation.
    FORMAT_BONUS = float(os.environ.get("FORMAT_BONUS", "0.1"))
    if formatted and FORMAT_BONUS != 0.0:
        score_out = float(score_out) + FORMAT_BONUS

    _debug_log({
        "data_source": str(data_source),
        "solution": solution_str[:500],
        "gt": ground_truth,
        "correctness": correctness,
        "faith": faith,
        "formatted": formatted,
        "composite": composite,
        "score_out": score_out,
        "iou_continuous": iou_continuous,
        "extracted": extra.get("raw_extracted") if extra else None,
    })
    # `acc` is a REPORTED METRIC ONLY: verl logs it as val-core/{data_source}/acc.
    # It does NOT feed the reward (`score`) or the tiebreaker (which reads the
    # `score` tensor + `faith` from non_tensor_batch, never `acc`). For iou we
    # report BINARY pass@0.5 so the logged metric is an honest accuracy; the
    # continuous IoU stays the reward (`score`) and is also logged as
    # `iou_continuous`. For digit/mcq/math, `signal` is already binary correctness.
    # Note: with continuous iou reward, exact ties are rare, so the faith probe is
    # mostly inert on iou (still active on digit/mcq/math where reward is binary).
    acc_metric = (1.0 if iou_continuous >= 0.5 else 0.0) if str(data_source) == "iou" else signal
    return {
        "score": score_out,
        "acc": acc_metric,
        "faith": float(faith),
        "formatted": int(bool(formatted)),
        "composite": float(composite),
        "iou_continuous": iou_continuous,
    }




# ============================================================================
# ONLINE PROBE REFRESH v3 (opt-in via FAITH_ONLINE_REFIT_EVERY_STEPS)
# ----------------------------------------------------------------------------
# Base probe is frozen on BASE-model features; the policy drifts, so it goes
# stale. v3 keeps it fresh WITHOUT data-starvation:
#   - KEEP-ALL buffer (never discard) -> n grows into the thousands, p<n, no
#     overfitting; plus EXPONENTIAL-DECAY recency weighting (half-life
#     FAITH_ONLINE_HALFLIFE_STEPS, default 30 steps) so recent rollouts dominate.
#   - PER-(verifier,class) BALANCING via sample weights (mirrors train_probe's
#     _balance_per_verifier, but as weights so no data is thrown away).
#   - GATE: integrate ONLY if the probe passes ALL 3 checks (c1 val_auroc>0.55,
#     c2 wg_std>0.05, c3 perm<0.55) OVERALL *and* for every evaluable verifier.
#     Else keep the prior probe and keep accumulating.
# Static arms are byte-identical (runs only when the flag is set).
# ============================================================================
_online = {"X": [], "y": [], "vt": [], "pid": [], "lock": threading.Lock(),
           "next_at": None, "n_attempts": 0, "n_swaps": 0, "enabled": None,
           "n_base": 0, "seeded": False}


def _online_seed_from_base():
    """Pre-load the buffer with the BASE probe's training rollouts (which pass all
    checks, wg_std~0.08). These anchor c2/c1/c3 so that as policy rollouts are added
    the probe KEEPS passing; base samples get a floor weight (never fully decayed),
    policy samples are recency-weighted on top. Path via FAITH_ONLINE_SEED_NPZ."""
    if _online["seeded"]:
        return
    _online["seeded"] = True
    path = os.environ.get("FAITH_ONLINE_SEED_NPZ", "")
    if not path or not os.path.exists(path):
        print(f"[faith-online] seed: no base npz ({path!r}) — buffer starts EMPTY", flush=True)
        return
    import numpy as _np
    d = _np.load(path, allow_pickle=True)
    X = d["X"]; y = d["y"].astype(int)
    vt = _np.asarray(d["verifier_type"]) if "verifier_type" in d.files else _np.array(["base"] * len(y))
    pid = _np.asarray(d["prompt_id"]) if "prompt_id" in d.files else _np.arange(len(y))
    # Subsample by WHOLE PROMPTS (preserve within-group structure so c2 stays valid)
    # to keep per-refit cost low — refits run inside the reward loop every N steps.
    seed_max = int(os.environ.get("FAITH_ONLINE_SEED_MAX", "8000"))
    if len(y) > seed_max:
        srng = _np.random.default_rng(0)
        uniq = _np.unique(pid); srng.shuffle(uniq)
        keep = set(); cnt = 0
        for p in uniq:
            keep.add(int(p)); cnt += int((pid == p).sum())
            if cnt >= seed_max:
                break
        mask = _np.array([int(p) in keep for p in pid])
        X, y, vt, pid = X[mask], y[mask], vt[mask], pid[mask]
        print(f"[faith-online] seed: subsampled to {len(y)} rows / {len(keep)} whole prompts (cap {seed_max})", flush=True)
    for i in range(len(y)):
        _online["X"].append(_np.asarray(X[i], dtype=_np.float32))
        _online["y"].append(int(y[i]))
        _online["vt"].append(str(vt[i]))
        _online["pid"].append(int(pid[i]) & 0x7fffffff)
    _online["n_base"] = int(len(y))
    print(f"[faith-online] SEEDED buffer with {len(y)} base rollouts "
          f"(pos={int(y.sum())}, anchors all 3 checks); policy rollouts accumulate on top.", flush=True)


def _online_enabled() -> bool:
    if _online["enabled"] is None:
        v = os.environ.get("FAITH_ONLINE_REFIT_EVERY_STEPS", "").strip()
        _online["enabled"] = bool(v) and v != "0"
        if _online["enabled"]:
            print(f"[faith-online] ENABLED v3: refit every {v} steps, KEEP-ALL buffer + exp-decay "
                  f"recency (half-life {os.environ.get('FAITH_ONLINE_HALFLIFE_STEPS','30')} steps) + "
                  f"per-(verifier,class) balancing; GATE=ALL 3 checks overall AND per-verifier.", flush=True)
    return _online["enabled"]


def _online_record(feat, correctness, data_source, question):
    if not _online_enabled() or correctness is None or data_source is None:
        return
    import numpy as _np
    with _online["lock"]:
        if not _online["seeded"]:
            _online_seed_from_base()  # pre-load base rollouts once, before the first policy sample
        _online["X"].append(_np.asarray(feat, dtype=_np.float32))
        _online["y"].append(int(round(float(correctness))))
        _online["vt"].append(str(data_source))
        _online["pid"].append(hash(str(question)) & 0x7fffffff)
        every = int(os.environ.get("FAITH_ONLINE_REFIT_EVERY_STEPS", "10"))
        per_step = int(os.environ.get("FAITH_ROLLOUTS_PER_STEP", "128"))
        thr = max(1, every * per_step)
        if _online["next_at"] is None:
            # first refit after `thr` POLICY samples past the seed (not on the seed itself)
            _online["next_at"] = _online["n_base"] + thr
        if len(_online["y"]) >= _online["next_at"]:
            _online["next_at"] += thr  # attempt every N steps on ALL history (keep-all)
            try:
                _online_refit_and_swap()
            except Exception as exc:  # never crash the reward loop
                print(f"[faith-online] refit FAILED (keeping prior probe): {exc}", flush=True)


def _online_auroc(y, s):
    import numpy as _np
    from sklearn.metrics import roc_auc_score
    y = _np.asarray(y); s = _np.asarray(s)
    if len(_np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _online_wg_std(pid, s):
    import numpy as _np
    pid = _np.asarray(pid); s = _np.asarray(s)
    stds = [float(s[pid == p].std()) for p in _np.unique(pid) if (pid == p).sum() >= 2]
    return float(_np.mean(stds)) if stds else 0.0


def _online_weights(vt, y, order, N, per_step, halflife, n_base):
    """sample_weight = per-(verifier,class) inverse-frequency balance x recency.
    Base-seed samples (original index < n_base) get a fixed FLOOR weight so they never
    decay away and keep anchoring the checks; policy samples get exp-decay recency."""
    import numpy as _np
    from collections import Counter
    order = _np.asarray(order, dtype=_np.float64)
    floor = float(os.environ.get("FAITH_ONLINE_BASE_FLOOR", "0.25"))
    age_steps = (N - 1 - order) / max(1, per_step)
    rec = 0.5 ** (age_steps / max(1e-6, halflife))
    rec = _np.where(order < n_base, floor, rec)  # base anchor: fixed floor, never decays
    cells = list(zip([str(v) for v in vt], [int(c) for c in y]))
    cnt = Counter(cells); ncell = max(1, len(cnt))
    bal = _np.array([1.0 / (ncell * cnt[c]) for c in cells], dtype=_np.float64)
    w = rec * bal
    m = float(w.mean())
    return w / m if m > 0 else w  # normalize to mean 1


def _online_fit(X, y, w):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, penalty="l2", n_jobs=1)
    clf.fit(X, y, sample_weight=w)
    return clf


def _online_eval_checks(clf, perm_clfs, Xva, yva, pva):
    import numpy as _np
    s = clf.predict_proba(Xva)[:, 1]
    va = _online_auroc(yva, s); wg = _online_wg_std(pva, s)
    perms = [_online_auroc(yva, cp.predict_proba(Xva)[:, 1]) for cp in perm_clfs]
    pm = float(_np.nanmean(perms)) if perms else float("nan")
    return {"val_auroc": round(va, 3), "wg_std": round(wg, 3), "perm": round(pm, 3),
            "c1": bool(va > 0.55), "c2": bool(wg > 0.05), "c3": bool(pm < 0.55)}


def _online_refit_and_swap():
    import numpy as _np
    X = _np.stack(_online["X"]); y = _np.asarray(_online["y"])
    vt = _np.asarray(_online["vt"]); pid = _np.asarray(_online["pid"])
    N = int(len(y)); order = _np.arange(N)
    per_step = int(os.environ.get("FAITH_ROLLOUTS_PER_STEP", "128"))
    halflife = float(os.environ.get("FAITH_ONLINE_HALFLIFE_STEPS", "30"))
    # c3 perm-null is chance-centered but wide (std ~0.1/shuffle; ~1/3 of single
    # shuffles exceed 0.55 by chance), so n=5 spuriously REJECTS good probes on an
    # unlucky draw. n=25 stabilizes the perm mean to ~0.51. (verified 30-shuffle run)
    n_perm = int(os.environ.get("FAITH_ONLINE_NPERM", "25"))
    n_base = int(_online["n_base"])
    _online["n_attempts"] += 1; a = _online["n_attempts"]; seed = 42 + a
    if int(y.sum()) < 10 or int((1 - y).sum()) < 10:
        print(f"[faith-online] attempt#{a}: SKIP (N={N} pos={int(y.sum())}, need >=10/class) — accumulating", flush=True)
        return
    rng = _np.random.default_rng(seed)
    uniq = _np.unique(pid); rng.shuffle(uniq)
    nval = max(1, int(len(uniq) * 0.2)); valp = set(uniq[:nval].tolist())
    vm = _np.array([p in valp for p in pid])
    if len(_np.unique(y[~vm])) < 2 or len(_np.unique(y[vm])) < 2:
        print(f"[faith-online] attempt#{a}: SKIP (one-class train/val split) — accumulating", flush=True)
        return
    Xtr, ytr, vttr, ordtr = X[~vm], y[~vm], vt[~vm], order[~vm]
    Xva, yva, vtva, pva = X[vm], y[vm], vt[vm], pid[vm]
    wtr = _online_weights(vttr, ytr, ordtr, N, per_step, halflife, n_base)
    clf = _online_fit(Xtr, ytr, wtr)
    perm_clfs = []
    for s in range(n_perm):
        prng = _np.random.default_rng(seed + s + 1); ysh = ytr.copy(); prng.shuffle(ysh)
        if len(_np.unique(ysh)) < 2:
            continue
        perm_clfs.append(_online_fit(Xtr, ysh, wtr))
    ov = _online_eval_checks(clf, perm_clfs, Xva, yva, pva)
    # c2 (within-group wg_std>0.05) is a MAGNITUDE bar; the argsort tiebreak uses only
    # ORDERING, so c2 is LOGGED but NOT gated (else a correctness/ranking probe, whose
    # within-group spread is intrinsically small, could never be accepted).
    ov_pass = ov["c1"] and ov["c3"]
    print(f"[faith-online] attempt#{a} OVERALL N={N} pos={int(y.sum())} "
          f"val_auroc={ov['val_auroc']} wg_std={ov['wg_std']} perm={ov['perm']} "
          f"CHECKS[c1={ov['c1']} c2={ov['c2']}(advisory) c3={ov['c3']}] PASS={ov_pass}", flush=True)
    pv_all_pass = True; pv_seen = 0
    for v in sorted(set(vtva.tolist())):
        m = vtva == v
        if int(m.sum()) < 20 or len(_np.unique(yva[m])) < 2:
            print(f"[faith-online] attempt#{a}   verifier={v}: n_val={int(m.sum())} untestable (skip, doesn't block)", flush=True)
            continue
        pv_seen += 1
        rv = _online_eval_checks(clf, perm_clfs, Xva[m], yva[m], pva[m])
        vp = rv["c1"] and rv["c3"]  # c2 logged, not gated (see overall note)
        if not vp:
            pv_all_pass = False
        print(f"[faith-online] attempt#{a}   verifier={v}: n_val={int(m.sum())} "
              f"val_auroc={rv['val_auroc']} wg_std={rv['wg_std']} perm={rv['perm']} "
              f"CHECKS[c1={rv['c1']} c2={rv['c2']}(advisory) c3={rv['c3']}] PASS={vp}", flush=True)
    gate = ov_pass and pv_all_pass
    if not gate:
        print(f"[faith-online] attempt#{a}: REJECTED (overall={ov_pass}, per_verifier_all={pv_all_pass}, "
              f"{pv_seen} verifiers tested) — keep prior probe", flush=True)
        return
    with _probe_lock:
        _probe_state["probe"] = clf
    _online["n_swaps"] += 1
    print(f"[faith-online] attempt#{a}: SWAPPED probe — ALL 3 checks passed OVERALL + {pv_seen} verifiers (N={N})", flush=True)
