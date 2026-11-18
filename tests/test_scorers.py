"""Smoke tests for per-verifier scorers — runs against real records from each split.

compute_score returns a FLAT dict: {score, acc, faith, formatted, composite, ...}.
Asserts the load-bearing invariants so this guards the scorer against regressions:
  - perfect formatted response   -> formatted==1, acc==1 (non-math)
  - wrong  formatted response    -> formatted==1, acc==0 (non-math)
  - unformatted response         -> formatted==0
  - iou acc is BINARY pass@0.5   -> in {0.0, 1.0}
Run with FAITH_PROBE_PATH unset (faith=0.0, no model load) so it's CPU-fast.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Keep the smoke test self-contained and fast: never load the faith model.
os.environ.pop("FAITH_PROBE_PATH", None)

sys.path.insert(0, str(Path(__file__).parent.parent))
from verl_faith.scorers import (
    compute_score, _MATH_VERIFY_OK,
    _find_boxed, decode_composite, _encode_composite, _iou, _parse_bbox,
    FAITH_FAIL_SENTINEL, _faith_unavailable,
)

MANIFEST = os.environ.get(
    "VLAA_SPLIT_MANIFEST",
    "data/vlaa_subset_5400/split_manifest.json")


def perfect_response(gt: str, vt: str) -> str:
    # Payload-EXACT bodies (strict mode requires the tag to contain ONLY the
    # answer; these also parse fine in lenient mode).
    if vt == "math":
        g = gt.strip()
        body = g if g.startswith("\\boxed{") else f"\\boxed{{{g}}}"
    elif vt == "iou":
        body = gt.strip()
    else:
        body = gt.strip()
    return f"<think>reasoning here</think>\n<answer>{body}</answer>"


def wrong_response(vt: str, gt: str) -> str:
    # Wrong ANSWER but payload-exact FORMAT (so formatted==1 in both modes).
    if vt == "mcq":
        body = "B" if gt.strip().upper() != "B" else "C"
    elif vt == "digit":
        body = "999" if gt.strip() != "999" else "998"
    elif vt == "math":
        body = r"\boxed{999999}" if "999999" not in gt else r"\boxed{999998}"
    elif vt == "iou":
        body = "[0.0, 0.0, 0.01, 0.01]"
    else:
        body = "garbage"
    return f"<think>...</think>\n<answer>{body}</answer>"


def sloppy_response(gt: str, vt: str) -> str:
    """RIGHT answer wrapped in prose INSIDE the tag — strict mode must score 0
    and withhold the format bonus; lenient mode may still salvage it."""
    g = gt.strip()
    body = {"digit": f"the count is {g}",
            "mcq": f"{g}) because of the angle",
            "math": f"The answer is \\boxed{{{g}}}" if not g.startswith("\\boxed") else f"The answer is {g}",
            "iou": f"the box is {g}"}.get(vt, g)
    return f"<think>...</think>\n<answer>{body}</answer>"


def unformatted_response(gt: str, vt: str) -> str:
    # math gt embeds \boxed{}, which the scorer (correctly) counts as formatted
    # per the R1 convention; use a marker-free string so this stays unformatted.
    if vt == "math":
        return "the answer is three probably idk"
    return f"the answer is {gt} probably idk"  # no <answer>/</think>/\\boxed


_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        _failures.append(msg)


def run_suite(seen: dict, strict: bool) -> None:
    os.environ["STRICT_CORRECTNESS"] = "1" if strict else "0"
    mode = "STRICT" if strict else "LENIENT"
    hdr = (f"[{mode}] {'verifier':<8} {'source':<12} {'perfect(s/acc/fmt)':<20} "
           f"{'wrong(s/acc/fmt)':<20} {'sloppy(acc/fmt)':<16} {'unfmt(fmt)':<10} gt")
    print("\n" + hdr)
    print("-" * 120)
    for (vt, src), r in sorted(seen.items()):
        gt = r["gt"]
        p = compute_score(vt, perfect_response(gt, vt), gt)
        w = compute_score(vt, wrong_response(vt, gt), gt)
        s = compute_score(vt, sloppy_response(gt, vt), gt)
        u = compute_score(vt, unformatted_response(gt, vt), gt)
        print(f"[{mode}] {vt:<8} {src:<12} "
              f"{p['score']:.2f}/{p['acc']:.2f}/{p['formatted']:<11} "
              f"{w['score']:.2f}/{w['acc']:.2f}/{w['formatted']:<11} "
              f"{s['acc']:.2f}/{s['formatted']:<11} "
              f"{u['formatted']:<10} {gt[:30]!r}")

        # --- invariants (both modes) ---
        check(p["formatted"] == 1, f"{mode} {vt}/{src}: perfect should be formatted==1, got {p['formatted']}")
        check(w["formatted"] == 1, f"{mode} {vt}/{src}: wrong should be formatted==1, got {w['formatted']}")
        check(u["formatted"] == 0, f"{mode} {vt}/{src}: unformatted should be formatted==0, got {u['formatted']}")
        if vt == "iou":
            check(p["acc"] in (0.0, 1.0), f"{mode} iou/{src}: acc must be binary pass@0.5, got {p['acc']}")
            check(w["acc"] in (0.0, 1.0), f"{mode} iou/{src}: acc must be binary pass@0.5, got {w['acc']}")
            check(p["acc"] == 1.0, f"{mode} iou/{src}: perfect (gt box) should pass@0.5, got acc={p['acc']}")
            check(w["acc"] == 0.0, f"{mode} iou/{src}: tiny wrong box should fail@0.5, got acc={w['acc']}")
        elif vt in ("digit", "mcq"):
            check(p["acc"] == 1.0, f"{mode} {vt}/{src}: perfect acc should be 1.0, got {p['acc']}")
            check(w["acc"] == 0.0, f"{mode} {vt}/{src}: wrong acc should be 0.0, got {w['acc']}")
        elif vt == "math" and _MATH_VERIFY_OK:
            check(p["acc"] >= w["acc"], f"{mode} math/{src}: perfect acc({p['acc']}) should be >= wrong acc({w['acc']})")
        # format bonus is added on top of acc for formatted rollouts
        check(p["score"] >= p["acc"], f"{mode} {vt}/{src}: formatted score({p['score']}) should be >= acc({p['acc']})")

        # --- strict-only invariants: payload-exact, no salvage parsing ---
        if strict:
            check(s["acc"] == 0.0,
                  f"STRICT {vt}/{src}: sloppy payload (prose in tag) must score 0, got {s['acc']}")
            check(s["formatted"] == 0,
                  f"STRICT {vt}/{src}: sloppy payload must lose the format bonus, got formatted={s['formatted']}")
            check(s["score"] == 0.0,
                  f"STRICT {vt}/{src}: sloppy total score must be 0, got {s['score']}")


def run_pure_tests() -> None:
    """Manifest-free, CPU-only unit tests — always runnable. Covers the load-bearing pure logic and, above
    all, the -1.0 faith sentinel convention the tiebreak estimator depends on."""
    print("\n[pure] manifest-free unit tests")

    # --- -1.0 faith sentinel (THE cross-file invariant) ---
    check(FAITH_FAIL_SENTINEL == -1.0, f"sentinel must be -1.0, got {FAITH_FAIL_SENTINEL}")
    check(_faith_unavailable("test") == -1.0, "_faith_unavailable must return the -1.0 sentinel")
    # With FAITH_PROBE_PATH unset (popped at import), compute_score must emit the
    # sentinel for faith, NOT 0.0 — 0.0 is a legitimate probability and would be
    # read as genuine low faith by any non-sentinel-aware consumer.
    os.environ["STRICT_CORRECTNESS"] = "1"
    r = compute_score("mcq", "<think>x</think>\n<answer>B</answer>", "B",
                      extra_info={"image": None, "question": "q"})
    fv = r["faith"] if isinstance(r, dict) else None
    check(fv == FAITH_FAIL_SENTINEL,
          f"faith with probe disabled must be the -1.0 sentinel, got {fv}")

    # --- nested-brace \boxed extraction ---
    nested = _find_boxed(r"\boxed{\frac{\sqrt{2}}{2}}")
    check(nested == r"\frac{\sqrt{2}}{2}", f"_find_boxed nested braces failed: {nested!r}")
    check(_find_boxed(r"x=\boxed{42}.") == "42", "_find_boxed simple failed")
    check(_find_boxed("no box here") is None, "_find_boxed should return None when absent")

    # --- composite encode/decode round-trip (formatted slot 10.0) ---
    for corr, faith, fmt in [(1.0, 0.7, True), (0.0, 0.2, True), (1.0, 0.9, False)]:
        c = _encode_composite(corr, faith, fmt)
        f_out, k_out, ft_out = decode_composite(c)
        if fmt:
            check(k_out == corr, f"composite corr round-trip: {corr} -> {k_out}")
            check(abs(ft_out - max(0.0, faith)) < 1e-2, f"composite faith round-trip: {faith} -> {ft_out}")
            check(f_out == 1.0, "composite formatted flag round-trip")
        else:
            check((f_out, k_out, ft_out) == (0.0, 0.0, 0.0), "unformatted composite -> (0,0,0)")
    # sentinel faith must not decode to a negative correctness (legacy clamp)
    f_out, k_out, ft_out = decode_composite(_encode_composite(1.0, -1.0, True))
    check(k_out == 1.0 and ft_out >= 0.0, "composite must clamp -1.0 faith, not corrupt correctness")

    # --- iou geometry + /1000 grid heuristic ---
    check(abs(_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9, "identical boxes IoU==1")
    check(_iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0, "disjoint boxes IoU==0")
    check(_parse_bbox("[0.1, 0.2, 0.3, 0.4]") == (0.1, 0.2, 0.3, 0.4), "_parse_bbox basic")
    # Scorer convention: gt is NORMALIZED [0,1]; Qwen emits the [0,1000] grid, so
    # pred /1000 must bridge to gt. Grid-space pred vs normalized gt -> IoU 1.0.
    grid = compute_score("iou", "<think>x</think>\n<answer>[100, 100, 200, 200]</answer>",
                         "[0.1, 0.1, 0.2, 0.2]")
    check(grid["acc"] == 1.0, f"iou /1000 grid bridge should pass, got {grid['acc']}")
    # A normalized-space pred (max coord <= 2) must NOT be divided by 1000.
    norm = compute_score("iou", "<think>x</think>\n<answer>[0.1, 0.1, 0.2, 0.2]</answer>",
                         "[0.1, 0.1, 0.2, 0.2]")
    check(norm["acc"] == 1.0, f"iou normalized-space perfect box should pass, got {norm['acc']}")

    # --- strict last-<answer>-tag rule + digit edge forms (stable interface) ---
    os.environ["STRICT_CORRECTNESS"] = "1"
    multi = compute_score("mcq", "<answer>A</answer> then <answer>B</answer>", "B")
    check(multi["acc"] == 1.0, f"last-<answer>-tag should win (B), got acc={multi['acc']}")
    d1 = compute_score("digit", "<think>x</think>\n<answer>5.</answer>", "5")
    d2 = compute_score("digit", "<think>x</think>\n<answer>+5</answer>", "5")
    check(d1["acc"] in (0.0, 1.0) and d2["acc"] in (0.0, 1.0),
          "digit edge forms must score a definite 0/1, not crash")


def main() -> None:
    print(f"math_verify available: {_MATH_VERIFY_OK}")

    # Pure tests ALWAYS run (no manifest / GPU needed) — the offline coverage the
    # 8B box relies on when the NFS split manifest is absent.
    run_pure_tests()

    if not os.path.isfile(MANIFEST):
        print(f"\n[smoke] MANIFEST not found: {MANIFEST}")
        print("[smoke] set VLAA_SPLIT_MANIFEST to this box's split_manifest.json to run the "
              "manifest-driven smoke suite; skipping it (pure tests still ran).")
    else:
        manifest = json.load(open(MANIFEST))
        seen: dict = {}
        # Guarantee all 4 base verifiers appear if present (not just the first 6
        # manifest-order combos).
        for r in manifest["records"]["test"]:
            key = (r["verifier_type"], r["image_source"])
            if key not in seen:
                seen[key] = r
        want_v = {"mcq", "math", "iou", "digit"}  # the 8B subset's 4 base verifiers
        have_v = {vt for vt, _ in seen}
        print(f"[smoke] verifier coverage: {sorted(have_v)} (missing: {sorted(want_v - have_v)})")
        for strict in (False, True):
            run_suite(seen, strict)
        print(f"\nChecked {len(seen)} verifier/source combos in BOTH modes.")

    if _failures:
        print(f"\nFAILED ({len(_failures)}):")
        for m in _failures:
            print("  -", m)
        sys.exit(1)
    print("\nALL ASSERTIONS PASSED ✓")


if __name__ == "__main__":
    main()
