"""Rewrite an existing v1/v2 vlaa_subset parquet to the v3 exact-payload prompt format.

Why this exists: regenerating the parquet from scratch (prep_vlaa_subset.py
--stage parquet) needs the full 53 GB image tree. The v1 parquets embed the image bytes, and the ONLY difference
between v1 and v3 rows is the format-hint sentence appended to the user prompt
— so we strip the known v1 hint suffix and append the v3 hint, byte-identical
everywhere else (system prompt, question, images, gt).

FAIL-LOUD: every row must end with its verifier's expected v1 or v2 suffix,
else we abort — no silent partial rewrites.

Usage:
    python rewrite_parquet_hints_v3.py \
        --in-dir data/vlaa_subset_5400_v1 \
        --out-dir data/vlaa_subset_5400
Writes {train,val,test,...}.parquet + parquet_format_version.json (v3 stamp).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_vlaa_subset import _FORMAT_HINT, _FORMAT_HINT_VERSION, _VL_THINKING_SYSTEM

# Exact v1 suffixes as they exist in the on-disk parquet (verified against
# the v1 train.parquet).
# NOTE the v1 iou wording differs from prep's later in-code hint (it carries
# the "or in the model's native coordinate space" clause) — strip THIS text.
_V1_HINT = {
    "digit": " Answer with a single number, inside <answer>...</answer>.",
    "mcq":   " End your response with the chosen letter (A, B, C, D, or E).",
    "math":  " Put the final answer inside <answer>\\boxed{...}</answer>.",
    "iou":   (" Answer with the bounding box as [x1, y1, x2, y2] (normalized to"
              " [0, 1] or in the model's native coordinate space), inside"
              " <answer>...</answer>."),
}

# Exact v2 suffixes (byte-identical to _FORMAT_HINT at the v2-mcq-tagged-2026-06
# vintage, git HEAD:scripts/prep_vlaa_subset.py) — a parquet staged from a v2
# prep run carries these instead of the v1 strings.
_V2_HINT = {
    "digit": " Answer with a single number, inside <answer>...</answer>.",
    "mcq":   " Put the chosen letter (A, B, C, D, or E) inside <answer>...</answer>.",
    "math":  " Put the final answer inside <answer>\\boxed{...}</answer>.",
    "iou":   " Answer with the bounding box as [x1, y1, x2, y2] normalized to [0, 1], inside <answer>...</answer>.",
}

# Known hint vintages, tried in order per row.
_KNOWN_HINTS = (("v1", _V1_HINT), ("v2", _V2_HINT))


def rewrite_split(in_path: Path, out_path: Path) -> dict:
    import pyarrow.parquet as pq
    from datasets import Dataset, Features, Value, Sequence, Image as ImageFeature

    df = pq.read_table(in_path).to_pandas()
    counts: dict[str, int] = {}
    rows = []
    for i, r in df.iterrows():
        vt = str(r["data_source"])
        if vt not in _V1_HINT:
            raise SystemExit(f"{in_path.name} row {i}: unknown verifier {vt!r}")
        prompt = list(r["prompt"])
        user = dict(prompt[1])
        content = user["content"]
        for _vintage, hints in _KNOWN_HINTS:
            hint = hints[vt]
            if content.endswith(hint):
                break
        else:
            raise SystemExit(
                f"{in_path.name} row {i} ({vt}): user content does not end with a "
                f"known v1/v2 hint — refusing a partial rewrite.\n  tail: {content[-160:]!r}")
        user["content"] = content[: -len(hint)] + _FORMAT_HINT[vt]
        sys_msg = dict(prompt[0])
        if sys_msg.get("content") != _VL_THINKING_SYSTEM:
            raise SystemExit(f"{in_path.name} row {i}: unexpected system prompt")
        # images arrive from pandas as {bytes,path} dicts — pass through untouched.
        rows.append({
            "data_source": vt,
            "prompt": [sys_msg, user],
            "images": [dict(img) for img in r["images"]],
            "ability": str(r["ability"]),
            "reward_model": dict(r["reward_model"]),
        })
        counts[vt] = counts.get(vt, 0) + 1

    features = Features({
        "data_source": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "images": Sequence(ImageFeature()),
        "ability": Value("string"),
        "reward_model": {"style": Value("string"), "ground_truth": Value("string")},
    })
    ds = Dataset.from_list(rows, features=features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out_path))
    print(f"[v3-rewrite] {in_path.name}: {len(rows)} rows -> {out_path}  per-verifier={counts}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    if in_dir.resolve() == out_dir.resolve():
        raise SystemExit("in-dir must differ from out-dir (keep the v1 originals)")

    parquets = sorted(in_dir.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquets in {in_dir}")
    for p in parquets:
        rewrite_split(p, out_dir / p.name)

    # Post-write verification FIRST: reload each output, confirm every row ends
    # with its verifier's v3 hint and row counts match. The stamp is what the four
    # phase-2 launchers grep for (v3-exact-payload + hint_sha1) before burning GPUs,
    # so it must be written ONLY after verification passes — a stamp on unverified
    # parquets would let every arm train on silently-wrong prompts.
    import pyarrow.parquet as pq
    stamp_path = out_dir / "parquet_format_version.json"
    if stamp_path.exists():
        stamp_path.unlink()  # clear any stale stamp so a mid-run failure can't leave a passing one
    for p in parquets:
        outp = out_dir / p.name
        din = pq.read_table(p, columns=["data_source"])
        dout = pq.read_table(outp).to_pandas()
        assert len(dout) == din.num_rows, (p.name, len(dout), din.num_rows)
        for i, r in dout.iterrows():
            c = r["prompt"][1]["content"]
            assert c.endswith(_FORMAT_HINT[str(r["data_source"])]), (p.name, i, c[-120:])
        print(f"[v3-rewrite] verified {outp.name}: {len(dout)} rows all end with v3 hints")
    print("[v3-rewrite] ALL SPLITS VERIFIED ✓")

    # Only now stamp — the guard file certifies verified-on-disk parquets.
    hint_sha1 = hashlib.sha1(
        (json.dumps(_FORMAT_HINT, sort_keys=True) + _VL_THINKING_SYSTEM).encode()
    ).hexdigest()[:12]
    stamp = {
        "format_hint_version": _FORMAT_HINT_VERSION,
        "hint_sha1": hint_sha1,
        "format_hints": _FORMAT_HINT,
        "rewritten_from": str(in_dir),
    }
    with open(stamp_path, "w") as f:
        json.dump(stamp, f, indent=2)
    print(f"[v3-rewrite] stamped {_FORMAT_HINT_VERSION} (hint_sha1={hint_sha1}) -> {stamp_path}")


if __name__ == "__main__":
    main()
