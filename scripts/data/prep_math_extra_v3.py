"""Build the extra-math parquet used for probe collection.

Samples N math prompts from the VLAA-Thinking annotations that are not in the
5,400-example subset, deduplicated by record id and question text against the
train, validation and test splits.

Usage:
    python scripts/data/prep_math_extra_v3.py \\
        --annotations data/annotations/vlaa_thinking_annotations.json \\
        --manifest data/vlaa_subset_5400/split_manifest.json \\
        --img-root data/vlaa_thinking/images_extracted \\
        --out-dir data/vlaa_math_extra \\
        --n 2000
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_vlaa_subset import _FORMAT_HINT, _FORMAT_HINT_VERSION, _VL_THINKING_SYSTEM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--manifest", required=True, help="split_manifest.json of the 5400 subset")
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    from datasets import Dataset, Features, Value, Sequence, Image as ImageFeature

    # Dedupe key: record id + (image, question) pair. NOT bare question text —
    # synthesis questions are templated (the same wording recurs across many
    # different images), and same-question-different-image is a genuinely
    # different task, so text-only dedupe would shrink the pool ~4x for nothing.
    manifest = json.load(open(args.manifest))
    used_ids, used_pairs = set(), set()
    for split, recs in manifest["records"].items():
        for r in recs:
            if r["verifier_type"] == "math":
                used_ids.add(r["id"])
                used_pairs.add((r["image"], r["question"].strip()))
    print(f"[math-extra] subset math records excluded (train+val+test): {len(used_ids)}")

    pool = []
    with open(args.annotations) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["verifier_type"] != "math":
                continue
            if d["id"] in used_ids or (d["image"], d["question"].strip()) in used_pairs:
                continue
            pool.append(d)
    print(f"[math-extra] spare math pool: {len(pool)}")
    if len(pool) < args.n:
        raise SystemExit(f"pool ({len(pool)}) smaller than requested n={args.n}")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(pool), size=args.n, replace=False)
    chosen = [pool[int(i)] for i in sorted(pick)]

    img_root = Path(args.img_root)
    rows, missing = [], 0
    for rec in chosen:
        p = img_root / rec["image"]
        if not p.is_file():
            missing += 1
            continue
        user_content = "<image>\n" + rec["question"] + _FORMAT_HINT["math"]
        rows.append({
            "data_source": "math",
            "prompt": [
                {"role": "system", "content": _VL_THINKING_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "images": [{"bytes": p.read_bytes(), "path": rec["image"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": rec["gt"]},
        })
    if missing:
        raise SystemExit(f"[math-extra] {missing}/{len(chosen)} images missing under "
                         f"{img_root} — refusing a silently-short parquet")

    features = Features({
        "data_source": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "images": Sequence(ImageFeature()),
        "ability": Value("string"),
        "reward_model": {"style": Value("string"), "ground_truth": Value("string")},
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows, features=features).to_parquet(str(out_dir / "train.parquet"))

    hint_sha1 = hashlib.sha1(
        (json.dumps(_FORMAT_HINT, sort_keys=True) + _VL_THINKING_SYSTEM).encode()
    ).hexdigest()[:12]
    with open(out_dir / "parquet_format_version.json", "w") as f:
        json.dump({"format_hint_version": _FORMAT_HINT_VERSION, "hint_sha1": hint_sha1,
                   "format_hints": _FORMAT_HINT, "note": "math_extra v3 rebuild",
                   "n": len(rows), "seed": args.seed}, f, indent=2)
    print(f"[math-extra] wrote {len(rows)} rows -> {out_dir/'train.parquet'} "
          f"(stamped {_FORMAT_HINT_VERSION})")


if __name__ == "__main__":
    main()
