#!/usr/bin/env python3
"""Build a stratified subset of VLAA-Thinking-verifiable-20K for GRPO families work.

Stages (run independently with --stage):
  split        — read JSONL, stratified sample, write split_manifest.json + iou_images_needed.txt
  parquet      — package full subset into verl-compatible {train,val,test}.parquet
  smoke_digit  — digit-only mini parquets for pipeline smoke testing

Stratification: (verifier_type, image_source), seed 42.

Per-verifier targets: 250 test / 1000 train / 100 val (1350 each, 5400 total).
Within mcq, sources are split proportionally (geoqa170k 87% / arxivqa 13%).
Within iou, sources are split 50/50 (coco / vg).

Output schema (verl-compatible, mirrors examples/data_preprocess/geo3k.py):
  data_source : str (constant "vlaa_thinking_verifiable")
  prompt      : list[{"role": "user", "content": str}]
  images      : list[{"bytes": bytes, "path": str}]   # one image per row
  ability     : str (verifier_type — used by reward dispatcher via reward_fn_key)
  reward_model: {"style": "rule", "ground_truth": str}
  extra_info  : {"id", "split", "index", "image_source", "verifier_type", "question"}
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import defaultdict
from pathlib import Path

SEED = 42
ANNOT = os.environ.get("VLAA_ANNOTATIONS", "data/annotations/vlaa_thinking_annotations.json")
IMG_ROOT = os.environ.get("VLAA_IMAGE_ROOT", "data/vlaa_thinking/images_extracted")
IOU_IMG_ROOT = IMG_ROOT  # coco/, vg/ go here too
OUT_DIR = os.environ.get("DATA_DIR", "data/vlaa_subset_5400")

# (verifier_type, image_source) -> {test, train, val}
TARGETS = {
    ("digit", "clevr_math"): {"test": 250, "train": 1000, "val": 100},
    ("mcq", "geoqa170k"):    {"test": 217, "train": 867,  "val": 87},
    ("mcq", "arxivqa"):      {"test": 33,  "train": 133,  "val": 13},
    ("math", "synthesis"):   {"test": 250, "train": 1000, "val": 100},
    ("iou", "coco"):         {"test": 125, "train": 500,  "val": 50},
    ("iou", "vg"):            {"test": 125, "train": 500,  "val": 50},
}


def stage_split(out_dir: Path) -> None:
    by_strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(ANNOT) as f:
        for line in f:
            d = json.loads(line)
            by_strata[(d["verifier_type"], d["image_source"])].append(d)

    rng = random.Random(SEED)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    summary: dict[str, dict[str, int]] = {}

    # Sort strata for deterministic iteration order regardless of dict insertion.
    for stratum in sorted(TARGETS):
        records = by_strata.get(stratum, [])
        target = TARGETS[stratum]
        needed = target["test"] + target["val"] + target["train"]
        if len(records) < needed:
            sys.exit(f"FATAL: stratum {stratum} has {len(records)} records but needs {needed}")
        records = list(records)
        rng.shuffle(records)
        cursor = 0
        for split in ("test", "val", "train"):
            n = target[split]
            chunk = records[cursor : cursor + n]
            cursor += n
            for r in chunk:
                r["split"] = split
                splits[split].append(r)
        summary[f"{stratum[0]}/{stratum[1]}"] = {
            "total_available": len(by_strata[stratum]),
            **target,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "annotation_source": ANNOT,
        "summary": summary,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "records": {k: v for k, v in splits.items()},
    }
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f)

    # IoU images needed (coco/vg) so we can selectively prefetch.
    iou_imgs = sorted({
        r["image"] for split in splits.values() for r in split
        if r["verifier_type"] == "iou"
    })
    with open(out_dir / "iou_images_needed.txt", "w") as f:
        for p in iou_imgs:
            f.write(p + "\n")

    print(f"[split] wrote {out_dir/'split_manifest.json'}")
    print(f"[split] split sizes: { {k: len(v) for k, v in splits.items()} }")
    print(f"[split] iou images needed: {len(iou_imgs)} unique (-> iou_images_needed.txt)")


# Official VL-Thinking system prompt (verbatim from VLAA-Thinking inference.py).
_VL_THINKING_SYSTEM = (
    "You are VL-Thinking\U0001F914, a helpful assistant with excellent reasoning ability. "
    "A user asks you a question, and you should try to solve it. You should first think "
    "about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>"
)


_FORMAT_HINT = {
    # v3 (2026-07): hints now state the PAYLOAD-EXACT rule explicitly, matching
    # the strict scorer (STRICT_CORRECTNESS=1): the <answer> tags must contain
    # ONLY the answer — any prose/extra characters inside the tags scores 0 and
    # forfeits the format bonus. mcq letter is UPPERCASE, case-sensitive.
    # (v2 history: mcq unified to tags so tag-required scoring applies across
    # all four verifiers; GATE on within-group mcq faith std still applies —
    # if it collapses vs digit/iou in the first ~30 steps, revisit.
    # OLD v1 soft mcq hint: " End your response with the chosen letter (A, B, C, D, or E).")
    "digit": (" Your <answer> tags must contain ONLY the final number and"
              " nothing else, e.g. <answer>5</answer>."),
    "mcq":   (" Your <answer> tags must contain ONLY the uppercase letter of the"
              " chosen option (A, B, C, D, or E) and nothing else, e.g."
              " <answer>B</answer>."),
    "math":  (" Your <answer> tags must contain ONLY \\boxed{...} with the final"
              " answer inside and nothing else, e.g. <answer>\\boxed{42}</answer>."),
    "iou":   (" Your <answer> tags must contain ONLY the bounding box as"
              " [x1, y1, x2, y2] normalized to [0, 1] and nothing else, e.g."
              " <answer>[0.12, 0.34, 0.56, 0.78]</answer>."),
}

# Bump when any _FORMAT_HINT or the system prompt changes — the hint is baked
# into every parquet row (user_content), so a stale on-disk parquet would train
# the model on an outdated prompt while the scorer/docs assume the new one. The
# rerun launchers assert the parquet's stamped version matches this.
_FORMAT_HINT_VERSION = "v3-exact-payload-2026-07"


def _build_row(rec: dict, split: str, index: int, img_bytes: bytes) -> dict:
    vt = rec["verifier_type"]
    question = rec["question"]
    # User content: <image> placeholder + question + per-verifier format hint.
    # System prompt is uniform (VL_THINKING_SYSTEM); verifier-specific guidance
    # lives entirely in the user prompt — TA's spec.
    hint = _FORMAT_HINT.get(vt, "")
    user_content = "<image>\n" + question + hint
    # Set data_source = verifier_type so the default verl reward_fn_key="data_source"
    # routes to the correct per-verifier scorer (digit/mcq/math/iou). The CLI
    # override `data.reward_fn_key=ability` doesn't propagate to the reward manager
    # in our verl version, so we encode the dispatch key in data_source directly.
    return {
        "data_source": vt,
        "prompt": [
            {"role": "system", "content": _VL_THINKING_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "images": [{"bytes": img_bytes, "path": rec["image"]}],
        "ability": vt,
        "reward_model": {"style": "rule", "ground_truth": rec["gt"]},
    }


def _load_image_bytes(rec: dict) -> bytes:
    path = Path(IMG_ROOT) / rec["image"]
    if not path.exists():
        raise FileNotFoundError(f"image missing: {path}")
    return path.read_bytes()


def _write_parquet(out_path: Path, records: list[dict], split: str, allow_missing: bool = False) -> int:
    """Write parquet via HF Datasets so the `images` column is a proper Image feature.

    Without this, raw pyarrow struct gets auto-decoded by HF Datasets at load time
    into a dict with both `bytes` AND `image` keys, which verl's process_image rejects.
    """
    from datasets import Dataset, Features, Value, Sequence, Image as ImageFeature

    rows: list[dict] = []
    missing = 0
    missing_examples: list[str] = []
    for idx, r in enumerate(records):
        try:
            img_bytes = _load_image_bytes(r)
        except FileNotFoundError:
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(str(r.get("image") or r.get("image_path") or r))
            continue
        rows.append(_build_row(r, split, idx, img_bytes))

    # Missing images silently desync the parquet from the stratified manifest and
    # corrupt per-stratum balance. Fatal by default; --allow-missing to override.
    if missing and not allow_missing:
        raise SystemExit(
            f"[parquet] {split}: {missing}/{len(records)} images missing — refusing to "
            f"write a desynced parquet (stratum balance would be wrong). Examples: "
            f"{missing_examples}. Re-run with --allow-missing to write anyway.")

    features = Features({
        "data_source": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "images": Sequence(ImageFeature()),
        "ability": Value("string"),
        "reward_model": {"style": Value("string"), "ground_truth": Value("string")},
    })
    ds = Dataset.from_list(rows, features=features)
    ds.to_parquet(str(out_path))
    print(f"[parquet] {split}: wrote {len(rows)} rows ({missing} missing) -> {out_path}")
    return len(rows)


def stage_parquet(out_dir: Path, allow_missing: bool = False) -> None:
    manifest = json.load(open(out_dir / "split_manifest.json"))
    for split, records in manifest["records"].items():
        _write_parquet(out_dir / f"{split}.parquet", records, split, allow_missing=allow_missing)
    # Stamp the format version + a hash of the hints/system prompt next to the
    # parquets so the rerun can assert it trained on the current prompt format.
    import hashlib
    hint_sha1 = hashlib.sha1(
        (json.dumps(_FORMAT_HINT, sort_keys=True) + _VL_THINKING_SYSTEM).encode()
    ).hexdigest()[:12]
    stamp = {
        "format_hint_version": _FORMAT_HINT_VERSION,
        "hint_sha1": hint_sha1,
        "format_hints": _FORMAT_HINT,
    }
    with open(out_dir / "parquet_format_version.json", "w") as f:
        json.dump(stamp, f, indent=2)
    print(f"[parquet] stamped format {_FORMAT_HINT_VERSION} (hint_sha1={hint_sha1}) "
          f"-> {out_dir/'parquet_format_version.json'}")


def stage_smoke_digit(out_dir: Path, smoke_dir: Path) -> None:
    """Digit-only mini split for verl pipeline smoke (200 train / 20 val / 50 test)."""
    manifest = json.load(open(out_dir / "split_manifest.json"))
    smoke_dir.mkdir(parents=True, exist_ok=True)
    targets = {"train": 200, "val": 20, "test": 50}
    for split, n in targets.items():
        digit_records = [r for r in manifest["records"][split] if r["verifier_type"] == "digit"][:n]
        _write_parquet(smoke_dir / f"{split}.parquet", digit_records, split)


def stage_smoke_verifier(out_dir: Path, smoke_dir: Path, verifier: str,
                          n_train: int = 500, n_val: int = 50, n_test: int = 100) -> None:
    """Build a per-verifier smoke slice for tiebreak experiments."""
    manifest = json.load(open(out_dir / "split_manifest.json"))
    smoke_dir.mkdir(parents=True, exist_ok=True)
    targets = {"train": n_train, "val": n_val, "test": n_test}
    for split, n in targets.items():
        records = [r for r in manifest["records"][split] if r["verifier_type"] == verifier][:n]
        _write_parquet(smoke_dir / f"{split}.parquet", records, split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["split", "parquet", "smoke_digit", "smoke_verifier"], required=True)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--smoke-out", default="data/vlaa_smoke_digit")
    ap.add_argument("--verifier", default="mcq", help="for smoke_verifier stage")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--allow-missing", action="store_true",
                    help="write the parquet even if some images are missing (default: fatal, "
                         "to avoid silently desyncing from the stratified manifest)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    if args.stage == "split":
        stage_split(out_dir)
    elif args.stage == "parquet":
        stage_parquet(out_dir, allow_missing=args.allow_missing)
    elif args.stage == "smoke_digit":
        stage_smoke_digit(out_dir, Path(args.smoke_out))
    elif args.stage == "smoke_verifier":
        stage_smoke_verifier(out_dir, Path(args.smoke_out), args.verifier, args.n_train, args.n_val, args.n_test)


if __name__ == "__main__":
    main()
