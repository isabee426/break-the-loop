"""Merge per-shard probe-feature npz files into a single balanced_features_1k.npz."""
from __future__ import annotations
import argparse
import glob
import re
from pathlib import Path

import numpy as np


def _shard_key(path: str):
    """Extract a shard index from a filename (…shard3…, …shard_3…, …_s3_…) so we
    can detect the same shard merged twice (the prompt-id-collision / split-leak
    hazard). Returns None if no shard number is parseable."""
    m = re.search(r"shard[_-]?(\d+)", Path(path).name) or re.search(r"_s(\d+)[_.]", Path(path).name)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="glob pattern, e.g. /path/balanced_features_1k_shard*.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-verifiers", default="",
                    help="comma list (e.g. mcq,vqa,math,iou); every one MUST appear in the merged set or merge aborts")
    ap.add_argument("--min-rows", type=int, default=0,
                    help="merged row count must be >= this, else abort (guards against a silently-truncated collection)")
    ap.add_argument("--require-disjoint", action="store_true",
                    help="hard-fail (not just warn) if two inputs share a parsed shard index")
    ap.add_argument("--extra-files", nargs="*", default=[],
                    help="explicit additional npz files to merge in beyond the glob "
                         "(e.g. a disjoint math_extra file the pattern doesn't match). "
                         "prompt_ids are offset per-file so they never collide.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    # Add explicit extra files (must exist; the glob-suffix trap is why they're
    # passed by path, not folded into the pattern).
    for xf in args.extra_files:
        if not Path(xf).is_file():
            raise SystemExit(f"--extra-files: {xf} does not exist")
        files.append(xf)
    if not files:
        raise SystemExit(f"no files match {args.pattern} (and no --extra-files)")
    # Exclude partial-checkpoint files — they carry the same keys and would merge
    # as duplicated rows.
    dropped = [f for f in files if f.endswith(".partial.npz")]
    files = [f for f in files if not f.endswith(".partial.npz")]
    if dropped:
        print(f"[merge] ignoring {len(dropped)} .partial.npz checkpoint file(s): {dropped}")
    if not files:
        raise SystemExit(f"no non-partial files match {args.pattern}")
    print(f"merging {len(files)} shards:")
    for f in files:
        print(f"  {f}")

    # DISJOINTNESS: the same shard index appearing twice means overlapping prompt
    # sets -> the offset-relabel below turns duplicates into "new" prompts and
    # leaks the prompt-level train/val split. prompt_id in each npz is LOCAL
    # (collector reset_index after iloc[shard::n]), so a content check isn't
    # possible; the filename shard index is the available signal.
    keys = [_shard_key(f) for f in files]
    seen: dict[int, str] = {}
    dup_msgs = []
    for f, k in zip(files, keys):
        if k is None:
            continue
        if k in seen:
            dup_msgs.append(f"shard index {k}: {seen[k]} AND {f}")
        seen[k] = f
    if dup_msgs:
        msg = "[merge] duplicate shard indices (overlapping prompt sets -> split leak):\n  " + "\n  ".join(dup_msgs)
        if args.require_disjoint or any(k is not None for k in keys):
            raise SystemExit(msg + "\n  Merge distinct shards only, or drop the re-collection.")
        print(msg)
    if any(k is None for k in keys):
        print("[merge][warn] some inputs have no parseable shard index — cannot verify "
              "disjointness for those; ensure they are distinct prompt sets.")

    Xs, ys, pids, vts = [], [], [], []
    extras: dict[str, list] = {}  # per-row analysis columns (y_lenient, formatted, feature_path)
    extras_missing: dict[str, list[str]] = {}  # column -> shards that lack it
    any_parity_failed = 0
    models: set[str] = set()  # collection model id — must be identical across shards
    offset = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if d["prompt_id"].size == 0:
            raise SystemExit(f"shard {f} is empty (zero rows) — remove or recollect it before merging")
        if "model" in d.files:
            models.add(str(d["model"]))
        Xs.append(d["X"])
        ys.append(d["y"])
        # Offset each shard's LOCAL prompt_ids so they're globally unique across
        # the (verified-disjoint) shards, keeping prompt-level splits valid.
        pids.append(d["prompt_id"] + offset)
        offset += int(d["prompt_id"].max()) + 1
        vts.append(d["verifier_type"])
        # feature_path (per-row batch-vs-serve flag) rides along so the fit can
        # drop parity-suspect rows; parity_failed (scalar) OR-reduces across shards.
        for k in ("y_lenient", "formatted", "feature_path"):
            if k in d.files:
                extras.setdefault(k, []).append(d[k])
            else:
                extras_missing.setdefault(k, []).append(f)
        if "parity_failed" in d.files:
            any_parity_failed = max(any_parity_failed, int(d["parity_failed"]))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    prompt_id = np.concatenate(pids, axis=0)
    verifier_type = np.concatenate(vts, axis=0)
    # Carry per-row columns through only if EVERY shard has them (a partial column
    # would misalign with the concatenated rows).
    for k, v in extras.items():
        if len(v) != len(files):
            print(f"[warn] dropping column '{k}': missing from shard(s) {extras_missing[k]}")
    extra_arrays = {k: np.concatenate(v, axis=0) for k, v in extras.items()
                    if len(v) == len(files)}

    # COVERAGE GUARDS (research-grade): a silently-dropped verifier or a truncated
    # collection changes the class/verifier balance the winning recipe depends on.
    present = set(str(v) for v in verifier_type.tolist())
    if args.expect_verifiers:
        want = {v.strip() for v in args.expect_verifiers.split(",") if v.strip()}
        missing = want - present
        if missing:
            raise SystemExit(
                f"[merge] FATAL: expected verifiers {sorted(want)} but merged set is "
                f"missing {sorted(missing)} (present: {sorted(present)}). A verifier's "
                f"shard likely failed collection — recollect it before fitting.")
    if args.min_rows and len(y) < args.min_rows:
        raise SystemExit(f"[merge] FATAL: merged {len(y)} rows < --min-rows {args.min_rows} "
                         f"— collection looks truncated.")

    # MODEL-IDENTITY GUARD (research-grade): the probe's features are architecture-
    # specific, so mixing shards from different collection models yields a garbage
    # probe. Refuse a mixed merge; propagate the single model id so train_probe
    # stamps it into the pickle and scorers.py can hard-check probe/model identity
    # at serve time (not just feat_dim, which any same-size model would pass).
    if len(models) > 1:
        raise SystemExit(f"[merge] FATAL: shards were collected on different models {sorted(models)} "
                         f"— features are model-specific; recollect on one model before merging.")
    model_kw = {"model": next(iter(models))} if models else {}
    if not models:
        print("[merge][warn] no shard carries a 'model' field — merged npz will be model-unstamped; "
              "serve-time probe/model identity check will degrade to feat_dim only.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, y=y, prompt_id=prompt_id, verifier_type=verifier_type,
             parity_failed=int(any_parity_failed), **model_kw, **extra_arrays)
    print(f"\nmerged: {X.shape} | pos={int(y.sum())}/{len(y)} ({y.mean():.3f}) | "
          f"unique_prompts={len(set(prompt_id.tolist()))} | parity_failed={any_parity_failed}")
    if any_parity_failed and "feature_path" not in extra_arrays:
        print("[merge][warn] parity_failed=1 but feature_path column absent — the fit "
              "cannot drop suspect rows. Recollect the affected shard(s).")
    from collections import Counter
    print(f"per-verifier: {dict(Counter(verifier_type.tolist()))}")


if __name__ == "__main__":
    main()
