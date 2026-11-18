"""Split-overlap audit (checks for train/test leakage).

Answers three questions about the seed-42 vlaa_subset_5400 split and the probe
corpus (= RL-train prompts + the disjoint extra-math prompts):

  1. Do any (image, question) pairs repeat across train / val / test?
  2. How many TEST records share an IMAGE with the probe corpus?
  3. Of those, how many also carry an identical ground-truth string?

Optionally re-scores an in-domain test eval json with the overlapping prompts
excluded, so the headline delta can be reported on a strictly image-disjoint
subset (--eval arm=path, repeatable).

Usage:
  python _audit_split_overlap.py \
      --manifest data/vlaa_subset_5400/split_manifest.json \
      --annotations data/annotations/vlaa_thinking_annotations.json \
      --eval vanilla=results/eval/vanilla_S150__test.json \
      --eval faith_a050=results/eval/a050_S150__test.json
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path

import numpy as np


def extra_math_images(annotations: str, recs: dict, n: int, seed: int) -> set[str]:
    """Reproduce prep_math_extra_v3.py's deterministic pick (id + (image, question) dedupe)."""
    used_ids, used_pairs = set(), set()
    for rs in recs.values():
        for r in rs:
            if r["verifier_type"] == "math":
                used_ids.add(r["id"])
                used_pairs.add((r["image"], r["question"].strip()))
    pool = []
    with open(annotations) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["verifier_type"] != "math":
                continue
            if d["id"] in used_ids or (d["image"], d["question"].strip()) in used_pairs:
                continue
            pool.append(d)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(pool), size=n, replace=False)
    print(f"[extra-math] spare pool {len(pool)} -> picked {n}")
    return {pool[int(i)]["image"] for i in sorted(pick)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--n-extra", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval", action="append", default=[], metavar="ARM=PATH",
                    help="in-domain test json with per-example records; repeatable")
    ap.add_argument("--out", default="split_overlap_audit.json")
    args = ap.parse_args()

    recs = json.load(open(args.manifest))["records"]

    # 1. exact (image, question) duplicates across splits
    pair_splits = collections.defaultdict(set)
    for split, rs in recs.items():
        for r in rs:
            pair_splits[(r["image"], r["question"].strip())].add(split)
    dup_pairs = [k for k, s in pair_splits.items() if len(s) > 1]
    print(f"[pairs] (image, question) appearing in >1 split: {len(dup_pairs)}")

    # 2. test records whose image is in the probe corpus
    train_by_img = collections.defaultdict(list)
    for r in recs["train"]:
        train_by_img[r["image"]].append(r)
    extra_imgs = extra_math_images(args.annotations, recs, args.n_extra, args.seed)
    probe_imgs = set(train_by_img) | extra_imgs
    hits = [r for r in recs["test"] if r["image"] in probe_imgs]
    print(f"[images] test records sharing an image with the probe corpus: "
          f"{len(hits)}/{len(recs['test'])} ({100 * len(hits) / len(recs['test']):.1f}%) "
          f"{dict(collections.Counter(r['image_source'] for r in hits))}")
    print(f"[images] test records sharing an image with extra-math only: "
          f"{sum(1 for r in recs['test'] if r['image'] in extra_imgs)}")

    # 3. of those, identical ground truth as well
    same_gt = [r for r in hits
               if str(r["gt"]).strip() in {str(t["gt"]).strip() for t in train_by_img[r["image"]]}]
    print(f"[gt] same image AND identical gt string: {len(same_gt)} "
          f"{dict(collections.Counter(r['image_source'] for r in same_gt))}")

    overlap_ids = {r["id"] for r in hits}
    json.dump({"overlap_test_ids": sorted(overlap_ids),
               "same_gt_ids": sorted(r["id"] for r in same_gt),
               "n_test": len(recs["test"]), "dup_pairs": len(dup_pairs)},
              open(args.out, "w"), indent=1)
    print(f"[out] wrote {args.out}")

    # 4. optional: accuracy with the overlapping prompts dropped
    for spec in args.eval:
        arm, path = spec.split("=", 1)
        d = json.load(open(path))
        rows = d.get("all_responses") or d.get("samples") or d.get("results") or []
        keyed = [r for r in rows if r.get("id") or r.get("idx") is not None]
        if not keyed:
            print(f"[eval:{arm}] no per-example ids in {Path(path).name}; skipping")
            continue
        def ok(r):
            v = r.get("correct", r.get("strict_correct", r.get("score")))
            return float(v) if v is not None else float("nan")
        keep = [r for r in keyed if r.get("id") not in overlap_ids]
        full = np.nanmean([ok(r) for r in keyed])
        excl = np.nanmean([ok(r) for r in keep])
        print(f"[eval:{arm}] full {full:.4f} (n={len(keyed)}) | image-disjoint {excl:.4f} "
              f"(n={len(keep)}) | delta {excl - full:+.4f}")


if __name__ == "__main__":
    main()
