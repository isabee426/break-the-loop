"""Aggregate v2 matched-step OOD evals into an A/B with PAIRED CIs.

Reads the per-seed eval JSONs written by eval_ood_vllm.py (which store
strict_correct + is_collapsed + is_truncated per example), groups by
(dataset, step, arm), and reports:

  - mean strict-F1 ± across-seed std (the seed spread only bounds vLLM
    nondeterminism at temp 0 — it is NOT the A/B significance test);
  - collapse rate (is_collapsed — content-based repetition, NOT truncation);
  - for each contrast (faith-vanilla, random-vanilla, length-vanilla,
    faith-random, faith-length): a PAIRED bootstrap 95% CI on ΔF1.

Why paired-bootstrap-over-examples and not sum-of-stdevs over 3 seeds: the arms
score the SAME examples in the SAME order (deterministic dataset load + seed-42
subsample, independent of the sampling --seed), so example i is paired across
arms. We average each example's strict_correct over the eval seeds (denoising
vLLM noise), then bootstrap-resample EXAMPLES to get a real CI on the mean
per-example delta. A contrast is significant iff its 95% CI excludes 0.

Usage:  python agg_ood_ci.py [results_dir]   (default results/ood)
"""
import json, glob, re, sys, os
import statistics as st
from collections import defaultdict
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "results/ood"
# Anchor the arm group and allow underscores (e.g. vanilla_ours) without the old
# greedy-[a-z]+ backtrack that silently relabeled multi-token arms.
PAT = re.compile(r"^(?P<arm>[a-z0-9_]+?)_step(?P<step>\d+)__(?P<ds>[a-z]+)_s(?P<seed>\d+)\.json$")
N_BOOT = 10000
BOOT_SEED = 0

# Per (ds, step, arm): seed -> per-example arrays (aligned by index across seeds/arms).
raw = defaultdict(lambda: defaultdict(dict))  # raw[(ds,step,arm)][seed] = {"f1": arr, "col": arr}
for f in sorted(glob.glob(os.path.join(D, "*.json"))):
    m = PAT.match(os.path.basename(f))
    if not m:
        continue
    obj = json.load(open(f))
    ex = obj.get("examples") or obj.get("per_example") or []
    if not ex:
        continue
    f1 = np.array([float(e.get("strict_correct", e.get("correct", 0.0))) for e in ex], dtype=float)
    col = np.array([1.0 if e.get("is_collapsed") else 0.0 for e in ex], dtype=float)
    raw[(m["ds"], int(m["step"]), m["arm"])][int(m["seed"])] = {"f1": f1, "col": col}


def per_example_mean(arm_seeds, key):
    """Stack a (ds,step,arm)'s seed arrays and mean over seeds -> per-example vector.
    Returns None if the seed arrays disagree in length (misaligned examples)."""
    arrs = [d[key] for d in arm_seeds.values()]
    if not arrs:
        return None
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        return None
    return np.mean(np.stack(arrs, axis=0), axis=0)  # (n_examples,)


def paired_ci(a_vec, b_vec, rng):
    """Paired bootstrap 95% CI on mean(a - b) over examples. a,b aligned by index."""
    if a_vec is None or b_vec is None or len(a_vec) != len(b_vec) or len(a_vec) == 0:
        return None
    delta = a_vec - b_vec
    n = len(delta)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi)


# Reduce to per (ds,step): arm -> {"f1_seed_means":[...], "col_seed_means":[...],
# "ex_f1": per-example-mean vector, "n_seeds": k}
rows = defaultdict(dict)
for (ds, step, arm), seeds in raw.items():
    f1_seed_means = [d["f1"].mean() for d in seeds.values()]
    col_seed_means = [d["col"].mean() for d in seeds.values()]
    rows[(ds, step)][arm] = {
        "f1_seed_means": f1_seed_means,
        "col_seed_means": col_seed_means,
        "ex_f1": per_example_mean(seeds, "f1"),
        "n_seeds": len(seeds),
    }

if not rows:
    print(f"no matching eval JSONs in {D} (expected <arm>_step<N>__<ds>_s<seed>.json)")
    sys.exit(0)

CONTRASTS = [("faith", "vanilla"), ("random", "vanilla"), ("length", "vanilla"),
             ("faith", "random"), ("faith", "length")]
rng = np.random.default_rng(BOOT_SEED)

for (ds, step) in sorted(rows):
    print(f"\n=== {ds} @ step {step} ===")
    a = rows[(ds, step)]
    for arm in sorted(a):
        fm = st.mean(a[arm]["f1_seed_means"])
        fs = st.stdev(a[arm]["f1_seed_means"]) if len(a[arm]["f1_seed_means"]) > 1 else 0.0
        cm = st.mean(a[arm]["col_seed_means"])
        n = a[arm]["n_seeds"]
        print(f"  {arm:8s} F1={fm:.3f}±{fs:.3f}  collapse={cm:.0%}  (n={n} seeds)")
    for hi_arm, base in CONTRASTS:
        if hi_arm not in a or base not in a:
            continue
        ci = paired_ci(a[hi_arm]["ex_f1"], a[base]["ex_f1"], rng)
        if ci is None:
            print(f"  {hi_arm}-{base}: SKIP (example counts misaligned across seeds/arms)")
            continue
        d, lo, hi = ci
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        tag = "  (faith-content effect)" if base in ("random", "length") else ""
        print(f"  {hi_arm}-{base:8s} ΔF1={d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}{tag}")
