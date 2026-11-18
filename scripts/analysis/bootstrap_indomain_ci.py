"""Paired, stratified bootstrap CI on the in-domain accuracy gains (no GPU, no model).
Puts an interval on the headline +3.4 (faith - vanilla) and on the 1.8-pt faith - additive
gap, using the SAME eval prompts for every arm. Also runs an exact McNemar test per contrast.

Run on the three in-domain test_full JSONs:
  python3 scripts/analysis/bootstrap_indomain_ci.py \
      --vanilla  results/eval/vanilla_S150__test.json \
      --faith    results/eval/faith_S150__test.json \
      --additive results/eval/additive_S150__test.json
"""
import json, argparse, math
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--vanilla",  default="results/eval/vanilla_S150__test.json")
ap.add_argument("--faith",    default="results/eval/faith_S150__test.json")
ap.add_argument("--additive", default="results/eval/additive_S150__test.json")
ap.add_argument("--reps", type=int, default=10000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

VERIFS = ["digit", "mcq", "math", "iou"]


def load(path):
    recs = json.load(open(path)).get("all_responses") or []
    corr, verif = {}, {}
    for r in recs:
        a = r.get("acc", r.get("score"))
        corr[r["idx"]] = 1 if (a or 0) >= 0.5 else 0
        verif[r["idx"]] = r.get("verifier")
    return corr, verif

van, vmap = load(args.vanilla)
fai, _ = load(args.faith)
add, _ = load(args.additive)
common = sorted(set(van) & set(fai) & set(add))
by_verif = defaultdict(list)
for i in common:
    by_verif[vmap[i]].append(i)
print(f"{len(common)} common prompts | " + " ".join(f"{v}={len(by_verif[v])}" for v in VERIFS))


def macro(corr, idxs_by_v):
    """paper 'Overall' = mean of the 4 per-verifier accuracies"""
    return float(np.mean([np.mean([corr[i] for i in idxs_by_v[v]]) for v in VERIFS]))


van_o, fai_o, add_o = (macro(c, by_verif) for c in (van, fai, add))
print(f"\npoint estimates (macro overall):  vanilla={van_o:.3f}  faith={fai_o:.3f}  additive={add_o:.3f}")
print(f"  faith - vanilla  = {100*(fai_o-van_o):+.1f} pts")
print(f"  faith - additive = {100*(fai_o-add_o):+.1f} pts")

# ---- paired stratified bootstrap: resample prompts WITHIN each verifier ----
rng = np.random.default_rng(args.seed)
arr = {v: np.array([[van[i], fai[i], add[i]] for i in by_verif[v]]) for v in VERIFS}
d_fv, d_fa = np.empty(args.reps), np.empty(args.reps)
for b in range(args.reps):
    vo = fo = ao = 0.0
    for v in VERIFS:
        n = len(arr[v]); s = arr[v][rng.integers(0, n, n)]
        vo += s[:, 0].mean(); fo += s[:, 1].mean(); ao += s[:, 2].mean()
    vo /= 4; fo /= 4; ao /= 4
    d_fv[b] = fo - vo; d_fa[b] = fo - ao


def ci(d):
    lo, hi = np.percentile(d, [2.5, 97.5]) * 100
    return lo, hi


lo1, hi1 = ci(d_fv); lo2, hi2 = ci(d_fa)
print(f"\n95% CI (stratified paired bootstrap, {args.reps} reps):")
print(f"  faith - vanilla : {100*(fai_o-van_o):+.1f}  [{lo1:+.1f}, {hi1:+.1f}]  "
      f"{'excludes 0' if lo1>0 else 'straddles 0'}")
print(f"  faith - additive: {100*(fai_o-add_o):+.1f}  [{lo2:+.1f}, {hi2:+.1f}]  "
      f"{'excludes 0' if lo2>0 else 'straddles 0'}")


# ---- exact McNemar per contrast (paired binary, discordant pairs) ----
def mcnemar(a, b):
    b01 = sum(1 for i in common if a[i] == 0 and b[i] == 1)  # b better
    b10 = sum(1 for i in common if a[i] == 1 and b[i] == 0)  # a better
    n = b01 + b10; k = min(b01, b10)
    p = sum(math.comb(n, j) for j in range(0, k + 1)) / 2**n
    return b01, b10, min(1.0, 2 * p)

for name, other in [("vanilla", van), ("additive", add)]:
    b01, b10, p = mcnemar(other, fai)
    print(f"  McNemar faith vs {name:8s}: faith-better={b01} {name}-better={b10} "
          f"(n_discordant={b01+b10}) exact p={p:.4f}")
