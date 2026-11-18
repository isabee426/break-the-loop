#!/usr/bin/env python
# Combined offline val-trajectory plot for ALL arms (full-FT 8B).
# Uses the OFFLINE fixed-math val jsons ({arm}_step{N}__val.json) — math is
# correctly credited here (unlike the in-loop metric). In-domain = 4-verifier
# mean acc (digit, mcq, math, iou) across every saved checkpoint.
#   Usage: plot_val_trajectory_all.py <eval_dir> <out.png> [sstar]
import sys, json, glob, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eval_dir = sys.argv[1]
outpng = sys.argv[2]
sstar = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].strip().isdigit() else None

# (label-in-file, display name, color). faith == a050 (α=0.5) — prefer the
# 'faith'-labeled jsons (user's canonical redo); fall back to a050 if absent.
ARMS = [
    ("vanilla",       "vanilla",          "#5B6B7B"),
    ("faith",         "faith-tiebreak α0.5", "#C2410C"),
    ("a025",          "faith-tiebreak α0.25", "#EA9A6B"),
    ("a100",          "faith-tiebreak α1.0",  "#7C2D12"),
    ("length",        "length penalty",   "#2563EB"),
    ("random",        "random tiebreak",  "#9CA3AF"),
    ("composite_a07", "composite α0.7",   "#15803D"),
]

VERIFIERS = ("digit", "mcq", "math", "iou")


def load_val(arm):
    d = {}
    for f in glob.glob(f"{eval_dir}/{arm}_step*__val.json"):
        m = re.search(r"_step(\d+)__val", f)
        if not m:
            continue
        try:
            j = json.load(open(f))
            accs = [j[v]["acc"] for v in VERIFIERS]
        except Exception:
            continue
        if all(a is not None for a in accs):
            d[int(m.group(1))] = sum(accs) / len(accs)
    return dict(sorted(d.items()))


fig, ax = plt.subplots(figsize=(9, 5.5))
plotted = []
for arm, name, color in ARMS:
    d = load_val(arm)
    if not d:
        continue
    xs, ys = list(d.keys()), list(d.values())
    ax.plot(xs, ys, "-o", ms=3.5, lw=1.8, color=color, label=name)
    plotted.append(name)

if sstar is not None:
    ax.axvline(sstar, ls="--", color="#111", lw=1.2, alpha=0.7)
    ax.annotate(f"S*={sstar}", (sstar, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(4, 6), fontsize=9, fontweight="bold")

ax.set_xlabel("training step")
ax.set_ylabel("in-domain val acc (4-verifier mean, fixed math)")
ax.set_title("Full-FT Qwen3-VL-8B — offline val trajectory, all arms", fontweight="bold")
ax.grid(alpha=0.25)
ax.legend(frameon=False, ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(outpng, dpi=140)
print(f"wrote {outpng}  (arms: {', '.join(plotted)})")
