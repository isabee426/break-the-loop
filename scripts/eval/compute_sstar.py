#!/usr/bin/env python
# S*: from the VANILLA arm's offline in-domain val (MATH NOW CORRECT),
# first step where the 3-point moving average of the 4-verifier mean stops
# increasing by >1% relative. Vanilla-only, blind to tiebreak arms.
# Prints S* to stdout; diagnostics to stderr.
import sys, json, glob, re
outdir, arm = sys.argv[1], sys.argv[2]
data = {}
for f in glob.glob(f"{outdir}/{arm}_step*__val.json"):
    m = re.search(r"_step(\d+)__val", f)
    if not m:
        continue
    try:
        j = json.load(open(f))
    except Exception:
        continue
    try:
        accs = [j[v]["acc"] for v in ("digit", "mcq", "math", "iou")]
    except Exception:
        continue
    if all(a is not None for a in accs):
        data[int(m.group(1))] = sum(accs) / 4
steps = sorted(data)
if len(steps) < 3:
    sys.stderr.write(f"[compute_sstar] too few val points: {len(steps)}\n")
    print("")
    sys.exit(0)
ma = {steps[i]: (data[steps[i-1]] + data[steps[i]] + data[steps[i+1]]) / 3
      for i in range(1, len(steps) - 1)}
mas = sorted(ma)
# Selection rule: "first checkpoint where the 3-pt MA STOPS increasing by >1% relative."
# "Stops" is SUSTAINED — a lone dip that later resumes rising >1% does NOT count.
# So S* = the first checkpoint AFTER the last >1% MA increase (plateau onset).
rels = {}
last_rise = None
for i in range(1, len(mas)):
    prev = ma[mas[i-1]]
    rels[mas[i]] = (ma[mas[i]] - prev) / prev if prev else 0.0
    if rels[mas[i]] > 0.01:
        last_rise = mas[i]
if last_rise is None:
    sstar = mas[1] if len(mas) > 1 else mas[0]
else:
    after = [s for s in mas if s > last_rise]
    sstar = after[0] if after else mas[-1]
sys.stderr.write("[compute_sstar] step: 4v-mean | 3ptMA relΔ\n")
for s in mas[1:]:
    mark = "  <== S* (plateau onset: MA never rises >1% after here)" if s == sstar else ""
    sys.stderr.write(f"  {s:>4}: {data[s]:.3f} | MA={ma[s]:.3f} relΔ={rels[s]*100:+.2f}%{mark}\n")
sys.stderr.write(f"[compute_sstar] last >1% MA rise at step {last_rise}; S* = {sstar}\n")
print(sstar)
