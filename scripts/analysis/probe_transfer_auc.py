"""Frozen-probe transfer test. Scores a set of rollouts through the FROZEN 8B
faith probe and measures how well faith predicts correctness (ROC-AUC), overall
and per-verifier. Run once on the Thinking baseline (vanilla_S150) and once on
the Instruct rollouts (same 8B geometry) to see if the probe transfers off the
distribution it was fit on. Needs one GPU; set PROBE_PATH and DATA_DIR."""
import os, sys, io, json, pickle, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--responses", required=True, help="*_test_full.json with all_responses")
ap.add_argument("--label", required=True, help="tag for output, e.g. instruct / thinking")
ap.add_argument("--gpu", default="0")
ap.add_argument("--cap-per-verifier", type=int, default=150)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from verl_faith.lookback import LookbackExtractor

OUT = os.environ.get("PROBE_TRANSFER_DIR", "results/probe_transfer"); os.makedirs(OUT, exist_ok=True)
d = json.load(open(args.responses))
recs = [r for r in (d.get("all_responses") or []) if r.get("response")]
df = pd.read_parquet(os.path.join(os.environ.get("DATA_DIR", "data/vlaa_subset_5400"), "test.parquet"))

# stratified cap per verifier, deterministic order (no RNG -> reproducible)
by_v = {}
for r in recs:
    by_v.setdefault(r.get("verifier"), []).append(r)
sample = []
for v, rs in by_v.items():
    sample += rs[:args.cap_per_verifier]
print(f"[{args.label}] scoring {len(sample)} rollouts across verifiers "
      f"{ {k: min(len(v), args.cap_per_verifier) for k, v in by_v.items()} }", flush=True)

probe = pickle.load(open(os.environ["PROBE_PATH"], "rb"))
ext = LookbackExtractor(model_path="Qwen/Qwen3-VL-8B-Thinking", device="cuda", attn_implementation="sdpa")

def acc_of(r):
    a = r.get("acc")
    if a is None:
        a = r.get("score")
    return None if a is None else int(float(a) >= 0.5)

rows = []
for k, r in enumerate(sample):
    idx = r["idx"]; row = df.iloc[idx]
    a = acc_of(r)
    if a is None:
        continue
    img = Image.open(io.BytesIO(row.images[0]["bytes"])).convert("RGB")
    q = row.prompt[1]["content"].replace("<image>\n", "")
    feat = ext.extract_efficient(image=img, question=q, response=r["response"], max_response_tokens=4096)
    faith = float(probe.predict_proba(feat[None, :])[0, 1])
    rows.append({"idx": idx, "verifier": r.get("verifier"), "faith": round(faith, 4),
                 "acc": a, "len_chars": len(r["response"])})
    if k % 50 == 0:
        print(f"  {k}/{len(sample)}", flush=True)

json.dump(rows, open(f"{OUT}/scored_{args.label}.json", "w"), indent=2)

def summarize(rs, name):
    y = np.array([x["acc"] for x in rs]); f = np.array([x["faith"] for x in rs])
    n, pos = len(y), int(y.sum())
    line = {"set": name, "n": n, "pos": pos, "neg": n - pos,
            "faith_mean_correct": round(float(f[y == 1].mean()), 4) if pos else None,
            "faith_mean_wrong": round(float(f[y == 0].mean()), 4) if (n - pos) else None}
    if 0 < pos < n:
        line["auc"] = round(float(roc_auc_score(y, f)), 4)
    else:
        line["auc"] = None  # degenerate: one class only
    return line

print(f"\n===== [{args.label}] frozen-probe faith -> correctness =====")
out = [summarize(rows, "ALL")]
for v in sorted(by_v):
    rv = [x for x in rows if x["verifier"] == v]
    if rv:
        out.append(summarize(rv, v))
for l in out:
    print(f"  {l['set']:6s} n={l['n']:4d} pos={l['pos']:4d} neg={l['neg']:4d} "
          f"AUC={l['auc']} faith(correct)={l['faith_mean_correct']} faith(wrong)={l['faith_mean_wrong']}")
json.dump(out, open(f"{OUT}/summary_{args.label}.json", "w"), indent=2)
print(f"\n[{args.label}] wrote {OUT}/summary_{args.label}.json", flush=True)
