#!/usr/bin/env python3
"""Aggregate IN-DOMAIN per-verifier collapse/truncation across arms.

Companion to agg_ood_ci.py (which does the OOD side). Reads one or more
eval_adapter_vllm.py result JSONs and prints a per-verifier table of:

  - acc            : content accuracy (Table 2)
  - trunc%         : finish_reason == "length"        (Table 4)
  - trunc_nothink% : no closing </think>              (paper-prose / data_loader defn)
  - collapse%      : is_collapsed (zlib>=4x | 12-gram) (collapse figures)
  - seq-rep-4      : mean duplicated-4-gram fraction   (Table 3)

By default it reads the aggregate fields the patched eval already wrote into
each result JSON. With --recompute it re-derives every metric from the saved
per-response text (`all_responses`, requires the eval to have been run with
--save-all-responses) using the exact same definitions, so it also works on
outputs produced before the patch and lets you re-define a metric without
re-running inference.

Usage:
    # from the embedded per-verifier fields (fast, needs the patched eval):
    python scripts/analysis/agg_indomain_collapse.py \
        vanilla__step150__test.json faith_a050__step150__test.json

    # recompute from raw saved responses (needs --save-all-responses eval):
    python scripts/analysis/agg_indomain_collapse.py --recompute results/eval/*step150*test.json

Arm label defaults to the file's basename (strip .json); override the whole
row set with --labels a,b,c (comma-separated, positional to the file args).
"""
from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

VERIFIERS = ("digit", "mcq", "math", "iou")


def is_collapsed(raw: str) -> bool:
    """Identical to eval_adapter_vllm._is_collapsed / eval_ood_vllm.is_collapsed."""
    t = raw or ""
    if not t:
        return False
    b = t.encode("utf-8", "ignore")
    if len(b) / max(1, len(zlib.compress(b, 6))) >= 4.0:
        return True
    w = t.split()
    if len(w) < 24:
        return False
    g = [" ".join(w[i:i + 12]) for i in range(len(w) - 11)]
    return bool(g) and Counter(g).most_common(1)[0][1] >= 5


def seq_rep_4(raw: str) -> float:
    """Welleck et al. (2020) seq-rep-4, whitespace-tokenized."""
    w = (raw or "").split()
    if len(w) < 4:
        return 0.0
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def stats_from_embedded(d: dict) -> dict:
    """Pull the per-verifier fields the patched eval wrote."""
    rows = {}
    for v in VERIFIERS + ("overall",):
        b = d.get(v)
        if not isinstance(b, dict):
            continue
        rows[v] = {
            "n": b.get("total"),
            "acc": b.get("acc"),
            "trunc": b.get("trunc_rate"),
            "trunc_nothink": b.get("trunc_rate_nothink"),
            "collapse": b.get("collapse_rate"),
            "seqrep": b.get("seq_rep_4"),
        }
    return rows


def stats_from_responses(d: dict) -> dict:
    """Recompute everything from all_responses (or the 4-per-verifier samples
    as a last resort, which only gives an unreliable small-sample estimate)."""
    resp = d.get("all_responses")
    source = "all_responses"
    if not resp:
        resp = d.get("samples")
        source = "samples(4/verifier — UNRELIABLE)"
    if not resp:
        raise SystemExit("no all_responses/samples in JSON — re-run eval with "
                         "--save-all-responses, or drop --recompute to use embedded fields")

    n = defaultdict(int); acc = defaultdict(float)
    tl = defaultdict(int); tt = defaultdict(int)
    cl = defaultdict(int); sr = defaultdict(float)
    for r in resp:
        v = r.get("verifier")
        if v not in VERIFIERS:
            continue
        raw = r.get("response", "")
        fr = r.get("finish_reason")
        n[v] += 1
        acc[v] += float(r.get("acc", 0.0) or 0.0)
        # prefer the recorded flags; fall back to recomputing from text
        tl[v] += int(r["is_truncated"]) if "is_truncated" in r else int(fr == "length")
        tt[v] += int(r["is_truncated_nothink"]) if "is_truncated_nothink" in r \
            else int("</think>" not in (raw or ""))
        cl[v] += int(r["is_collapsed"]) if "is_collapsed" in r else int(is_collapsed(raw))
        sr[v] += float(r["seq_rep_4"]) if "seq_rep_4" in r else seq_rep_4(raw)

    rows = {}
    tot = defaultdict(float)
    N = 0
    for v in VERIFIERS:
        if not n[v]:
            continue
        rows[v] = {
            "n": n[v],
            "acc": round(acc[v] / n[v], 4),
            "trunc": round(100 * tl[v] / n[v], 2),
            "trunc_nothink": round(100 * tt[v] / n[v], 2),
            "collapse": round(100 * cl[v] / n[v], 2),
            "seqrep": round(sr[v] / n[v], 4),
        }
        N += n[v]
        tot["acc"] += acc[v]; tot["tl"] += tl[v]; tot["tt"] += tt[v]
        tot["cl"] += cl[v]; tot["sr"] += sr[v]
    if N:
        rows["overall"] = {
            "n": N,
            "acc": round(tot["acc"] / N, 4),
            "trunc": round(100 * tot["tl"] / N, 2),
            "trunc_nothink": round(100 * tot["tt"] / N, 2),
            "collapse": round(100 * tot["cl"] / N, 2),
            "seqrep": round(tot["sr"] / N, 4),
        }
    rows["_source"] = source  # noqa: for the header note
    return rows


def fmt(x, pct=False):
    if x is None:
        return "  -  "
    return f"{x:5.2f}{'%' if pct else ''}" if pct else f"{x:.4f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="eval_adapter_vllm result JSON(s)")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute from saved per-response text instead of embedded fields")
    ap.add_argument("--labels", default=None,
                    help="comma-separated arm labels, positional to files (default: basenames)")
    ap.add_argument("--metric", default="all",
                    choices=["all", "trunc", "trunc_nothink", "collapse", "seqrep"],
                    help="which collapse metric to tabulate (default all)")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        raise SystemExit(f"--labels count ({len(labels)}) != files ({len(args.files)})")

    per_arm = {}
    src_note = "embedded per-verifier fields"
    for lab, f in zip(labels, args.files):
        d = json.loads(Path(f).read_text())
        if args.recompute:
            rows = stats_from_responses(d)
            src_note = rows.pop("_source", src_note)
        else:
            rows = stats_from_embedded(d)
            if not any(rows[v].get("trunc") is not None for v in rows):
                raise SystemExit(f"{f}: no collapse fields — run the patched eval, "
                                 f"or pass --recompute (needs --save-all-responses)")
        per_arm[lab] = rows

    cols = VERIFIERS + ("overall",)
    metrics = (["trunc", "trunc_nothink", "collapse", "seqrep"]
               if args.metric == "all" else [args.metric])
    label_w = max(len(l) for l in labels) + 1

    print(f"# In-domain per-verifier collapse   (source: {src_note})")
    for m in metrics:
        pct = m != "seqrep"
        unit = " (%)" if pct else " (mean frac)"
        title = {"trunc": "Truncation rate — finish_reason==length (Table 4)",
                 "trunc_nothink": "Truncation rate — no </think> (paper prose)",
                 "collapse": "Collapse rate — is_collapsed (figures)",
                 "seqrep": "seq-rep-4 (Table 3)"}[m]
        print(f"\n## {title}{unit}")
        print(f"{'arm':<{label_w}} | " + " | ".join(f"{c:>7s}" for c in cols))
        print("-" * (label_w + 3 + 10 * len(cols)))
        for lab in labels:
            rows = per_arm[lab]
            cells = []
            for c in cols:
                v = rows.get(c, {}).get(m)
                cells.append("   -   " if v is None else (f"{v:7.2f}" if pct else f"{v:7.4f}"))
            print(f"{lab:<{label_w}} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
