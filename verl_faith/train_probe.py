"""Phase C: train Lookback-Lens logistic regression probe.

Loads features.npz produced by collect_probe_data.py, fits a balanced LR with L2,
runs the 3-check validation protocol from attn-probe-SLAKE:
  1. Val AUROC on a prompt-level split (no leakage)
  2. Within-group std (per-prompt rollout score variance must be > 0.05 to be useful as
     a tiebreaker — degenerate near-constant scores fail this)
  3. Permutation AUROC (shuffle labels, refit, AUROC must drop to ~0.5; > 0.55 suggests
     prompt-id leakage / circularity)

Saves the fitted classifier to probe.pkl + a JSON report next to it.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _balance(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Balance pos/neg by undersampling the majority class to match the minority."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n = min(len(pos_idx), len(neg_idx))
    if n == 0:
        return X, y
    pos_pick = rng.choice(pos_idx, size=n, replace=False)
    neg_pick = rng.choice(neg_idx, size=n, replace=False)
    keep = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(keep)
    return X[keep], y[keep]


def _balance_per_verifier(X: np.ndarray, y: np.ndarray, vtype: np.ndarray,
                          rng: np.random.Generator,
                          cap_per_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Balance pos/neg within each verifier slice. Concatenates the per-verifier
    balanced chunks.

    By default each verifier contributes `min(pos_v, neg_v)` rows of each class
    — preserves the natural-prevalence weighting between verifiers (so digit
    won't be artificially shrunk to math's tiny size and starve the LR).

    If `cap_per_class` is set, further cap each verifier to that many pos+neg
    rows (use this when you want strict verifier-type balance and accept the
    data loss).
    """
    chunks_X, chunks_y = [], []
    per_v: dict[str, dict[str, int]] = {}
    for v in np.unique(vtype):
        mask = vtype == v
        n_pos = int((y[mask] == 1).sum())
        n_neg = int((y[mask] == 0).sum())
        n = min(n_pos, n_neg)
        if cap_per_class is not None:
            n = min(n, cap_per_class)
        per_v[str(v)] = {"pos": n_pos, "neg": n_neg, "kept_per_class": n}
        if n == 0:
            continue
        pos_idx = np.where((y == 1) & mask)[0]
        neg_idx = np.where((y == 0) & mask)[0]
        pos_pick = rng.choice(pos_idx, size=n, replace=False)
        neg_pick = rng.choice(neg_idx, size=n, replace=False)
        keep = np.concatenate([pos_pick, neg_pick])
        chunks_X.append(X[keep])
        chunks_y.append(y[keep])
    print(f"[balance] per-verifier sizes: {per_v}")
    if not chunks_y:
        return _balance(X, y, rng)
    Xb = np.concatenate(chunks_X, axis=0)
    yb = np.concatenate(chunks_y, axis=0)
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]


def _prompt_split(prompt_id: np.ndarray, vtypes: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-verifier, prompt-level split. Each verifier's unique prompts are split
    independently then unioned, so a low-count verifier (e.g. math, the bottleneck)
    can NEVER have all its prompts land in val and silently vanish from training.
    A verifier with a single prompt keeps it in train (n_val=0)."""
    rng = np.random.default_rng(seed)
    val_prompts: set = set()
    for v in np.unique(vtypes):
        vp = np.unique(prompt_id[vtypes == v])
        rng.shuffle(vp)
        n_val = max(1, int(round(val_frac * len(vp)))) if len(vp) > 1 else 0
        val_prompts.update(vp[:n_val].tolist())
    val_mask = np.array([p in val_prompts for p in prompt_id])
    return ~val_mask, val_mask


def _within_group_std(prompt_id: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Mean and median of per-prompt rollout-score std."""
    stds = []
    for p in np.unique(prompt_id):
        s = scores[prompt_id == p]
        if len(s) > 1:
            stds.append(float(np.std(s)))
    if not stds:
        return 0.0, 0.0
    return float(np.mean(stds)), float(np.median(stds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True, help="output probe.pkl path")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c", type=float, default=1.0, help="LR L2 inverse strength")
    ap.add_argument("--n-perm", type=int, default=10, help="number of label-shuffle refits")
    ap.add_argument("--verifier", default=None, help="train on this verifier only (digit/mcq/math/iou)")
    ap.add_argument("--label-key", default="y", choices=["y", "y_lenient"],
                    help="npz column to train on: 'y' (strict, the live scorer) or "
                         "'y_lenient' (content-only, for the strict-vs-lenient AUROC comparison)")
    ap.add_argument("--per-verifier-balance", action="store_true", default=True,
                    help="balance pos/neg within each verifier AND equalize verifier contributions (default on for unified probes; auto-disabled by --verifier)")
    ap.add_argument("--no-per-verifier-balance", action="store_false", dest="per_verifier_balance")
    ap.add_argument("--cap-per-class", type=int, default=None,
                    help="optional cap on pos+neg rows per verifier (otherwise uses min available)")
    ap.add_argument("--allow-failing", action="store_true",
                    help="save the probe even if the 3-check validation gate fails "
                         "(default: a failing probe is NOT saved and the run exits non-zero)")
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = data["X"]
    if args.label_key not in data.files:
        raise SystemExit(f"label key '{args.label_key}' not in {args.features} "
                         f"(available: {data.files}) — recollect with the vllm collector "
                         f"or use --label-key y")
    y = data[args.label_key].astype(np.int64)
    prompt_id = data["prompt_id"].astype(np.int64)
    vtypes = data["verifier_type"]
    if args.verifier:
        mask = vtypes == args.verifier
        X, y, prompt_id, vtypes = X[mask], y[mask], prompt_id[mask], vtypes[mask]
        print(f"filtered to verifier={args.verifier}: {X.shape}")
    print(f"loaded {X.shape}; pos={int(y.sum())}/{len(y)} ({y.mean():.3f}); unique_prompts={len(np.unique(prompt_id))}")

    rng = np.random.default_rng(args.seed)
    train_mask, val_mask = _prompt_split(prompt_id, vtypes, args.val_frac, args.seed)
    Xtr, ytr, ptr = X[train_mask], y[train_mask], prompt_id[train_mask]
    Xva, yva, pva = X[val_mask], y[val_mask], prompt_id[val_mask]
    vtr = vtypes[train_mask]
    print(f"train: n={len(ytr)} pos={int(ytr.sum())} | val: n={len(yva)} pos={int(yva.sum())}")
    # Belt-and-suspenders: no verifier may be missing from train (the bug this split fixes).
    missing = set(np.unique(vtypes).tolist()) - set(np.unique(vtr).tolist())
    if missing:
        raise SystemExit(f"verifier(s) {missing} absent from train split — aborting (would train a probe blind to them)")
    if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
        raise SystemExit("train or val split has only one class; collect more data or rebalance")

    if args.per_verifier_balance and not args.verifier and len(np.unique(vtr)) > 1:
        Xtr_b, ytr_b = _balance_per_verifier(Xtr, ytr, vtr, rng, cap_per_class=args.cap_per_class)
        print(f"train (per-verifier balanced): n={len(ytr_b)}")
    else:
        Xtr_b, ytr_b = _balance(Xtr, ytr, rng)
        print(f"train (globally balanced): n={len(ytr_b)}")

    clf = LogisticRegression(max_iter=2000, C=args.c, penalty="l2", n_jobs=1)
    clf.fit(Xtr_b, ytr_b)

    # Check 1: val AUROC (prompt-split, no balancing on val)
    val_scores = clf.predict_proba(Xva)[:, 1]
    val_auroc = float(roc_auc_score(yva, val_scores))
    train_scores = clf.predict_proba(Xtr_b)[:, 1]
    train_auroc = float(roc_auc_score(ytr_b, train_scores))

    # Check 2: within-group std on val
    wg_mean, wg_median = _within_group_std(pva, val_scores)

    # Check 3: permutation AUROC (refit on shuffled labels, take mean)
    perm_aurocs = []
    for s in range(args.n_perm):
        prng = np.random.default_rng(args.seed + s + 1)
        ytr_b_shuf = ytr_b.copy()
        prng.shuffle(ytr_b_shuf)
        clf_p = LogisticRegression(max_iter=2000, C=args.c, penalty="l2", n_jobs=1)
        clf_p.fit(Xtr_b, ytr_b_shuf)
        scores_p = clf_p.predict_proba(Xva)[:, 1]
        perm_aurocs.append(float(roc_auc_score(yva, scores_p)))
    perm_mean = float(np.mean(perm_aurocs))
    perm_std = float(np.std(perm_aurocs))

    report = {
        "n_train": int(len(ytr_b)),
        "n_val": int(len(yva)),
        "n_train_prompts": int(len(np.unique(ptr))),
        "n_val_prompts": int(len(np.unique(pva))),
        "train_auroc": train_auroc,
        "val_auroc": val_auroc,
        "within_group_std_mean": wg_mean,
        "within_group_std_median": wg_median,
        "permutation_auroc_mean": perm_mean,
        "permutation_auroc_std": perm_std,
        "real_minus_perm_gap": val_auroc - perm_mean,
        "checks": {
            "check1_val_auroc_above_0.55": val_auroc > 0.55,
            "check2_within_group_std_above_0.05": wg_mean > 0.05,
            "check3_permutation_below_0.55": perm_mean < 0.55,
        },
    }
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print("=" * 60)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Always write the report JSON (even on failure) so the gate decision is auditable.
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(report, f, indent=2)

    # Enforce the 3-check validation gate. The docstring/README advertise these
    # checks as a guarantee the probe is real and non-circular; enforcing them
    # means a probe that fails (e.g. permutation AUROC >= 0.55 -> label leakage,
    # or within-group std <= 0.05 -> no usable tiebreak signal) is NEVER shipped.
    failed = [k for k, ok in report["checks"].items() if not ok]
    if failed and not args.allow_failing:
        raise SystemExit(
            f"probe FAILED validation gate: {failed}\n"
            f"report: {out.with_suffix('.json')}\n"
            f"re-run with --allow-failing to override (and document why).")
    if failed:
        print(f"WARNING: probe failed checks {failed} but --allow-failing set; shipping anyway.")

    import pickle
    with open(out, "wb") as f:
        pickle.dump(clf, f)
    # Sidecar metadata (<pkl>.meta.json). The pickle itself stays a bare sklearn
    # LR — scorers.py's serve-time load depends on that format — but the sidecar
    # records what the probe was fit on so a pickle is never provenance-free.
    from datetime import datetime, timezone
    meta = {
        "feat_dim": int(X.shape[1]),
        "label_key": args.label_key,
        "features_npz": str(Path(args.features).resolve()),
        "model": str(data["model"]) if "model" in data.files else None,
        "verifier_filter": args.verifier,
        "n_train": int(len(ytr_b)),
        "n_val": int(len(yva)),
        "val_auroc": val_auroc,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(str(out) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved probe to {out} (+ {out.name}.meta.json)"
          + (" (validation gate PASSED)" if not failed else " (gate OVERRIDDEN)"))


if __name__ == "__main__":
    main()
