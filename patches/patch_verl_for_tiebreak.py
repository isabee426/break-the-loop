"""Apply verl source patches needed for the tiebreak adv estimator + faith plumbing.

1. naive.py reward manager: when calling compute_score, also pass `image` and
   `question` in extra_info (lifted from non_tensor_batch["images"]/["prompt"]).

2. core_algos.py: register a new adv estimator `tiebreak_centered_rank` that
   decodes (formatted, correctness, faith) from the composite reward via the
   `verl_faith.scorers.decode_composite` helper, lex-sorts per group, and
   assigns centered-rank advantages.

Idempotent — safe to re-run."""
from __future__ import annotations
import os
import re
from pathlib import Path


def _resolve_verl() -> Path:
    """Locate the verl source to patch. Precedence: $VERL_HOME -> the installed
    `verl` package location -> the legacy hardcoded path. Portable across machines
    (no machine-specific default)."""
    env = os.environ.get("VERL_HOME")
    if env:
        return Path(env)
    try:
        import importlib.util
        spec = importlib.util.find_spec("verl")
        if spec is not None:
            origin = spec.origin or (spec.submodule_search_locations or [None])[0]
            if origin:
                return Path(origin).resolve().parent.parent  # .../verl/__init__.py -> repo root
    except Exception:
        pass
    return Path("verl")


VERL = _resolve_verl()

# ---------------------------------------------------------------------------
# Patch 1: naive.py — pass image+question into extra_info before compute_score
# ---------------------------------------------------------------------------

NAIVE = VERL / "verl/workers/reward_manager/naive.py"
OLD_NAIVE = '''            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores'''

NEW_NAIVE = '''            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            # FAITH PATCH: pass image + question to compute_score so the faith
            # probe can re-forward. Multi-source key grab (same order as the
            # reward-loop patch, which is the empirically-verified chain):
            # faith_image (rl_dataset stash) -> multi_modal_data -> images;
            # question from raw_prompt -> prompt. Plain "images" is popped by
            # RLHFDataset before reward time and "prompt" is usually absent —
            # reading only those (the old version of this patch) found nothing.
            try:
                _ntb = data_item.non_tensor_batch
                _img = None
                _fi = _ntb.get("faith_image")
                if _fi is not None and len(_fi) > 0:
                    _img = _fi[0]
                if _img is None:
                    _mmd = _ntb.get("multi_modal_data")
                    if isinstance(_mmd, dict):
                        _il = _mmd.get("image") or _mmd.get("images")
                        if _il is not None and len(_il) > 0:
                            _img = _il[0]
                if _img is None:
                    _ims = _ntb.get("images")
                    if _ims is not None and len(_ims) > 0:
                        _img = _ims[0]
                if _img is not None:
                    if isinstance(_img, dict) and "bytes" in _img:
                        from io import BytesIO as _BIO
                        from PIL import Image as _PIL
                        _img = _PIL.open(_BIO(_img["bytes"])).convert("RGB")
                    extra_info["image"] = _img
                _msgs = _ntb.get("raw_prompt")
                if _msgs is None:
                    _msgs = _ntb.get("prompt")
                if _msgs is not None:
                    for _m in list(_msgs):
                        if isinstance(_m, dict) and _m.get("role") == "user":
                            _c = _m.get("content")
                            if isinstance(_c, str):
                                _q = _c
                                if _q.startswith("<image>"):
                                    _q = _q.split("\\n", 1)[1] if "\\n" in _q else _q
                                extra_info["question"] = _q
                            elif isinstance(_c, (list, tuple)):
                                _parts = [p.get("text", "") for p in _c if isinstance(p, dict) and p.get("text")]
                                extra_info["question"] = " ".join(t for t in _parts if t)
                            break
            except Exception:
                pass'''


_NAIVE_FAITH_MARK = "            # FAITH PATCH: pass image"
_NAIVE_END_MARK = "            except Exception:\n                pass"


def patch_naive() -> bool:
    src = NAIVE.read_text()
    if "FAITH PATCH" in src:
        if "faith_image" in src:
            return False  # already the multi-source version
        # OLD single-source FAITH PATCH present — swap just the faith block.
        start = src.index(_NAIVE_FAITH_MARK)
        end = src.index(_NAIVE_END_MARK, start) + len(_NAIVE_END_MARK)
        new_block = NEW_NAIVE[NEW_NAIVE.index(_NAIVE_FAITH_MARK):]
        NAIVE.write_text(src[:start] + new_block + src[end:])
        return True
    if OLD_NAIVE not in src:
        raise SystemExit("naive.py: OLD block not found verbatim")
    NAIVE.write_text(src.replace(OLD_NAIVE, NEW_NAIVE, 1))
    return True


# ---------------------------------------------------------------------------
# Patch 2: core_algos.py — register tiebreak_centered_rank adv estimator
# ---------------------------------------------------------------------------

CORE = VERL / "verl/trainer/ppo/core_algos.py"
TIEBREAK_BLOCK = '''
# ---------------------------------------------------------------------------
# Shared faith-sentinel handling (research-grade, 8B A/B).
# The scorer emits faith in [0,1], or the -1.0 UNAVAILABLE sentinel when the
# probe/extractor failed for a rollout (extract error, missing image, probe
# disabled). -1.0 must NOT be read as "very low faith": that would rank every
# failed-extraction rollout at the bottom of its correctness tie -- a length-
# correlated bias (extraction fails preferentially on long responses) that is
# exactly the confound the length-tiebreak control exists to isolate. So the
# faith path EXCLUDES faith<0 members from within-tie jitter / rescue.
#
# The random/length control arms call the faith estimator with a SUBSTITUTED
# signal (uniform noise / -length). Those signals are legitimately negative
# (length) and are never the sentinel, so substituted=True disables both the
# sentinel exclusion and the FAITH_STRICT guard, keeping the controls' validated
# behaviour bitwise-identical to before.
# ---------------------------------------------------------------------------
def _tb_read_faith(non_tensor_batch, bsz, substituted=False):
    """Return (faith[bsz], genuine_mask[bsz]).

    genuine_mask is True where faith carries a real probe probability in [0,1]
    and False where it is the -1.0 unavailable sentinel. On the substituted-
    signal control path every member is genuine. Under FAITH_STRICT (default on)
    a missing 'faith' key, or an all-unavailable/all-zero signal sustained for
    32 consecutive steps, raises rather than silently degrading to vanilla GRPO
    (the pre-Jun-2 dead-faith failure class).
    """
    import numpy as _np
    import os as _os
    raw = None
    if non_tensor_batch is not None and "faith" in non_tensor_batch:
        raw = _np.asarray(non_tensor_batch["faith"], dtype=_np.float64)
    if substituted:
        f = raw if raw is not None else _np.zeros(bsz, dtype=_np.float64)
        f = _np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        return f, _np.ones(bsz, dtype=bool)
    strict = _os.environ.get("FAITH_STRICT", "1") == "1"
    if raw is None:
        if strict:
            raise RuntimeError(
                "FAITH_STRICT: faith-tiebreak arm but non_tensor_batch has no "
                "'faith' key -> the reward plumbing is broken and this arm is "
                "training as bitwise-vanilla GRPO (cf. the pre-Jun-2 dead-faith "
                "incident). Fix the reward loop, or set FAITH_STRICT=0 to allow "
                "the vanilla fallback intentionally.")
        return _np.zeros(bsz, dtype=_np.float64), _np.zeros(bsz, dtype=bool)
    # nan/inf -> sentinel (excluded), NOT 0.0 (which would read as genuine faith).
    faith = _np.where(_np.isfinite(raw), raw, -1.0)
    genuine = faith >= 0.0
    _g = globals()
    gen_frac = float(genuine.mean())
    dead = (not genuine.any()) or bool(_np.all(faith[genuine] == 0.0))
    _g["_TB_FAITH_DEAD_STREAK"] = (_g.get("_TB_FAITH_DEAD_STREAK", 0) + 1) if dead else 0
    # Partial-degrade guard. The dead-streak above only fires on a FULLY dead
    # signal; a SUSTAINED partial sentinel rate (e.g. the extractor choking on
    # long sequences for a subset of rollouts) leaves genuine.any() True, so the
    # dead-streak never accrues and the treatment silently weakens on the
    # affected rollouts with no alarm. Track a separate streak on
    # genuine_fraction < floor and, under strict, fatal on a sustained breach.
    PARTIAL_FLOOR = float(_os.environ.get("FAITH_GENUINE_FLOOR", "0.5"))
    partial = (not dead) and gen_frac < PARTIAL_FLOOR
    _g["_TB_FAITH_PARTIAL_STREAK"] = (_g.get("_TB_FAITH_PARTIAL_STREAK", 0) + 1) if partial else 0
    # Per-step visibility (one line; mirrors the mass_frac print). Makes a
    # creeping sentinel rate observable instead of silent.
    print(f"[faith-health] genuine_frac={gen_frac:.3f} ({int(genuine.sum())}/{bsz}) "
          f"dead_streak={_g['_TB_FAITH_DEAD_STREAK']} "
          f"partial_streak={_g['_TB_FAITH_PARTIAL_STREAK']} (floor={PARTIAL_FLOOR})",
          flush=True)
    if strict and _g["_TB_FAITH_DEAD_STREAK"] >= 32:
        raise RuntimeError(
            "FAITH_STRICT: faith has been unavailable/all-zero for 32 consecutive "
            "steps -> the faith signal is dead (cf. pre-Jun-2 incident). "
            "Investigate the probe/extractor; set FAITH_STRICT=0 to suppress.")
    if strict and _g["_TB_FAITH_PARTIAL_STREAK"] >= 32:
        raise RuntimeError(
            f"FAITH_STRICT: genuine faith fraction < {PARTIAL_FLOOR} for 32 consecutive "
            f"steps (now {gen_frac:.3f}) -> the extractor is PARTIALLY degraded (e.g. "
            "choking on long sequences), silently weakening the treatment on the "
            "sentinel rollouts. Investigate the extractor; lower FAITH_GENUINE_FLOOR "
            "or set FAITH_STRICT=0 to suppress.")
    return faith, genuine


def _tb_neutralize_sentinel(faith, neutral):
    """For ablation estimators that are not sentinel-aware: replace -1.0
    unavailable sentinels with a neutral value so they cannot poison ranking /
    modulation. Not used by the 8B A/B arms (those use the exclusion path)."""
    import numpy as _np
    faith = _np.where(_np.isfinite(faith), faith, -1.0)
    return _np.where(faith < 0.0, neutral, faith)


# ---------------------------------------------------------------------------
# Tiebreak centered-rank advantage estimator (bbox-free Option D).
# PHASE-2 (no_drop_unformatted): format signal comes from the scorer's
# literature-standard +0.1 format reward, so vanilla and tiebreak have the
# same format pressure. The ONLY mechanistic difference between them is how
# advantage is computed from rewards (mean/std vs centered-rank + faith
# lex-sort tiebreak). ALL N rollouts participate in the centered rank —
# unformatted (correctness=0, faith=0 per scorer) naturally fall to the
# bottom of the rank and get strongly-negative advantage, symmetric with how
# vanilla GRPO's mean/std subtraction handles them.
# Reads (correctness, faith) directly from non_tensor_batch.
# ---------------------------------------------------------------------------

@register_adv_est("tiebreak_centered_rank")
def compute_tiebreak_centered_rank_advantage(
    token_level_rewards,
    response_mask,
    index,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    non_tensor_batch=None,
    **kwargs,
):
    """Centered-rank advantage with (correctness, faith) lex-sort tiebreak.

    Phase-2 variant: NO drop_unformatted. All rollouts participate in the
    centered rank. Unformatted rollouts land at the bottom of the rank
    (correctness=0, faith=0 per scorer) -> strongly-negative advantage ->
    active push-away signal. Symmetric with how vanilla GRPO treats them.
    Format incentive comes from the scorer's +0.1 format bonus (literature
    standard) applied to both methods.
    """
    import numpy as _np
    import torch as _torch

    per_rollout = token_level_rewards.sum(dim=-1)  # (bsz,)
    bsz = per_rollout.shape[0]

    # Pull correctness/faith from non_tensor_batch (populated from the scorer's
    # return dict keys). Fall back to legacy composite-decode if the batch
    # doesn't have them (e.g. an older scorer pickle is still in use).
    correctness = _np.zeros(bsz, dtype=_np.float64)
    faith = _np.zeros(bsz, dtype=_np.float64)
    have_extras = (
        non_tensor_batch is not None
        and "faith" in non_tensor_batch
    )
    if have_extras:
        # token_level_rewards carries the per-rollout reward (correctness +
        # format bonus). Used directly as the primary sort key — unformatted
        # rollouts have score=0 here (scorer assigns nothing) so they fall
        # to the bottom of the rank naturally.
        correctness = per_rollout.detach().cpu().to(_torch.float64).numpy()
        faith = _np.asarray(non_tensor_batch["faith"], dtype=_np.float64)
    else:
        from verl_faith.scorers import decode_composite
        composites = per_rollout.detach().cpu().to(_torch.float64).numpy()
        for i, c in enumerate(composites):
            _f, k, ft = decode_composite(float(c))
            correctness[i], faith[i] = k, ft
    faith = _tb_neutralize_sentinel(faith, 0.5)  # ablation: -1.0 sentinel -> neutral 0.5

    if isinstance(index, _torch.Tensor):
        idx_arr = index.detach().cpu().numpy()
    else:
        idx_arr = _np.asarray(index)
    # Index may be UUID strings (one per prompt) — use as-is for grouping; no int cast.

    centered_rank = _np.zeros(bsz, dtype=_np.float64)
    for gid in _np.unique(idx_arr):
        members = _np.where(idx_arr == gid)[0]
        n = len(members)
        if n <= 1:
            continue
        # Lex-sort all N rollouts in this prompt-group by (correctness desc,
        # faith desc). NO drop_unformatted — unformatted rollouts (correctness=0,
        # faith=0) naturally fall to the bottom of the rank and receive strongly-
        # negative advantage.
        sort_key = list(zip(-correctness[members], -faith[members], _np.arange(n)))
        order = sorted(range(n), key=lambda i: sort_key[i])
        mean_rank = (n - 1) / 2.0
        for pos, ord_i in enumerate(order):
            global_idx = members[ord_i]
            rank = (n - 1) - pos
            centered_rank[global_idx] = (rank - mean_rank) / max(n - 1, 1)

    if norm_adv_by_std_in_grpo:
        for gid in _np.unique(idx_arr):
            members = _np.where(idx_arr == gid)[0]
            vals = centered_rank[members]
            std = vals.std()
            if std > 1e-8:
                centered_rank[members] = vals / std

    advantages_per_rollout = _torch.from_numpy(centered_rank).to(token_level_rewards.device).to(token_level_rewards.dtype)
    advantages = advantages_per_rollout.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    returns = advantages
    return advantages, returns


# ---------------------------------------------------------------------------
# GOPO — rank-based advantage, pure ordinal (arXiv:2602.03876).
# A_i = 2 - (rank(i) - 1) * 4/(G-1), where rank 1 = highest reward.
# ---------------------------------------------------------------------------
@register_adv_est("gopo")
def compute_gopo_advantage(
    token_level_rewards, response_mask, index,
    epsilon: float = 1e-6, norm_adv_by_std_in_grpo: bool = True,
    config=None, **kwargs,
):
    import numpy as _np
    import torch as _torch
    per = token_level_rewards.sum(dim=-1).detach().cpu().to(_torch.float64).numpy()
    bsz = per.shape[0]
    idx_arr = index.detach().cpu().numpy() if isinstance(index, _torch.Tensor) else _np.asarray(index)
    adv = _np.zeros(bsz, dtype=_np.float64)
    for gid in _np.unique(idx_arr):
        members = _np.where(idx_arr == gid)[0]
        n = len(members)
        if n <= 1:
            continue
        order = _np.argsort(-per[members], kind="stable")
        ranks = _np.empty(n, dtype=_np.float64)
        ranks[order] = _np.arange(1, n + 1)
        adv[members] = 2.0 - (ranks - 1.0) * (4.0 / (n - 1))
    out = _torch.from_numpy(adv).to(token_level_rewards.device).to(token_level_rewards.dtype)
    advantages = out.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    return advantages, advantages


# ---------------------------------------------------------------------------
# Unlikeliness reward (arXiv:2506.02355).
# Rank rollouts by current-policy probability (rank 1 = highest probability).
# Modified reward: r'_i = r_i * (1 - beta * (G - rank_i)/G).
# Then standard GRPO advantage on modified rewards.
# beta defaults to 0.25 per the paper.
# ---------------------------------------------------------------------------
@register_adv_est("unlikeliness_grpo")
def compute_unlikeliness_advantage(
    token_level_rewards, response_mask, index,
    epsilon: float = 1e-6, norm_adv_by_std_in_grpo: bool = True,
    config=None, batch=None, **kwargs,
):
    import numpy as _np
    import torch as _torch
    import os as _os
    beta = float(_os.environ.get("UNLIKELINESS_BETA", "0.25"))

    per = token_level_rewards.sum(dim=-1).detach().cpu().to(_torch.float64).numpy()
    bsz = per.shape[0]
    idx_arr = index.detach().cpu().numpy() if isinstance(index, _torch.Tensor) else _np.asarray(index)

    # Sum response log-probs per rollout (proxy for policy probability).
    if batch is None or "old_log_probs" not in batch:
        # Fall back to vanilla GRPO if log_probs unavailable.
        scaled = per
    else:
        lp = batch["old_log_probs"].detach()
        mask = response_mask.detach()
        lp_sum = ((lp * mask).sum(dim=-1)).cpu().to(_torch.float64).numpy()
        scaled = _np.zeros(bsz, dtype=_np.float64)
        for gid in _np.unique(idx_arr):
            members = _np.where(idx_arr == gid)[0]
            n = len(members)
            if n <= 1:
                scaled[members] = per[members]
                continue
            order = _np.argsort(-lp_sum[members], kind="stable")  # highest prob first
            ranks = _np.empty(n, dtype=_np.float64)
            ranks[order] = _np.arange(1, n + 1)
            scaled[members] = per[members] * (1.0 - beta * (n - ranks) / n)

    # Standard GRPO advantage on the modified rewards.
    adv = _np.zeros(bsz, dtype=_np.float64)
    for gid in _np.unique(idx_arr):
        members = _np.where(idx_arr == gid)[0]
        n = len(members)
        if n <= 1:
            continue
        m = scaled[members].mean()
        s = scaled[members].std() + epsilon if norm_adv_by_std_in_grpo else 1.0
        adv[members] = (scaled[members] - m) / s
    out = _torch.from_numpy(adv).to(token_level_rewards.device).to(token_level_rewards.dtype)
    advantages = out.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    return advantages, advantages


# ---------------------------------------------------------------------------
# FaithRL — Faithfulness-Aware Advantage Modulation (arXiv:2602.03507).
# Adapted to our setup: faith comes from the lookback-lens probe (continuous
# [0,1]) instead of the paper's binary verifier.
#   A_i' = A_i * M_i
#   M_i = (1-alpha)*faith_i + alpha          if A_i > 0
#   M_i = (1-alpha)*(1-faith_i) + alpha      if A_i <= 0
# Geometric reward (+y0 / 0 / -x0) collapses to plain correctness in our case
# since we don't have separate "miss" vs "hallucination" categories.
# ---------------------------------------------------------------------------
@register_adv_est("faithrl_grpo")
def compute_faithrl_advantage(
    token_level_rewards, response_mask, index,
    epsilon: float = 1e-6, norm_adv_by_std_in_grpo: bool = True,
    config=None, non_tensor_batch=None, **kwargs,
):
    import numpy as _np
    import torch as _torch
    import os as _os
    alpha = float(_os.environ.get("FAITHRL_ALPHA", "0.0"))

    per = token_level_rewards.sum(dim=-1).detach().cpu().to(_torch.float64).numpy()
    bsz = per.shape[0]
    idx_arr = index.detach().cpu().numpy() if isinstance(index, _torch.Tensor) else _np.asarray(index)

    if non_tensor_batch is None or "faith" not in non_tensor_batch:
        faith = _np.full(bsz, 0.5, dtype=_np.float64)
    else:
        faith = _np.asarray(non_tensor_batch["faith"], dtype=_np.float64)
    faith = _tb_neutralize_sentinel(faith, 0.5)  # ablation: -1.0 sentinel -> neutral 0.5

    # Standard GRPO advantage first.
    adv = _np.zeros(bsz, dtype=_np.float64)
    for gid in _np.unique(idx_arr):
        members = _np.where(idx_arr == gid)[0]
        n = len(members)
        if n <= 1:
            continue
        m = per[members].mean()
        s = per[members].std() + epsilon if norm_adv_by_std_in_grpo else 1.0
        adv[members] = (per[members] - m) / s

    # Modulate by faith.
    pos = adv > 0
    mod = _np.where(pos,
                    (1.0 - alpha) * faith + alpha,
                    (1.0 - alpha) * (1.0 - faith) + alpha)
    adv = adv * mod

    out = _torch.from_numpy(adv).to(token_level_rewards.device).to(token_level_rewards.dtype)
    advantages = out.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    return advantages, advantages


# ---------------------------------------------------------------------------
# Tiebreak rescue-ties-only advantage estimator (Phase-2 ablation).
# Vanilla mean-subtracted advantage on batches with reward variance;
# falls back to centered-rank-on-faith ONLY for fully-tied batches (where
# vanilla would assign zero advantage). Isolates the "rescue zero-variance
# batches" mechanism from the rank-vs-mean confound. Directly comparable to
# vanilla on every batch with non-zero reward variance.
# ---------------------------------------------------------------------------
@register_adv_est("tiebreak_rescue_ties_only")
def compute_tiebreak_rescue_ties_only_advantage(
    token_level_rewards,
    response_mask,
    index,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    non_tensor_batch=None,
    **kwargs,
):
    """Mean-subtracted advantage (matches vanilla GRPO) on batches with
    reward variance; falls back to faith-ranked centered rank ONLY on
    fully-tied batches where vanilla would give zero advantage.

    Isolates the "rescue zero-variance batches" mechanism from the rank-
    vs-mean confound. On every batch with std(reward) > epsilon, this is
    identical to vanilla GRPO. Only batches where all rollouts tie on
    reward get the faith-rescue treatment.
    """
    import numpy as _np
    import torch as _torch

    per_rollout = token_level_rewards.sum(dim=-1)
    bsz = per_rollout.shape[0]
    rewards = per_rollout.detach().cpu().to(_torch.float64).numpy()

    faith = _np.zeros(bsz, dtype=_np.float64)
    if non_tensor_batch is not None and "faith" in non_tensor_batch:
        faith = _np.asarray(non_tensor_batch["faith"], dtype=_np.float64)
    # Gradient-poisoning guard: a single NaN/inf faith would propagate through
    # mean/std/z-score into the advantage tensor with no error. Turn it inert.
    faith = _tb_neutralize_sentinel(faith, 0.5)  # ablation: -1.0 sentinel -> neutral 0.5 (not a real faith arm)

    if isinstance(index, _torch.Tensor):
        idx_arr = index.detach().cpu().numpy()
    else:
        idx_arr = _np.asarray(index)

    advantages = _np.zeros(bsz, dtype=_np.float64)
    for gid in _np.unique(idx_arr):
        members = _np.where(idx_arr == gid)[0]
        n = len(members)
        if n <= 1:
            continue
        member_rewards = rewards[members]
        std = member_rewards.std(ddof=1)  # unbiased std -> matches verl GRPO's torch.std
        if std > epsilon:
            # IDENTICAL to vanilla GRPO mean-subtracted advantage.
            mean = member_rewards.mean()
            advantages[members] = member_rewards - mean
            if norm_adv_by_std_in_grpo:
                advantages[members] = advantages[members] / (std + epsilon)  # exactly verl GRPO: (x-mean)/(std+eps)
        else:
            # All tied on reward — rescue with faith-ranked centered rank.
            member_faiths = faith[members]
            sort_key = list(zip(-member_faiths, _np.arange(n)))
            order = sorted(range(n), key=lambda i: sort_key[i])
            mean_rank = (n - 1) / 2.0
            for pos, ord_i in enumerate(order):
                rank = (n - 1) - pos
                advantages[members[ord_i]] = (rank - mean_rank) / max(n - 1, 1)
            if norm_adv_by_std_in_grpo:
                rescue_std = advantages[members].std()
                if rescue_std > 1e-8:
                    advantages[members] = advantages[members] / rescue_std

    out = _torch.from_numpy(advantages).to(token_level_rewards.device).to(token_level_rewards.dtype)
    advantages_full = out.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    return advantages_full, advantages_full


# ---------------------------------------------------------------------------
# Tiebreak break-all-ties advantage estimator (Phase-2, mean-subtracted base).
# Mean-subtracted advantage as the base; within each tied-reward subgroup,
# adds a faith-ranked centered-rank perturbation. ALL ties within correctness
# groups get broken by faith (the structural intent of the tiebreak mechanism)
# while keeping reward magnitudes (mean-subtracted) for between-group spread.
#
# Per-band perturbation magnitude is capped so that the perturbation NEVER
# pushes a tied rollout's advantage past the boundary of an adjacent band.
# Guarantees: rollouts in a higher-correctness band always end with strictly
# greater advantage than rollouts in a lower-correctness band, regardless
# of faith values. Within-band order is fully determined by faith.
#
# User-requested ε via env var FAITH_TIE_EPSILON (default 0.1). Actual ε per
# band = min(user_ε, safe_ε_to_neighbors).
# ---------------------------------------------------------------------------
@register_adv_est("tiebreak_break_all_ties")
def compute_tiebreak_break_all_ties_advantage(
    token_level_rewards,
    response_mask,
    index,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    non_tensor_batch=None,
    **kwargs,
):
    """Mean-subtracted advantage + faith-broken sub-group ties, with
    correctness-preserving safety cap on per-band perturbation.

    For each prompt-group:
      1. Compute mean-subtracted advantage on rewards (same as vanilla GRPO).
      2. Identify subgroups of rollouts with tied rewards (bands).
      3. For each tied band, compute the safe perturbation magnitude that
         guarantees the band stays strictly between its neighbors after
         perturbation. Use min(user_eps, safe_eps).
      4. Apply faith-ranked centered-rank perturbation scaled by actual_eps.
      5. Std-normalize the final per-group advantages.

    Cross-band correctness invariant: every rollout in a higher-reward band
    ends with strictly greater advantage than every rollout in a lower-reward
    band, regardless of faith values.
    """
    import os as _os
    import numpy as _np
    import torch as _torch

    tie_eps = float(_os.environ.get("FAITH_TIE_EPSILON", "0.1"))

    per_rollout = token_level_rewards.sum(dim=-1)
    bsz = per_rollout.shape[0]
    rewards = per_rollout.detach().cpu().to(_torch.float64).numpy()

    faith = _np.zeros(bsz, dtype=_np.float64)
    if non_tensor_batch is not None and "faith" in non_tensor_batch:
        faith = _np.asarray(non_tensor_batch["faith"], dtype=_np.float64)
    # Gradient-poisoning guard: a single NaN/inf faith would propagate through
    # mean/std/z-score into the advantage tensor with no error. Turn it inert.
    faith = _tb_neutralize_sentinel(faith, 0.5)  # ablation: -1.0 sentinel -> neutral 0.5 (not a real faith arm)

    if isinstance(index, _torch.Tensor):
        idx_arr = index.detach().cpu().numpy()
    else:
        idx_arr = _np.asarray(index)

    advantages = _np.zeros(bsz, dtype=_np.float64)
    for gid in _np.unique(idx_arr):
        members = _np.where(idx_arr == gid)[0]
        n = len(members)
        if n <= 1:
            continue
        member_rewards = rewards[members]
        member_faiths = faith[members]
        # Mean-subtracted base.
        group_mean = member_rewards.mean()
        local_adv = member_rewards - group_mean

        # Build band info: sorted unique rewards (descending) and per-band size.
        # Bands sorted desc: highest correctness first.
        unique_r = sorted(set(member_rewards.tolist()), reverse=True)
        band_size = {r: int((member_rewards == r).sum()) for r in unique_r}

        # For each band, compute safe ε against its adjacent neighbors so the
        # perturbation can NEVER push a rollout past a neighboring band.
        # Constraint for adjacency between band B (size N_B at adv A_B) and
        # band B+1 (size N_{B+1} at adv A_{B+1}, with A_B > A_{B+1}):
        #   A_B - eps*(N_B-1)/2  >  A_{B+1} + eps*(N_{B+1}-1)/2
        #   eps < 2*(A_B - A_{B+1}) / (N_B + N_{B+1} - 2)
        # Compute the max safe eps for each band considering both neighbors.
        for r in unique_r:
            sub_mask = member_rewards == r
            sub_n = band_size[r]
            if sub_n <= 1:
                continue  # solo band, no perturbation needed
            sub_idx = _np.where(sub_mask)[0]
            sub_faiths = member_faiths[sub_idx]

            constraints = [tie_eps]
            # Higher neighbor: the next band up (greater reward)
            higher = [r2 for r2 in unique_r if r2 > r]
            if higher:
                r_high = min(higher)  # nearest higher band
                n_high = band_size[r_high]
                gap = r_high - r  # in mean-subtracted space, gap == reward gap
                denom = (sub_n - 1) + (n_high - 1) if (n_high > 1) else (sub_n - 1)
                if denom > 0:
                    safe = (2.0 * gap / denom if (n_high > 1) else (2.0 * gap / (sub_n - 1))) * 0.99
                    constraints.append(safe)
            # Lower neighbor
            lower = [r2 for r2 in unique_r if r2 < r]
            if lower:
                r_low = max(lower)
                n_low = band_size[r_low]
                gap = r - r_low
                denom = (sub_n - 1) + (n_low - 1) if (n_low > 1) else (sub_n - 1)
                if denom > 0:
                    safe = (2.0 * gap / denom if (n_low > 1) else (2.0 * gap / (sub_n - 1))) * 0.99
                    constraints.append(safe)

            actual_eps = min(constraints)

            # Apply faith-ranked centered-rank perturbation with capped ε.
            sort_order = _np.argsort(-sub_faiths, kind="stable")
            mean_rank = (sub_n - 1) / 2.0
            perturb = _np.zeros(sub_n, dtype=_np.float64)
            for new_pos, ord_i in enumerate(sort_order):
                rank = (sub_n - 1) - new_pos  # highest-faith → rank N-1 (top)
                perturb[ord_i] = (rank - mean_rank) * actual_eps
            local_adv[sub_idx] = local_adv[sub_idx] + perturb

        # Std-normalize per-group (matches vanilla GRPO behavior).
        if norm_adv_by_std_in_grpo:
            std = local_adv.std()
            if std > 1e-8:
                local_adv = local_adv / std
        advantages[members] = local_adv

    out = _torch.from_numpy(advantages).to(token_level_rewards.device).to(token_level_rewards.dtype)
    advantages_full = out.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    return advantages_full, advantages_full


# ---------------------------------------------------------------------------
# Tiebreak fill-bands-lex advantage estimator (Phase-2, parameter-free).
# Lex-sort by (correctness DESC, faith DESC); within each tied-reward band,
# spread members across the half-gap to adjacent bands automatically.
# No epsilon to tune — the within-band spread is bounded by the gap structure
# of the actual reward distribution. Highest-faith rollout in a band gets the
# top of the band's slot; lowest-faith gets the bottom.
#
# All-tied (degenerate) batches fall back to faith-ranked centered rank.
# Cross-band ordering preserved by construction.
# ---------------------------------------------------------------------------
@register_adv_est("tiebreak_fill_bands_lex")
def compute_tiebreak_fill_bands_lex_advantage(
    token_level_rewards,
    response_mask,
    index,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    non_tensor_batch=None,
    **kwargs,
):
    """Lex-sort with auto-fill within each band.

    Each band's members are distributed across [lower_bound, upper_bound]
    where the bounds are midpoints to adjacent bands. Top and bottom bands
    use symmetric spread equal to half-gap to their only neighbor. Single-
    band (all-tied) groups fall back to faith-ranked centered rank.
    """
    import numpy as _np
    import torch as _torch

    per_rollout = token_level_rewards.sum(dim=-1)
    bsz = per_rollout.shape[0]
    rewards = per_rollout.detach().cpu().to(_torch.float64).numpy()

    faith = _np.zeros(bsz, dtype=_np.float64)
    if non_tensor_batch is not None and "faith" in non_tensor_batch:
        faith = _np.asarray(non_tensor_batch["faith"], dtype=_np.float64)
    # Gradient-poisoning guard: a single NaN/inf faith would propagate through
    # mean/std/z-score into the advantage tensor with no error. Turn it inert.
    faith = _tb_neutralize_sentinel(faith, 0.5)  # ablation: -1.0 sentinel -> neutral 0.5 (not a real faith arm)

    if isinstance(index, _torch.Tensor):
        idx_arr = index.detach().cpu().numpy()
    else:
        idx_arr = _np.asarray(index)

    advantages = _np.zeros(bsz, dtype=_np.float64)

    for gid in _np.unique(idx_arr):
        members = _np.where(idx_arr == gid)[0]
        n = len(members)
        if n <= 1:
            continue
        member_rewards = rewards[members]
        member_faiths = faith[members]
        group_mean = member_rewards.mean()
        local_adv = member_rewards - group_mean  # mean-subtracted base

        unique_r = sorted(set(member_rewards.tolist()), reverse=True)
        n_bands = len(unique_r)

        # All-tied case: only one band → fall back to centered-rank-by-faith.
        if n_bands == 1:
            sort_order = _np.argsort(-member_faiths, kind="stable")
            mean_rank = (n - 1) / 2.0
            for new_pos, ord_i in enumerate(sort_order):
                rank = (n - 1) - new_pos
                local_adv[ord_i] = (rank - mean_rank) / max(n - 1, 1)
            if norm_adv_by_std_in_grpo:
                std = local_adv.std()
                if std > 1e-8:
                    local_adv = local_adv / std
            advantages[members] = local_adv
            continue

        # Compute (lower, upper) bound for each band.
        band_bounds = {}
        for i, r in enumerate(unique_r):
            a_b = r - group_mean
            if i > 0:
                a_h = unique_r[i - 1] - group_mean
                half_above = (a_h - a_b) / 2.0
            else:
                half_above = None
            if i < n_bands - 1:
                a_l = unique_r[i + 1] - group_mean
                half_below = (a_b - a_l) / 2.0
            else:
                half_below = None

            if half_above is None:  # top band — symmetric spread by half_below
                band_bounds[r] = (a_b - half_below, a_b + half_below)
            elif half_below is None:  # bottom band — symmetric spread by half_above
                band_bounds[r] = (a_b - half_above, a_b + half_above)
            else:  # middle band — asymmetric to neighbors
                band_bounds[r] = (a_b - half_below, a_b + half_above)

        # Spread each band's members across its bounds by faith rank.
        for r in unique_r:
            sub_mask = member_rewards == r
            sub_n = int(sub_mask.sum())
            if sub_n <= 1:
                continue  # solo band — leave at base advantage
            sub_idx = _np.where(sub_mask)[0]
            sub_faiths = member_faiths[sub_idx]
            lo, hi = band_bounds[r]
            if hi - lo < 1e-9:
                continue  # degenerate; skip
            sort_order = _np.argsort(-sub_faiths, kind="stable")
            # Top of slot for highest-faith, bottom for lowest-faith.
            for new_pos, ord_i in enumerate(sort_order):
                frac = new_pos / (sub_n - 1)  # 0 at top, 1 at bottom
                local_adv[sub_idx[ord_i]] = hi - frac * (hi - lo)

        if norm_adv_by_std_in_grpo:
            std = local_adv.std()
            if std > 1e-8:
                local_adv = local_adv / std

        advantages[members] = local_adv

    out = _torch.from_numpy(advantages).to(token_level_rewards.device).to(token_level_rewards.dtype)
    advantages_full = out.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    return advantages_full, advantages_full


# ---------------------------------------------------------------------------
# Tiebreak faith-only-for-ties advantage estimator (Phase-2, pure tiebreaker).
# Non-tied rollouts: advantage is NUMERICALLY IDENTICAL to vanilla GRPO
# (same formula: (reward - mean) / (unbiased_std + epsilon)).
# Tied rollouts: faith breaks the tie via zero-sum jitter bounded by band gap.
# All-tied groups: rescued via faith z-score, magnitude calibrated to the
# batch's mean |adv| (no arbitrary epsilon).
# ---------------------------------------------------------------------------
@register_adv_est("tiebreak_faith_only_for_ties")
def compute_tiebreak_faith_only_for_ties_advantage(
    token_level_rewards,
    response_mask,
    index,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    non_tensor_batch=None,
    _substituted_signal=False,
    **kwargs,
):
    """Pure tiebreaker: faith touches only tied rollouts; non-tied
    advantages are numerically identical to vanilla GRPO
    ((reward - mean) / (unbiased_std + epsilon)).

    Partial-tie regime (group has reward variance, some rollouts tied):
        Base advantage = vanilla GRPO (mean-subtracted, std-normalized).
        Within each tied cluster of size k>=2: add a zero-sum faith-ordered
        jitter. Spread = ALPHA * min(g_lo, g_up) where g_lo, g_up are gaps
        to the cluster's nearest-distinct lower/upper advantage in
        normalized space (only the existing gap is used for top/bottom
        clusters). ALPHA=0.5 (half-band fill). Linear interpolation:
        lowest-faith gets -spread/2, highest gets +spread/2.

    Full-tie regime (group has zero reward variance — vanilla discards it):
        adv_i = beta * (faith_i - mean(faith)) / max(std(faith),
        STD_FAITH_FLOOR), where beta = mean(|adv|) across rollouts in
        non-tied groups in the same batch. Clamping std(faith) at 0.01
        prevents magnitude blowup when the policy has converged to similar
        faith across rollouts. If the WHOLE batch is all-tied (rare),
        beta=0 and all groups stay at zero — no signal fabricated when
        none is available.

    Properties:
        - Non-tied rollouts: EXACTLY vanilla GRPO.
        - Tied clusters: zero-sum (group mean preserved); bounded by
          existing band structure (no tied rollout crosses a non-tied
          neighbor).
        - All-tied groups: zero-sum z-scores; magnitude calibrated to
          batch, never to an arbitrary constant.
        - STD_FAITH_FLOOR is hard-coded with rationale. ALPHA defaults to
          0.5 (pre-registered main arm) and is overridable via
          FAITH_TIE_ALPHA for the α-sensitivity ablation only — the
          order-preservation proof holds for any ALPHA in (0,1], so no
          setting of it can let faith overtake correctness.
    """
    import numpy as _np
    import os as _os
    import torch as _torch

    # ALPHA = within-tie band-fill fraction (half-band default). Overridable via
    # FAITH_TIE_ALPHA for the α-sensitivity ablation ONLY; the order-preservation
    # proof holds for any ALPHA in (0,1], so this cannot let faith overtake
    # correctness. Default 0.5 reproduces the pre-registered main arm.
    ALPHA = float(_os.environ.get("FAITH_TIE_ALPHA", "0.5"))
    STD_FAITH_FLOOR = 0.01

    per_rollout = token_level_rewards.sum(dim=-1)
    bsz = per_rollout.shape[0]
    rewards = per_rollout.detach().cpu().to(_torch.float64).numpy()

    # faith in [0,1] with -1.0 unavailable sentinels; `genuine` masks the real
    # values. nan/inf are folded into the sentinel by the helper. The control
    # arms pass _substituted_signal=True -> every member genuine, no guard.
    faith, genuine = _tb_read_faith(non_tensor_batch, bsz,
                                    substituted=_substituted_signal)

    if isinstance(index, _torch.Tensor):
        idx_arr = index.detach().cpu().numpy()
    else:
        idx_arr = _np.asarray(index)

    # Pass 1: compute base advantages (vanilla GRPO for non-tied groups;
    # zeros for all-tied groups), tag groups by tie status.
    base_adv = _np.zeros(bsz, dtype=_np.float64)
    is_all_tied = {}
    for gid in _np.unique(idx_arr):
        members = _np.where(idx_arr == gid)[0]
        n = len(members)
        if n <= 1:
            is_all_tied[gid] = True
            continue
        member_rewards = rewards[members]
        std_r = member_rewards.std(ddof=1)  # unbiased std — matches verl GRPO's torch.std
        if std_r <= epsilon:
            is_all_tied[gid] = True
        else:
            is_all_tied[gid] = False
            adv = member_rewards - member_rewards.mean()
            if norm_adv_by_std_in_grpo:
                adv = adv / (std_r + epsilon)  # exactly verl GRPO: (x-mean)/(std+eps)
            base_adv[members] = adv

    # beta = batch-wide mean(|adv|) over non-tied rollouts. Calibrates the
    # all-tied rescue magnitude to the policy's current advantage scale.
    non_tied_mask = _np.zeros(bsz, dtype=bool)
    for gid, all_tied in is_all_tied.items():
        if not all_tied:
            non_tied_mask[_np.where(idx_arr == gid)[0]] = True
    if non_tied_mask.any():
        beta = float(_np.abs(base_adv[non_tied_mask]).mean())
    else:
        beta = 0.0

    advantages = base_adv.copy()

    # Pass 2: tiebreak per group.
    for gid in _np.unique(idx_arr):
        members = _np.where(idx_arr == gid)[0]
        n = len(members)
        if n <= 1:
            continue
        member_faiths = faith[members]
        member_genuine = genuine[members]

        if is_all_tied[gid]:
            if beta <= 0.0:
                continue  # whole-batch all-tied — leave at zero
            # Rescue only over genuine-faith members; unavailable (-1.0 sentinel)
            # members stay at 0. z is zero-mean over the genuine subset, so the
            # group advantage still sums to 0 (mean preserved).
            gf = member_faiths[member_genuine]
            if gf.shape[0] < 2:
                continue  # <2 genuine faith values -> no rescue signal
            f_std = max(float(gf.std(ddof=1)), STD_FAITH_FLOOR)  # unbiased, same operator as vanilla's z-score
            z = (gf - gf.mean()) / f_std
            rescue = _np.zeros(n, dtype=_np.float64)
            rescue[member_genuine] = beta * z
            advantages[members] = rescue
            continue

        # Partial-tie: jitter within tied clusters of base_adv.
        local_adv = base_adv[members].copy()
        unique_vals = sorted(set(local_adv.tolist()))
        if len(unique_vals) == n:
            continue  # no ties

        for tied_val in unique_vals:
            cluster_local = _np.where(local_adv == tied_val)[0]
            k = len(cluster_local)
            if k <= 1:
                continue
            lower = [v for v in unique_vals if v < tied_val]
            upper = [v for v in unique_vals if v > tied_val]
            g_lo = (tied_val - max(lower)) if lower else None
            g_up = (min(upper) - tied_val) if upper else None
            if g_lo is None and g_up is None:
                continue  # unreachable: partial-tied => >=2 distinct values
            if g_lo is None:
                gap = g_up
            elif g_up is None:
                gap = g_lo
            else:
                gap = min(g_lo, g_up)
            spread = ALPHA * gap

            # Jitter only genuine-faith members of this tied cluster; -1.0
            # sentinel members keep the exact vanilla (tied) advantage. Offsets
            # over the kg genuine members are a symmetric arithmetic sequence
            # (sum 0), so the cluster stays zero-sum and no member exceeds
            # +/- spread/2 = +/- 0.25*gap -> correctness ordering never flips.
            cluster_faiths = member_faiths[cluster_local]
            cluster_genuine = member_genuine[cluster_local]
            gpos = _np.where(cluster_genuine)[0]  # positions within cluster_local
            kg = gpos.shape[0]
            if kg <= 1:
                continue  # <2 genuine faith values -> ranking undefined -> stay vanilla
            gf = cluster_faiths[gpos]
            if _np.ptp(gf) == 0.0:
                continue  # uniform genuine faith -> no info -> no jitter (stays exactly vanilla)
            sort_order = _np.argsort(gf, kind="stable")  # indices into gpos
            for new_pos, oo in enumerate(sort_order):
                offset = -spread / 2.0 + (new_pos / (kg - 1)) * spread
                local_adv[cluster_local[gpos[oo]]] += offset

        advantages[members] = local_adv
        # NOTE: no per-group std re-normalization. base_adv is already in
        # normalized space, and the jitter is zero-sum so non-tied entries
        # remain bitwise identical to vanilla GRPO.

    # ── Tie-regime telemetry (diagnostic: how much of the gradient is the
    # faith perturbation vs vanilla GRPO). Printed once per advantage call (≈ per
    # step). faith_adv_mass_frac = L1 mass of (advantages - base_adv) / total
    # |advantages|: 0 = pure vanilla, 1 = faith drives everything. all_tied groups
    # are where vanilla would give exactly zero and faith is the SOLE driver;
    # partial = genuine within-tie jitter; no_tie = bitwise-vanilla.
    _n_no_tie = _n_partial = _n_all_tied = _n_singleton = 0
    for _gid in _np.unique(idx_arr):
        _members = _np.where(idx_arr == _gid)[0]
        _n = len(_members)
        if _n <= 1:
            _n_singleton += 1
        elif is_all_tied[_gid]:
            _n_all_tied += 1
        elif len(set(base_adv[_members].tolist())) == _n:
            _n_no_tie += 1
        else:
            _n_partial += 1
    _total_mass = float(_np.abs(advantages).sum())
    _faith_mass = float(_np.abs(advantages - base_adv).sum())
    _frac = (_faith_mass / _total_mass) if _total_mass > 0.0 else 0.0
    print(f"[tie-regime] groups={_n_no_tie + _n_partial + _n_all_tied + _n_singleton} "
          f"no_tie={_n_no_tie} partial={_n_partial} all_tied={_n_all_tied} "
          f"singleton={_n_singleton} faith_adv_mass_frac={_frac:.3f} beta={beta:.4f}", flush=True)

    out = _torch.from_numpy(advantages).to(token_level_rewards.device).to(token_level_rewards.dtype)
    advantages_full = out.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    return advantages_full, advantages_full


@register_adv_est("tiebreak_random_for_ties")
def compute_tiebreak_random_for_ties_advantage(
    token_level_rewards,
    response_mask,
    index,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    non_tensor_batch=None,
    **kwargs,
):
    """RANDOM-tiebreak CONTROL for tiebreak_faith_only_for_ties.

    Byte-identical estimator logic — it literally calls the faith estimator —
    but the per-rollout faith vector is replaced by seeded uniform[0,1) noise of
    the same scale. Isolates *faith content* from the structural effect of the
    estimator (within-tie zero-sum jitter + all-tied rescue gradient): if random
    matches faith, the gain is "any signal revives tied/dead groups"; if faith
    beats random, the faith content is what matters. Reproducible via
    FAITH_RANDOM_SEED (default 42); the RNG persists at module scope so
    successive steps draw independent (not repeated) noise.
    """
    import numpy as _np
    import os as _os
    _g = globals()
    if "_TB_RANDOM_RNG" not in _g:
        _g["_TB_RANDOM_RNG"] = _np.random.default_rng(
            int(_os.environ.get("FAITH_RANDOM_SEED", "42")))
    bsz = int(token_level_rewards.shape[0])
    noise = _g["_TB_RANDOM_RNG"].random(bsz)
    nt = dict(non_tensor_batch) if non_tensor_batch is not None else {}
    nt["faith"] = noise
    return compute_tiebreak_faith_only_for_ties_advantage(
        token_level_rewards,
        response_mask,
        index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
        non_tensor_batch=nt,
        _substituted_signal=True,  # control arm injected its own signal; no sentinel logic
        **kwargs,
    )


@register_adv_est("tiebreak_neglength_for_ties")
def compute_tiebreak_neglength_for_ties_advantage(
    token_level_rewards,
    response_mask,
    index,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    non_tensor_batch=None,
    **kwargs,
):
    """LENGTH-tiebreaker CONTROL for tiebreak_faith_only_for_ties.

    Identical estimator, but the per-rollout 'faith' is replaced by -(response
    length in tokens) — i.e. break correctness ties toward SHORTER responses. On
    SLAKE the faith score correlates r=-0.61 with response length (vs only +0.26
    with correctness), so a natural question is whether the faith gain is just an
    anti-rambling / brevity effect. If length-tiebreak ≈ faith-tiebreak, the
    grounding *content* adds nothing beyond "prefer the shorter, committed answer";
    if faith beats length, the probe content matters. Needs no probe.
    """
    import torch as _torch
    lengths = response_mask.sum(dim=-1).detach().cpu().to(_torch.float64).numpy()
    nt = dict(non_tensor_batch) if non_tensor_batch is not None else {}
    nt["faith"] = -lengths  # shorter -> higher -> plays the role of high faith
    return compute_tiebreak_faith_only_for_ties_advantage(
        token_level_rewards,
        response_mask,
        index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
        non_tensor_batch=nt,
        _substituted_signal=True,  # control arm injected its own signal; no sentinel logic
        **kwargs,
    )
'''


def patch_core_algos() -> bool:
    src = CORE.read_text()
    block = TIEBREAK_BLOCK
    sig = "compute_tiebreak_centered_rank_advantage"
    has_all = all(name in src for name in (
        "compute_tiebreak_centered_rank_advantage",
        "compute_gopo_advantage",
        "compute_unlikeliness_advantage",
        "compute_faithrl_advantage",
        "compute_tiebreak_rescue_ties_only_advantage",
        "compute_tiebreak_break_all_ties_advantage",
        "compute_tiebreak_fill_bands_lex_advantage",
        "compute_tiebreak_faith_only_for_ties_advantage",
        "compute_tiebreak_random_for_ties_advantage",
        "compute_tiebreak_neglength_for_ties_advantage",
    ))
    # Phase-2 marker — present only in the no-drop variant. If missing, the
    # source still has the old drop_unformatted version and needs re-patching.
    PHASE2_MARKER = "Phase-2 variant: NO drop_unformatted"
    SAFETY_CAP_MARKER = "correctness-preserving safety cap on per-band perturbation"
    # Margin marker: only present in the 0.99-margin (strict inequality) version
    SAFETY_MARGIN_MARKER = "(2.0 * gap / (sub_n - 1))) * 0.99"
    # Alpha marker: present only in the env-overridable version (FAITH_TIE_ALPHA).
    # An already-patched source with the old hard-coded `ALPHA = 0.5` lacks it and
    # must be re-injected so the α-sensitivity ablation can vary it.
    ALPHA_ENV_MARKER = "FAITH_TIE_ALPHA"
    # Faith-health marker: present only once the per-step genuine-fraction log +
    # partial-degrade guard are in. An already-patched source lacking it must be
    # re-injected so the silent partial-degrade hole is closed.
    FAITH_HEALTH_MARKER = "[faith-health]"
    needs_update = has_all and (PHASE2_MARKER not in src or SAFETY_CAP_MARKER not in src
                                or SAFETY_MARGIN_MARKER not in src or ALPHA_ENV_MARKER not in src
                                or FAITH_HEALTH_MARKER not in src)
    if has_all and not needs_update:
        return False
    if sig in src:
        # Old (or phase-1) version present — remove the old block before appending the new one.
        start = src.find("# ---------------------------------------------------------------------------\n# Tiebreak centered-rank advantage estimator")
        if start >= 0:
            src = src[:start]
    CORE.write_text(src.rstrip() + "\n" + block + "\n")
    return True


# ---------------------------------------------------------------------------
# Patch 3: ray_trainer.py compute_advantage — also pass non_tensor_batch to
# the tiebreak adv estimator so it can read correctness/faith/formatted.
# ---------------------------------------------------------------------------

RAY = VERL / "verl/trainer/ppo/ray_trainer.py"
OLD_RAY = '''        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch'''
NEW_RAY = '''        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # FAITH PATCH: tiebreak variants + faithrl read faith from
        # non_tensor_batch; unlikeliness reads old_log_probs from batch.
        if adv_estimator in ("tiebreak_centered_rank", "faithrl_grpo", "tiebreak_rescue_ties_only", "tiebreak_break_all_ties", "tiebreak_fill_bands_lex", "tiebreak_faith_only_for_ties", "tiebreak_random_for_ties", "tiebreak_neglength_for_ties"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
        if adv_estimator == "unlikeliness_grpo":
            adv_kwargs["batch"] = data.batch'''


def patch_ray() -> bool:
    src = RAY.read_text()
    if all(s in src for s in ("unlikeliness_grpo", "faithrl_grpo", "tiebreak_rescue_ties_only", "tiebreak_break_all_ties", "tiebreak_fill_bands_lex", "tiebreak_faith_only_for_ties", "tiebreak_random_for_ties", "tiebreak_neglength_for_ties")):
        return False
    # Strip any previous (partial) FAITH PATCH block so we always end up with
    # the latest version.
    OLD_VARIANT = '''        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # FAITH PATCH: tiebreak reads correctness/faith/formatted from non_tensor_batch
        if adv_estimator == "tiebreak_centered_rank":
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch'''
    if OLD_VARIANT in src:
        src = src.replace(OLD_VARIANT, OLD_RAY, 1)
    if OLD_RAY not in src:
        raise SystemExit("ray_trainer.py: GDPO block not found verbatim — verl version drift?")
    RAY.write_text(src.replace(OLD_RAY, NEW_RAY, 1))
    return True


def main() -> None:
    a = patch_naive()
    b = patch_core_algos()
    c = patch_ray()
    print(f"naive.py patched: {a}")
    print(f"core_algos.py patched: {b}")
    print(f"ray_trainer.py patched: {c}")


if __name__ == "__main__":
    main()
