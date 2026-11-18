"""Patch verl's RewardLoopManager so each reward worker can request a GPU slot.

Without this, the faith probe's HF re-forward inside compute_score fails with
'No CUDA GPUs are available' because Ray actors default to num_gpus=0.

Patch: reads `config.reward.num_gpus_per_worker` (defaults to 0). When set >0,
each reward worker is created via ray.remote(num_gpus=val)(RewardLoopWorker).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_verl_for_tiebreak import _resolve_verl  # $VERL_HOME -> installed verl -> legacy path

REWARD_LOOP = _resolve_verl() / "verl/experimental/reward_loop/reward_loop.py"

OLD = "        self.reward_loop_workers_class = ray.remote(RewardLoopWorker)"
NEW = (
    "        # FAITH PATCH: optionally reserve a GPU per reward worker so the lookback\n"
    "        # probe (HF re-forward in compute_score) can run inside the actor.\n"
    "        _ng = float(getattr(self.config.reward, 'num_gpus_per_worker', 0) or 0)\n"
    "        if _ng > 0:\n"
    "            self.reward_loop_workers_class = ray.remote(num_gpus=_ng)(RewardLoopWorker)\n"
    "        else:\n"
    "            self.reward_loop_workers_class = ray.remote(RewardLoopWorker)"
)

src = REWARD_LOOP.read_text()
if "FAITH PATCH" in src:
    print("already patched")
elif OLD in src:
    REWARD_LOOP.write_text(src.replace(OLD, NEW, 1))
    print("patched")
else:
    raise SystemExit("OLD block not found verbatim")
