"""Apply the COMPLETE verl patch chain for the faith-tiebreak pipeline.

One command on a fresh cluster:

    VERL_HOME=/path/to/verl python patches/apply_all_verl_patches.py

(or leave VERL_HOME unset to patch the pip-installed verl). All patches are
idempotent; re-running is safe. Pin: verl @ 3467d90aeb3a00f0de84040e984fe37cf1a25021.

Order matters only for the reward-loop pair (v1 injects the block, v2 upgrades
it to the multi-source image grab + one-shot [faith-dbg] diagnostic).

The chain (why each link exists):
  1. patch_verl_for_tiebreak   — registers all advantage estimators in
                                 core_algos.py, plumbs non_tensor_batch to them
                                 in ray_trainer.py, and injects image/question
                                 into extra_info in workers/reward_manager/naive.py.
  2. patch_verl_union          — protocol.py: skip the extra_info equality
                                 assertion that fires on every train step.
  3. patch_verl_reward_gpu     — reward_loop.py: honor reward.num_gpus_per_worker
                                 so reward workers can see a GPU (else the faith
                                 extractor gets "No CUDA GPUs are available").
  4. _patch_rl_dataset_faithimage — rl_dataset.py: stash the raw image as
                                 faith_image before RLHFDataset pops `images`.
  5. _patch_rewardloop_faith + v2 — experimental/reward_loop/reward_manager/naive.py:
                                 the TRAINING-path image/question injection
                                 (multi-source: faith_image -> multi_modal_data
                                 -> images; raw_prompt -> prompt).

NOT here (runtime, not source patches): verl_lora_save_patch.py — adapter-only
save + FSDP2 DTensor adapter resume — is applied at process start by
run_train_lora.py.
"""
from __future__ import annotations
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from patch_verl_for_tiebreak import _resolve_verl

STEPS = [
    "patch_verl_for_tiebreak.py",
    "patch_verl_union.py",
    "patch_verl_reward_gpu.py",
    "_patch_rl_dataset_faithimage.py",
    "_patch_rewardloop_faith.py",
    "_patch_rewardloop_faith_v2.py",  # must run AFTER _patch_rewardloop_faith
    "patch_verl_qwen3vl_conv.py",  # vision patch-embed Conv3d -> matmul (hang fix)
    "patch_verl_qwen3vl_posembed.py",  # vision fast_pos_embed_interpolate FSDP2 multi-GPU device fix
]

# (verl-relative path, marker substring, why) — every marker must be present for
# the chain to be considered installed. Markers are chosen to prove the CURRENT
# block is in place, not a stale earlier version:
#   * `_tb_read_faith` proves the sentinel-aware advantage block (-1.0 exclusion
#     + FAITH_STRICT guard) is installed, not the old zeros-default block.
#   * `tiebreak_neglength_for_ties` proves the newest ray_trainer plumbing (all
#     four A/B estimators wired to non_tensor_batch).
#   * `faith-dbg` proves reward-loop naive.py has the v2 multi-source image grab
#     (v2 requires v1, so it implies both).
VERIFY = [
    ("verl/trainer/ppo/core_algos.py", "_tb_read_faith",
     "sentinel-aware advantage block"),
    ("verl/trainer/ppo/core_algos.py", "tiebreak_faith_only_for_ties",
     "canonical faith estimator registered"),
    ("verl/trainer/ppo/ray_trainer.py", "tiebreak_neglength_for_ties",
     "advantage-kwargs plumbing (all 4 arms)"),
    ("verl/protocol.py", "skip equality check for extra_info",
     "extra_info assertion skip"),
    ("verl/experimental/reward_loop/reward_loop.py", "FAITH PATCH",
     "reward-loop GPU routing"),
    ("verl/utils/dataset/rl_dataset.py", "faith_image",
     "raw image stashed before images pop"),
    ("verl/workers/reward_manager/naive.py", "faith_image",
     "train-path image/question injection"),
    ("verl/experimental/reward_loop/reward_manager/naive.py", "faith-dbg",
     "reward-loop naive v2 multi-source grab (implies v1)"),
    ("verl/models/transformers/qwen3_vl.py", "_patched_patchembed_forward",
     "Qwen3-VL vision Conv3d -> matmul (batch>=8 hang fix)"),
    ("verl/models/transformers/qwen3_vl.py", "_patched_fast_pos_embed",
     "Qwen3-VL vision pos-embed FSDP2 multi-GPU device fix"),
]


def verify_only() -> None:
    """Check every patch marker is present in the verl tree; print the exact
    sentinel line ALL_PATCHES_VERIFIED on success. Launchers grep for that line
    and abort otherwise, so a missing/unrecognized result is the safe direction.
    """
    verl = _resolve_verl()
    if not (verl / "verl").is_dir():
        raise SystemExit(f"[verify] verl source not found at {verl} — set VERL_HOME.")
    missing = []
    for rel, marker, why in VERIFY:
        target = verl / rel
        try:
            text = target.read_text()
        except OSError as e:
            missing.append(f"{rel}: unreadable ({e})")
            continue
        if marker not in text:
            missing.append(f"{rel}: missing marker {marker!r} ({why})")
    if missing:
        print(f"[verify] verl at {verl}: {len(missing)} marker(s) MISSING:")
        for m in missing:
            print(f"  ❌ {m}")
        raise SystemExit("[verify] patch chain INCOMPLETE — run "
                         "`python patches/apply_all_verl_patches.py` to (re)apply.")
    print(f"[verify] verl at {verl}: all {len(VERIFY)} patch markers present ✓")
    print("ALL_PATCHES_VERIFIED")


def main() -> None:
    verl = _resolve_verl()
    if not (verl / "verl").is_dir():
        raise SystemExit(
            f"verl source not found at {verl} — set VERL_HOME to the verl repo root "
            f"(clone volcengine/verl @ 3467d90a, pip install -e it, then re-run).")
    print(f"[apply-all] patching verl at: {verl}\n")
    failed = []
    for step in STEPS:
        print(f"--- {step} ---")
        try:
            # Each patch script is written as a standalone __main__; several
            # exit via SystemExit(None/msg) on the already-patched path.
            runpy.run_path(str(HERE / step), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0):
                failed.append((step, str(e)))
                print(f"[apply-all] {step} FAILED: {e}")
        except Exception as e:  # noqa: BLE001 — report and continue to show full picture
            failed.append((step, f"{type(e).__name__}: {e}"))
            print(f"[apply-all] {step} CRASHED: {e}")
        print()
    if failed:
        raise SystemExit("[apply-all] FAILURES:\n" + "\n".join(f"  {s}: {m}" for s, m in failed))
    print("[apply-all] all verl patches applied ✓")
    # Confirm on disk (catches a patch that reported success but no-op'd).
    verify_only()


if __name__ == "__main__":
    if "--verify-only" in sys.argv[1:]:
        verify_only()
    else:
        main()
