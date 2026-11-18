"""Entry-point wrapper: install the LoRA-only-save patch, then hand control to
verl's main_ppo. Drop-in replacement for `python -m verl.trainer.main_ppo`.

The wrapper preserves sys.argv unchanged, so all hydra overrides flow through
to verl as if it were invoked directly.
"""
from __future__ import annotations
import os
import sys


def _ensure_scripts_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def _seed_everything() -> None:
    """Seed python/numpy/torch from $SEED (default 42) so the A/B arms share
    LoRA init and any host-side RNG. verl's dataloader sampler is seeded
    separately via the hydra `data.seed` override in the launchers; the old
    `+trainer.seed` was a no-op (not in verl's config schema) — this closes that
    gap on the process side. CUDA determinism is NOT forced (cudnn.deterministic
    would slow training and verl's kernels are not all deterministic anyway);
    the goal is matched initialization + data order across arms, verified
    empirically by the per-step BATCH_ID_LOG parity lines, not bitwise replay.
    """
    seed = int(os.environ.get("SEED", "42"))
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # numpy always present in this env; guard is belt-and-suspenders
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # PYTHONHASHSEED affects set/dict ordering in subprocesses; set for children.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    print(f"[run_train_lora] seeded python/numpy/torch with SEED={seed}", flush=True)


def main() -> None:
    _ensure_scripts_on_path()
    _seed_everything()
    import verl_lora_save_patch
    verl_lora_save_patch.apply()
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
