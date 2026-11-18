"""Runtime patch: when training a LoRA model with verl + FSDP, skip writing
the heavy full-model FSDP shards and save only the LoRA adapter instead —
and on load, restore that adapter into the FSDP2 (DTensor) model so
auto-resume works (see _load_adapter; this replaces the manual
fsdp_checkpoint_manager.py distribute_tensor edit).

For our LoRA(rank=16) sweep this drops each checkpoint from ~4 GB × world_size
of model shards to ~10 MB of adapter weights. Optimizer + extra_state shards
are still written (they're already small for LoRA) so resume is possible if
needed.

Apply once at process start (before any verl trainer code imports):

    import verl_lora_save_patch
    verl_lora_save_patch.apply()

Mechanism: monkey-patches FSDPCheckpointManager.save_checkpoint. When the
underlying model has a peft_config:
  1. Temporarily flip should_save_model=False so the original save skips the
     heavy model shard write but still writes optimizer + extra + HF config.
  2. Gather the full model state on rank 0 (FSDP all-gather), extract the
     PEFT adapter params, and write adapter_model.safetensors + adapter_config.json
     to <checkpoint>/lora_adapter/.

If anything in the LoRA path fails we fall through to the original behaviour
so a misconfiguration can't silently lose a checkpoint.
"""
from __future__ import annotations
import logging
import os
from typing import Any

logger = logging.getLogger("verl_lora_save_patch")
logging.basicConfig(level=logging.INFO)

_PATCHED = False


def _unwrap(model: Any) -> Any:
    """Return the PEFT module (has peft_config) inside whatever wrapping is in
    play. FSDP1 wraps in `_fsdp_wrapped_module`; FSDP2 (fully_shard) modifies the
    module IN PLACE (no wrapper attr), so the old FSDP1-only unwrap missed it and
    the patch silently no-oped -> full 23GB model saves. This walks the common
    wrapper attrs AND, as a backstop, deep-searches submodules for peft_config."""
    seen = set()
    def _walk(m, depth=0):
        if m is None or id(m) in seen or depth > 6:
            return None
        seen.add(id(m))
        if hasattr(m, "peft_config"):
            return m
        for attr in ("_fsdp_wrapped_module", "module", "_orig_mod",
                     "_checkpoint_wrapped_module", "_fsdp_module"):
            found = _walk(getattr(m, attr, None), depth + 1)
            if found is not None:
                return found
        return None
    direct = _walk(model)
    if direct is not None:
        return direct
    # Backstop: deep named_modules search (FSDP2 in-place wrap can bury it).
    try:
        for _, sub in model.named_modules():
            if hasattr(sub, "peft_config"):
                return sub
    except Exception:
        pass
    return model  # not found — caller's _has_peft will report False


def _materialize_tensor(v):
    """Return a plain contiguous CPU tensor with VALID storage, safe for
    safetensors.save_file. The FSDP-gathered / torch-loaded shards are DTensor
    or have special/shared storage — save_file raised 'invalid python storage'
    on them (the silent crash behind the full-model saves). full_tensor()
    de-shards a DTensor; .clone() forces fresh owned storage; bf16→fp32 for a
    portable adapter file."""
    import torch
    if type(v).__name__ == "DTensor":
        try:
            v = v.full_tensor()
        except Exception:
            v = v.to_local()
    v = v.detach().cpu()
    if v.dtype == torch.bfloat16:
        v = v.to(torch.float32)
    return v.contiguous().clone()


def _lora_adapter_from_state(full_state: dict) -> dict:
    """Pull the LoRA adapter tensors out of a gathered full state dict by NAME,
    stripping the '.default' adapter infix so keys are standard PEFT adapter
    format (base_model.model...lora_A.weight) that vLLM/PEFT loads, and
    MATERIALIZING each to valid CPU storage (DTensor-safe). Returns {} for a
    non-LoRA (full-FT) state dict. Pure + unit-tested."""
    return {k.replace(".default", ""): _materialize_tensor(v)
            for k, v in full_state.items() if "lora_" in k.lower()}


def _has_peft(model: Any) -> bool:
    """Detect a LoRA model ROBUSTLY. `apply_fsdp2` (fully_shard) hides
    `peft_config` from both attribute access and named_modules on this verl,
    so the attribute check silently returned False -> full 23GB saves. LoRA
    PARAMETERS, however, are always named with 'lora_' regardless of wrapping —
    that's the unfoolable signal. Attribute check kept as a secondary path."""
    try:
        for n, _ in model.named_parameters():
            if "lora_" in n:
                return True
    except Exception:
        pass
    return hasattr(_unwrap(model), "peft_config")


def _save_adapter(self, local_path: str) -> bool:
    """Gather full state on rank 0, extract+save the LoRA adapter, and return a
    result that is CONSISTENT across all ranks (broadcast from rank 0). All-rank
    consistency is essential: the caller's recovery path runs a *collective* FSDP
    save, so it must execute on all ranks or none — otherwise a rank-0-only entry
    deadlocks every other rank. Returns True on success."""
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from verl.utils.fsdp_utils import get_fsdp_full_state_dict

    inner = _unwrap(self.model)

    # Collective: ALL ranks must participate in the all-gather; only rank 0 gets data.
    full_state = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

    ok = True
    if self.rank == 0:
        try:
            # Extract the adapter DIRECTLY from the gathered state dict by name
            # (module-level, unit-tested fn). The one path proven reliable under
            # FSDP2: the state dict has the lora_ keys (confirmed: 288) even when
            # the live model's peft_config/named_parameters do not expose them.
            adapter_state = _lora_adapter_from_state(full_state)

            if not adapter_state:
                logger.error("[lora-patch] adapter extraction yielded 0 lora_ tensors "
                             "in the gathered state dict — genuine full-FT? aborting prune")
                ok = False
            else:
                adapter_dir = os.path.join(local_path, "lora_adapter")
                os.makedirs(adapter_dir, exist_ok=True)
                # adapter_state tensors are already materialized (valid CPU
                # storage, fp32) by _lora_adapter_from_state -> save_file is safe.
                save_file(
                    adapter_state,
                    os.path.join(adapter_dir, "adapter_model.safetensors"),
                )
                logger.info(f"[lora-patch] wrote {len(adapter_state)} adapter tensors "
                            f"({sum(v.numel() for v in adapter_state.values()) * 4 // (1024*1024)}MB) "
                            f"-> {adapter_dir}")
                # adapter_config.json — needed by vLLM/PEFT to load the adapter at
                # eval. Prefer the real peft_config; if _unwrap can't reach it under
                # FSDP2, reconstruct a minimal-but-correct LoraConfig from the actual
                # adapter tensors (rank/target-modules inferred from the saved keys).
                cfg_written = False
                try:
                    inner2 = _unwrap(self.model)
                    peft_config = inner2.peft_config.get("default") or next(iter(inner2.peft_config.values()))
                    peft_config.save_pretrained(adapter_dir)
                    cfg_written = True
                except Exception as e:
                    logger.warning(f"[lora-patch] peft_config unreachable ({e}); "
                                   f"reconstructing adapter_config.json from tensors")
                if not cfg_written:
                    try:
                        import re as _re, json as _json
                        # infer rank from a lora_A tensor shape (r = out_features of A)
                        rank = None
                        targets = set()
                        for k, v in adapter_state.items():
                            m = _re.search(r"\.([a-z_]+)\.lora_A", k)
                            if m:
                                targets.add(m.group(1))
                            if rank is None and "lora_A" in k and hasattr(v, "shape"):
                                rank = int(v.shape[0])
                        cfg = {
                            "peft_type": "LORA", "task_type": "CAUSAL_LM",
                            "r": rank or int(os.environ.get("LORA_RANK", "16")),
                            "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
                            "lora_dropout": 0.0, "bias": "none",
                            "target_modules": sorted(targets) or ["q_proj", "k_proj", "v_proj", "o_proj"],
                            "inference_mode": True,
                        }
                        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as _f:
                            _json.dump(cfg, _f, indent=2)
                        logger.info(f"[lora-patch] wrote reconstructed adapter_config.json "
                                    f"(r={cfg['r']}, targets={cfg['target_modules']})")
                    except Exception as e2:
                        logger.warning(f"[lora-patch] failed to reconstruct adapter_config.json: {e2}")
                logger.info(f"[lora-patch] wrote {len(adapter_state)} adapter tensors → {adapter_dir}")
        except Exception as e:
            logger.error(f"[lora-patch] adapter save crashed on rank 0: {e}")
            ok = False

    # Broadcast rank-0's result so EVERY rank agrees whether to run the (collective)
    # recovery save. This is the sync point (replaces the old per-rank barriers).
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        flag = torch.tensor([1 if ok else 0], device=dev)
        dist.broadcast(flag, src=0)
        ok = bool(flag.item())
    return ok


def _load_adapter(self, local_path: str) -> bool:
    """Load lora_adapter/adapter_model.safetensors back into the (FSDP2-sharded)
    PEFT model. Companion to _save_adapter for auto-resume.

    Why this exists: with the save patch active, a checkpoint contains ONLY the
    adapter safetensors — no model_world_size_*.pt shards — so verl's stock
    load_checkpoint can't resume it. Worse, under FSDP2 the live parameters are
    DTensors; a plain load_state_dict with the full (unsharded) torch.Tensors
    hits the mixed Tensor/DTensor copy path and silently leaves the adapter at
    its fresh init — the run "resumes" as an untrained adapter. Fix: convert
    each full tensor to a DTensor matching the target param's mesh/placements
    via distribute_tensor, then copy_. FAILS LOUD on any unmapped key rather
    than resuming a partially-restored adapter.
    """
    import torch
    from safetensors.torch import load_file

    adapter_file = os.path.join(local_path, "lora_adapter", "adapter_model.safetensors")
    if not os.path.isfile(adapter_file):
        return False
    adapter_state = load_file(adapter_file)  # full (unsharded) tensors, identical on every rank

    named = dict(self.model.named_parameters())

    def _candidates(k: str):
        """PEFT saves keys without the '.default' adapter infix and (version-
        dependent) with/without the 'base_model.model.' prefix — try all forms."""
        forms = {k}
        for a in ("lora_A", "lora_B"):
            if f".{a}.weight" in k:
                forms.add(k.replace(f".{a}.weight", f".{a}.default.weight"))
        out = []
        for f in forms:
            out.append(f)
            out.append("base_model.model." + f)
            if f.startswith("base_model.model."):
                out.append(f[len("base_model.model."):])
        return out

    n_loaded, misses = 0, []
    for k, v in adapter_state.items():
        param = next((named[c] for c in _candidates(k) if c in named), None)
        if param is None:
            misses.append(k)
            continue
        with torch.no_grad():
            src = v.to(dtype=param.dtype)
            if type(param.data).__name__ == "DTensor":
                from torch.distributed.tensor import distribute_tensor
                src = distribute_tensor(
                    src, device_mesh=param.data.device_mesh, placements=param.data.placements)
            param.data.copy_(src)
        n_loaded += 1

    if misses or n_loaded == 0:
        raise RuntimeError(
            f"[lora-patch] adapter resume FAILED: loaded {n_loaded}, "
            f"unmapped keys (first 5): {misses[:5]} — refusing to resume with a "
            f"partially-restored adapter (would silently train from re-init).")
    print(f"[lora-patch] RESUMED {n_loaded} adapter tensors from {adapter_file} (DTensor reload OK)", flush=True)
    return True


def apply() -> None:
    """Install the monkey-patch. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    # Full fine-tune ablation (FAITH_FULL_FT=1): there is NO LoRA adapter, so the
    # adapter save/load rebind is pointless — it would gather a 0-tensor adapter,
    # log an error, and fall through to a full save anyway. Skip the rebind so
    # stock verl full-model FSDP save/load runs directly. The launcher grep-asserts
    # only check that this file/run_train_lora.py REFERENCE the patch, not that the
    # rebind ran, so they stay satisfied.
    if os.environ.get("FAITH_FULL_FT") == "1":
        logger.warning("[lora-patch] FAITH_FULL_FT=1 -> NOT installing the LoRA save/load "
                       "patch; full fine-tune uses stock verl full-model checkpoints")
        _PATCHED = True
        return

    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    orig_save = FSDPCheckpointManager.save_checkpoint
    orig_load = FSDPCheckpointManager.load_checkpoint

    def patched_load(self, local_path, hdfs_path=None, del_local_after_load=False):
        # Do NOT gate on _has_peft(self.model) — same FSDP2 false-negative as the
        # save side (it hid the adapter and made resume silently re-init from a
        # fresh adapter, discarding trained steps). Gate on what's ON DISK: a full
        # model shard -> stock loader; else an adapter file -> adapter path.
        if local_path is None:
            return orig_load(self, local_path, hdfs_path=hdfs_path,
                             del_local_after_load=del_local_after_load)
        shard = os.path.join(
            local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        if os.path.isfile(shard):
            # Full FSDP shards exist (save patch was off / recovery save) —
            # stock path handles them.
            return orig_load(self, local_path, hdfs_path=hdfs_path,
                             del_local_after_load=del_local_after_load)
        if not _load_adapter(self, local_path):
            return orig_load(self, local_path, hdfs_path=hdfs_path,
                             del_local_after_load=del_local_after_load)
        # Adapter restored. Let the stock loader pick up optimizer/extra IF those
        # files exist — via checkpoint_load_contents (should_load_* are read-only
        # properties over it, no setters). Exclude "model" (adapter loaded above).
        prev_lc = list(self.checkpoint_load_contents)
        opt_f = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        ext_f = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        want = []
        if "optimizer" in prev_lc and os.path.isfile(opt_f):
            want.append("optimizer")
        if "extra" in prev_lc and os.path.isfile(ext_f):
            want.append("extra")
        try:
            self.checkpoint_load_contents = want   # NOT "model"
            if want:
                orig_load(self, local_path, hdfs_path=hdfs_path,
                          del_local_after_load=del_local_after_load)
            else:
                print("[lora-patch] resume: adapter weights only "
                      "(no optimizer/extra state in checkpoint — fresh optimizer)", flush=True)
        finally:
            self.checkpoint_load_contents = prev_lc

    def patched_save(self, local_path, *args, **kwargs):
        # Do NOT gate on _has_peft(self.model): under FSDP2 (fully_shard),
        # named_parameters/peft_config on self.model do NOT expose the LoRA
        # (confirmed — the gathered STATE DICT has 288 lora_ keys but the live
        # model's introspection shows none), so any live-model check
        # false-negatives and silently full-saves. Instead always run the
        # adapter path; _save_adapter gathers the full state (which DOES contain
        # lora_ keys) and returns False when there are none (genuine full-FT),
        # which triggers the recovery full save below.
        if local_path is None:
            return orig_save(self, local_path, *args, **kwargs)

        # Skip the heavy model shard write inside orig_save by temporarily
        # dropping "model" from checkpoint_save_contents. should_save_model is a
        # READ-ONLY property (returns "model" in self.checkpoint_save_contents) —
        # it has no setter on this verl, so we edit the underlying list.
        prev_contents = list(self.checkpoint_save_contents)
        self.checkpoint_save_contents = [c for c in prev_contents if c != "model"]
        try:
            orig_save(self, local_path, *args, **kwargs)   # optim/extra/configs, NO model shard
        finally:
            self.checkpoint_save_contents = prev_contents

        try:
            ok = _save_adapter(self, local_path)
        except Exception as e:
            import traceback as _tb
            # print(flush=True): the module logger goes to Ray WORKER logs (invisible
            # in the driver stdout) — the earlier "silent" full saves were this crash
            # never reaching the console. print forwards to the driver.
            print(f"[lora-patch] adapter save CRASHED: {e}\n{_tb.format_exc()}", flush=True)
            ok = False

        if not ok:
            # Recovery: checkpoint_save_contents is restored (includes "model"),
            # so orig_save writes the full shard -> checkpoint still resumable.
            print("[lora-patch] recovery: writing FULL model shard "
                  "(adapter path failed — this is the 23GB save)", flush=True)
            orig_save(self, local_path, *args, **kwargs)
        else:
            print("[lora-patch] adapter-only save OK (no full model shard written)", flush=True)

    FSDPCheckpointManager.save_checkpoint = patched_save
    FSDPCheckpointManager.load_checkpoint = patched_load
    _PATCHED = True
    logger.warning("[lora-patch] installed — LoRA checkpoints save adapter only; "
                   "resume loads the adapter back via DTensor distribute_tensor")
