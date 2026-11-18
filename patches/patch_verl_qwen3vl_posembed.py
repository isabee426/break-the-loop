"""Patch the Qwen3-VL vision `fast_pos_embed_interpolate` for FSDP2 multi-GPU.

Root cause (found by the batch-16 full-FT smoke, 2026-07-10): under FSDP2 sharded
across >1 GPU, `Qwen3VLVisionModel.fast_pos_embed_interpolate` builds BOTH its
index tensor and its bilinear-weight tensor with `device=self.pos_embed.weight.device`,
but at that point the sharded `pos_embed.weight` is NOT yet gathered — the device
reads CPU. FSDP then gathers the weight to GPU inside the very next line:

    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]

so the embedding runs with a CPU index and a cuda:0 weight (first crash:
"index is on cpu ... wrapper_CUDA__index_select"), and even once the index is
fixed the `* weight_tensor` multiply is cuda:0 * cpu (second crash: "found at
least two devices, cuda:0 and cpu"). The 1-GPU LoRA runs never hit it: with
world_size==1 FSDP keeps the param resident, so `weight.device` already reads
cuda:0. It only appears with the multi-GPU FSDP shard — i.e. exactly the full-FT
config — and would hit every FT arm (shared vision forward).

Fix: reproduce the method (values IDENTICAL — pinned transformers 4.57.6) but
place both tensors on the *gathered* device. The gathered device is only known
after `self.pos_embed(...)` runs (FSDP's pre-forward hook materializes the weight
on GPU there — the crash itself proves the weight is cuda:0 at that call). So:
  * do the embedding lookup through a one-shot forward wrapper that moves its
    input onto `pos_embed.weight.device` at call time (post-gather -> GPU), and
  * move `weight_tensor` onto the resulting `pos_embeds.device` before the multiply.
Purely tensor placement — no numeric change — and a no-op on the 1-GPU path.

Installs a module-level monkeypatch in verl/models/transformers/qwen3_vl.py that
replaces `Qwen3VLVisionModel.fast_pos_embed_interpolate` at import time. The patch
is UPDATE-SAFE: it strips any prior version of this block before injecting the
current one (so re-running upgrades v1 -> v2 in place). Idempotent in effect.
"""
from __future__ import annotations

from patch_verl_for_tiebreak import _resolve_verl  # reuse verl-root resolver

MARKER = "_patched_fast_pos_embed"
_BEGIN = "# ---- BEGIN faith-patch: Qwen3-VL fast_pos_embed_interpolate FSDP2 device fix"
_END = "# ---- END faith-patch: Qwen3-VL fast_pos_embed_interpolate FSDP2 device fix"

BLOCK = '''
# ---- BEGIN faith-patch: Qwen3-VL fast_pos_embed_interpolate FSDP2 device fix ---------
# Under FSDP2 sharded across >1 GPU the vision pos-embed weight is not gathered when
# fast_pos_embed_interpolate reads self.pos_embed.weight.device (reads CPU), so BOTH
# idx_tensor and weight_tensor are built on CPU; FSDP gathers the weight to GPU inside
# `self.pos_embed(idx_tensor) * weight_tensor[...]`, so the embedding (cpu index vs cuda
# weight) and the multiply (cuda * cpu) both fail. 1-GPU (LoRA) never hit it. This
# reproduces the method (values identical; pinned transformers 4.57.6) but places both
# tensors on the GATHERED device. See patch_verl_qwen3vl_posembed.py.
try:
    from transformers.models.qwen3_vl import modeling_qwen3_vl as _q3vl_pe_mod

    _Q3VL_VIS = _q3vl_pe_mod.Qwen3VLVisionModel
    if not getattr(_Q3VL_VIS, "_patched_fast_pos_embed", False):

        def _q3vl_patched_fast_pos_embed(self, grid_thw):
            grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

            idx_list = [[] for _ in range(4)]
            weight_list = [[] for _ in range(4)]

            for t, h, w in zip(grid_ts, grid_hs, grid_ws):
                h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
                w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

                h_idxs_floor = h_idxs.int()
                w_idxs_floor = w_idxs.int()
                h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
                w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

                dh = h_idxs - h_idxs_floor
                dw = w_idxs - w_idxs_floor

                base_h = h_idxs_floor * self.num_grid_per_side
                base_h_ceil = h_idxs_ceil * self.num_grid_per_side

                indices = [
                    (base_h[None].T + w_idxs_floor[None]).flatten(),
                    (base_h[None].T + w_idxs_ceil[None]).flatten(),
                    (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                    (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
                ]

                weights = [
                    ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                    ((1 - dh)[None].T * dw[None]).flatten(),
                    (dh[None].T * (1 - dw)[None]).flatten(),
                    (dh[None].T * dw[None]).flatten(),
                ]

                for i in range(4):
                    idx_list[i].extend(indices[i].tolist())
                    weight_list[i].extend(weights[i].tolist())

            # Build on CPU; the correct (gathered) device is only knowable after the
            # embedding forward runs, so defer placement to use-time.
            idx_tensor = torch.tensor(idx_list, dtype=torch.long)
            weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype)

            # Embedding lookup: move the index onto pos_embed.weight.device AT CALL TIME
            # (FSDP's pre-forward hook has gathered the weight to GPU by then).
            _emb = self.pos_embed
            _emb_fwd = _emb.forward

            def _dev_safe(_inp, *_a, **_k):
                return _emb_fwd(_inp.to(_emb.weight.device), *_a, **_k)

            _emb.forward = _dev_safe
            try:
                pos_embeds = _emb(idx_tensor)
            finally:
                _emb.forward = _emb_fwd

            # pos_embeds is now on the gathered (GPU) device — put weight_tensor there too.
            pos_embeds = pos_embeds * weight_tensor.to(pos_embeds.device)[:, :, None]
            patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

            patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

            patch_pos_embeds_permute = []
            merge_size = self.config.spatial_merge_size
            for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
                pos_embed = pos_embed.repeat(t, 1)
                pos_embed = (
                    pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                    .permute(0, 1, 3, 2, 4, 5)
                    .flatten(0, 4)
                )
                patch_pos_embeds_permute.append(pos_embed)
            patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
            return patch_pos_embeds

        _Q3VL_VIS.fast_pos_embed_interpolate = _q3vl_patched_fast_pos_embed
        _Q3VL_VIS._patched_fast_pos_embed = True
        print("[q3vl-patch] fast_pos_embed_interpolate FSDP2 device fix installed", flush=True)
except Exception as _q3vl_pe_e:  # noqa: BLE001
    print(f"[q3vl-patch] fast_pos_embed device fix FAILED to install: {_q3vl_pe_e}", flush=True)
# ---- END faith-patch: Qwen3-VL fast_pos_embed_interpolate FSDP2 device fix -----------
'''


def patch_qwen3vl_posembed() -> bool:
    verl = _resolve_verl()
    target = verl / "verl" / "models" / "transformers" / "qwen3_vl.py"
    if not target.is_file():
        raise SystemExit(f"[q3vl-patch] not found: {target}")
    src = target.read_text()
    # Update-safe: strip any prior version of this block, then (re)inject the current one.
    if _BEGIN in src and _END in src:
        i = src.index(_BEGIN)
        j = src.index("\n", src.index(_END)) + 1  # through end of the END line
        src = src[:i].rstrip("\n") + "\n" + src[j:]
    anchor = "\nimport torch\n"
    if anchor not in src:
        raise SystemExit("[q3vl-patch] no 'import torch' anchor in qwen3_vl.py")
    src = src.replace(anchor, anchor + BLOCK, 1)
    target.write_text(src)
    return True


def main() -> None:
    changed = patch_qwen3vl_posembed()
    print("patched qwen3_vl.py (fast_pos_embed device fix)" if changed else "already patched")


if __name__ == "__main__":
    main()
