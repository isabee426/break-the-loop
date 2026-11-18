"""Patch the Qwen3-VL vision patch-embedding Conv3d -> an equivalent matmul.

Root cause (found by faulthandler stack-dump of the frozen FSDP actor, 2026-07-09):
`compute_log_prob`'s forward froze in `torch/nn/modules/conv.py:_conv_forward`,
inside the Qwen3-VL vision patch-embed. That layer is
`nn.Conv3d(in_ch, embed_dim, kernel=stride=[t_patch, patch, patch])`. Because
kernel == stride == the full input spatial extent, the output is 1x1x1 — the
"convolution" is exactly a per-patch linear projection. On GPU at batch scale
(~30k patches, batch>=8) this degenerate Conv3d HANGS: the kernel spins at low
power and never returns, under BOTH cuDNN and the native backend
(`cudnn.flags(enabled=False)` did NOT help). The input is small (~45M elements),
so it is not a size/OOM issue — cuDNN/native just choke on the degenerate 3D-conv
shape.

`F.linear` is mathematically identical (same element order: the Conv3d weight
(out, in, kT, kH, kW) flattens to (out, in*kT*kH*kW) and the input flattens the
same way) and cannot hang. Verified end-to-end: with the Conv3d the actor freezes
at batch 8; with the matmul a full step completes (generation -> log_prob ->
update -> checkpoint).

This installs a module-level monkeypatch in verl/models/transformers/qwen3_vl.py
that replaces `Qwen3VLVisionPatchEmbed.forward` at import time. Idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

from patch_verl_for_tiebreak import _resolve_verl  # reuse verl-root resolver

MARKER = "_patched_patchembed_forward"

BLOCK = '''
# ---- BEGIN faith-patch: Qwen3-VL vision patch-embed Conv3d -> matmul ------------------
# self.proj is Conv3d(in, embed, kernel=stride=[t,p,p]); kernel==stride==full input
# spatial => output 1x1x1 => a per-patch linear projection. The degenerate Conv3d
# HANGS on GPU at batch scale (~30k patches, both cuDNN and native); F.linear is
# identical (same element order) and cannot hang. See patch_verl_qwen3vl_conv.py.
import torch.nn.functional as _F_q3vl


def _patched_patchembed_forward(self, hidden_states):
    target_dtype = self.proj.weight.dtype
    x = hidden_states.reshape(
        -1, self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    ).to(target_dtype)
    w = self.proj.weight.reshape(self.embed_dim, -1)
    return _F_q3vl.linear(x, w, self.proj.bias)


try:
    from transformers.models.qwen3_vl import modeling_qwen3_vl as _q3vl_mod

    _q3vl_mod.Qwen3VLVisionPatchEmbed.forward = _patched_patchembed_forward
    print("[q3vl-patch] vision patch-embed Conv3d -> matmul installed", flush=True)
except Exception as _q3vl_e:  # noqa: BLE001
    print(f"[q3vl-patch] FAILED to install: {_q3vl_e}", flush=True)
# ---- END faith-patch: Qwen3-VL vision patch-embed Conv3d -> matmul --------------------
'''


def patch_qwen3vl() -> bool:
    verl = _resolve_verl()
    target = verl / "verl" / "models" / "transformers" / "qwen3_vl.py"
    if not target.is_file():
        raise SystemExit(f"[q3vl-patch] not found: {target}")
    src = target.read_text()
    if MARKER in src:
        return False  # already patched
    anchor = "\nimport torch\n"
    if anchor not in src:
        raise SystemExit("[q3vl-patch] no 'import torch' anchor in qwen3_vl.py")
    src = src.replace(anchor, anchor + BLOCK, 1)
    target.write_text(src)
    return True


def main() -> None:
    changed = patch_qwen3vl()
    print("patched qwen3_vl.py" if changed else "already patched")


if __name__ == "__main__":
    main()
