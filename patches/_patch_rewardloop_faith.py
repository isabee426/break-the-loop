"""Inject image+question into extra_info in the REWARD-LOOP naive manager's
run_single (verl/experimental/reward_loop/reward_manager/naive.py), so the faith
probe can re-forward. The non-reward-loop naive.py was already patched; this is
the path the RewardLoopWorker actually uses. Idempotent."""
import ast, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_verl_for_tiebreak import _resolve_verl  # $VERL_HOME -> installed verl -> legacy path

P = str(_resolve_verl() / "verl/experimental/reward_loop/reward_manager/naive.py")
src = open(P).read()

if "FAITH PATCH" in src:
    print("already patched")
    ast.parse(src); print("OK parse")
    raise SystemExit

anchor = '        extra_info["rollout_reward_scores"] = rollout_reward_scores\n'
assert src.count(anchor) == 1, f"anchor not unique (count={src.count(anchor)})"

inject = anchor + '''        # FAITH PATCH: pass image + question into extra_info so the faith probe
        # (lookback re-forward in compute_score) can run. The reward-loop naive
        # manager omits this by default.
        try:
            _ntb = data_item.non_tensor_batch
            _imgs = _ntb.get("images")
            if _imgs is not None and len(_imgs) > 0:
                _img = _imgs[0]
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
            pass
'''
shutil.copy(P, P + ".bak_pre_faithpatch")
new = src.replace(anchor, inject, 1)
ast.parse(new)
open(P, "w").write(new)
print("FAITH PATCH applied to reward-loop naive.py")
print("OK parse")
