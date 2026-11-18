"""Replace the reward-loop FAITH PATCH block with a multi-source image grab
(faith_image -> multi_modal_data['image'] -> images) + a one-shot diagnostic
that prints the real non_tensor_batch keys. Idempotent (keyed on 'faith-dbg')."""
import ast, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_verl_for_tiebreak import _resolve_verl  # $VERL_HOME -> installed verl -> legacy path

F = str(_resolve_verl() / "verl/experimental/reward_loop/reward_manager/naive.py")
s = open(F).read()
if "faith-dbg" in s:
    print("already v2-patched"); ast.parse(s); print("OK parse"); raise SystemExit

start = s.index("        # FAITH PATCH: pass image")
end_marker = "        except Exception:\n            pass\n"
end = s.index(end_marker, start) + len(end_marker)

new = (
    '        # FAITH PATCH: pass image + question to the faith probe (multi-source).\n'
    '        try:\n'
    '            _ntb = data_item.non_tensor_batch\n'
    '            _img = None\n'
    '            _fi = _ntb.get("faith_image")\n'
    '            if _fi is not None and len(_fi) > 0:\n'
    '                _img = _fi[0]\n'
    '            if _img is None:\n'
    '                _mmd = _ntb.get("multi_modal_data")\n'
    '                if isinstance(_mmd, dict):\n'
    '                    _il = _mmd.get("image") or _mmd.get("images")\n'
    '                    if _il is not None and len(_il) > 0:\n'
    '                        _img = _il[0]\n'
    '            if _img is None:\n'
    '                _ims = _ntb.get("images")\n'
    '                if _ims is not None and len(_ims) > 0:\n'
    '                    _img = _ims[0]\n'
    '            if _img is not None:\n'
    '                if isinstance(_img, dict) and "bytes" in _img:\n'
    '                    from io import BytesIO as _BIO\n'
    '                    from PIL import Image as _PIL\n'
    '                    _img = _PIL.open(_BIO(_img["bytes"])).convert("RGB")\n'
    '                extra_info["image"] = _img\n'
    '            _msgs = _ntb.get("raw_prompt")\n'
    '            if _msgs is None:\n'
    '                _msgs = _ntb.get("prompt")\n'
    '            if _msgs is not None:\n'
    '                for _m in list(_msgs):\n'
    '                    if isinstance(_m, dict) and _m.get("role") == "user":\n'
    '                        _c = _m.get("content")\n'
    '                        if isinstance(_c, str):\n'
    '                            _q = _c\n'
    '                            if _q.startswith("<image>"):\n'
    '                                _q = _q.split("\\n", 1)[1] if "\\n" in _q else _q\n'
    '                            extra_info["question"] = _q\n'
    '                        elif isinstance(_c, (list, tuple)):\n'
    '                            _parts = [p.get("text", "") for p in _c if isinstance(p, dict) and p.get("text")]\n'
    '                            extra_info["question"] = " ".join(t for t in _parts if t)\n'
    '                        break\n'
    '            if not getattr(NaiveRewardManager, "_faith_dbg", False):\n'
    '                NaiveRewardManager._faith_dbg = True\n'
    '                _mmd = _ntb.get("multi_modal_data")\n'
    '                _mmk = list(_mmd.keys()) if isinstance(_mmd, dict) else None\n'
    '                _hasimg = "image" in extra_info\n'
    '                _hasq = "question" in extra_info\n'
    '                print("[faith-dbg] ntb_keys=" + str(list(_ntb.keys())) + " mmd_keys=" + str(_mmk)\n'
    '                      + " image_set=" + str(_hasimg) + " question_set=" + str(_hasq), flush=True)\n'
    '        except Exception as _e:\n'
    '            print(f"[faith-dbg] err {_e}", flush=True)\n'
)

out = s[:start] + new + s[end:]
ast.parse(out)
open(F, "w").write(out)
print("v2 patch applied; OK parse")
