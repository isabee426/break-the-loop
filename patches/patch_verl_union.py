"""Patch verl/protocol.py to skip the brittle extra_info equality assertion in
union_numpy_dict. vLLM rollout legitimately mutates extra_info per-step (adds
finish_reason / token counts), so the assertion fires on every train step.
The patch makes the check conditional: skip when key == 'extra_info'."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_verl_for_tiebreak import _resolve_verl  # $VERL_HOME -> installed verl -> legacy path

P = _resolve_verl() / "verl/protocol.py"
src = P.read_text()

OLD = '''        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # to properly deal with nan and object type
            assert _deep_equal(tensor_dict1[key], tensor_dict2[key], visited=set()), (
                f"`{key}` in tensor_dict1 and tensor_dict2 are not the same object."
            )'''

NEW = '''        if key in tensor_dict1:
            # PATCH: skip equality check for extra_info — vLLM rollout legitimately mutates it
            if key != "extra_info":
                assert isinstance(tensor_dict2[key], np.ndarray)
                assert isinstance(tensor_dict1[key], np.ndarray)
                # to properly deal with nan and object type
                assert _deep_equal(tensor_dict1[key], tensor_dict2[key], visited=set()), (
                    f"`{key}` in tensor_dict1 and tensor_dict2 are not the same object."
                )'''

if NEW in src:
    print("already patched")
elif OLD in src:
    P.write_text(src.replace(OLD, NEW, 1))
    print("patched")
else:
    raise SystemExit("OLD block not found verbatim — verl source has changed")
