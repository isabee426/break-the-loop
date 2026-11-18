"""Preserve the raw image for the lookback faith probe. verl's RLHFDataset pops
self.image_key ('images') from row_dict after building multimodal inputs, so the
raw image is gone by reward time. Stash the first image under 'faith_image'
(which is NOT popped) so the reward-loop manager can pass it to the faith probe.
Idempotent."""
import ast, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_verl_for_tiebreak import _resolve_verl  # $VERL_HOME -> installed verl -> legacy path

P = str(_resolve_verl() / "verl/utils/dataset/rl_dataset.py")
src = open(P).read()
if "faith_image" in src:
    print("already patched"); ast.parse(src); print("OK parse"); raise SystemExit

old = "        row_dict.pop(self.image_key, None)\n"
new = (
    "        # FAITH PATCH: stash the first raw image for the lookback faith probe\n"
    "        # before it is popped (it's gone from non_tensor_batch otherwise).\n"
    "        _fi = row_dict.get(self.image_key)\n"
    "        if _fi is not None and len(_fi) > 0:\n"
    "            row_dict[\"faith_image\"] = [_fi[0]]\n"
    "        row_dict.pop(self.image_key, None)\n"
)
assert src.count(old) == 1, f"pop line not unique (count={src.count(old)})"
shutil.copy(P, P + ".bak_pre_faithimage")
open(P, "w").write(src.replace(old, new, 1))
ast.parse(open(P).read())
print("patched rl_dataset.py (faith_image preserved); OK parse")
