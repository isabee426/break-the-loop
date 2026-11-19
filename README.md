# Break the Loop with Good Attention

Code for **FT-GRPO**, a tie-constrained variant of GRPO for vision-language models.

Vanilla GRPO gives identical advantages to rollouts with the same reward, so it has no signal inside a
reward tie, and none at all when every rollout in a group is tied. FT-GRPO uses a frozen logistic-regression
probe on visual-attention features to order rollouts **only within those ties**. The perturbation is bounded
and zero-sum, so a probe score can never move a rollout past one with a higher reward.

## Status

| Component | State |
|---|---|
| Training, probe, evaluation and analysis code | Included |
| verl patch chain | Applies cleanly to the pinned commit |
| Out-of-domain summary for all nine arms | Included in `results/` |
| Train/val/test split manifest (seed 42) | Included in `artifacts/split_manifest.json` |
| Frozen probe | Not yet included |
| Repetition control estimator | Not yet included, so `train.sh repetition` exits with an error |
| Exact environment lockfile | Not yet included; `requirements.txt` lists core versions |
| Trained checkpoints | Not included |

## Setup

```bash
git clone https://github.com/volcengine/verl.git
git -C verl checkout 3467d90aeb3a00f0de84040e984fe37cf1a25021
pip install -e ./verl
pip install -r requirements.txt

export VERL_HOME=$PWD/verl VERL_DIR=$PWD/verl
python patches/apply_all_verl_patches.py
python tests/test_scorers.py
```

The paper's runs used Qwen3-VL-8B-Thinking on two NVIDIA H200 GPUs.

## Data

Training uses a stratified 5,400-example subset of VL-Thinking across four verifier types (digit, MCQ, math,
IoU), seed 42. With the annotations and images available:

```bash
export VLAA_ANNOTATIONS=data/annotations/vlaa_thinking_annotations.json
export VLAA_IMAGE_ROOT=data/vlaa_thinking/images_extracted
python scripts/data/prep_vlaa_subset.py --stage split
python scripts/data/prep_vlaa_subset.py --stage parquet
```

Parquets from the older prompt format can be upgraded with `scripts/data/rewrite_parquet_hints_v3.py`.
The out-of-domain benchmarks are loaded by `scripts/eval/eval_ood_vllm.py`. SLAKE is read from `SLAKE_DIR`.

## Training

```bash
export FAITH_PROBE_PATH=/path/to/probe_8b_strict_contentlabel.pkl   # probe-based arms only
bash scripts/train.sh vanilla
bash scripts/train.sh faith
```

| Arm | Advantage within ties |
|---|---|
| `vanilla` | None (GRPO) |
| `faith` | **FT-GRPO**, probe rank with α = 0.5 |
| `faith_a025`, `faith_a075`, `faith_a100` | FT-GRPO with α = 0.25, 0.75, 1.0 |
| `random` | Seeded random rank |
| `length` | Shorter responses first |
| `additive` | Vanilla GRPO on a reward blended with the probe score (λ = 0.7) |

The paper's repetition control is not runnable from this release yet.

All settings are in `configs/full_ft.env`. Use `SEED=123` for the replication seed.

## Evaluation

```bash
GPU=0 bash scripts/evaluate.sh vanilla 150 all
GPU=1 bash scripts/evaluate.sh faith 150 all
```

The evaluation step is chosen from the vanilla arm only, as the first checkpoint after the last relative
improvement above 1% in the three-point moving average of validation accuracy. In the paper that step is 150.
Out-of-domain evaluation uses `eval_ood_vllm.py`'s default prompt, the one used for the saved results.

```bash
python scripts/eval/compute_sstar.py results/eval vanilla
```

| Analysis | Script |
|---|---|
| In-domain accuracy, truncation, collapse, repetition | `scripts/analysis/agg_indomain_collapse.py` |
| In-domain paired bootstrap CIs and McNemar tests | `scripts/analysis/bootstrap_indomain_ci.py` |
| Out-of-domain paired bootstrap CIs | `scripts/analysis/agg_ood_ci.py` |
| Validation curves | `scripts/analysis/plot_val_trajectory_all.py` |
| Train/test overlap audit | `scripts/analysis/audit_split_overlap.py` |
| Probe transfer to other policies | `scripts/analysis/probe_transfer_auc.py` |

## Probe

The probe uses 1,152 attention features (36 layers × 32 heads) from eight rollouts per question on 6,000
questions: the 4,000 training prompts plus 2,000 extra math prompts.

```bash
python scripts/data/prep_math_extra_v3.py \
  --annotations data/annotations/vlaa_thinking_annotations.json \
  --manifest artifacts/split_manifest.json \
  --img-root data/vlaa_thinking/images_extracted --out-dir data/vlaa_math_extra --n 2000

# Repeat for digit, mcq, math and iou, and for data/vlaa_math_extra/train.parquet.
python scripts/probe/collect_probe_data_vllm.py --model Qwen/Qwen3-VL-8B-Thinking \
  --parquet data/vlaa_subset_5400/train.parquet --filter-verifier mcq \
  --temperatures 0.7,1.0,1.0,1.0,1.0,1.0,1.3,1.3 --no-batch --out outputs/probe/mcq.npz

python verl_faith/merge_probe_shards.py --pattern 'outputs/probe/*.npz' --out outputs/probe_merged.npz
python verl_faith/train_probe.py --features outputs/probe_merged.npz --label-key y_lenient \
  --cap-per-class 474 --out outputs/probe.pkl
```

A pickle can run code when loaded, so only load probe files you trust.

## License

Apache-2.0. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
