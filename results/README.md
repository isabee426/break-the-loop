# Results

- `ood_summary.csv` has out-of-domain results at step 150 for all nine arms on all eight benchmarks. Its
  accuracy and truncation values match the paper's tables.
- `is_collapsed_pct` is a separate compression and 12-gram collapse check. It is **not** the paper's 4-gram
  repetition metric.
- `probe_transfer/` has frozen-probe AUROC on vanilla GRPO step-150 responses (`vanilla_s150.json`) and on
  Qwen3-VL-8B-Instruct responses (`instruct.json`).

Per-example model outputs are not included. `scripts/evaluate.sh` writes new runs to `results/eval/` and
`results/ood/`.
