# Artifacts

`split_manifest.json` lists the seed-42 train/validation/test split (4,000/400/1,000 records) of the
VLAA-Thinking subset, with each record's ID, image path, question and ground truth. The 2,000 extra math
prompts used only for probe fitting are reproduced deterministically from this manifest by
`scripts/data/prep_math_extra_v3.py`.

The frozen 8B probe and per-arm result files will be added here and in `results/`.
Trained checkpoints are not included.
