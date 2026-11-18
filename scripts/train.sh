#!/usr/bin/env bash
# Train one arm of the paper:  bash scripts/train.sh <arm>
# Arms: vanilla faith faith_a025 faith_a075 faith_a100 random length additive
set -euo pipefail

ARM=${1:?usage: train.sh <vanilla|faith|faith_a025|faith_a075|faith_a100|random|length|additive>}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1090
source "${FTGRPO_CONFIG:-$REPO/configs/full_ft.env}"

USES_PROBE=0
case "$ARM" in
    vanilla)    ADV_ESTIMATOR=grpo ;;
    faith)      ADV_ESTIMATOR=tiebreak_faith_only_for_ties; USES_PROBE=1; export FAITH_TIE_ALPHA=${FAITH_TIE_ALPHA:-0.5} ;;
    faith_a025) ADV_ESTIMATOR=tiebreak_faith_only_for_ties; USES_PROBE=1; export FAITH_TIE_ALPHA=0.25 ;;
    faith_a075) ADV_ESTIMATOR=tiebreak_faith_only_for_ties; USES_PROBE=1; export FAITH_TIE_ALPHA=0.75 ;;
    faith_a100) ADV_ESTIMATOR=tiebreak_faith_only_for_ties; USES_PROBE=1; export FAITH_TIE_ALPHA=1.0 ;;
    random)     ADV_ESTIMATOR=tiebreak_random_for_ties; export FAITH_RANDOM_SEED=${FAITH_RANDOM_SEED:-42} ;;
    length)     ADV_ESTIMATOR=tiebreak_neglength_for_ties ;;
    repetition) echo "The repetition control estimator is not included in this release yet." >&2; exit 2 ;;
    additive)   ADV_ESTIMATOR=grpo; USES_PROBE=1; export FAITH_REWARD_ALPHA=${FAITH_REWARD_ALPHA:-0.7} ;;
    *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

PYTHON=${PYTHON:-python}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs}
DATA_DIR=${DATA_DIR:-$REPO/data/vlaa_subset_5400}
EXPERIMENT=${EXPERIMENT:-ft_${ARM}$([ "$SEED" = 42 ] || echo "_s$SEED")}
CKPT_DIR=${CKPT_DIR:-$OUTPUT_DIR/checkpoints/$EXPERIMENT}
LOG_DIR=${LOG_DIR:-$OUTPUT_DIR/logs}
TMPBASE=${TMPBASE:-$OUTPUT_DIR/tmp/$EXPERIMENT}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
RESUME=${RESUME:-1}
N_GPUS=$(echo "$GPUS" | awk -F, '{print NF}')

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$TMPBASE"/{ray,triton,vllm,general}
export CUDA_VISIBLE_DEVICES="$GPUS"
export FAITH_FULL_FT=1
export PYTHONPATH="$REPO:$REPO/scripts/train:${PYTHONPATH:-}"
export TMPDIR="$TMPBASE/general" RAY_TMPDIR="$TMPBASE/ray"
export TRITON_CACHE_DIR="$TMPBASE/triton" VLLM_CACHE_ROOT="$TMPBASE/vllm"

die() { echo "FATAL: $*" >&2; exit 1; }

"$PYTHON" "$REPO/patches/apply_all_verl_patches.py" --verify-only 2>&1 | grep -q ALL_PATCHES_VERIFIED \
    || die "verl is not patched; run: python patches/apply_all_verl_patches.py"
grep -q v3-exact-payload "$DATA_DIR/parquet_format_version.json" 2>/dev/null \
    || die "$DATA_DIR is missing the v3 parquet stamp; see README (Data)."

REWARD_ARGS=()
if [ "$USES_PROBE" = 1 ]; then
    export FAITH_PROBE_PATH=${FAITH_PROBE_PATH:-$REPO/artifacts/probe_8b_strict_contentlabel.pkl}
    [ -f "$FAITH_PROBE_PATH" ] || die "probe not found: $FAITH_PROBE_PATH (set FAITH_PROBE_PATH)"
    export FAITH_PROBE_MODEL=${FAITH_PROBE_MODEL:-$MODEL_PATH}
    export FAITH_SERVE_WINDOW=$MAX_RESPONSE
    # Features are extracted one request at a time; the batched path is not used for paper runs.
    export FAITH_ALLOW_BATCH_EXTRACT=${FAITH_ALLOW_BATCH_EXTRACT:-0}
    export FAITH_BATCH_WINDOW_S=${FAITH_BATCH_WINDOW_S:-0}
    # The extractor shares the GPUs, so probe arms offload parameters and optimizer state.
    PARAM_OFFLOAD=True
    OPTIMIZER_OFFLOAD=True
    REWARD_ARGS=(reward_model.num_workers=1 +reward.num_gpus_per_worker=0)
fi

[ "$RESUME" = 0 ] && RESUME_MODE=disable || RESUME_MODE=auto
LOG=$LOG_DIR/${EXPERIMENT}_$(date +%m%d_%H%M).log
echo "arm=$ARM estimator=$ADV_ESTIMATOR steps=$TOTAL_STEPS seed=$SEED gpus=$GPUS ckpt=$CKPT_DIR" | tee -a "$LOG"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    set +e
    "$PYTHON" "$REPO/scripts/train/run_train_lora.py" \
        algorithm.adv_estimator=$ADV_ESTIMATOR \
        algorithm.use_kl_in_reward=False \
        data.train_files="$DATA_DIR/train.parquet" \
        data.val_files="$DATA_DIR/val.parquet" \
        data.image_key=images \
        data.reward_fn_key=data_source \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.max_prompt_length=4096 \
        data.max_response_length=$MAX_RESPONSE \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        ++data.seed=$SEED \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=$GRAD_CKPT \
        actor_rollout_ref.actor.optim.lr=$LR \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_RATIO \
        actor_rollout_ref.actor.optim.weight_decay=0.01 \
        actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
        actor_rollout_ref.actor.optim.total_training_steps="$TOTAL_STEPS" \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
        actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.actor.checkpoint.save_contents="$SAVE_CONTENTS" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
        actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
        actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
        actor_rollout_ref.rollout.n=$ROLLOUT_N \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.rollout.multi_stage_wake_up=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=$CHUNKED_PREFILL \
        actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.ref.fsdp_config.param_offload=$PARAM_OFFLOAD \
        custom_reward_function.path="$REPO/verl_faith/scorers.py" \
        custom_reward_function.name=compute_score \
        +reward_model.reward_kwargs.num_examine=2 \
        ${REWARD_ARGS[@]+"${REWARD_ARGS[@]}"} \
        trainer.balance_batch=True \
        trainer.logger='["console"]' \
        trainer.project_name=ft_grpo \
        trainer.experiment_name="$EXPERIMENT" \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.test_freq=$TEST_FREQ \
        trainer.val_before_train=False \
        trainer.total_epochs=999 \
        trainer.total_training_steps="$TOTAL_STEPS" \
        trainer.default_local_dir="$CKPT_DIR" \
        trainer.resume_mode="$RESUME_MODE" 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    set -e
    [ "$status" -eq 0 ] && exit 0
    echo "attempt $attempt failed with exit $status" | tee -a "$LOG"
    pkill -f "$TMPBASE/ray" 2>/dev/null || true
    sleep 30
done
exit 1
