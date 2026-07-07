#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Flow-GRPO output layout:
#   ${CHECKPOINT_DIR}/checkpoint-*/lora/adapter_model.safetensors
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/outputs/flow_grpo/checkpoints}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/FLUX.1-Kontext-dev}"
GT_ROOT="${GT_ROOT:-${REPO_ROOT}/datasets/P3M-10k}"
FILE_LIST="${FILE_LIST:-${REPO_ROOT}/data_split/P3M_matting/filenames_val_NP.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/flow_grpo_eval}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_NAME="${DATASET_NAME:-p3m-np}"
EVAL_LOG="${EVAL_LOG:-${OUTPUT_DIR}/evaluation_all_checkpoints.txt}"
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-${REPO_ROOT}/batch_inference.py}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-${REPO_ROOT}/convert_flux_peft_lora_to_e2p.py}"
LORA_RANK="${LORA_RANK:-64}"

mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${EVAL_LOG}" ]; then
    echo "================ Evaluation Results ================" > "${EVAL_LOG}"
fi

shopt -s nullglob
checkpoint_paths=("${CHECKPOINT_DIR}"/checkpoint-*)
if [ ${#checkpoint_paths[@]} -eq 0 ]; then
    echo "No checkpoint-* directories found under ${CHECKPOINT_DIR}"
    exit 1
fi

for ckpt_dir in $(printf "%s\n" "${checkpoint_paths[@]}" | sort -V); do
    ckpt_name="$(basename "${ckpt_dir}")"
    ckpt_step="${ckpt_name#checkpoint-}"
    lora_dir="${ckpt_dir}/lora"
    converted_lora="${ckpt_dir}/e2p_fused_lora.safetensors"
    out_root="${OUTPUT_DIR}/predictions_${DATASET_NAME}_${ckpt_step}"

    echo "------------------------------------------------------------------"
    echo "Checking ${ckpt_name}"

    if [ -f "${out_root}/.eval_done" ]; then
        echo "${ckpt_name} already evaluated, skipping."
        continue
    fi

    if [ ! -d "${lora_dir}" ]; then
        echo "Missing LoRA directory: ${lora_dir}"
        continue
    fi

    echo "[1/3] Converting Flow-GRPO LoRA to E2P format"
    if [ ! -f "${converted_lora}" ]; then
        python "${CONVERT_SCRIPT}" \
            --input "${lora_dir}" \
            --output "${converted_lora}" \
            --rank "${LORA_RANK}"
    else
        echo "Converted LoRA already exists: ${converted_lora}"
    fi

    echo "[2/3] Running batch inference"
    python "${INFERENCE_SCRIPT}" \
        --lora_path "${converted_lora}" \
        --out_root "${out_root}" \
        --model_root "${MODEL_ROOT}" \
        --gt_root "${GT_ROOT}" \
        --file_list "${FILE_LIST}" \
        --device "${DEVICE}"

    echo "[3/3] Running matting evaluation"
    echo "" >> "${EVAL_LOG}"
    echo "=== Results for ${ckpt_name} ===" >> "${EVAL_LOG}"

    python -m utils.eval_matting \
        --pred_path "${out_root}" \
        --gt_path "${GT_ROOT}" \
        --dataset "${DATASET_NAME}" | tee -a "${EVAL_LOG}"

    touch "${out_root}/.eval_done"
    echo "${ckpt_name} finished."
done

echo "All checkpoints processed."
