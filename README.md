# Matting Flow-GRPO Training

This repository documents the Flow-GRPO training command for the matting task with `FLUX.1-Kontext-dev`.

Use the matting configuration only:

```text
config/grpo.py:matting_flux_kontext_full_safe
```

## Environment

```bash
conda create -n flow_grpo python=3.10.16
conda activate flow_grpo
pip install -e .
pip install git+https://github.com/huggingface/diffusers.git
```

## Initial LoRA Weights

Download the initial LoRA weights from Hugging Face:

```bash
hf download Renz-7/grpo \
  --local-dir ./checkpoints/Renz-7/grpo
```

Use the downloaded LoRA directory as `config.train.lora_path` in `config/grpo.py`.

## Dataset Metadata

The matting dataset is read from JSONL metadata files:

```text
dataset/e2p_matting_grpo_new/
├── train_metadata.jsonl
└── test_metadata.jsonl
```

Each row should contain:

```json
{"prompt": "...", "image": "/path/to/original.jpg", "trimap": "/path/to/trimap.png", "alpha": "/path/to/mask.png"}
```

If metadata needs to be regenerated:

```bash
python scripts/prepare_matting_metadata.py \
  --split-file /path/to/filenames_train.txt \
  --output-dir dataset/e2p_matting_grpo_new \
  --split train \
  --root /path/to/dataset/root
```

For validation:

```bash
python scripts/prepare_matting_metadata.py \
  --split-file /path/to/filenames_val.txt \
  --output-dir dataset/e2p_matting_grpo_new \
  --split test \
  --root /path/to/validation/root
```

## Paths to Modify

Before training, update the matting configuration in `config/grpo.py`:

```text
matting_flux_kontext_full_safe()
├── config.dataset            # dataset/e2p_matting_grpo_new
├── config.pretrained.model   # local FLUX.1-Kontext-dev path
├── config.train.lora_path    # downloaded initial LoRA weights from Renz-7/grpo
└── config.save_dir           # output directory
```

The matting reward is enabled by:

```python
config.reward_fn = {"matting": 1.0}
```

## Single-GPU Training

Run with `matting_flux_kontext_full_safe`:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 accelerate launch \
  --config_file scripts/accelerate_configs/single_gpu.yaml \
  --num_processes=1 \
  --main_process_port 29501 \
  scripts/train_flux_kontext.py \
  --config config/grpo.py:matting_flux_kontext_full_safe
```

## Multi-GPU Training

Single-node multi-GPU example:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
  --num_processes=8 \
  --main_process_port 29501 \
  scripts/train_flux_kontext.py \
  --config config/grpo.py:matting_flux_kontext_full_safe
```

Multi-node example:

```bash
# Node 0
WANDB_MODE=offline accelerate launch \
  --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
  --num_machines 4 \
  --num_processes 28 \
  --machine_rank 0 \
  --main_process_ip MASTER_NODE_IP \
  --main_process_port 19001 \
  scripts/train_flux_kontext.py \
  --config config/grpo.py:matting_flux_kontext_full_safe

# Node 1/2/3: change --machine_rank to 1, 2, or 3.
```

## Outputs

Training outputs are saved under `config.save_dir`:

```text
config.save_dir/
├── checkpoints/checkpoint-*/lora/
├── reward_history.jsonl
└── debug_images/
```

## Matting Evaluation

Use `matting_eval/` to evaluate Flow-GRPO checkpoints with the E2P/matting inference code. Flow-GRPO saves PEFT LoRA weights, so each checkpoint must first be converted to the E2P LoRA format before inference.

Expected Flow-GRPO checkpoint layout:

```text
config.save_dir/
└── checkpoints/
    ├── checkpoint-100/
    │   └── lora/
    │       └── adapter_model.safetensors
    └── checkpoint-200/
        └── lora/
            └── adapter_model.safetensors
```

Batch evaluation:

```bash
cd /path/to/grpo

CHECKPOINT_DIR=/path/to/grpo/output/checkpoints \
MATTING_REPO=/path/to/matting \
MODEL_ROOT=/path/to/FLUX.1-Kontext-dev \
GT_ROOT=/path/to/P3M-10k \
FILE_LIST=/path/to/matting/data_split/P3M_matting/filenames_val_NP.txt \
OUTPUT_DIR=/path/to/eval_outputs \
DEVICE=cuda:0 \
DATASET_NAME=p3m-np \
LORA_RANK=64 \
bash matting_eval/run_all_checkpoints.sh
```

For each checkpoint, `run_all_checkpoints.sh` runs:

```text
1. matting_eval/convert_flux_peft_lora_to_e2p.py
2. matting_eval/batch_inference.py
3. python -m utils.eval_matting
```

Single-checkpoint evaluation:

```bash
cd /path/to/grpo

python matting_eval/convert_flux_peft_lora_to_e2p.py \
  --input /path/to/checkpoint-100/lora \
  --output /path/to/checkpoint-100/e2p_fused_lora.safetensors \
  --rank 64

MATTING_REPO=/path/to/matting python matting_eval/batch_inference.py \
  --lora_path /path/to/checkpoint-100/e2p_fused_lora.safetensors \
  --out_root /path/to/eval_outputs/predictions_p3m-np_100 \
  --model_root /path/to/FLUX.1-Kontext-dev \
  --gt_root /path/to/P3M-10k \
  --file_list /path/to/matting/data_split/P3M_matting/filenames_val_NP.txt \
  --device cuda:0

PYTHONPATH=/path/to/matting python -m utils.eval_matting \
  --pred_path /path/to/eval_outputs/predictions_p3m-np_100 \
  --gt_path /path/to/P3M-10k \
  --dataset p3m-np
```

Batch evaluation results are appended to:

```text
OUTPUT_DIR/evaluation_all_checkpoints.txt
```
