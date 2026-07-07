# 8 GPU
CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux.py --config config/grpo.py:pickscore_flux_8gpu
