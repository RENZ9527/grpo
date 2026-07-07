# export NCCL_P2P_LEVEL=2
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_TIMEOUT=22
# export TORCH_NCCL_BLOCKING_WAIT=0

export TORCH_NCCL_TIMEOUT=7200000

python -m accelerate.commands.launch --config_file configs/accelerate_config.yaml scripts/train.py \
    --dataset_base_path /nvmedata/workspace2/users/rzc/datasets/merged_matting_dataset/train \
    --dataset_metadata_path /nvmedata/workspace2/users/rzc/datasets/merged_matting_dataset/train/filenames_train.txt \
    --data_file_keys kontext_images,image \
    --model_paths /nvmedata/workspace2/share_model/FLUX.1-Kontext-dev \
    --learning_rate 1e-5 \
    --num_epochs 30 \
    --remove_prefix_in_ckpt pipe.dit. \
    --trainable_models dit \
    --extra_inputs kontext_images,trimap \
    --multi_res_noise \
    --default_caption "Transform to matting map while maintaining original composition" \
    --with_mask \
    --batch_size 2 \
    --save_steps 2000 \
    --matting_prompt trimap \
    --output_path /nvmedata/workspace2/users/rzc/output/kontext_matting/20260707_e2p_setting \
    --eval_file_list ./data_split/AIM_matting/filenames_val.txt \
    --eval_steps 500 \
    --resume  \
    --height 512 \
    --width 512 \
    --adamw8bit \
    --lora_base_model "dit" \
    --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
    --lora_rank 64 \
    --align_to_opensource_format \
    --extra_loss cycle_consistency_matting_estimation \
    --use_gradient_checkpointing \
    # --extra_inputs kontext_images,trimap,visual_prompt_coords \
    # --extra_loss cycle_consistency_matting_estimation \

    # --with_mask  --use_coor_input \
    # --matting_prompt bbox \
    # --deterministic_flow


