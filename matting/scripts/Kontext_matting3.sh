# export NCCL_P2P_LEVEL=2
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_TIMEOUT=22
# export TORCH_NCCL_BLOCKING_WAIT=0

export TORCH_NCCL_TIMEOUT=1800

accelerate launch --config_file configs/accelerate_config.yaml scripts/train.py \
    --dataset_base_path /nvmedata/workspace2/users/rzc/datasets/e2p_matting/train \
    --dataset_metadata_path /nvmedata/workspace2/users/rzc/datasets/e2p_matting/train/filenames_train.txt \
    --data_file_keys kontext_images,image \
    --model_paths /nvmedata/workspace2/share_model/FLUX.1-Kontext-dev \
    --learning_rate 1e-5 \
    --num_epochs 30 \
    --remove_prefix_in_ckpt pipe.dit. \
    --trainable_models dit \
    --extra_inputs kontext_images,trimap \
    --use_gradient_checkpointing \
    --multi_res_noise \
    --default_caption "Transform to matting map while maintaining original composition" \
    --batch_size 2 \
    --save_steps 5000 \
    --matting_prompt points \
    --output_path /nvmedata/workspace2/users/rzc/output/kontext_matting/20260408_e2p_setting \
    --eval_file_list /nvmedata/workspace2/users/rzc/datasets/AIM-500/filenames_val.txt \
    --eval_steps 500 \
    --resume \
    --adamw8bit \
    --dataset_num_workers 0 \
    --resume \
    --height 512 \
    --width 512 \
    --extra_loss cycle_consistency_matting_estimation \
    # --with_mask \

    # --extra_loss cycle_consistency_cycle_consistency_matting_estimation \
    # --extra_inputs kontext_images,trimap,visual_prompt_coords \
    # --extra_loss cycle_consistency_matting_estimation \
    # --with_mask  --use_coor_input \
    # --matting_prompt bbox \
    # --deterministic_flow


