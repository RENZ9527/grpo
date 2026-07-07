import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
GRPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MATTING_REPO = GRPO_ROOT.parent / "matting"
MATTING_REPO = Path(os.environ.get("MATTING_REPO", DEFAULT_MATTING_REPO)).resolve()
if str(MATTING_REPO) not in sys.path:
    sys.path.insert(0, str(MATTING_REPO))

from pipelines.flux_image_new import FluxImagePipeline
from models.utils import DiffusionTrainingModule, load_state_dict, parse_flux_model_configs
from models.unified_dataset import UnifiedDataset
from lora.flux_lora import FluxLoRALoader

DEFAULT_MODEL_ROOT = MATTING_REPO / "FLUX.1-Kontext-dev"
DEFAULT_GT_ROOT = MATTING_REPO / "datasets" / "P3M-10k"
DEFAULT_FILE_LIST = MATTING_REPO / "data_split" / "P3M_matting" / "filenames_val_NP.txt"

PROMPT = "Transform to matting map while maintaining original composition"
RESOLUTION = 768
SEED = 42
NUM_INFERENCE_STEPS = 1
CFG_SCALE = 1

LORA_TARGET_MODULES = [
    "a_to_qkv",
    "b_to_qkv",
    "ff_a.0",
    "ff_a.2",
    "ff_b.0",
    "ff_b.2",
    "a_to_out",
    "b_to_out",
    "proj_out",
    "norm.linear",
    "norm1_a.linear",
    "norm1_b.linear",
    "to_qkv_mlp",
]
LORA_RANK = 64


def load_training_style_lora(pipe, lora_path, torch_dtype, device):
    """Load LoRA through the same PEFT adapter path used by E2P training."""
    helper = DiffusionTrainingModule()
    pipe.dit = helper.add_lora_to_model(
        pipe.dit,
        target_modules=LORA_TARGET_MODULES,
        lora_rank=LORA_RANK,
        upcast_dtype=pipe.torch_dtype,
    )

    lora_path = Path(lora_path)
    if lora_path.is_dir():
        raise ValueError(
            "batch_inference.py expects a converted E2P LoRA .safetensors file. "
            "Run convert_flux_peft_lora_to_e2p.py first for Flow-GRPO checkpoint lora directories."
        )
    state_dict = load_state_dict(str(lora_path))
    loader = FluxLoRALoader(torch_dtype=torch_dtype, device=device)
    state_dict = loader.convert_state_dict(state_dict)
    state_dict = helper.mapping_lora_state_dict(state_dict)

    load_result = pipe.dit.load_state_dict(state_dict, strict=False)
    missing = getattr(load_result, "missing_keys", load_result[0])
    unexpected = getattr(load_result, "unexpected_keys", load_result[1])
    lora_missing = [key for key in missing if "lora_" in key]

    print(
        f"[load] PEFT LoRA loaded from {lora_path}: "
        f"keys={len(state_dict)}, missing={len(missing)}, "
        f"missing_lora={len(lora_missing)}, unexpected={len(unexpected)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference for Matting with Resume Capability")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to converted E2P LoRA .safetensors")
    parser.add_argument("--out_root", type=str, required=True, help="Directory to save the resulting .npy files")
    parser.add_argument("--model_root", type=str, default=str(DEFAULT_MODEL_ROOT), help="Local FLUX.1-Kontext-dev directory")
    parser.add_argument("--gt_root", type=str, default=str(DEFAULT_GT_ROOT), help="Evaluation dataset root")
    parser.add_argument("--file_list", type=str, default=str(DEFAULT_FILE_LIST), help="Evaluation split file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for inference")
    parser.add_argument("--resolution", type=int, default=RESOLUTION, help="Inference resolution")
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--cfg_scale", type=float, default=CFG_SCALE)
    return parser.parse_args()


def main():
    args = parse_args()
    lora_path = args.lora_path
    out_root = Path(args.out_root)
    model_root = Path(args.model_root)
    gt_root = Path(args.gt_root)
    file_list = Path(args.file_list)
    
    device = args.device
    torch_dtype = torch.bfloat16

    print("[load] pipeline")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=parse_flux_model_configs(str(model_root)),
        model_base_path=str(model_root),
    )

    print(f"[load] lora: {lora_path}")
    load_training_style_lora(pipe, lora_path, torch_dtype, device)

    transform = UnifiedDataset.default_image_operator(
        height=args.resolution,
        width=args.resolution,
    )

    lines = [x.strip() for x in file_list.read_text().splitlines() if x.strip()]
    out_root.mkdir(parents=True, exist_ok=True)

    failed = []
    done = 0
    skipped = 0

    for i, line in enumerate(lines, 1):
        items = line.split()
        merged_rel = items[0]
        trimap_rel = items[1]

        image_path = gt_root / merged_rel
        trimap_path = gt_root / trimap_rel

        save_to = out_root / Path(merged_rel).with_suffix(".npy")
        save_to.parent.mkdir(parents=True, exist_ok=True)

        # ================= 断点续推核心逻辑 =================
        # 如果对应的 numpy 文件已经生成过，则直接跳过，不再进行模型前向推理
        if save_to.exists():
            skipped += 1
            if i % 20 == 0 or i == len(lines):
                print(f"[{i}/{len(lines)}] skipped={skipped}, done={done}, failed={len(failed)}")
            continue
        # ====================================================

        try:
            kontext_images = [
                transform(str(image_path)),
                transform(str(trimap_path)),
            ]

            with torch.no_grad():
                out_np = pipe(
                    prompt=PROMPT,
                    kontext_images=kontext_images,
                    height=args.resolution,
                    width=args.resolution,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    seed=SEED,
                    output_type="np",
                    rand_device=pipe.device,
                    deterministic_flow=False,
                    task="matting",
                )

            np.save(save_to, out_np)
            done += 1

            if i % 20 == 0 or i == len(lines):
                print(f"[{i}/{len(lines)}] skipped={skipped}, done={done}, failed={len(failed)}")

        except Exception as e:
            failed.append((merged_rel, str(e)))
            print(f"[ERROR] {merged_rel}: {e}")

    print(f"\nfinished. done={done}, skipped={skipped}, failed={len(failed)}, out={out_root}")
    if failed:
        print("first failures:")
        for name, err in failed[:20]:
            print(name, err)


if __name__ == "__main__":
    main()
