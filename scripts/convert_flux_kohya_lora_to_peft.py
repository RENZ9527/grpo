import argparse
import os

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file


TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "norm1.linear",
    "norm1_context.linear",
    "norm.linear",
    "proj_mlp",
    "proj_out",
]


def find_key(state_dict, module_name, side):
    suffix = f"lora_{side}.default.weight"
    marker = f".{module_name}.{suffix}"
    matches = [
        key
        for key in state_dict
        if key.endswith(marker) or key == f"{module_name}.{suffix}"
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected one PEFT key for {module_name}.{suffix}, found {len(matches)}")
    return matches[0]


def assign_pair(peft_state, out_state, module_name, down, up):
    key_a = find_key(peft_state, module_name, "A")
    key_b = find_key(peft_state, module_name, "B")
    if peft_state[key_a].shape != down.shape:
        raise ValueError(f"{module_name} A shape mismatch: {down.shape} vs {peft_state[key_a].shape}")
    if peft_state[key_b].shape != up.shape:
        raise ValueError(f"{module_name} B shape mismatch: {up.shape} vs {peft_state[key_b].shape}")
    out_state[key_a] = down
    out_state[key_b] = up


def get_pair(kohya_state, prefix):
    return (
        kohya_state[f"{prefix}.lora_down.weight"].float(),
        kohya_state[f"{prefix}.lora_up.weight"].float(),
    )


def maybe_pair(kohya_state, prefix):
    down_key = f"{prefix}.lora_down.weight"
    up_key = f"{prefix}.lora_up.weight"
    if down_key not in kohya_state or up_key not in kohya_state:
        return None
    return get_pair(kohya_state, prefix)


def convert(kohya_state, peft_state):
    out_state = {}

    for block_id in range(19):
        mappings = [
            ("img_mod_lin", f"transformer_blocks.{block_id}.norm1.linear"),
            ("txt_mod_lin", f"transformer_blocks.{block_id}.norm1_context.linear"),
            ("img_attn_proj", f"transformer_blocks.{block_id}.attn.to_out.0"),
            ("txt_attn_proj", f"transformer_blocks.{block_id}.attn.to_add_out"),
            ("img_mlp_0", f"transformer_blocks.{block_id}.ff.net.0.proj"),
            ("img_mlp_2", f"transformer_blocks.{block_id}.ff.net.2"),
            ("txt_mlp_0", f"transformer_blocks.{block_id}.ff_context.net.0.proj"),
            ("txt_mlp_2", f"transformer_blocks.{block_id}.ff_context.net.2"),
        ]
        for kohya_suffix, module_name in mappings:
            pair = maybe_pair(kohya_state, f"lora_unet_double_blocks_{block_id}_{kohya_suffix}")
            if pair:
                assign_pair(peft_state, out_state, module_name, *pair)

        pair = maybe_pair(kohya_state, f"lora_unet_double_blocks_{block_id}_img_attn_qkv")
        if pair:
            down, up = pair
            for name, up_chunk in zip(
                ["attn.to_q", "attn.to_k", "attn.to_v"],
                torch.chunk(up, 3, dim=0),
            ):
                assign_pair(peft_state, out_state, f"transformer_blocks.{block_id}.{name}", down, up_chunk)

        pair = maybe_pair(kohya_state, f"lora_unet_double_blocks_{block_id}_txt_attn_qkv")
        if pair:
            down, up = pair
            for name, up_chunk in zip(
                ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
                torch.chunk(up, 3, dim=0),
            ):
                assign_pair(peft_state, out_state, f"transformer_blocks.{block_id}.{name}", down, up_chunk)

    for block_id in range(38):
        pair = maybe_pair(kohya_state, f"lora_unet_single_blocks_{block_id}_modulation_lin")
        if pair:
            assign_pair(peft_state, out_state, f"single_transformer_blocks.{block_id}.norm.linear", *pair)

        pair = maybe_pair(kohya_state, f"lora_unet_single_blocks_{block_id}_linear2")
        if pair:
            assign_pair(peft_state, out_state, f"single_transformer_blocks.{block_id}.proj_out", *pair)

        pair = maybe_pair(kohya_state, f"lora_unet_single_blocks_{block_id}_linear1")
        if pair:
            down, up = pair
            q, k, v, mlp = torch.split(up, [3072, 3072, 3072, up.shape[0] - 9216], dim=0)
            for module_suffix, up_chunk in [
                ("attn.to_q", q),
                ("attn.to_k", k),
                ("attn.to_v", v),
                ("proj_mlp", mlp),
            ]:
                assign_pair(
                    peft_state,
                    out_state,
                    f"single_transformer_blocks.{block_id}.{module_suffix}",
                    down,
                    up_chunk,
                )

    return out_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        from diffusers import FluxKontextPipeline
    except ImportError as exc:
        raise ImportError(
            "Your installed diffusers package does not provide FluxKontextPipeline. "
            "Install a Kontext-capable diffusers build first, for example: "
            "pip uninstall -y diffusers && pip install git+https://github.com/huggingface/diffusers.git"
        ) from exc

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    pipe = FluxKontextPipeline.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=False)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        init_lora_weights="gaussian",
        target_modules=TARGET_MODULES,
    )
    transformer = get_peft_model(pipe.transformer, lora_config)
    peft_state = transformer.state_dict()

    kohya_state = load_file(args.input, device="cpu")
    converted = convert(kohya_state, peft_state)
    missing, unexpected = transformer.load_state_dict(converted, strict=False)

    os.makedirs(args.output, exist_ok=True)
    transformer.save_pretrained(args.output)
    print(f"Saved PEFT adapter to {args.output}")
    print(f"Converted tensors: {len(converted)}")
    print(f"Missing tensors after partial load: {len(missing)}")
    print(f"Unexpected tensors after partial load: {len(unexpected)}")


if __name__ == "__main__":
    main()
