import argparse
import os

import torch
from safetensors.torch import load_file, save_file


PREFIXES = ("base_model.model.", "model.")


def strip_prefix(key):
    for prefix in PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def pair(state, module_name):
    candidates = [
        (
            f"{module_name}.lora_A.weight",
            f"{module_name}.lora_B.weight",
        ),
        (
            f"{module_name}.lora_A.default.weight",
            f"{module_name}.lora_B.default.weight",
        ),
    ]
    for key_a, key_b in candidates:
        if key_a in state and key_b in state:
            return state[key_a].float(), state[key_b].float()
    return None


def set_pair(out, name, a, b):
    out[f"{name}.lora_A.default.weight"] = a.contiguous()
    out[f"{name}.lora_B.default.weight"] = b.contiguous()


def svd_lora(delta, rank):
    delta = delta.float()
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    rank = min(rank, s.numel())
    sqrt_s = torch.sqrt(s[:rank])
    b = u[:, :rank] * sqrt_s.unsqueeze(0)
    a = sqrt_s.unsqueeze(1) * vh[:rank, :]
    return a, b


def fuse_modules(state, modules, rank):
    deltas = []
    missing = []
    for module in modules:
        weights = pair(state, module)
        if weights is None:
            missing.append(module)
            continue
        a, b = weights
        deltas.append(b @ a)
    if missing:
        return None, missing
    delta = torch.cat(deltas, dim=0)
    return svd_lora(delta, rank), []


def copy_direct(state, out, src, dst):
    weights = pair(state, src)
    if weights is None:
        return False
    set_pair(out, dst, *weights)
    return True


def normalize_state_keys(state):
    return {strip_prefix(key): value for key, value in state.items()}


def convert(state, rank):
    state = normalize_state_keys(state)
    out = {}
    missing = []
    approximated = []

    for block_id in range(19):
        direct = [
            (
                f"transformer_blocks.{block_id}.norm1.linear",
                f"blocks.{block_id}.norm1_a.linear",
            ),
            (
                f"transformer_blocks.{block_id}.norm1_context.linear",
                f"blocks.{block_id}.norm1_b.linear",
            ),
            (
                f"transformer_blocks.{block_id}.attn.to_out.0",
                f"blocks.{block_id}.attn.a_to_out",
            ),
            (
                f"transformer_blocks.{block_id}.attn.to_add_out",
                f"blocks.{block_id}.attn.b_to_out",
            ),
            (
                f"transformer_blocks.{block_id}.ff.net.0.proj",
                f"blocks.{block_id}.ff_a.0",
            ),
            (
                f"transformer_blocks.{block_id}.ff.net.2",
                f"blocks.{block_id}.ff_a.2",
            ),
            (
                f"transformer_blocks.{block_id}.ff_context.net.0.proj",
                f"blocks.{block_id}.ff_b.0",
            ),
            (
                f"transformer_blocks.{block_id}.ff_context.net.2",
                f"blocks.{block_id}.ff_b.2",
            ),
        ]
        for src, dst in direct:
            if not copy_direct(state, out, src, dst):
                missing.append(src)

        fused_groups = [
            (
                [
                    f"transformer_blocks.{block_id}.attn.to_q",
                    f"transformer_blocks.{block_id}.attn.to_k",
                    f"transformer_blocks.{block_id}.attn.to_v",
                ],
                f"blocks.{block_id}.attn.a_to_qkv",
            ),
            (
                [
                    f"transformer_blocks.{block_id}.attn.add_q_proj",
                    f"transformer_blocks.{block_id}.attn.add_k_proj",
                    f"transformer_blocks.{block_id}.attn.add_v_proj",
                ],
                f"blocks.{block_id}.attn.b_to_qkv",
            ),
        ]
        for modules, dst in fused_groups:
            fused, group_missing = fuse_modules(state, modules, rank)
            if group_missing:
                missing.extend(group_missing)
                continue
            set_pair(out, dst, *fused)
            approximated.append(dst)

    for block_id in range(38):
        direct = [
            (
                f"single_transformer_blocks.{block_id}.norm.linear",
                f"single_blocks.{block_id}.norm.linear",
            ),
            (
                f"single_transformer_blocks.{block_id}.proj_out",
                f"single_blocks.{block_id}.proj_out",
            ),
        ]
        for src, dst in direct:
            if not copy_direct(state, out, src, dst):
                missing.append(src)

        modules = [
            f"single_transformer_blocks.{block_id}.attn.to_q",
            f"single_transformer_blocks.{block_id}.attn.to_k",
            f"single_transformer_blocks.{block_id}.attn.to_v",
            f"single_transformer_blocks.{block_id}.proj_mlp",
        ]
        fused, group_missing = fuse_modules(state, modules, rank)
        if group_missing:
            missing.extend(group_missing)
        else:
            dst = f"single_blocks.{block_id}.to_qkv_mlp"
            set_pair(out, dst, *fused)
            approximated.append(dst)

    return out, missing, approximated


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Flow-GRPO/PEFT FLUX adapter to e2p native fused LoRA keys. "
            "Split q/k/v/mlp PEFT LoRAs are compressed back to the requested rank with SVD."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to checkpoint-*/lora directory or adapter_model.safetensors.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .safetensors path for e2p.",
    )
    parser.add_argument("--rank", type=int, default=64)
    args = parser.parse_args()

    input_path = args.input
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, "adapter_model.safetensors")
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    state = load_file(input_path, device="cpu")
    converted, missing, approximated = convert(state, args.rank)
    if not converted:
        raise ValueError(f"No LoRA tensors converted from {input_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(converted, args.output)

    print(f"Loaded: {input_path}")
    print(f"Saved: {args.output}")
    print(f"Converted tensors: {len(converted)}")
    print(f"SVD-compressed fused modules: {len(approximated)}")
    print(f"Missing source modules: {len(missing)}")
    if missing:
        print("First missing modules:")
        for item in missing[:16]:
            print(f"  {item}")
    print("Example keys:")
    for key in list(converted.keys())[:12]:
        print(f"  {key} {tuple(converted[key].shape)}")


if __name__ == "__main__":
    main()
