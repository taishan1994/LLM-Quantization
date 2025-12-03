import sys
# sys.path.insert(0, "/nfs/FM/liqi/vllm_online_rotate-main")
import shutil
import argparse
import json
import copy
from pathlib import Path
import random
import torch.nn.functional as F
import torch
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# 现在导入vllm模块
# from vllm.model_executor.layers.quantization.awq_triton_w4a8 import (
#     awq_gemm_triton,
#     awq_scaled_gemm_triton,
# )


SAVE_IN_LOW_BITS = None


def gemm_pack(weight: torch.Tensor, scales: torch.Tensor, n_bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack int8 weights into AWQ 4-bit format for efficient storage and 
    computation.
    
    Converts unpacked int8 weights to AWQ's packed int32 format following the
    specific bit packing order used by AutoAWQ. Each int32 value stores 8 
    4-bit weights using AWQ's custom ordering scheme.
    
    Args:
        weight: [N, K] int8 weight tensor in range [-8, 7]
        scales: [K, 1] per-channel scaling factors  
        n_bits: Number of bits per weight (must be 4)
        
    Returns:
        tuple containing:
        - int_weight: [N, K//8] packed int32 weights in AWQ format
        - scales: [1, K] transposed and contiguous scales tensor
        
    Raises:
        AssertionError: If n_bits is not 4
    """
    assert n_bits == 4, f"Only 4-bit are supported for now, got {n_bits}"
    scales = scales.t().contiguous()
    weight = weight.to(torch.int32).t().contiguous()

    pack_num = 32 // n_bits

    int_weight = torch.zeros(
        (weight.shape[0], weight.shape[1] // pack_num),
        dtype=torch.int32,
        device=weight.device,
    )

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    for col in range(weight.shape[1] // pack_num):
        for i in range(pack_num):
            # Ensure 4-bit values before shifting to avoid sign-extension
            int_weight_col = weight[:, col * pack_num + order_map[i]] & (
                (1 << n_bits) - 1
            )
            int_weight[:, col] |= int_weight_col << (i * n_bits)

    return int_weight, scales


def normalize_to_lower_4bit(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize 4-bit values to lower nibble and adjust scales accordingly.
    
    Detects whether 4-bit quantized values are stored in the high nibble 
    (bits 4-7) or low nibble (bits 0-3) and normalizes them to the low nibble 
    format expected by AWQ packing. If values are in high nibble, divides 
    weights by 16 and multiplies scales by 16 to maintain mathematical 
    equivalence.
    
    Args:
        weight: [N, K] int8 weight tensor with 4-bit quantized values
        scales: [K, 1] per-channel scaling factors
        
    Returns:
        tuple containing:
        - weight: Normalized weight tensor with values in [-8, 7] range
        - scales: Adjusted scaling factors
        
    Raises:
        NotImplementedError: If weight values are outside expected ranges
        AssertionError: If inconsistent bit storage is detected across tensors
    """

    sample = weight.flatten()[:1000]

    # Check if values are sign-extended 4-bit values (-8 to 7)
    unique_vals = torch.unique(sample)
    assert unique_vals.numel() <= 16, (
        f"Expected 16 unique values, got {unique_vals.numel()}"
    )
    global SAVE_IN_LOW_BITS
    if unique_vals.min() >= -8 and unique_vals.max() <= 7:
        # directly return
        if SAVE_IN_LOW_BITS is None:
            SAVE_IN_LOW_BITS = True
            print("the data is stored in low bits")
        else:
            assert SAVE_IN_LOW_BITS
    elif unique_vals.min() >= -128 and unique_vals.max() <= 127:
        if SAVE_IN_LOW_BITS is None:
            SAVE_IN_LOW_BITS = False
            print("the data is stored in high bits")
        else:
            assert not SAVE_IN_LOW_BITS
        weight //= 16
        scales *= 16
    else:
        raise NotImplementedError(f"Unexpected values: {unique_vals}")
    return weight, scales


# def verify_gemm_pack(
#     weight: torch.Tensor,
#     scales: torch.Tensor,
#     qweight: torch.Tensor,
#     qscales: torch.Tensor,
# ) -> None:
#     """Verify the correctness of AWQ packing by comparing GEMM outputs.
    
#     Performs a numerical verification that the packed AWQ weights produce
#     the same results as the original int8 weights when used in matrix
#     multiplication. Uses random activations to test the equivalence.
    
#     Args:
#         weight: [N, K] original int8 weight tensor
#         scales: [K, 1] original scaling factors  
#         qweight: [N, K//8] packed AWQ weight tensor
#         qscales: [1, K] packed scaling factors
        
#     Raises:
#         ValueError: If GEMM outputs don't match exactly
#         AssertionError: If weight dtype is not int8
#     """
#     assert weight.dtype == torch.int8, f"weight dtype is {weight.dtype}"
#     out_ch, in_ch = weight.shape
#     # Randomize sequence length to diversify coverage
#     seq_len = random.randint(1, 32)
#     fake_act = torch.randn(seq_len, in_ch) * (1 << 3 - 1)
#     fake_act = fake_act.to(weight)
#     # Use float approximation since mm_cuda doesn't support int8
#     ori_out_int32 = (
#         F.linear(fake_act.float(), weight.float()).round().to(torch.int32)
#     )

#     # Get int32 result from our pure integer GEMM
#     awq_out_int32 = awq_gemm_triton(fake_act, qweight)
#     if not torch.equal(ori_out_int32, awq_out_int32):
#         raise ValueError("int8 gemm outputs don't match exactly")


# def verify_gemm_pack_scaled(
#     weight: torch.Tensor,
#     scales: torch.Tensor,
#     qweight: torch.Tensor,
#     qscales: torch.Tensor,
# ) -> None:
#     """Verify the correctness of AWQ packing with scaled GEMM operations.
    
#     Performs a more comprehensive verification that includes both activation
#     and weight scaling, which is the typical usage pattern in inference.
#     Tests that the packed format produces numerically equivalent results
#     to the original int8 weights when scaling is applied.
    
#     Args:
#         weight: [N, K] original int8 weight tensor
#         scales: [K, 1] original weight scaling factors
#         qweight: [N, K//8] packed AWQ weight tensor  
#         qscales: [1, K] packed weight scaling factors
        
#     Raises:
#         ValueError: If scaled GEMM outputs don't match within tolerance
#         AssertionError: If weight dtype is not int8
#     """
#     assert weight.dtype == torch.int8, f"weight dtype is {weight.dtype}"
#     out_ch, in_ch = weight.shape
#     # Randomize sequence length to diversify coverage
#     seq_len = random.randint(1, 32)
#     fake_act = torch.randn(seq_len, in_ch) * (1 << 3 - 1)
#     fake_act = fake_act.to(weight)
#     act_scales = torch.randn(seq_len, 1).cuda()

#     # Use float approximation since mm_cuda doesn't support int8
#     ori_out = F.linear(fake_act.float(), weight.float())
#     ori_out = ori_out.round().to(torch.int32) * act_scales * scales.t()

#     # Get int32 result from our pure integer GEMM
#     awq_out = awq_scaled_gemm_triton(fake_act, qweight, act_scales, qscales)
#     if not torch.equal(ori_out, awq_out):
#         diff = (ori_out - awq_out).abs().max()
#         raise ValueError(
#             f"int8 scaled gemm outputs don't match exactly, "
#             f"Max diff: {diff.abs().max():.6f}"
#         )


def process_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    verify_prob: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process a weight tensor for AWQ packing with optional verification.
    
    Complete pipeline to convert fake-int8 weights to AWQ format:
    1. Move tensors to CUDA for processing
    2. Normalize 4-bit values to lower nibble format  
    3. Pack weights using AWQ bit packing scheme
    4. Optionally verify packing correctness with probability verify_prob
    
    Args:
        weight: [N, K] int8 weight tensor to be packed
        scales: [K, 1] per-channel scaling factors
        verify_prob: Probability of running verification (0.0 to 1.0)
        
    Returns:
        tuple containing:
        - qweight: [N, K//8] packed int32 weights in AWQ format
        - qscales: [1, K] transposed scaling factors
        
    Raises:
        ValueError: If verification fails (when verification is run)
    """

    scales = scales.cuda()
    weight = weight.cuda()

    # Detect 4-bit position
    weight, scales = normalize_to_lower_4bit(weight, scales)

    # Pack to int32 in AWQ format
    qweight, qscales = gemm_pack(weight, scales)

    # Probabilistic verification controlled by caller
    # if random.random() < verify_prob:
    #     verify_gemm_pack_scaled(weight, scales, qweight, qscales)

    return qweight, qscales


def update_safetensors_index(index_file: Path, output_dir: Path) -> None:
    """Update model.safetensors.index.json to reflect AWQ weight names.
    
    Converts fake-int8 format weight names to AWQ format:
    - ".weight_scale" -> both ".qweight" and ".scales" entries
    - Removes original ".weight" entries (they become ".qweight")
    - Preserves all other tensor entries unchanged
    
    Args:
        index_file: Path to the original model.safetensors.index.json file
        output_dir: Directory where the updated index file will be saved
        
    Raises:
        AssertionError: If weight_zeros are found (asymmetric quantization not supported)
        FileNotFoundError: If index_file doesn't exist
    """
    print(f"Updating safetensors index file: {index_file}")

    with open(index_file) as f:
        index_data = json.load(f)

    # Update weight_map to reflect new AWQ format names
    assert "weight_map" in index_data
    new_weight_map = {}
    for key, file_name in index_data["weight_map"].items():
        assert "weight_zeros" not in key

        # For int8 weights, map to qweight and scales
        if "visual" not in key:
            if key.endswith(".weight_scale"):
                key_stem = key.rsplit(".", 1)[0]
                # Add both qweight and scales entries
                new_weight_map[key_stem + ".qweight"] = file_name
                new_weight_map[key_stem + ".scales"] = file_name
            elif key + "_scale" in index_data["weight_map"]:
                continue
            else:
                # Keep other entries as-is
                new_weight_map[key] = file_name
        else:
            new_weight_map[key] = file_name

    index_data["weight_map"] = new_weight_map

    # Save updated index file
    output_index_file = output_dir / index_file.name
    with open(output_index_file, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Updated index file saved to: {output_index_file}")


def main() -> None:
    """Main function to pack fake-int8 model to AWQ format.
    
    Processes command line arguments and orchestrates the packing process:
    1. Validates input and output directories
    2. Processes all .safetensors and .bin files in the model directory
    3. Packs int8 weights to AWQ format with optional verification
    4. Updates safetensors index file if present
    5. Creates appropriate quantization config for awq_triton_w4a8 method
    
    Command line arguments:
        --model_dir: Path to fake-int8 model directory (required)
        --output: Output path for packed model (optional, defaults to input + '_awq_packed')
        --group-size: Group size for quantization, must be -1 for channel-wise (default: -1)
        --verify-prob: Probability of verifying each weight tensor (default: 0.1)
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pack int8 weights to AWQ format for awq_triton_w4a8 inference"
        )
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to input model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for packed model (default: input_path + '_awq_packed')"
        ),
    )
    parser.add_argument(
        "--quant_visual",
        action = "store_true",
        help=(
            "whether visual blocks are quantized"
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=-1,
        help="Group size for quantization (-1 for channel-wise)",
    )
    parser.add_argument(
        "--verify-prob",
        type=float,
        default=0.1,
        help=(
            "Probability [0,1] to verify each weight with random seq_len "
            "(default: 0.1)"
        ),
    )
    args = parser.parse_args()
    print(args.quant_visual)
    # Determine input file
    model_dir = Path(args.model_dir)
    print(f"{model_dir=}")
    if not model_dir.exists():
        raise FileNotFoundError("model_dir does not exist")

    # Determine output path
    if args.output is None:
        output_dir = model_dir.with_name(model_dir.name + "_awq_packed")
        print(f"output_dir not specified, using {output_dir}")
    else:
        output_dir = Path(args.output)
        print(f"{output_dir=}")
    if output_dir.exists():
        print(f"[INFO] output_dir already exists: {output_dir}. Nothing to do, exit normally.")
        sys.exit(0)
    output_dir.mkdir(parents=True, exist_ok=False)

    # Process weights
    assert args.group_size == -1, (
        "Group size must be -1 for channel-wise quantization"
    )
    print(f"Group size: {args.group_size}")
    for file in model_dir.iterdir():
        if file.is_dir():
            continue

        if file.suffix in [".safetensors", ".bin"]:
            packed_tensors = {}
            with safe_open(file, framework="pt", device="cuda") as f:
                f: safetensors.safe_open
                for key in tqdm(f.keys(), desc=file.stem):
                    # Visual weights: save as-is (keep compressed-tensors format)
                    if "visual" in key:
                        weight = f.get_tensor(key)
                        packed_tensors[key] = weight.cpu()
                        continue
                    
                    # Skip intermediate weight_scale and weight_zeros (will be packed)
                    if "weight_scale" in key or "weight_zeros" in key:
                        continue

                    weight = f.get_tensor(key)
                    # LLM weights: pack int8 to AWQ format
                    if weight.dtype == torch.int8:
                        scale = f.get_tensor(key + "_scale")

                        qweight, scales = process_weight(
                            weight, scale, args.verify_prob
                        )
                        key_stem = key.rsplit(".", 1)[0]
                        packed_tensors[key_stem + ".qweight"] = qweight.cpu()
                        packed_tensors[key_stem + ".scales"] = scales.cpu()
                    else:
                        packed_tensors[key] = weight.cpu()
            save_file(packed_tensors, output_dir / file.name)
        elif file.name == "tokenizer.json":
            # Special handling for tokenizer.json: set truncation to null
            print(f"Processing {file.name}: setting truncation to null")
            with open(file) as f:
                tokenizer_config = json.load(f)
            tokenizer_config["truncation"] = None
            with open(output_dir / file.name, "w") as f:
                json.dump(tokenizer_config, f, indent=2)
        else:
            shutil.copy(file, output_dir / file.name)

    # Update safetensors index file if it exists
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        update_safetensors_index(index_file, output_dir)

    # Create quantization config file for awq_triton_w4a8
    quant_config = {
        "bits": 4,
        "group_size": args.group_size,
        "zero_point": False,  # Symmetric quantization
        "quant_method": "awq_triton_w4a8",
        "version": "gemm",
    }

    if not args.quant_visual:
        quant_config["modules_to_not_convert"] = ["visual"]
        
    # Read the config.json that was copied from model_dir to output_dir
    config_file = output_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"config.json not found in output directory: {config_file}"
        )
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Save original quantization config before modifying
    original_quant_config = config.get("quantization_config", {})
    
    # Move visual-related quantization config to vision_config
    if args.quant_visual:
        if "vision_config" in config and "config_groups" in original_quant_config:
            # Extract group_0 (visual quantization config)
            group_0 = original_quant_config["config_groups"].get("group_0", {})
            
            if group_0:
                # Deep copy group_0 to avoid modifying the original
                group_0 = copy.deepcopy(group_0)
                
                # Add alternative target patterns without "model." prefix
                # This ensures compatibility with different model loading paths
                if "targets" in group_0:
                    original_targets = group_0["targets"]
                    additional_targets = []
                    for target in original_targets:
                        # For targets starting with "re:model.visual", add "re:visual" version
                        if target.startswith("re:model.visual"):
                            additional_target = target.replace("re:model.visual", "re:visual")
                            additional_targets.append(additional_target)
                    # Append additional targets to maintain both patterns
                    group_0["targets"] = original_targets + additional_targets
                
                # Create vision quantization config with only group_0
                vision_quant_config = {
                    "config_groups": {
                        "group_0": group_0
                    },
                    "format": original_quant_config.get("format", "int-quantized"),
                    "global_compression_ratio": original_quant_config.get("global_compression_ratio"),
                    "ignore": original_quant_config.get("ignore", []),
                    "kv_cache_scheme": original_quant_config.get("kv_cache_scheme"),
                    "quant_method": original_quant_config.get("quant_method", "compressed-tensors"),
                    "quantization_status": original_quant_config.get("quantization_status", "compressed"),
                    "sparsity_config": original_quant_config.get("sparsity_config", {}),
                    "transform_config": original_quant_config.get("transform_config", {}),
                    "version": original_quant_config.get("version")
                }
                
                # Add to vision_config
                config["vision_config"]["quantization_config"] = vision_quant_config
    
    # Replace root level quantization_config with AWQ config for LLM
    config.pop("compression_config", None)
    config["quantization_config"] = quant_config
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done! AWQ format model saved to: {output_dir}")


if __name__ == "__main__":
    main()
