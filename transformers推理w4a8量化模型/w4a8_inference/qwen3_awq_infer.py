from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import torch
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    Qwen3ForCausalLM,
)
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from .awq_linear import AwqW4A8Linear


@dataclass
class ModelPaths:
    model_dir: str


def _get_parent_and_attr(module: torch.nn.Module, path: str) -> Tuple[torch.nn.Module, str]:
    parts = path.split(".")
    parent = module
    for name in parts[:-1]:
        if name.isdigit():
            parent = parent[int(name)]  # type: ignore[index]
        else:
            parent = getattr(parent, name)
    return parent, parts[-1]


def _replace_linears_with_awq(
    model: Qwen3ForCausalLM,
    qweight_keys: Iterable[str],
    *,
    dtype: torch.dtype,
    use_triton: bool,
) -> None:
    """Replace Linear modules with AwqW4A8Linear, passing use_triton flag."""

    for key in qweight_keys:
        if not key.endswith(".qweight"):
            continue
        module_path = key[: -len(".qweight")]
        parent, attr_name = _get_parent_and_attr(model, module_path)
        orig_linear = getattr(parent, attr_name)
        if not isinstance(orig_linear, torch.nn.Linear):
            continue

        awq_linear = AwqW4A8Linear(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            dtype=dtype,
            use_triton=use_triton,  # Pass the flag here
        )
        setattr(parent, attr_name, awq_linear)


def _collect_qweight_keys(model_dir: str) -> List[str]:
    """Collect all `.qweight` keys from single or sharded safetensors (names only)."""

    single_path = os.path.join(model_dir, "model.safetensors")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.exists(single_path):
        keys: List[str] = []
        with safe_open(single_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".qweight"):
                    keys.append(k)
        return keys

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Neither model.safetensors nor model.safetensors.index.json found in {model_dir}"
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"Empty weight_map in {index_path}")

    return [k for k in weight_map.keys() if k.endswith(".qweight")]


def _load_awq_into_model(
    model: Qwen3ForCausalLM,
    model_dir: str,
    device: torch.device,
) -> None:
    """Stream AWQ weights into model without building a giant state_dict.

    - 遍历单文件或分片 safetensors
    - 对普通权重/缓冲区：用 set_module_tensor_to_device 直接写入模型
    - 对 `.qweight` / `.scales`：直接调用 AwqW4A8Linear.set_awq_params

    这样避免一次性在内存中构造完整 state_dict，适合 32B 这类大模型。
    同时会打印简单的加载进度日志，便于观察加载过程不会“卡死”。
    """

    pending_q: Dict[str, torch.Tensor] = {}
    pending_s: Dict[str, torch.Tensor] = {}

    def maybe_set_awq(module_path: str) -> None:
        if module_path in pending_q and module_path in pending_s:
            q = pending_q.pop(module_path)
            s = pending_s.pop(module_path)
            parent, attr_name = _get_parent_and_attr(model, module_path)
            module = getattr(parent, attr_name)
            if isinstance(module, AwqW4A8Linear):
                module.set_awq_params(q.to(device), s.to(device))

    single_path = os.path.join(model_dir, "model.safetensors")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    target = str(device)

    # 单文件场景
    if os.path.exists(single_path):
        print(f"[LOAD] Found single safetensors file: {single_path}")
        with safe_open(single_path, framework="pt", device=target) as f:
            keys = list(f.keys())
            total = len(keys)
            print(f"[LOAD] Total tensors: {total}")
            for i, key in enumerate(keys, 1):
                tensor = f.get_tensor(key)
                if key.endswith(".qweight"):
                    module_path = key[: -len(".qweight")]
                    pending_q[module_path] = tensor
                    maybe_set_awq(module_path)
                    continue
                if key.endswith(".scales"):
                    module_path = key[: -len(".scales")]
                    pending_s[module_path] = tensor
                    maybe_set_awq(module_path)
                    continue

                try:
                    set_module_tensor_to_device(
                        model,
                        key,
                        device=device,
                        value=tensor,
                    )
                except ValueError:
                    # key not present as param/buffer; ignore
                    pass

                if i % 500 == 0 or i == total:
                    print(f"[LOAD] Loaded {i}/{total} tensors", flush=True)
        return

    # 分片场景
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Neither model.safetensors nor model.safetensors.index.json found in {model_dir}"
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"Empty weight_map in {index_path}")

    shard_to_keys: Dict[str, List[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    shard_names = sorted(shard_to_keys.keys())
    num_shards = len(shard_names)
    print(f"[LOAD] Found {num_shards} shards in index")

    for shard_idx, shard in enumerate(shard_names, 1):
        keys = shard_to_keys[shard]
        shard_path = os.path.join(model_dir, shard)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file {shard_path} referenced in index but not found")

        print(
            f"[LOAD] Loading shard {shard_idx}/{num_shards}: {shard} "
            f"({len(keys)} tensors)",
            flush=True,
        )

        with safe_open(shard_path, framework="pt", device=target) as f:
            total = len(keys)
            for i, key in enumerate(keys, 1):
                tensor = f.get_tensor(key)

                if key.endswith(".qweight"):
                    module_path = key[: -len(".qweight")]
                    pending_q[module_path] = tensor
                    maybe_set_awq(module_path)
                    continue
                if key.endswith(".scales"):
                    module_path = key[: -len(".scales")]
                    pending_s[module_path] = tensor
                    maybe_set_awq(module_path)
                    continue

                try:
                    set_module_tensor_to_device(
                        model,
                        key,
                        device=device,
                        value=tensor,
                    )
                except ValueError:
                    pass

                if i % 500 == 0 or i == total:
                    print(
                        f"[LOAD]   shard {shard_idx}/{num_shards}: "
                        f"{i}/{total} tensors loaded",
                        flush=True,
                    )


def build_qwen3_awq_model(
    model_dir: str,
    device: Optional[torch.device] = None,
    use_triton: bool = True,  # Add use_triton flag
) -> Qwen3ForCausalLM:
    """Build Qwen3 AWQ model with streaming weight loading and低内存初始化.

    - 使用 accelerate.init_empty_weights 在 meta device 上构建模型骨架
    - 根据 `.qweight` key 先替换对应 Linear 为 AwqW4A8Linear
    - 再按 shard 流式加载权重到目标 device（CUDA），边加载边赋值
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(model_dir)
    if config.model_type != "qwen3":
        raise ValueError(f"Expected model_type 'qwen3', got {config.model_type}")

    # 1) 在 meta device 上构建空模型，避免先分配一整份 32B 随机权重
    with init_empty_weights():
        model = Qwen3ForCausalLM(config)

    # 2) 收集所有 `.qweight` key，并替换对应 Linear 为 AwqW4A8Linear
    qweight_keys = _collect_qweight_keys(model_dir)
    dtype = (
        getattr(torch, str(config.torch_dtype))
        if isinstance(config.torch_dtype, str)
        else torch.bfloat16
    )
    _replace_linears_with_awq(model, qweight_keys, dtype=dtype, use_triton=use_triton)

    # 3) 流式将权重加载到目标 device
    print(f"[LOAD] Streaming weights into model on {device} ...")
    _load_awq_into_model(model, model_dir, device=device)

    model.to(device)
    return model


def run_chat_inference(paths: ModelPaths, prompt: str, max_new_tokens: int = 64, use_triton: bool = True) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading tokenizer from {paths.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(paths.model_dir, use_fast=False)

    print(f"[INFO] Building Qwen3 AWQ model from {paths.model_dir} on {device}")
    model = build_qwen3_awq_model(paths.model_dir, device=device, use_triton=use_triton)

    print(f"[INFO] Loading generation config")
    try:
        gen_config = GenerationConfig.from_pretrained(paths.model_dir)
        model.generation_config = gen_config
    except Exception as exc:  # pragma: no cover - optional
        print(f"[WARN] Failed to load GenerationConfig: {exc!r}")

    model.eval()

    messages = [
        {"role": "system", "content": "You are Qwen3-0.6B W4A8 quantized model."},
        {"role": "user", "content": prompt},
    ]

    print("[INFO] Applying chat_template ...")
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    print(f"[INFO] Starting generation (use_triton={use_triton})...")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    duration_ms = (t1 - t0) * 1000
    print(f"[INFO] Generation finished in {duration_ms:.2f} ms.")

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("===== MODEL OUTPUT =====")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3 W4A8 AWQ inference (chat_template)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to packed AWQ model directory (e.g. vllm_quant_model_packed)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，简单自我介绍一下。",
        help="User prompt for chat.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--use-triton",
        action="store_true",
        help="Enable Triton-based GEMM kernel for inference.",
    )
    parser.add_argument(
        "--no-triton",
        action="store_false",
        dest="use_triton",
        help="Disable Triton-based GEMM, use PyTorch fallback.",
    )
    parser.set_defaults(use_triton=True)  # Triton is on by default
    args = parser.parse_args()

    paths = ModelPaths(model_dir=args.model_dir)
    run_chat_inference(
        paths,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        use_triton=args.use_triton,  # Pass parsed arg
    )


if __name__ == "__main__":
    main()
