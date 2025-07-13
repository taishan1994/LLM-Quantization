# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import typing
import torch


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            print("layernorm存在bias")
            if linear.bias is None:
                print("线性层不存在bias")
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)

    W_norm = layernorm.weight.data
    layernorm.weight.data = torch.ones_like(W_norm)


def fuse_layer_norms(model):
    kwargs = {"model": model}

    # Embedding fusion
    # if hasattr(model.language_model.model, "embed_tokens"): # for llama-vision
    #     for W in [model.language_model.model.embed_tokens]:
    #         W_ = W.weight.data.double()
    #         W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    #     layers = [layer for layer in model.language_model.model.layers]

    # else:
    for W in [model.model.embed_tokens]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = [layer for layer in model.model.layers]

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
        if hasattr(layer, "self_attn"):
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
        elif hasattr(layer, "cross_attn"):
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.cross_attn.q_proj,
                ],
            )


        # W_norm = layer.post_attention_layernorm.weight.data
        # layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
        # W_norm = layer.input_layernorm.weight.data
        # layer.input_layernorm.weight.data = torch.ones_like(W_norm)

        
    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
    )

    # W_norm = model.model.norm.weight.data
    # model.model.norm.weight.data = torch.ones_like(W_norm)

def merge_rmsnorm_in_model(model):
    """
    将模型中所有 LlamaRMSNorm 层的参数合并到其后的线性层中。

    这个函数会遍历模型的所有 decoder layer，并将以下配对的参数进行合并：
    1. input_layernorm -> q_proj, k_proj, v_proj
    2. post_attention_layernorm -> gate_proj, up_proj

    合并后，RMSNorm 层的 weight 会被设置为 1.0，使其在数学上成为一个纯粹的归一化层，
    而其缩放效果已经提前应用到了线性层的权重上。

    Args:
        model (nn.Module): Hugging Face Transformer 模型。
    """
    print("开始合并 RMSNorm 参数...")

    # 检查模型是否包含预期的 decoder layers 结构
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        print("警告: 模型结构不符合预期 (如 Llama, Mistral)。跳过合并。")
        return

    for layer in model.model.layers:
        # 1. 合并 Attention 块中的 RMSNorm
        # input_layernorm -> q_proj, k_proj, v_proj

        # 获取 RMSNorm 层的 gamma 参数 (weight)
        # 使用 .data 避免梯度追踪
        gamma_attn = layer.input_layernorm.weight.data

        # 获取线性层的权重
        q_weight = layer.self_attn.q_proj.weight.data
        k_weight = layer.self_attn.k_proj.weight.data
        v_weight = layer.self_attn.v_proj.weight.data

        # 执行合并：将 gamma 乘到线性层的权重上
        # PyTorch 的广播机制会自动处理 (out_features, in_features) * (in_features,)
        q_weight *= gamma_attn
        k_weight *= gamma_attn
        v_weight *= gamma_attn

        # 将 RMSNorm 的 gamma 参数重置为 1
        layer.input_layernorm.weight.data.fill_(1.0)

        # 2. 合并 MLP 块中的 RMSNorm
        # post_attention_layernorm -> gate_proj, up_proj

        gamma_mlp = layer.post_attention_layernorm.weight.data

        gate_weight = layer.mlp.gate_proj.weight.data
        up_weight = layer.mlp.up_proj.weight.data

        gate_weight *= gamma_mlp
        up_weight *= gamma_mlp

        layer.post_attention_layernorm.weight.data.fill_(1.0)

    # 还要对lm_head之间的model_norm进行合并
    model_norm_weight = model.model.norm.weight.data
    print(model_norm_weight.shape)
    print(model.lm_head.weight.data.shape)
    model.lm_head.weight.data = model.lm_head.weight.data * model_norm_weight
    model.model.norm.weight.data.fill_(1.0)
    print("RMSNorm 参数合并完成！")