# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import transformers

from eval_utils import rotation_utils
from train_utils import rtn_utils
import torch
from utils import fuse_norm_utils, hadamard_utils, quant_utils, utils
from utils.hadamard_utils import (
    random_orthogonal_matrix,
)

def prepare_model(args, model):
    low_frac, high_frac = args.low_fraction, args.high_fraction
    model_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model_dim // num_heads
    low_length_head, high_length_head = int(low_frac * head_dim), int(high_frac * head_dim)

    transformers.set_seed(args.seed)
    model.eval()
    assert args.rotate_mode == "resq"

    # Rotate the weights
    fuse_norm_utils.fuse_layer_norms(model)
    rotation_utils.fuse_basis_to_model(model, args)
    utils.cleanup_memory(verbos=True)
    quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had

    if args.w_bits < 16:
        quantizers = rtn_utils.rtn_fwrd(model, "cuda", args)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads
            mlp_dim = model.config.intermediate_size
            v_groupsize = head_dim
            high_bits_fraction = args.high_fraction
            high_bits_length = int(high_bits_fraction * model_dim)
            low_bits_fraction = args.low_fraction
            low_bits_length = int(low_bits_fraction * model_dim)

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                # per group residual
                v_high_bits_length = int(v_groupsize * high_bits_fraction)
                v_low_bits_length = int(v_groupsize * low_bits_fraction)

                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                    high_bits_length=v_high_bits_length,
                    high_bits=args.high_bits,
                    low_bits_length=v_low_bits_length,
                    low_bits=args.low_bits,
                )
            if "o_proj" in name:
                layer_groupsize = head_dim
                # per group residual
                high_bits_length = int(v_groupsize * high_bits_fraction)
                low_bits_length = int(v_groupsize * low_bits_fraction)

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16
                high_bits_length = 0
                low_bits_length = 0

            if "basis_change" in name:
                layer_input_bits = 8  #####
                high_bits_length = 0
                low_bits_length = 0
            
            if "visual" in name: ###### Qwen-2-VL (vision part is not quantized)
                layer_input_bits = 16
                high_bits_length = 0
                low_bits_length = 0

            if "down_proj" in name:  # Set the down_proj precision
                high_bits_length = 0
                low_bits_length = 0
                # layer_input_bits = 4

                if args.int8_down_proj:
                    layer_input_bits = 8

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
                high_bits_length=high_bits_length,
                high_bits=args.high_bits,
                low_bits_length=low_bits_length,
                low_bits=args.low_bits,
            )

    R_dict = {}

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            if hasattr(model, "visual"):
                rope_function_name = "apply_multimodal_rotary_pos_emb"
            else:
                rope_function_name = "apply_rotary_pos_emb"
            
            layers = model.model.layers
            U_cpk = torch.load(args.optimized_basis_path)
            # residual_length_k = int(args.residual_fraction * head_dim)
            high_bits_length = int(args.high_fraction * head_dim)
            low_bits_length = int(args.low_fraction * head_dim)

            k_quant_config = {
                "k_bits": args.k_bits,
                "k_bits_high": args.high_bits,
                "k_bits_low": args.low_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
                "high_bits_length": high_bits_length,
                "low_bits_length": low_bits_length,
            }

            RK_1 = random_orthogonal_matrix(
                head_dim - high_length_head - low_length_head,
                "cuda",
            )
            RK_2 = random_orthogonal_matrix(high_length_head, "cuda")
            if low_length_head != 0 :
                RK_0 = random_orthogonal_matrix(low_length_head, "cuda")
            else:
                RK_0 = None
            R_dict["R2_1"] = RK_1
            R_dict["R2_2"] = RK_2
            RK = torch.block_diag(RK_1, RK_2)
            R_dict["R2_0"] = RK_0
            if RK_0 is not None:
                RK = torch.block_diag(RK_0, RK)
            for idx, layer in enumerate(layers):
                k_rotation = U_cpk[f"layer.{idx}.self_attn.key_pos"].cuda()
                k_rotation = torch.matmul(k_rotation, RK)
                quantizer = quant_utils.WeightQuantizer()
                quantizer.configure(8)
                quantizer.find_params(k_rotation)
                k_rotation = quantizer.quantize(k_rotation)
                k_had = False
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    k_rotation=k_rotation,
                    k_had=k_had,
                    **k_quant_config,
                )

    return model, R_dict
