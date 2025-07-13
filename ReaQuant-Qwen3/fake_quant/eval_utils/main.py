# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import transformers

from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils
from utils.hadamard_utils import (
    random_orthogonal_matrix,
)


def ptq_model(args, model, model_args=None, tokenizer=None):
    transformers.set_seed(args.seed)
    model.eval()
    # Rotate the weights
    if not args.rotate_mode == "none":
        # fuse_norm_utils.fuse_layer_norms(model)
        # model.to("cuda:0")
        # inps = [{"role": "user", "content": "你是谁？"}]
        # inps = tokenizer.apply_chat_template(inps, tokenize=False, add_generation_prompt=True)
        # print(inps)
        # inps = tokenizer.encode(inps, return_tensors="pt").to(model.device)
        # outputs = model.generate(inps, max_new_tokens=256, temperature=0.01)
        # print(tokenizer.decode(outputs[0]))
        # # ================================
        # ==========================
        # tokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path=model_args.input_model,
        # )
        # print("fuse_basis_to_model之前")
        # message = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
        # input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)
        #
        # direct_output = model.generate(input_ids, max_new_tokens=256, do_sample=False, temperature=0.01)
        # direct_text = tokenizer.decode(direct_output[0])
        # print(direct_text)
        # ==========================

        if args.rotate_mode == "resq" or args.rotate_mode == "quik":
            rotation_utils.fuse_basis_to_model(model, args)
        else:
            rotation_utils.rotate_model(model, args)

        if args.rotate_mode == "resq" or args.rotate_mode == "quik":
            rotation_utils.rearrange_columns(model, args, False)

        utils.cleanup_memory(verbos=True)
        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        # 这里加入在线旋转的hadamard
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had

    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            # quantizers = gptq_utils.lwc_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)
            tokenizer.save_pretrained(args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )
        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            # head_dim = model_dim // num_heads
            head_dim = model.config.head_dim
            mlp_dim = model.config.intermediate_size
            v_groupsize = head_dim
            if args.rotate_mode == "resq" or args.rotate_mode == "quik":
                high_bits_fraction = args.high_fraction
                high_bits_length = int(high_bits_fraction * model_dim)
                low_bits_fraction = args.low_fraction
                low_bits_length = int(low_bits_fraction * model_dim)

            else:
                high_bits_length = 0
                low_bits_length = 0

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                if args.rotate_mode == "resq" or args.rotate_mode == "quik":
                    # per group residual
                    v_high_bits_length = int(v_groupsize * high_bits_fraction)
                    v_low_bits_length = int(v_groupsize * low_bits_fraction)
                else:
                    v_high_bits_length = 0
                    v_low_bits_length = 0

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
                if args.rotate_mode == "resq" or args.rotate_mode == "quik":
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

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            if hasattr(model, "visual"):
                rope_function_name = "apply_multimodal_rotary_pos_emb"
            else:
                rope_function_name = "apply_rotary_pos_emb"
            
            layers = model.model.layers
            if args.rotate_mode == "resq" or args.rotate_mode == "quik":
                U_cpk = torch.load(args.optimized_basis_path)
                # residual_length_k = int(args.residual_fraction * head_dim)
                high_bits_length = int(args.high_fraction * head_dim)
                low_bits_length = int(args.low_fraction * head_dim)
            else:
                # residual_length_k = 0
                high_bits_length = 0
                low_bits_length = 0

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

            for idx, layer in enumerate(layers):
                if args.rotate_mode == "resq" or args.rotate_mode == "quik":
                    R_dict = torch.load(args.optimized_rotation_path)
                    R2_1 = R_dict["R2_1"].cuda().to(torch.float64)
                    R2_2 = R_dict["R2_2"].cuda().to(torch.float64)
                    R2 = torch.block_diag(R2_1, R2_2)
                    R2_0 = R_dict["R2_0"]
                    if R2_0 is not None:
                        R2 = torch.block_diag(R2_0.cuda().to(torch.float64), R2)
                    k_rotation = U_cpk[f"layer.{idx}.self_attn.key_pos"].cuda()
                    k_rotation = torch.matmul(k_rotation, R2)
                    quantizer = quant_utils.WeightQuantizer()
                    quantizer.configure(8)
                    quantizer.find_params(k_rotation)
                    k_rotation = quantizer.quantize(k_rotation)
                    k_had = False
                else:
                    k_rotation = None
                    k_had = True
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    k_rotation=k_rotation,
                    k_had=k_had,
                    **k_quant_config,
                )


    return model
