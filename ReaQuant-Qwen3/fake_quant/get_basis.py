# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText, AutoConfig
from utils.data_utils import get_data
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as pos_emb_llama
from transformers.models.mllama.modeling_mllama import apply_rotary_pos_emb as pos_emb_mllama
# from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as pos_emb_qwen
from eval_utils.modeling_qwen2 import apply_rotary_pos_emb as pos_emb_qwen
from eval_utils.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb as pos_emb_qwen2_vl
from train_utils.optimizer import SGDG
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict
from utils import utils, data_utils, fuse_norm_utils, quant_utils
from tqdm import tqdm
import transformers
from utils.hadamard_utils import random_orthogonal_matrix
import time

log: Logger = get_logger("resq", "get_basis.log")


@torch.no_grad()
def get_outlier_rotations(model_args, training_args, ptq_args) -> None:

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=torch.float32,
        trust_remote_code=True

    )
    model.seqlen = training_args.model_max_length

    transformers.set_seed(ptq_args.seed)
    model.eval()

    # Rotate the weights
    fuse_norm_utils.fuse_layer_norms(model)

    utils.cleanup_memory(verbos=True)

    log.info("Model init completed for training {}".format(model))
    log.info("Start to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    seqlen = model.seqlen
    train_data = data_utils.get_data(
        seed=ptq_args.seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
        eval_mode=False,
        nsamples=ptq_args.nsamples,
        calib_dataset=ptq_args.calib_dataset,
    )
    nbatches = len(train_data)
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(utils.DEV)
    model.model.rotary_emb = model.model.rotary_emb.to(utils.DEV)
    # model.R1.weight.data.copy_() = model.R1.to(utils.DEV)

    layers[0] = layers[0].to(utils.DEV)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=utils.DEV,
    )
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nbatches):
        batch = train_data[i][0].to(utils.DEV)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()

    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches

    basis_dict = {}
    eval_dict = {}
    rotation_dict = {}
    attention_mask = cache["attention_mask"]

    hidden_dim = model.config.hidden_size
    # head_dim = hidden_dim // model.config.num_attention_heads
    head_dim = model.config.head_dim
    kv_heads = model.config.num_key_value_heads
    down_proj_blocksize = ptq_args.down_proj_blocksize
    nlayers = len(layers)
    residual_fraction = ptq_args.residual_fraction
    residual_length_hidden = int(residual_fraction * hidden_dim)
    residual_length_head = int(residual_fraction * head_dim)
    residual_length_down_proj = int(residual_fraction * down_proj_blocksize)
    rotation_granularity = ptq_args.rotation_granularity

    for i in tqdm(range(nlayers), desc="(Collecting Outliers) Layers"):
        layer = layers[i].to(utils.DEV)

        hooks = []

        input_up_proj = []
        output_vproj = []
        output_kpos = []
        input_qkv_proj = []

        def hook_fn_upproj(module, input, output):
            input_up_proj.append(input[0].cpu())

        def hook_fn_vproj(module, input, output):
            global out_vp
            out_vp = output
            output_vproj.append(output.cpu())

        def hook_fn_kproj(module, input, output):
            global output_kproj
            output_kproj = output

        def hook_fn_qproj(module, input, output):
            global output_qproj
            output_qproj = output
            input_qkv_proj.append(input[0].cpu())

        hooks.append(layer.mlp.up_proj.register_forward_hook(hook_fn_upproj))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_fn_vproj))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(hook_fn_kproj))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_fn_qproj))

        for j in range(nbatches):
            with torch.no_grad():
                # register forward hook for getting inputs to MLP

                # 1 sample at a time
                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]

                # rope cos, sin
                cos, sin = layer.self_attn.rotary_emb(out_vp, position_ids)

                # reshape to get key states per head
                key_states = output_kproj.view(
                    1,
                    seqlen,
                    layer.self_attn.num_key_value_heads,
                    layer.self_attn.head_dim,
                ).transpose(1, 2)
                # key_states = repeat_kv(key_states, layer.self_attn.num_key_value_groups)

                # reshape to get query states per head
                query_states = output_qproj.view(
                    1,
                    seqlen,
                    layer.self_attn.num_heads,
                    layer.self_attn.head_dim,
                ).transpose(1, 2)

                # apply rotary embedding
                query_states_pos, key_states_pos = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
                output_kpos.append(key_states_pos.cpu())

        for hook in hooks:
            hook.remove()
        # reshape to get value states per head
        value_states = torch.stack(output_vproj).view(
            -1,
            layer.self_attn.num_key_value_heads,
            layer.self_attn.head_dim,
        )

        input_qkv_proj = torch.stack(input_qkv_proj).view(-1, hidden_dim)
        input_up_proj = torch.stack(input_up_proj).view(-1, hidden_dim)
        output_kpos = torch.stack(output_kpos).view(
            -1,
            layer.self_attn.num_key_value_heads,
            layer.self_attn.head_dim,
        )

        layers[i] = layers[i].cpu()

        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # find outlier channels
        basis_dict["layer." + str(i) + ".mlp"] = torch.eye(
            hidden_dim, dtype=torch.float64
        )[torch.sort(input_up_proj.abs().sum(0), descending=False).indices.cpu()].t()
        basis_dict["layer." + str(i) + ".self_attn"] = (
            torch.eye(hidden_dim, dtype=torch.float64)[
                torch.sort(input_qkv_proj.abs().sum(0), descending=False).indices.cpu()
            ]
            .cpu()
            .t()
        )

        basis_v_proj = []
        basis_k_pos = []
        for j in range(layer.self_attn.num_key_value_heads):
            basis_v_proj.append(
                torch.eye(layer.self_attn.head_dim, dtype=torch.float64)[
                    torch.sort(value_states[:, j].abs().sum(0), descending=False)
                    .indices.cpu()
                    .t()
                ]
            )

            basis_k_pos.append(
                torch.eye(layer.self_attn.head_dim, dtype=torch.float64)[
                    torch.sort(output_kpos[:, j].abs().sum(0), descending=False)
                    .indices.cpu()
                    .t()
                ]
            )

        basis_dict["layer." + str(i) + ".self_attn.value"] = torch.stack(
            basis_v_proj
        ).cpu()
        basis_dict["layer." + str(i) + ".self_attn.key_pos"] = torch.stack(
            basis_k_pos
        ).cpu()

    torch.cuda.empty_cache()
    os.makedirs(model_args.output_rotation_path, exist_ok=True)
    path = os.path.join(
        model_args.output_rotation_path,
        "U-" + "outliers-" + model_args.input_model.split("/")[1] + ".bin",
    )

    torch.save(
        basis_dict,
        path,
    )


@torch.no_grad()
def get_basis(model_args, training_args, ptq_args) -> None:
    vision = False
    if 'vl' in model_args.input_model.lower() or 'vision' in model_args.input_model.lower():
        tokenizer = AutoProcessor.from_pretrained(model_args.input_model)
        model = AutoModelForImageTextToText.from_pretrained(model_args.input_model, torch_dtype=torch.float32)
        vision = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            # cache_dir=training_args.cache_dir,
            # model_max_length=training_args.model_max_length,
            # padding_side="right",
            # use_fast=True,
            # add_eos_token=False,
            # add_bos_token=False,
        )
        config = AutoConfig.from_pretrained(model_args.input_model)
        config._attn_implementation = "eager"

        # from eval_utils.modeling_qwen3  import Qwen3ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            torch_dtype=torch.float32,
            config=config
        )
    model.seqlen = training_args.model_max_length
    transformers.set_seed(ptq_args.seed)
    model.eval()

    if hasattr(model, "language_model"): #for vision llama
        llm = model.language_model
    else:
        llm = model

    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    if process_word_embeddings:
        llm.lm_head.weight.data = llm.model.embed_tokens.weight.data.clone()

    # Fuse Norm
    # fuse_norm_utils.fuse_layer_norms(llm)
    # ==========================
    # message = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
    # input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)
    #
    # direct_output = model.generate(input_ids, max_new_tokens=256, do_sample=False, temperature=0.01)
    # direct_text = tokenizer.decode(direct_output[0])
    # print(direct_text)
    # ==========================

    # fuse_norm_utils.merge_rmsnorm_in_model(llm)

    # for k,v in llm.named_parameters():
    #     if "norm" in k:
    #         print(k, v)

    # ==========================
    message = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)

    direct_output = llm.generate(input_ids, max_new_tokens=256, do_sample=False, temperature=0.01)
    direct_text = tokenizer.decode(direct_output[0])
    print(direct_text)
    # ==========================

    utils.cleanup_memory(verbos=True)

    log.info("Model init completed {}".format(model))

    llm.config.use_cache = False
    seqlen = model.seqlen

    train_data = data_utils.get_data(
        seed=ptq_args.seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
        eval_mode=False,
        nsamples=ptq_args.nsamples,
        calib_dataset=ptq_args.calib_dataset,
        vision=vision,
    )
    nbatches = len(train_data)
    layers = llm.model.layers
    llm.model.embed_tokens = llm.model.embed_tokens.to(utils.DEV)
    if hasattr(llm.model, "rotary_emb"):
        llm.model.rotary_emb = llm.model.rotary_emb.to(utils.DEV)

    layers[0] = layers[0].to(utils.DEV)

    dtype = next(iter(llm.parameters())).dtype

    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, model.seqlen, llm.config.hidden_size),
        dtype=dtype,
        device=utils.DEV,
    )
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            if "position_embeddings" in kwargs:
                cache["position_embeddings"] = kwargs["position_embeddings"]
            else:
                cache["position_embeddings"] = None
            cache["cross_attention_states"] = kwargs["cross_attention_states"] if "cross_attention_states" in kwargs.keys() else None
            cache["cross_attention_mask"] = kwargs["cross_attention_mask"] if "cross_attention_mask" in kwargs.keys() else None
            cache["full_text_row_masked_out_mask"] = kwargs["full_text_row_masked_out_mask"] if "full_text_row_masked_out_mask" in kwargs.keys() else None
            cache["cache_position"] = kwargs["cache_position"] if "cache_position" in kwargs.keys() else None
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nbatches):
        batch = train_data[i][0].to(utils.DEV)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    llm.model.embed_tokens = llm.model.embed_tokens.cpu()
    if hasattr(llm.model, "rotary_emb"):
        llm.model.rotary_emb = llm.model.rotary_emb.cpu()
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches

    basis_dict = {}
    eval_dict = {}
    rotation_dict = {}
    attention_mask = cache["attention_mask"]

    hidden_dim = llm.config.hidden_size
    num_attention_heads = llm.config.num_attention_heads
    # head_dim = hidden_dim // num_attention_heads
    head_dim = llm.config.head_dim
    kv_heads = llm.config.num_key_value_heads
    down_proj_blocksize = ptq_args.down_proj_blocksize
    nlayers = len(layers)
    
    low_frac, high_frac = ptq_args.low_fraction, ptq_args.high_fraction
    low_length_hidden, high_length_hidden = int(low_frac * hidden_dim), int(high_frac * hidden_dim)
    low_length_head, high_length_head = int(low_frac * head_dim), int(high_frac * head_dim)
    low_length_down_proj, high_length_down_proj = int(low_frac * down_proj_blocksize), int(high_frac * down_proj_blocksize)

    rotation_granularity = ptq_args.rotation_granularity

    sparse_fraction = ptq_args.sparse_fraction
    sparse_length_hidden = int(sparse_fraction * low_length_hidden)
    sparse_length_head = int(sparse_fraction * low_length_head)

    if '70b' in model_args.input_model.lower() or "72b" in model_args.input_model.lower() :
        cov_device = 'cpu'
    else:
        cov_device = utils.DEV

    # initialize covariance matrices
    H_attn = torch.zeros((len(layers), hidden_dim, hidden_dim), device=cov_device)
    H_mlp = torch.zeros((len(layers), hidden_dim, hidden_dim), device=cov_device)
    H_down_proj = torch.zeros(
        (
            nlayers,
            down_proj_blocksize,
            down_proj_blocksize,
        ),
        device=cov_device,
    )  # block down_proj with certain dimension
    H_value = torch.zeros(
        (
            nlayers,
            llm.config.num_key_value_heads,
            head_dim,
            head_dim,
        ),
        device=cov_device,
    )
    H_key_pos = torch.zeros(
        (
            nlayers,
            kv_heads,
            head_dim,
            head_dim,
        ),
        device=cov_device,
    )
    R1_1 = random_orthogonal_matrix(
        hidden_dim - high_length_hidden - low_length_hidden - sparse_length_hidden, "cuda"
    )

    rotation_dict["R1_1"] = R1_1
    rotation_dict["R1_2"] = random_orthogonal_matrix(high_length_hidden, "cuda")
    if low_length_hidden != 0 :
        R1_0 = random_orthogonal_matrix(low_length_hidden - sparse_length_hidden, "cuda")
        if sparse_length_hidden > 0:
            zeros = torch.zeros(
                (sparse_length_hidden, sparse_length_hidden),
                device=R1_1.device,
                dtype=R1_1.dtype,
            )
            R1_0 = torch.block_diag(zeros, R1_0)
        rotation_dict["R1_0"] = R1_0
    else:
        rotation_dict["R1_0"] = None


    R2_1 = random_orthogonal_matrix(
        head_dim - high_length_head - low_length_head,
        "cuda",
    )
    rotation_dict["R2_1"] = R2_1
    rotation_dict["R2_2"] = random_orthogonal_matrix(high_length_head, "cuda")
    if low_length_head != 0 :
        R2_0 = random_orthogonal_matrix(low_length_head - sparse_length_head, "cuda")
        if sparse_length_hidden > 0:
            zeros = torch.zeros(
                (sparse_length_head, sparse_length_head),
                device=R2_1.device,
                dtype=R2_1.dtype,
            )
            R2_0 = torch.block_diag(zeros, R2_0)
        rotation_dict["R2_0"] = R2_0
    else:
        rotation_dict["R2_0"] = None


    os.makedirs(model_args.output_rotation_path, exist_ok=True)
    rotation_path = os.path.join(
        model_args.output_rotation_path,
        "R-high-"
        + str(high_frac)
        + "-low-"
        +str(low_frac)
        + "-sparse-"
        + str(sparse_fraction)
        + "-"
        + model_args.input_model.split("/")[1]
        + ".bin",
    )
    if not os.path.exists(rotation_path):
        torch.save(
            rotation_dict,
            rotation_path,
        )
    basis_path = os.path.join(
        model_args.output_rotation_path,
        "U-"
        + str(ptq_args.calib_dataset)
        + "-"
        + str(ptq_args.nsamples)
        + "-"
        + model_args.input_model.split("/")[1]
        + ".bin",
    )

    eval_path = os.path.join(
        model_args.output_rotation_path,
        "E-"
        + str(ptq_args.calib_dataset)
        + "-"
        + str(ptq_args.nsamples)
        + "-"
        + model_args.input_model.split("/")[1]
        + ".bin",
    )
    if not os.path.exists(basis_path):
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        for i in tqdm(range(nlayers), desc="(Collecting Cov matrices) Layers"):
            layer = layers[i].to(utils.DEV)

            hooks = []

            def hook_fn_upproj(module, input, output):
                global input_up_proj
                input_up_proj = input[0]

            def hook_fn_vproj(module, input, output):
                global output_vproj
                output_vproj = output

            # def hook_fn_kproj(module, input, output):
            #     global output_kproj
            #     output_kproj = output
            #
            # def hook_fn_qproj(module, input, output):
            #     global output_qproj, input_qkv_proj
            #     output_qproj = output
            #     input_qkv_proj = input[0]

            # NEW: Hook for q_proj's INPUT to get hidden_states for H_attn
            def hook_fn_q_proj_input(module, input, output):
                global input_qkv_proj
                input_qkv_proj = input[0]

            # NEW: Hook for q_norm's OUTPUT to get the normalized Q states
            def hook_fn_q_norm_output(module, input, output):
                global output_qproj
                # The output here is (bsz, q_len, num_heads, head_dim)
                output_qproj = output

            # NEW: Hook for k_norm's OUTPUT to get the normalized K states
            def hook_fn_k_norm_output(module, input, output):
                global output_kproj
                # The output here is (bsz, q_len, num_key_value_heads, head_dim)
                output_kproj = output

            def hook_fn_downproj(module, input, output):
                global input_down_proj
                input_down_proj = input[0]

            hooks.append(layer.mlp.up_proj.register_forward_hook(hook_fn_upproj))
            hooks.append(layer.mlp.down_proj.register_forward_hook(hook_fn_downproj))
            if hasattr(layer, "self_attn"):
                # hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_fn_vproj))
                # hooks.append(layer.self_attn.k_proj.register_forward_hook(hook_fn_kproj))
                # hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_fn_qproj))

                # Hook V-proj output (no norm on V)
                hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_fn_vproj))
                # Hook q_proj input (to get hidden_states)
                hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_fn_q_proj_input))
                # Hook q_norm output (to get normalized Q)
                hooks.append(layer.self_attn.q_norm.register_forward_hook(hook_fn_q_norm_output))
                # Hook k_norm output (to get normalized K)
                hooks.append(layer.self_attn.k_norm.register_forward_hook(hook_fn_k_norm_output))
            # elif hasattr(layer, "cross_attn"):
            #     hooks.append(layer.cross_attn.v_proj.register_forward_hook(hook_fn_vproj))
            #     hooks.append(layer.cross_attn.k_proj.register_forward_hook(hook_fn_kproj))
            #     hooks.append(layer.cross_attn.q_proj.register_forward_hook(hook_fn_qproj))

            for j in range(nbatches):
                with torch.no_grad():
                    # register forward hook for getting inputs to MLP

                    # 1 sample at a time
                    if hasattr(layer, "cross_attn"):
                        outs[j] = layer(
                            inps[j],
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            cross_attention_states = cache["cross_attention_states"],
                            cross_attention_mask = cache["cross_attention_mask"],
                            full_text_row_masked_out_mask = cache["full_text_row_masked_out_mask"],
                            cache_position = cache["cache_position"],
                        )[0]
                    else:
                        outs[j] = layer(
                            inps[j],
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )[0]
                    
                    # reshape to get value states per head
                    value_states = output_vproj.view(
                        1,
                        seqlen,
                        kv_heads,
                        head_dim,
                    ).transpose(1, 2)
                    # rope cos, sin
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                    else:
                        if "llama" in model_args.input_model.lower():
                            cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)
                        elif "qwen" in model_args.input_model.lower():
                            cos, sin = layer.self_attn.rotary_emb(value_states, seqlen)
                    # reshape to get key states per head
                    key_states = output_kproj.transpose(1, 2)
                    query_states = output_qproj.transpose(1, 2)
                    # key_states = output_kproj.view(
                    #     1,
                    #     seqlen,
                    #     kv_heads,
                    #     head_dim,
                    # ).transpose(1, 2)
                    # # key_states = repeat_kv(key_states, layer.self_attn.num_key_value_groups)
                    #
                    # # reshape to get query states per head
                    # query_states = output_qproj.view(
                    #     1,
                    #     seqlen,
                    #     num_attention_heads,
                    #     head_dim,
                    # ).transpose(1, 2)

                    # apply rotary embedding
                    if hasattr(layer, "self_attn"):
                        if "llama" in model_args.input_model.lower() and "vision" not in model_args.input_model.lower():
                            query_states_pos, key_states_pos = pos_emb_llama(
                                query_states, key_states, cos, sin
                            )
                        elif "llama" in model_args.input_model.lower() and "vision" in model_args.input_model.lower():
                            query_states_pos, key_states_pos = pos_emb_mllama(
                                query_states, key_states, cos, sin
                            )
                        elif "qwen" in model_args.input_model.lower() and "vl" not in model_args.input_model.lower():
                            query_states_pos, key_states_pos = pos_emb_qwen(
                                query_states, key_states, cos, sin, position_ids
                            )
                        elif "qwen" in model_args.input_model.lower() and "vl" in model_args.input_model.lower():
                            query_states_pos, key_states_pos = pos_emb_qwen2_vl(
                                query_states, key_states, cos, sin, layer.self_attn.rope_scaling["mrope_section"]
                            )
                    else:
                        #cross attn does not have rotary embedding
                        key_states_pos = key_states.clone()
                        query_states_pos = query_states.clone()
                    
                    # calculate covariance
                    H_mlp[i] += torch.sum(
                        input_up_proj.double().mT @ input_up_proj.double(), dim=0
                    ).to(cov_device)  # shape : [hidden_dim, hidden_dim]

                    H_attn[i] += torch.sum(
                        input_qkv_proj.double().mT @ input_qkv_proj.double(), dim=0
                    ).to(cov_device)  # shape : [hidden_dim, hidden_dim]
                    H_value[i] += torch.sum(
                        value_states.double().mT @ value_states.double(), dim=0
                    ).to(cov_device)  # shape : [num_heads, head_dim, head_dim]

                    H_key_pos[i] += torch.sum(
                        key_states_pos.double().mT @ key_states_pos.double(), dim=(0)
                    ).to(cov_device)  # shape : [num_kv_heads, head_dim, head_dim]

                    H_down_proj[i] += torch.sum(
                        input_down_proj.view(
                            input_down_proj.shape[0], -1, ptq_args.down_proj_blocksize
                        )
                        .double()
                        .mT
                        @ input_down_proj.view(
                            input_down_proj.shape[0], -1, ptq_args.down_proj_blocksize
                        ).double(),
                        dim=(0),
                    ).to(cov_device)  # shape : [1024, 1024]
            for hook in hooks:
                hook.remove()

            layers[i] = layers[i].cpu()

            torch.cuda.empty_cache()

            inps, outs = outs, inps

        if "per_layer" in rotation_granularity.lower():
            for i in tqdm(range(nlayers), desc="(Getting Basis) Layers"):

                # eigen decomposition of attn
                eval_attn, evec_attn = perform_eigen_decomp(
                    H_attn[i] / (nbatches * seqlen)
                )

                # eigen decomposition of up proj
                eval_mlp, evec_mlp = perform_eigen_decomp(
                    H_mlp[i] / (nbatches * seqlen)
                )

                # eigen decomposition of down proj
                eval_down_proj, evec_down_proj = perform_eigen_decomp(
                    H_down_proj[i] / (nbatches * seqlen)
                )

                # eigen decomposition of value states
                eval_value, evec_value = perform_eigen_decomp(
                    H_value[i] / (seqlen * nbatches),
                    per_head=True,
                    num_heads=H_value.shape[0],
                )

                # eigen decomposition of key states after rope embedding
                eval_k_pos, evec_k_pos = perform_eigen_decomp(
                    H_key_pos[i] / (seqlen * nbatches),
                    per_head=True,
                    num_heads=H_key_pos.shape[0],
                )

                print(
                    "up proj:",
                    (eval_mlp[0:100].sum() / eval_mlp.sum()).item(),
                    "down proj:",
                    (eval_down_proj[0:100].sum() / eval_down_proj.sum()).item(),
                    ", hidden_attn:",
                    (eval_attn[0:100].sum() / eval_attn.sum()).item(),
                    ", v_proj:",
                    (eval_value[:, :32].sum(1) / eval_value.sum(1)).mean().item(),
                    # ", kq_proj:",
                    # (eval_kq[:, :32].sum(1) / eval_kq.sum(1)).mean().item(),
                    ", k_proj_pos:",
                    (eval_k_pos[:, :32].sum(1) / eval_k_pos.sum(1)).mean().item(),
                )

                basis_dict["config"] = "per_layer_rotation"
                basis_dict["layer." + str(i) + ".mlp"] = evec_mlp.cpu()
                basis_dict["layer." + str(i) + ".mlp.down_proj"] = evec_down_proj.cpu()
                basis_dict["layer." + str(i) + ".self_attn"] = evec_attn.cpu()
                basis_dict["layer." + str(i) + ".self_attn.value"] = evec_value.cpu()
                basis_dict["layer." + str(i) + ".self_attn.key_pos"] = evec_k_pos.cpu()

        elif "full_shared" in rotation_granularity.lower():
            # eigen decomposition of attn and mlp
            eval_attn_mlp, evec_attn_mlp = perform_eigen_decomp(
                (H_attn.sum(0) + H_mlp.sum(0)) / (2 * nbatches * nlayers * seqlen)
            )

            basis_dict["config"] = "full_shared_rotation"
            basis_dict["attn_mlp"] = evec_attn_mlp.cpu()

            eval_dict["config"] = "full_shared_rotation"
            eval_dict["attn_mlp"] = eval_attn_mlp.cpu()

            for i in tqdm(range(nlayers), desc="(Getting Rotations) Layers"):
                # eigen decomposition of down proj
                eval_down_proj, evec_down_proj = perform_eigen_decomp(
                    H_down_proj[i] / (nbatches * seqlen)
                )
                # eigen decomposition of value states
                eval_value, evec_value = perform_eigen_decomp(
                    (H_value[i] / (seqlen * nbatches)), per_head=True, num_heads=kv_heads
                )
                # eigen decomposition of key states after rope embedding
                eval_k_pos, evec_k_pos = perform_eigen_decomp(
                    H_key_pos[i].sum(0) / (kv_heads * nbatches * seqlen),
                )

                basis_dict["layer." + str(i) + ".self_attn.value"] = evec_value.cpu()
                basis_dict["layer." + str(i) + ".self_attn.key_pos"] = evec_k_pos.cpu()
                basis_dict["layer." + str(i) + ".mlp.down_proj"] = evec_down_proj.cpu()

                eval_dict["layer." + str(i) + ".self_attn.value"] = eval_value.cpu()
                eval_dict["layer." + str(i) + ".self_attn.key_pos"] = eval_k_pos.cpu()
                eval_dict["layer." + str(i) + ".mlp.down_proj"] = eval_down_proj.cpu()

        elif "one_per_decoder" in rotation_granularity.lower():

            for i in tqdm(range(nlayers), desc="(Getting Basis) Layers"):
                # eigen decomposition of attn and mlp
                eval_attn_mlp, evec_attn_mlp = perform_eigen_decomp(
                    (H_attn[i] + H_mlp[i]) / (2 * nbatches * seqlen)
                )
                # eigen decomposition of down proj
                eval_down_proj, evec_down_proj = perform_eigen_decomp(
                    H_down_proj[i] / (nbatches * seqlen)
                )

                # eigen decomposition of value states
                eval_value, evec_value = perform_eigen_decomp(
                    H_value[i] / (seqlen * nbatches), per_head=True, num_heads=kv_heads
                )

                # eigen decomposition of key states after rope embedding
                eval_k_pos, evec_k_pos = perform_eigen_decomp(
                    H_key_pos[i] / (seqlen * nbatches),
                    per_head=True,
                    num_heads=kv_heads,
                )

                print(
                    "down proj:",
                    (eval_down_proj[0:100].sum() / eval_down_proj.sum()).item(),
                    ", attn_mlp:",
                    (eval_attn_mlp[0:100].sum() / eval_attn_mlp.sum()).item(),
                    ", v_proj:",
                    (eval_value[:, :32].sum(1) / eval_value.sum(1)).mean().item(),
                    # ", kq_proj:",
                    # (eval_kq[:, :32].sum(1) / eval_kq.sum(1)).mean().item(),
                    ", k_proj_pos:",
                    (eval_k_pos[:, :32].sum(1) / eval_k_pos.sum(1)).mean().item(),
                )

                basis_dict["config"] = "one_per_decoder"
                basis_dict["layer." + str(i) + ".mlp.down_proj"] = evec_down_proj.cpu()
                basis_dict["layer." + str(i) + ".self_attn_mlp"] = evec_attn_mlp.cpu()
                basis_dict["layer." + str(i) + ".self_attn.value"] = evec_value.cpu()
                basis_dict["layer." + str(i) + ".self_attn.key_pos"] = evec_k_pos.cpu()

        torch.cuda.empty_cache()

        torch.save(
            basis_dict,
            basis_path,
        )

        torch.save(
            eval_dict,
            eval_path,
        )
    else:
        print(f"Basis rotations already exist at {basis_path}")


def perform_eigen_decomp(Cov_matrix, per_head=False, num_heads=0):
    # performs eigen decomposition and returns
    # the sorted eigen values and eigen vectors
    Cov_matrix = Cov_matrix.to(utils.DEV)
    if per_head:
        assert num_heads != 0  # cannot use per head and not pass num_heads
        eval = []
        evec = []
        for hd in range(num_heads):
            H = Cov_matrix[hd]
            damp = 0.01 * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[-1]).to(device=H.device)
            H[diag, diag] = H[diag, diag] + damp
            X = torch.linalg.eigh(H.to(torch.float64))
            index = torch.argsort(X[0])
            eval.append(X[0][index])
            evec.append(X[1][:, index])
        eval = torch.stack(eval)
        evec = torch.stack(evec)
    else:
        H = Cov_matrix
        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[-1]).to(device=H.device)
        H[diag, diag] = H[diag, diag] + damp
        X = torch.linalg.eigh(H.to(torch.float64))
        index = torch.argsort(X[0])
        eval = X[0][index]
        evec = X[1][:, index]

    return eval, evec


if __name__ == "__main__":
    model_args, training_args, ptq_args = process_args_ptq()
    if ptq_args.rotate_mode == "quik":
        get_outlier_rotations(model_args, training_args, ptq_args)
    else:
        get_basis(model_args, training_args, ptq_args)
