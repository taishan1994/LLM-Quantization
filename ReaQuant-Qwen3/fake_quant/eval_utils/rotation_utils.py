# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm

from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
    matmul_hadU_cuda,
    get_hadK,
)
from utils.utils import HadamardTransform
from utils.utils import get_local_rank
import logging, os


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)
    if hasattr(model, "visual"):
        #for VLMs, also rotate the output of patch Merger layer
        W = model.visual.merger.mlp[-1]
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device="cuda", dtype=torch.float64)
            W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1, R4, no_had=False):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    if R1 is not None:
        W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if R4 is not None:
        if not no_had:
            device = W.weight.device
            W.weight.data = matmul_hadU_cuda(
                W.weight.data.float().cuda(),
                R4.shape[0] * torch.inverse(R4).t(),
                R4.shape[0],
            ).to(device=device, dtype=dtype)
        else:
            device = W.weight.device
            n = W.weight.data.shape[-1]
            K = R4.shape[0]
            W_ = W.weight.data.view(-1, n // K, K).to(torch.float64).cuda()
            W_ = torch.matmul(W_, R4).to(device=device, dtype=dtype)
            W.weight.data = W_.reshape(W.weight.data.shape)
    else:
        apply_exact_had_to_linear(
            W, had_dim=-1, output=False
        )  # apply exact (inverse) hadamard on the weights of mlp output

    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_basis_change(
    layer, R1_prev: torch.Tensor, R1_attn: torch.Tensor, R1_mlp: torch.Tensor
) -> None:
    # Rotate the head.
    W = layer.basis_change_1
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    # W_ = torch.eye(4096, device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1_attn.T, torch.matmul(W_, R1_prev)).to(
        device="cpu", dtype=dtype
    )

    W = layer.basis_change_2
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    # W_ = torch.eye(4096, device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1_mlp.T, torch.matmul(W_, R1_attn)).to(
        device="cpu", dtype=dtype
    )


def rotate_ov_proj(layer, head_num, head_dim, R2=None, per_head=False):
    # per head means that R2 is different for different heads. If per_head is False, R2 is shared across heads.
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(
        v_proj, had_dim=head_dim, output=True, R2=R2, per_head=per_head
    )
    apply_exact_had_to_linear(
        o_proj, had_dim=head_dim, output=False, R2=R2, per_head=per_head
    )


def create_orthogonal(rot):
    """
    helper function to convert any random matrix to orthogonal matrix. Leveraging Cayley transform
        1. Convert input A to a skew symmetric matrix by sk_sm = (A - A.t())/2
        2. Use cayley transform to convert skew symmetric matrix to orthogonal matrix orth = (I-A)((I+A)^-1)
    """
    dtype = rot.dtype
    device = rot.device
    sk_sm = 0.5 * (rot - rot.t())
    eye = torch.eye(rot.shape[0], device=device, dtype=dtype)
    p1 = eye - sk_sm
    p2 = torch.inverse((eye + sk_sm).to(torch.float32)).to(dtype)
    orth = torch.matmul(p1, p2)
    orth = orth / orth.norm(p=2, dim=0)

    return orth


@torch.inference_mode()
def rotate_model(model, args):
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = config.head_dim
    if args.rotate_mode == "quarot":
        R1 = get_orthogonal_matrix(model.config.hidden_size, "hadamard")
        R2 = get_orthogonal_matrix(head_dim, "hadamard")
    elif args.rotate_mode == "spinquant":
        if args.optimized_rotation_path is not None:
            R_cpk = args.optimized_rotation_path
            R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
        else:
            print("You have not provided the rotation path")
            raise ValueError

    R4 = None
    no_had = False

    rotate_embeddings(model, R1)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]

    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        if args.rotate_mode == "spinquant":
            if args.optimized_rotation_path is not None:
                key = f"model.layers.{idx}.self_attn.R2"
                R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
            else:
                print("You have not provided the rotation path")
                raise ValueError
        rotate_attention_inputs(layers[idx], R1)
        rotate_attention_output(layers[idx], R1)
        rotate_mlp_input(layers[idx], R1)
        rotate_mlp_output(layers[idx], R1, R4, no_had=no_had)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)
        if layers[idx].basis_change_1 is not None:
            rotate_basis_change(layers[idx], R1, R1, R1)

    rotate_head(model, R1)


def rearrange_o_proj(layer, high_bits_length, low_bits_length, head_dim, training):
    o_proj = layer.self_attn.o_proj

    in_dim = o_proj.weight.shape[-1]
    num_replicated_heads = in_dim // head_dim
    high_length_per_head = high_bits_length // num_replicated_heads
    low_length_per_head = low_bits_length // num_replicated_heads
    # rearrange dimensions
    chunk_starts = torch.arange(0, in_dim, head_dim)
    high_precision_columns = torch.arange(head_dim - high_length_per_head, head_dim, 1)
    columns_to_end = (chunk_starts.unsqueeze(1) + high_precision_columns).flatten()
    
    low_precision_columns = torch.arange(0, low_length_per_head, 1)
    columns_to_beginning = (chunk_starts.unsqueeze(1) + low_precision_columns).flatten()

    all_columns = torch.arange(in_dim)
    mask = torch.ones(in_dim, dtype=torch.bool)
    mask[columns_to_end] = False
    mask[columns_to_beginning] = False

    remaining_columns = all_columns[mask]
    new_column_order = torch.cat((columns_to_beginning, remaining_columns, columns_to_end))
    permutation_matrix = torch.eye(in_dim)[:, new_column_order]

    # rearrange weights of o_proj
    if not training:
        Wo = o_proj.weight.data
        o_proj.weight.data = Wo[:, new_column_order]

    # save new_column_order for rearranging attn_out on the fly
    if training:
        layer.self_attn.new_column_order = permutation_matrix
    else:
        layer.self_attn.new_column_order = new_column_order


def rearrange_down_proj(layer, residual_length, block_dim, training):
    down_proj = layer.mlp.down_proj

    in_dim = down_proj.weight.shape[-1]
    num_replicated_block = in_dim // block_dim
    residual_per_block = residual_length // num_replicated_block
    # rearrange dimensions
    chunk_starts = torch.arange(0, in_dim, block_dim)
    first_four = torch.arange(residual_per_block)
    columns_to_front = (chunk_starts.unsqueeze(1) + first_four).flatten()
    all_columns = torch.arange(in_dim)
    mask = torch.ones(in_dim, dtype=torch.bool)
    mask[columns_to_front] = False
    remaining_columns = all_columns[mask]
    new_column_order = torch.cat((columns_to_front, remaining_columns))
    permutation_matrix = torch.eye(in_dim)[:, new_column_order]

    # rearrange weights of o_proj
    if not training:
        Wd = down_proj.weight.data
        down_proj.weight.data = Wd[:, new_column_order]

    # save new_column_order for rearranging attn_out on the fly
    if training:
        layer.mlp.new_column_order = permutation_matrix
    else:
        layer.mlp.new_column_order = new_column_order


def fuse_basis_to_model(model, args):
    if args.rotate_mode == "quik":
        fuse_basis_per_layer(model, args)
    else:
        if "full_shared" in args.rotation_granularity.lower():
            fuse_basis_shared(model, args)
        elif "one_per_decoder" in args.rotation_granularity.lower():
            fuse_basis_one_per_decoder(model, args)
        else:
            fuse_basis_per_layer(model, args)


def fuse_basis_one_per_decoder(model, args):
    layers = [layer for layer in model.model.layers]
    dtype = model.dtype
    hidden_dim = model.config.hidden_size
    # add per layer basis change
    for idx in range(len(layers)):
        # this config does not have basis change 1
        layers[idx].basis_change_2 = torch.nn.Linear(
            hidden_dim, hidden_dim, bias=False, dtype=dtype
        )

    R_dict = torch.load(args.optimized_rotation_path)
    R1_1 = R_dict["R1_1"].cuda().to(torch.float64)
    R1_2 = R_dict["R1_2"].cuda().to(torch.float64)
    R2_1 = R_dict["R2_1"].cuda().to(torch.float64)
    R2_2 = R_dict["R2_2"].cuda().to(torch.float64)
    R1 = torch.block_diag(R1_1, R1_2)
    R2 = torch.block_diag(R2_1, R2_2)

    U_cpk = torch.load(args.optimized_basis_path)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = config.head_dim
    dev = model.model.embed_tokens.weight.device
    U_attn = torch.matmul(U_cpk["layer.0.self_attn_mlp"].cuda(), R1)

    rotate_embeddings(model, U_attn)
    rotate_head(model, R1)
    utils.cleanup_memory()

    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Fusing Basis")):
        key = f"layer.{idx}.self_attn_mlp"
        U_attn_mlp = torch.matmul(U_cpk[key].cuda(), R1)
        rotate_attention_inputs(layers[idx], U_attn_mlp)

        key = f"layer.{idx}.self_attn.value"
        U_value = torch.matmul(U_cpk[key].cuda(), R2)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=U_value, per_head=True)
        rotate_attention_output(layers[idx], U_attn_mlp)

        rotate_mlp_input(layers[idx], U_attn_mlp)

        # rotate MLP output for every layer except last layer
        if idx == model.config.num_hidden_layers - 1:
            rotate_mlp_output(layers[idx], R1=R1, R4=None)

            dtype = layer.basis_change_2.weight.dtype
            W_rot_change = torch.matmul(R1.T, U_attn_mlp).to(device="cpu", dtype=dtype)
            layer.basis_change_2.weight.data.copy_(W_rot_change)
        else:
            key = f"layer.{idx+1}.self_attn_mlp"
            U_attn_next = torch.matmul(U_cpk[key].cuda(), R1)
            rotate_mlp_output(layers[idx], R1=U_attn_next, R4=None)

            dtype = layer.basis_change_2.weight.dtype
            W_rot_change = torch.matmul(U_attn_next.T, U_attn_mlp).to(
                device="cpu", dtype=dtype
            )
            layer.basis_change_2.weight.data.copy_(W_rot_change)


def fuse_basis_shared(model, args):
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    high_bits_length = int(args.high_fraction * model_dim)
    head_dim = config.head_dim
    
    U_cpk = torch.load(args.optimized_basis_path)
    U_attn = U_cpk["attn_mlp"].cuda()

    if not args.train_rotations:
        R_dict = torch.load(args.optimized_rotation_path)
        R1_1 = R_dict["R1_1"].cuda().to(torch.float64)
        R1_2 = R_dict["R1_2"].cuda().to(torch.float64)

        assert (
            R1_2.shape[0] == high_bits_length
        )  # checking if rotation dimensions align with residual length
            
        R1 = torch.block_diag(R1_1, R1_2)
        R1_0 = R_dict["R1_0"]
        if R1_0 is not None:
            R1 = torch.block_diag(R1_0.cuda().to(torch.float64), R1)
        
        

        U_attn = torch.matmul(U_attn, R1)
    
    torch.distributed.barrier()
    
    rotate_embeddings(model, U_attn)
    rotate_head(model, U_attn)

    utils.cleanup_memory()

    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):

        rotate_attention_inputs(layers[idx], U_attn)

        key = f"layer.{idx}.self_attn.value"
        U_value = U_cpk[key].cuda()
        if not args.train_rotations:
            if 'trained' in args.optimized_rotation_path:
                print("here")
                R2_1 = R_dict[f"model.layers.{idx}.self_attn.R2_1"].cuda().to(torch.float64)
                R2_2 = R_dict[f"model.layers.{idx}.self_attn.R2_2"].cuda().to(torch.float64)
                R2 = torch.block_diag(R2_1, R2_2)
                if f"model.layers.{idx}.self_attn.R2_0" in R_dict.keys():
                    R2_0 = R_dict[f"model.layers.{idx}.self_attn.R2_0"].cuda().to(torch.float64)
                    R2 = torch.block_diag(R2_0, R2)
            else:
                R2_1 = R_dict["R2_1"].cuda().to(torch.float64)
                R2_2 = R_dict["R2_2"].cuda().to(torch.float64)
                R2 = torch.block_diag(R2_1, R2_2)
                R2_0 = R_dict["R2_0"]
                if R2_0 is not None:
                    R2 = torch.block_diag(R2_0.cuda().to(torch.float64), R2)
            
            
            U_value = torch.matmul(U_value, R2)

        # rotate_ov_proj(layers[idx], num_heads, head_dim, R2=U_value, per_head=True)

        rotate_attention_output(layers[idx], U_attn)

        rotate_mlp_input(layers[idx], U_attn)

        rotate_mlp_output(layers[idx], R1=U_attn, R4=None)


def fuse_basis_per_layer(model, args):
    layers = [layer for layer in model.model.layers]
    dtype = model.dtype
    hidden_dim = model.config.hidden_size
    # add per layer basis change
    for idx in range(len(layers)):
        layers[idx].basis_change_1 = torch.nn.Linear(
            hidden_dim, hidden_dim, bias=False, dtype=dtype
        )

        layers[idx].basis_change_2 = torch.nn.Linear(
            hidden_dim, hidden_dim, bias=False, dtype=dtype
        )

    U_cpk = torch.load(args.optimized_basis_path)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = config.head_dim
    dev = model.model.embed_tokens.weight.device
    if args.optimized_rotation_path is not None:
        R_dict = torch.load(args.optimized_rotation_path)
        R1_1 = R_dict["R1_1"].cuda().to(torch.float64)
        R1_2 = R_dict["R1_2"].cuda().to(torch.float64)
        R2_1 = R_dict["R2_1"].cuda().to(torch.float64)
        R2_2 = R_dict["R2_2"].cuda().to(torch.float64)
        R1 = torch.block_diag(R1_1, R1_2)
        R2 = torch.block_diag(R2_1, R2_2)
    else:
        logging.info("No random rotation path provided")
        R1 = torch.eye(hidden_dim).cuda().to(torch.float64)
        R2 = torch.eye(head_dim).cuda().to(torch.float64)
    U_attn = torch.matmul(U_cpk["layer.0.self_attn"].cuda(), R1)
    rotate_embeddings(model, U_attn)
    rotate_head(model, R1)
    utils.cleanup_memory()

    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Fusing Basis")):
        key = f"layer.{idx}.self_attn"
        U_attn = torch.matmul(U_cpk[key].cuda(), R1)
        rotate_attention_inputs(layers[idx], U_attn)

        key = f"layer.{idx}.self_attn.value"
        U_value = torch.matmul(U_cpk[key].cuda(), R2)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=U_value, per_head=True)

        key = f"layer.{idx}.mlp"
        U_mlp = torch.matmul(U_cpk[key].cuda(), R1)
        rotate_attention_output(layers[idx], U_mlp)
        rotate_mlp_input(layers[idx], U_mlp)
        dtype = layer.basis_change_1.weight.dtype
        layer.basis_change_1.weight.data.copy_(
            torch.matmul(U_mlp.T, U_attn).to(device="cpu", dtype=dtype)
        )
        # rotate MLP output for every layer except last layer
        if idx == model.config.num_hidden_layers - 1:
            rotate_mlp_output(layers[idx], R1=R1, R4=None)

            dtype = layer.basis_change_2.weight.dtype
            W_rot_change = torch.matmul(R1.T, U_mlp).to(device="cpu", dtype=dtype)
            layer.basis_change_2.weight.data.copy_(W_rot_change)
        else:
            key = f"layer.{idx+1}.self_attn"
            U_attn_next = torch.matmul(U_cpk[key].cuda(), R1)
            rotate_mlp_output(layers[idx], R1=U_attn_next, R4=None)

            dtype = layer.basis_change_2.weight.dtype
            W_rot_change = torch.matmul(U_attn_next.T, U_mlp).to(
                device="cpu", dtype=dtype
            )
            layer.basis_change_2.weight.data.copy_(W_rot_change)


def rearrange_columns(model, args, training):

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = config.head_dim
    mlp_dim = model.config.intermediate_size
    high_bits_length = int(args.high_fraction * model_dim)
    low_bits_length = int(args.low_fraction * model_dim)
    torch.distributed.barrier()
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(
        tqdm.tqdm(layers, unit="layer", desc="Rearranging O_proj rows")
    ):
        rearrange_o_proj(layers[idx], high_bits_length, low_bits_length, head_dim, training)

        # residual_length = int(args.residual_fraction * mlp_dim)
        # rearrange_down_proj(
        #     layers[idx], residual_length, args.down_proj_blocksize, training
        # )


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, k_rotation, k_had, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = config.head_dim
        self.num_kv_groups = num_heads // config.num_key_value_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_residual_quantizer = quant_utils.ActQuantizer()
        self.pre_rotation_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        self.k_rotation = k_rotation
        self.k_had = k_had  # if you want to perform hadamard rotation for key and query
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_bits_high = kwargs["k_bits_high"]
            self.k_bits_low = kwargs["k_bits_low"]

            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]

            self.high_bits_length = kwargs["high_bits_length"]
            self.low_bits_length = kwargs["low_bits_length"]

            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
                high_bits_length=self.high_bits_length,
                high_bits=self.k_bits_high,
                low_bits_length=self.low_bits_length,
                low_bits=self.k_bits_low,
            )
            self.pre_rotation_quantizer.configure(
                bits=8,
                groupsize=-1,
                sym=self.k_sym,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        if self.k_had:
            q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
            k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        else:
            self.pre_rotation_quantizer.find_params(q)
            q = self.pre_rotation_quantizer(q).to(dtype)
            q = torch.matmul(q, self.k_rotation.to(q))
            self.pre_rotation_quantizer.free()

            self.pre_rotation_quantizer.find_params(k)
            k = self.pre_rotation_quantizer(k).to(dtype)
            k = torch.matmul(k, self.k_rotation.to(k))

        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            raise NotImplementedError  # todo
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.contiguous().view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.pre_rotation_quantizer.free()
        self.k_quantizer.free()
        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
