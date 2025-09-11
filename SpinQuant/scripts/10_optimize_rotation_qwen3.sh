# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py \
--input_model "/data/gongoubo/checkpoints/Qwen/Qwen3-0___6B"  \
--output_rotation_path "qwen3-0.6b" \
--output_dir "outputs/" \
--logging_dir "logs/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing False \
--save_safetensors False \
--max_steps 100 \
--w_bits 4 \
--a_bits 8 \
--k_bits 16 \
--v_bits 16 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
