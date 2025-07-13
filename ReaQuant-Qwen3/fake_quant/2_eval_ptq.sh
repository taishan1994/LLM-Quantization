# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
### k_groupsize, v_groupsize = 64 only for Llama-3.2-1B else 128
#### Storing config for 70B model
#/data/gongoubo/other_project/project-resq-bak/fake_quant/LLM-Research/Llama-3___2-1B-Instruct
# /data/gongoubo/checkpoints/Qwen/Qwen3-0___6B/
#/data/gongoubo/Qwen-1.5-Factory/model_hub/Qwen/Qwen2___5-1___5B-Instruct
export CUDA_VISIBLE_DEVICES=4

torchrun --nnodes=1 --nproc_per_node=1 --master_port=24544 ptq.py \
--input_model /data/gongoubo/checkpoints/Qwen/Qwen3-0.6B-merge-norm \
--per_device_eval_batch_size 1 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--w_bits 4 \
--a_bits 8 \
--k_bits 16 \
--v_bits 16 \
--high_bits 8 \
--low_bits 4 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--high_fraction 0.25 \
--low_fraction 0.0 \
--rotate_mode "resq" \
--optimized_rotation_path ./rotation_qwen3_0.6b_change_r0.5/R-high-0.5-low-0.0-sparse-0.0-data.bin \
--optimized_basis_path ./rotation_qwen3_0.6b_change_r0.5/U-wikitext-512-data.bin \
--rotation_granularity 'full_shared' \
--rotate \
--tasks "mmlu,boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"  \
#--long_bench_tasks "dureader" \
#--save_qmodel_path "qwen3_0.6b_w8a8"
