# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# /data/gongoubo/other_project/project-resq-bak/fake_quant/LLM-Research/Llama-3___2-1B-Instruct
# /data/gongoubo/Qwen-1.5-Factory/model_hub/Qwen/Qwen2___5-1___5B-Instruct
# /data/gongoubo/checkpoints/Qwen/Qwen3-0___6B/
python get_basis.py \
--input_model "/data/gongoubo/checkpoints/Qwen/Qwen3-0.6B-merge-norm" \
--output_rotation_path "rotation_qwen3_0.6b_change_r0.5" \
--model_max_length 2048 \
--down_proj_blocksize 256 \
--high_fraction 0.125 \
--low_fraction 0.0 \
--rotation_granularity "full_shared" \
--rotate_mode "resq" \
--nsamples 512 \
--calib_dataset "wikitext" \
--sparse_fraction 0.0 \