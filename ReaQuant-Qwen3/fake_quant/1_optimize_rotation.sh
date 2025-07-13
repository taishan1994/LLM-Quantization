# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### k_groupsize, v_groupsize = 64 only for Llama-3.2-1B else 128
#### Storing config for 70B model
torchrun --nnodes=1 --nproc_per_node=1 --master_port=24544 optimize_rotation.py \
--input_model Qwen/Qwen2.5-1.5B \
--per_device_eval_batch_size 8 \
--per_device_train_batch_size 1 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--w_bits 16 \
--a_bits 16 \
--k_bits 16 \
--v_bits 16 \
--high_bits 8 \
--low_bits 2 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--high_fraction 0.125 \
--low_fraction 0.0 \
--rotate_mode "resq" \
--output_rotation_path "rotation" \
--optimized_basis_path ./rotation/U-wikitext-512-Qwen2.5-1.5B.bin \
--rotation_granularity 'full_shared' \
--rotate \
--train_rotations \
--learning_rate 1.5 \
--max_steps 100 \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--logging_steps 1 \
--weight_decay 0. \

