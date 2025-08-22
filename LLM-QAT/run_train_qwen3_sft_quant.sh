# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --train_data_local_path "gen_data.jsonl" \
# --eval_data_local_path "wiki2.jsonl" \
#--w_bits $1 \
#--a_bits $2 \
#--kv_bits $3 \
#--use_kd True \

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=15001 train_qwen3_sft.py \
--local_dir "model_hub/Qwen/Qwen3-4B" \
--input_model_filename "Qwen3/Qwen3-4B" \
--output_model_filename "qwen3-4B-gen-quant-finetuned2" \
--do_train True \
--do_eval False \
--max_length 4096 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir ./train_log \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_strategy "steps" \
--save_steps 2000 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing True \
--w_bits 8 \
--a_bits 8 \
--kv_bits 16 \
--qat True \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'Qwen3DecoderLayer'
