# SpinQuant

This repository contains the code of SpinQuant introduced: "[SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/pdf/2405.16406)"

将其适配量化Qwen3，使用Qwen3-0.6B作为基座模型。

## Run

### 1. Requirements:
* python 3.9, pytorch >= 2.0
* install pytorch with cuda from https://pytorch.org/get-started/locally/, it is prerequisite for fast-hadamard-transform package.
* pip install -r requirement.txt
* git clone https://github.com/Dao-AILab/fast-hadamard-transform.git  
* cd fast-hadamard-transform  
* pip install .
  
### 2. Steps to run:
Step 1: 优化旋转矩阵

```shell
bash scripts/10_optimize_rotation_qwen3.sh

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

```

Step 2: 使用GPTQ量化并评估

```shell
bash scripts/2_eval_ptq_qwen3.sh

torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model "/data/gongoubo/checkpoints/Qwen/Qwen3-0___6B" \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
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
--rotate \
--optimized_rotation_path "qwen3-0.6b/R.bin" \
```



