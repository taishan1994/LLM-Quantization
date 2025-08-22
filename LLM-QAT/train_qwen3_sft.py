# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Add quantization and knowledge distillialtion
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import math
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from models.configuration_qwen3 import Qwen3Config
from models.modeling_qwen3_quant import (
    Qwen3ForCausalLM as Qwen3ForCausalLMQuant,
)

import copy
import torch
import transformers
from utils import utils
from utils import datautils

# from utils.kd_trainer import KDTrainer

from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer
from datasets import load_dataset

log = utils.get_logger("clm")

def _is_linear(m, n):
    if isinstance(m, torch.nn.Linear) and 'lm_head' not in n:
        return True


def train():
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    if training_args.qat:
        # if training_args.use_torchao:
        #     from torchao.quantization import quantize_
        #     from torchao.quantization.qat import (
        #         FakeQuantizeConfig,
        #         FromIntXQuantizationAwareTrainingConfig,
        #         IntXQuantizationAwareTrainingConfig,
        #         NVFP4FakeQuantizeConfig
        #     )
        #     model = transformers.Qwen3ForCausalLM.from_pretrained(
        #         model_args.local_dir,
        #         torch_dtype=dtype,
        #         low_cpu_mem_usage=True,
        #     )
        #     # torch只支持激活per token的非对称量化
        #     activation_config = FakeQuantizeConfig(
        #         dtype=torch.int8, granularity="per_token", is_symmetric=False
        #     )
        #     # weight_config = FakeQuantizeConfig(dtype=torch.int4, granularity = "per_group",group_size=64)
        #     weight_config = FakeQuantizeConfig(dtype=torch.int4, granularity="per_channel", is_symmetric=True)
        #     linear_quantize_config = IntXQuantizationAwareTrainingConfig(
        #         activation_config=activation_config,
        #         weight_config=weight_config,
        #     )
        #     quantize_(model, linear_quantize_config, filter_fn=_is_linear)

        # config = LlamaConfig.from_pretrained(model_args.input_model_filename)
        config = Qwen3Config.from_pretrained(model_args.local_dir)
        student_config = copy.deepcopy(config)
        student_config.w_bits = model_args.w_bits
        student_config.a_bits = model_args.a_bits
        student_config.kv_bits = model_args.kv_bits
        # model = LlamaForCausalLMQuant.from_pretrained(
        #     pretrained_model_name_or_path=model_args.input_model_filename,
        #     config=student_config,
        #     cache_dir=training_args.cache_dir,
        #     torch_dtype=dtype,
        #     low_cpu_mem_usage=True,
        #     device_map=None if len(training_args.fsdp) > 0 else "auto",
        # )
        model = Qwen3ForCausalLMQuant.from_pretrained(
            model_args.local_dir,
            config=student_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            # device_map=None if len(training_args.fsdp) > 0 else "auto",
        )
    else:
        # model = transformers.LlamaForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path=model_args.input_model_filename,
        #     cache_dir=training_args.cache_dir,
        #     torch_dtype=dtype,
        #     low_cpu_mem_usage=True,
        #     device_map=None if len(training_args.fsdp) > 0 else "auto",
        # )
        model = transformers.Qwen3ForCausalLM.from_pretrained(
            model_args.local_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None if len(training_args.fsdp) > 0 else "auto",
        )
    model.cuda()
    if training_args.use_kd:
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.local_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            #device_map=None if len(training_args.fsdp) > 0 else "auto",
        )
        teacher_model.eval()
        teacher_model.cuda()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        model.kd_loss_scale = training_args.kd_loss_scale
        model.teacher = teacher_model
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    # tokenizer = transformers.LlamaTokenizer.from_pretrained(
    #     pretrained_model_name_or_path=model_args.input_model_filename,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     use_fast=False,
    # )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.local_dir,
        padding_side="right",
        use_fast=False,
    )

    log.info("Complete tokenizer loading...")

    # train_dataset, valid_dataset = datautils.get_train_val_dataset(
    #     train_path=data_args.train_data_local_path,
    #     valid_path=data_args.eval_data_local_path
    #     if data_args.eval_data_local_path is not None
    #     else None,
    # )
    # train_data = datautils.CustomJsonDataset(
    #     train_dataset, tokenizer, block_size=training_args.model_max_length
    # )
    # valid_data = datautils.CustomJsonDataset(
    #     valid_dataset, tokenizer, block_size=min(training_args.model_max_length, 1024)
    # )

    dataset = load_dataset("/home/gongoubo/project/LLM-QAT/data/medical_o1_gen")

    # 划分成 train 和 valid，比如 90% 训练 10% 验证
    dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=False, seed=42)
    # 这样 dataset 就是一个字典，包含 'train' 和 'test' 两个 Dataset 对象
    # 如果想叫 'validation' 而不是 'test'
    dataset["valid"] = dataset.pop("test")

    def preprocess_sft_function(example):
        return {
            "prompt": [{"role": "user", "content": example["Question"]}],
            "completion": [
                {"role": "assistant", "content": f"<think>{example['Complex_CoT']}</think>{example['Response']}"} if example['Complex_CoT'] != "" else {"role": "assistant", "content": f"{example['Response']}"}
            ],
        }

    def preprocess_kd_function(example):
        return {
            "messages": [
                {"role": "user", "content": example["Question"]},
                {"role": "assistant", "content": f"<think>{example['Complex_CoT']}</think>{example['Response']}"} if example['Complex_CoT'] != "" else {"role": "assistant", "content": f"{example['Response']}"}
            ]
        }

    if training_args.use_kd:
        dataset = dataset.map(preprocess_kd_function, remove_columns=["Question", "Response", "Complex_CoT"])
    else:
        dataset = dataset.map(preprocess_sft_function, remove_columns=["Question", "Response", "Complex_CoT"])

    print(dataset["train"][0])
    train_data = dataset["train"]
    valid_data = dataset["valid"]

    model.config.use_cache = False

    print(training_args)

    if training_args.use_kd:
        # myTrainer = KDTrainer
        from trl import GKDTrainer, GKDConfig
        training_args_dict = training_args.to_dict()
        training_args_dict.pop("cache_dir", None)  # 移除多余参数
        training_args_dict.pop("qat", None)  # 移除多余参数
        training_args_dict.pop("use_kd", None)  # 移除多余参数
        training_args_dict.pop("kd_loss_scale", None)  # 移除多余参数
        trainer = GKDTrainer(
            model=model,
            teacher_model=teacher_model,
            processing_class=tokenizer,
            args=GKDConfig(**training_args_dict),
            train_dataset=train_data,
            eval_dataset=valid_data,
        )
    else:
        # myTrainer = Trainer
        from trl import SFTTrainer, SFTConfig
        myTrainer = SFTTrainer

        print(training_args)
        training_args_dict = training_args.to_dict()
        training_args_dict.pop("cache_dir", None)  # 移除多余参数
        training_args_dict.pop("qat", None)  # 移除多余参数
        training_args_dict.pop("use_kd", None)  # 移除多余参数
        training_args_dict.pop("kd_loss_scale", None)  # 移除多余参数
        training_args_dict.pop("lmbda", None)  # 移除多余参数
        training_args_dict.pop("beta", None)  # 移除多余参数
        training_args_dict.pop("seq_kd", None)  # 移除多余参数

        trainer = myTrainer(
            model=model,
            args=SFTConfig(**training_args_dict),
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=valid_data if training_args.do_eval else None,
        )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()
        utils.safe_save_model_for_hf_trainer(trainer, model_args.output_model_local_path)

    # Evaluation
    if training_args.do_eval:
        model.to("cuda")
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    torch.distributed.barrier()


if __name__ == "__main__":
    train()
