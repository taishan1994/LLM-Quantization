# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    local_dir: str = field(
        default=None, metadata={"help": "Local Path of storing inputs and outputs "}
    )
    input_model_filename: Optional[str] = field(
        default="test-input", metadata={"help": "Input model relative manifold path"}
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    output_model_local_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    w_bits: Optional[int] = field(
        default=16,
        metadata={
            "help": "#bits to use for quantization; use 16 for evaluating base model. choices=[4, 8, 32]"
        },
    )
    a_bits: Optional[int] = field(
        default=16,
        metadata={"help": "Activation quantization bits."},
    )
    kv_bits: Optional[int] = field(
        default=16,
        metadata={"help": "KV_cache quantization bits."},
    )


@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    qat: Optional[bool] = field(default=False)
    use_kd: Optional[bool] = field(default=False)
    kd_loss_scale: Optional[float] = field(
        default=1.0,
        metadata={"help": "Scale of KD loss."},
    )
    lmbda : Optional[float] = field(
        default=0.0,
        metadata={"help": " Lambda 参数，用于控制学生数据部分（即同策略下学生生成输出的比例）"},
    )
    beta: Optional[float] = field(
        default=0.0,
        metadata={"help": "介于 0.0 和 1.0 之间。当 beta 为 0.0 时，损失为 KL 散度。当 beta 为 1.0 时，损失为逆 KL 散度。"},
    )
    seq_kd : Optional[bool] = field(default=False, metadata={"help":"用于控制是否执行序列级知识蒸馏（Sequence-Level KD），可视为在教师生成的输出上进行监督微调。"})




def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(model_args.local_dir, exist_ok=True)

    assert model_args.output_model_local_path is None

    model_args.output_model_local_path = os.path.join(
        model_args.local_dir, "models", str(model_args.output_model_filename)
    )

    return model_args, data_args, training_args
