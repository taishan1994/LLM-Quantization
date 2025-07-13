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
from transformers import (
    LlamaTokenizerFast,
    Trainer,
    default_data_collator,
    TrainingArguments,
)
from utils.data_utils import get_wikitext2
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_train import LlamaForCausalLM
from train_utils.modeling_qwen2_train import Qwen2ForCausalLM
# from eval_utils.modeling_llama_2 import LlamaForCausalLM
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import (
    random_orthogonal_matrix,
)
import math
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict

from transformers import AutoConfig, AutoTokenizer
import numpy as np
import random
# import eval_utils, utils.utils, utils.data_utils
from utils import data_utils, eval_utils, utils

log: Logger = utils.get_logger("resq", "log.log")



def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss)
    return {"perplexity": perplexity.item()}


class RotateModule(nn.Module):
    def __init__(self, R_init):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    
    seed_everything(ptq_args.seed)
    config = AutoConfig.from_pretrained(
        model_args.input_model,
    )
    # ResQ is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    
    if "llama" in model_args.input_model.lower():
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            torch_dtype=dtype,
            config=config,
        )
    elif "qwen2" in model_args.input_model.lower() and "vl" not in model_args.input_model.lower():
        model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            torch_dtype=dtype,
            config=config,
        )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model, R_dict = prepare_model(ptq_args, model)
    model_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model_dim // num_heads

    #mixed precision quant
    low_frac, high_frac = ptq_args.low_fraction, ptq_args.high_fraction
    sparse_fraction = ptq_args.sparse_fraction
    low_length_hidden, high_length_hidden = int(low_frac * model_dim), int(high_frac * model_dim)
    low_length_head, high_length_head = int(low_frac * head_dim), int(high_frac * head_dim)

    rotation_granularity = ptq_args.rotation_granularity
    assert rotation_granularity == "full_shared" # only full shared is supported at this moment


    for param in model.parameters():
        param.requires_grad = False
    
    R1_1 = random_orthogonal_matrix(
        model_dim - high_length_hidden - low_length_hidden, "cuda"
    )
    R1_2 = random_orthogonal_matrix(high_length_hidden, "cuda")
    if low_length_hidden != 0 :
        R1_0 = random_orthogonal_matrix(low_length_hidden, "cuda")

    model.R1_1 = RotateModule(R1_1)
    model.R1_2 = RotateModule(R1_2)
    if low_length_hidden != 0:
        model.R1_0 = RotateModule(R1_0) 

    R2_1 = random_orthogonal_matrix(
        head_dim - high_length_head - low_length_head,
        "cuda",
    )
    R2_2 = random_orthogonal_matrix(high_length_head, "cuda")
    if low_length_head != 0 :
        R2_0 = random_orthogonal_matrix(low_length_head, "cuda")
    else:
        R2_0 = None 

    for i in range(model.config.num_hidden_layers):
        # Each head dim = 128 for Llama model
        model.model.layers[i].self_attn.R2_1 = RotateModule(R2_1)
        model.model.layers[i].self_attn.R2_2 = RotateModule(R2_2)
        if R2_0 is not None:
            model.model.layers[i].self_attn.R2_0 = RotateModule(R2_0)

    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    #####
    model.seqlen = training_args.model_max_length
    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    with torch.no_grad():
        dataset_ppl = eval_utils.evaluator_cuda(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    testset = testloader.input_ids
    nsamples = testset.numel() // model.seqlen
    testset = (
        testset[:, : nsamples * model.seqlen].view(nsamples, model.seqlen).to(utils.DEV)
    )  # (nsamples, seqlen)
    batch_size = ptq_args.bsz
    testset = [testset[i : i + batch_size] for i in range(0, nsamples, batch_size)]

    #####
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    test_data = CustomJsonDataset(
        calibration_datasets["test"], tokenizer, block_size=model.seqlen
    )

    trainable_parameters = []
    trainable_parameters.append(
        {
            "params": [model.R1_1.weight, model.R1_2.weight],
            "stiefel": True,
            "lr": training_args.learning_rate,
            "momentum": 0.9,
            "nesterov": True,
        }
    )
    if low_length_hidden != 0:
        trainable_parameters.append(
        {
            "params": [model.R1_0.weight],
            "stiefel": True,
            "lr": training_args.learning_rate,
            "momentum": 0.9,
            "nesterov": True,
        }
    )
        
    for i in range(model.config.num_hidden_layers):
        trainable_parameters.append(
            {
                "params": [model.model.layers[i].self_attn.R2_1.weight, model.model.layers[i].self_attn.R2_2.weight],
                "stiefel": True,
                "lr": training_args.learning_rate,
                "momentum": 0.9,
                "nesterov": True,
            }
        )
        if low_length_head !=0 :
            trainable_parameters.append(
            {
                "params": model.model.layers[i].self_attn.R2_0.weight,
                "stiefel": True,
                "lr": training_args.learning_rate,
                "momentum": 0.9,
                "nesterov": True,
            }
        )

    for name, p in model.named_parameters():
        if (
            "R1_1.weight" in name
            or "R1_2.weight" in name
            or "R1_0.weight" in name
            or "R2_0.weight" in name
            or "R2_1.weight" in name
            or "R2_2.weight" in name
        ):
            p.requires_grad = True

    model.seqlen = training_args.model_max_length
    optimizer = SGDG(
        trainable_parameters,
        lr=training_args.learning_rate,
    )
    MyTrainer = Trainer
    # Use FSDP for 70B rotation training
    if training_args.fsdp != "" and training_args.fsdp != []:
        MyTrainer = FSDPTrainer

    # training_args = TrainingArguments(
    #     output_dir="./checkpoint",
    #     save_strategy="steps",
    #     save_steps=0.1,
    #     save_safetensors=False,
    #     logging_steps=0.1,
    #     gradient_checkpointing=True,
    #     per_device_train_batch_size=training_args.per_device_train_batch_size,
    #     per_device_eval_batch_size=training_args.per_device_eval_batch_size,
    #     max_steps=training_args.max_steps,
    #     auto_find_batch_size=True,
    #     save_total_limit=3,
    # )

    model.gradient_checkpointing_enable()
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,  # Make sure this is included
    )
    torch.distributed.barrier()

    torch.cuda.empty_cache()
    torch.distributed.barrier()
    trainer.train()

    if training_args.fsdp != "" and training_args.fsdp != []:
        cpu_state = pt_fsdp_state_dict(trainer.model)
    else:
        cpu_state = trainer.model.state_dict()

    R_dict = R_dict | {
        key.replace(".weight", ""): value
        for key, value in cpu_state.items()
        if (
            "R1_1.weight" in key
            or "R1_2.weight" in key
            or "R1_0.weight" in key
            or "R2_0.weight" in key
            or "R2_1.weight" in key
            or "R2_2.weight" in key
        )
    }
    if 'R1_0' not in R_dict.keys():
        R_dict["R1_0"] = None

    print(R_dict.keys())
    if local_rank == 0:
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
            + "-trained"
            + ".bin",
        )
        torch.save(
            R_dict,
            rotation_path,
        )
        # also remove the checkpoint folder, we dont need it not
        # os.rmdir(training_args.output_dir)
    
    dist.barrier()
    with torch.no_grad():
        dataset_ppl = eval_utils.evaluator_cuda(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))


if __name__ == "__main__":
    train()
