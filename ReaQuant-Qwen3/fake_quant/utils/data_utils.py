# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import random
from typing import Any, Dict

import datasets
import torch
import transformers


def get_wikitext2(
    nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False, vision=False,
):
    print("get_wikitext2")
    
    if tokenizer is None:
        if vision:
            tokenizer = transformers.AutoProcessor.from_pretrained(model)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if eval_mode:
        testdata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "test"
        ]
        testenc = tokenizer(text="\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "train"
        ]
        trainenc = tokenizer(text="\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4(
    nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False, vision=False,
):
    print("get_c4")

    if tokenizer is None:
        if vision:
            tokenizer = transformers.AutoProcessor.from_pretrained(model)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if eval_mode:
        valdata = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(text=valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        return valenc
    else:
        random.seed(seed)
        traindata = datasets.load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(text=traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_ptb(
    nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False, vision=False,
):
    print("get_ptb")
    
    if tokenizer is None:
        if vision:
            tokenizer = transformers.AutoProcessor.from_pretrained(model)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if eval_mode:
        testdata = datasets.load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(text="\n\n".join(testdata["sentence"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset("ptb_text_only", "penn_treebank", split="train", trust_remote_code=True)
        trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_alpaca(
    nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False, vision=False,
):
    print("get_alpaca")
    
    if tokenizer is None:
        if vision:
            tokenizer = transformers.AutoProcessor.from_pretrained(model)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if eval_mode:
        raise ValueError # no eval set for alpaca
    else:
        ds = datasets.load_dataset('tatsu-lab/alpaca')
        ds = ds.remove_columns(['input', 'output', 'instruction'])
        traindata = ds["train"]
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader


def get_data(
    calib_dataset="wikitext",
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    tokenizer=None,
    eval_mode=False,
    vision=False,
):
    if "wikitext" in calib_dataset:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer, eval_mode, vision)
    elif "c4" in calib_dataset:
        return get_c4(nsamples, seed, seqlen, model, tokenizer, eval_mode, vision)
    elif "ptb" in calib_dataset:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer, eval_mode, vision)
    elif "alpaca" in calib_dataset:
        return get_alpaca(nsamples, seed, seqlen, model, tokenizer, eval_mode, vision)
    else:
        raise NotImplementedError


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size: int = 1024) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
