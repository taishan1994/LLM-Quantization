# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, Qwen2Config

from eval_utils.main import ptq_model
from eval_utils.modeling_llama_2 import LlamaForCausalLM
from eval_utils.modeling_qwen2 import Qwen2ForCausalLM
from eval_utils.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
# from eval_utils.modeling_qwen3 import Qwen3ForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from lm_eval import evaluator
import numpy as np
from lm_eval.utils import make_table
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import random
from utils.parallel_utils import map_layers_to_multi_gpus
from lm_eval.models.huggingface import HFLM
from lm_eval.models.hf_vlms import HFMultimodalLM



from utils.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

log: Logger = utils.get_logger("resq", "log.log")


@torch.no_grad()
def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


@torch.no_grad()
def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        # print(prompt)
        prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        # if "chatglm3" in model:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        # if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
        #     prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        # pred = post_process(pred, model_name)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


@torch.no_grad()
def evaluate(model, tokenizer, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    results = {}

    testloader = data_utils.get_wikitext2(
        seed=args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
        vision=args.vision_lm,
    )

    # ==========================
    # local_rank = utils.get_local_rank()
    # if local_rank == 0:
    #     model.to("cuda:0")
    #     message = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
    #     input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)
    #
    #     direct_output = model.generate(input_ids, max_new_tokens=256, do_sample=False, temperature=0.01)
    #     direct_text = tokenizer.decode(direct_output[0])
    #     print(direct_text)
    # ==========================

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    model.config.use_cache = use_cache


    if args.multigpu:
        map_layers_to_multi_gpus(model.model.layers)
        input_device = model.model.layers[0].device
        output_device = model.model.layers[-1].device
        assert input_device == output_device
        model.model.embed_tokens.to(input_device)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(input_device)
        if hasattr(model, "visual"):
            model.visual = model.visual.to(input_device)
        model.model.norm.to(output_device)
        model.lm_head.to(output_device)
    else:
        input_device = utils.DEV
        model.to(utils.DEV)
    if args.tasks != "":
        if args.vision_lm:
            lm = HFMultimodalLM(pretrained=model, processor=tokenizer)
        else:
            lm = HFLM(pretrained=model, tokenizer=tokenizer)
        lm._device = input_device
        model_args={}
        model_args['parallelize'] = True if args.multigpu else False
        
        # hack for setting trust remote code = True taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py:377
        model_args['trust_remote_code'] = True
        import datasets
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        apply_chat_template = "mmmu" in args.tasks #Run MMMU separately because apply_chat_template is False for every other task.

        # apply_chat_template = True
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks.split(","),
            model_args=model_args,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
            batch_size = args.bsz,
            apply_chat_template=apply_chat_template,
            #log_samples=True,
            write_out=True,
            #predict_only=True,
        )
        results.update(t_results)
        log.info(make_table(t_results))
        print(make_table(t_results))

    if args.long_bench_tasks != "":
        model2path = json.load(open("config_longbench/model2path.json", "r"))
        model2maxlen = json.load(open("config_longbench/model2maxlen.json", "r"))
        # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
        dataset2prompt = json.load(open("config_longbench/dataset2prompt.json", "r"))
        dataset2maxlen = json.load(open("config_longbench/dataset2maxlen.json", "r"))
        if args.tasks != "":
            model = lm.model
            tokenizer = lm.tokenizer
        # model_name = args.model_name
        model_name = "qwen3-0.6b"
        max_length = model2maxlen[model_name]
        
        all_tasks = args.long_bench_tasks.split(",")
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            # "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            # "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        for dataset in all_tasks:
            data = load_dataset('THUDM/LongBench', f"{dataset}", split='test')
            if not os.path.exists(f"pred/{model_name}_{max_length}"):
                os.makedirs(f"pred/{model_name}_{max_length}")
            out_path = f"pred/{model_name}_{max_length}/{dataset}.jsonl"

            if os.path.exists(out_path):
                continue
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, utils.DEV, model_name)
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
            
        path = f"pred/{model_name}_{max_length}/"
        all_files = os.listdir(path)
        scores = dict()

        print("Evaluating on:", all_files)
        for filename in all_files:
            if not filename.endswith("jsonl"):
                continue
            predictions, answers, lengths = [], [], []
            dataset = filename.split('.')[0]
            with open(f"{path}{filename}", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    all_classes = data["all_classes"]
                    if "length" in data:
                        lengths.append(data["length"])
            score = scorer(dataset, predictions, answers, all_classes)
            scores[dataset] = score
        out_path = f"pred/{model_name}_{max_length}/result.json"
        with open(out_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

    return results

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
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()
    seed_everything(ptq_args.seed)
    config = Qwen2Config.from_pretrained(
        model_args.input_model,
    )
    
    if ptq_args.flash_attn:
        config._attn_implementation = "flash_attention_2"
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    # ResQ is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    vision = False
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
    elif "qwen2" in model_args.input_model.lower() and "vl" in model_args.input_model.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            torch_dtype=dtype,
            config=config,
        )
        vision = True

    config._attn_implementation = "eager"
    # print(config)

    model = Qwen2ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=dtype,
        config=config,
    )


    #elif "qwen3" in model_args.input_model.lower():
    #    model = Qwen3ForCausalLM.from_pretrained(
    #        pretrained_model_name_or_path=model_args.input_model,
    #        torch_dtype=dtype,
    #        config=config,
    #    )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    for name, m in model.named_modules():
        if "basis_change" in name:
            m.weight.data.copy_(torch.eye(model.config.hidden_size))

    # ================================
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

    # model = ptq_model(ptq_args, model, model_args, tokenizer)
    print(model)
    model.seqlen = training_args.model_max_length

    # if local_rank == 0:
    #     print("开始验证量化后的模型！！！")
    #
    #     model.to("cuda:0")
    #     inps = [{"role": "user", "content": "who are you？"}]
    #     inps = tokenizer.apply_chat_template(inps, tokenize=False, add_generation_prompt=True)
    #     print(inps)
    #     inps = tokenizer.encode(inps, return_tensors="pt").to(model.device)
    #     outputs = model.generate(inps, max_new_tokens=256, temperature=0.01)
    #     print(tokenizer.decode(outputs[0]))
    #     # ================================


    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    if vision:
        tokenizer = AutoProcessor.from_pretrained(model_args.input_model)
    else:
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

    ptq_args.vision_lm = vision
    ptq_args.model_name = model_args.input_model.split('/')[-1]
    results = evaluate(model, tokenizer, ptq_args)
    dist.barrier()


if __name__ == "__main__":
    train()