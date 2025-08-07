import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from transformers import AutoTokenizer, Qwen3Config
# 确保下面的导入指向您修改后的文件
# 例如，如果您将上面的代码保存为 `modeling_qwen3_w8a8.py`
from modeling_qwen3 import Qwen3ForCausalLM

# 1. 配置模型
# 确保你使用的 config 和你的权重是匹配的
model_path = "/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B"
config = Qwen3Config.from_pretrained(model_path)

model = Qwen3ForCausalLM.from_pretrained(model_path,
                                         torch_dtype=torch.bfloat16,
                                         device_map="auto",
                                         attn_implementation="sdpa")

print(model)

tokenizer = AutoTokenizer.from_pretrained(model_path)


for k,v in model.named_parameters():
    print(k, v.dtype)

model.eval() # 设置为评估模式

# 5. 进行推理
tokenizer = AutoTokenizer.from_pretrained(model_path)
text = """背景
虽然现在可以通过大语言模型、多模态大模型做端到端的任务。但依然存在一些大模型无法直接处理的场景。例如比较常见的 RAG 任务，从海量文档数据中找回目标数据。常用的手段就是多路召回，其中就不乏有基于 Embedding 的稠密召回操作，对于召回的内容总得有一个“相似度”评判，就是一个 Reranking 模型。

预备知识-benchmark
评判 Embedding、Reranker 模型的性能效果的 benchmark 主要有：

MMTEB(Massive Multilingual Text Embedding Benchmark)[1]. 相关论文介绍：MMTEB: Massive Multilingual Text Embedding Benchmark[2]：

C-MTEB(Chinese Massive Text Embedding Benchmark)[3]. 相关论文：C-Pack: Packed Resources For General Chinese Embeddings[4]:

Qwen3-Embedding
基本信息
官方博客：Qwen3 Embedding：新一代文本表征与排序模型[5]

论文：Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models[6]

github:https://github.com/QwenLM/Qwen3-Embedding[7]

概述
Qwen3 Embedding 系列模型分别有 0.6b, 4b, 8b 的 Embedding 和 Reranker 模型。该系列模型专为文本表征、检索与排序任务设计，基于 Qwen3 基础模型进行训练，充分继承了 Qwen3 在多语言文本理解能力方面的优势。在多项基准测试中，Qwen3 Embedding 系列在文本表征和排序任务中展现了卓越的性能。主要特点如下：

卓越的多功能性：该嵌入模型在广泛的下游应用评估中达到了最先进的性能。8B 大小的嵌入模型在 MTEB 多语言排行榜上排名第 1（截至 2025 年 6 月 5 日，得分为 70.58），而重排序模型在各种文本检索场景中表现出色。
全面的灵活性：Qwen3 Embedding 系列提供了从 0.6B 到 8B 的全尺寸范围的嵌入和重排序模型，适用于重视效率和效果的各种使用场景。开发人员可以无缝地组合这两个模块。此外，嵌入模型允许在所有维度上灵活定义向量，并且嵌入和重排序模型都支持用户定义的指令，以增强特定任务、语言或场景的性能。
多语言能力：得益于 Qwen3 模型的多语言能力，Qwen3 Embedding 系列支持超过 100 种语言。这包括多种编程语言，并提供了强大的多语言、跨语言和代码检索能力。
模型参数：


效果与指标
Embedding 模型对比：



ReRanker 模型对比：


模型
模型架构
基于 Qwen3 基础模型， Embedding 模型和 Reranking 模型分别采用了双塔结构和单塔结构的设计。通过 LoRA 微调，最大限度地保留并继承了基础模型的文本理解能力。具体实现如下：1) Embedding 模型接收单段文本作为输入，取模型最后一层[EOS]标记对应的隐藏状态向量，作为输入文本的语义表示；2) Reranking 模型则接收文本对（例如用户查询与候选文档）作为输入，利用单塔结构计算并输出两个文本的相关性得分。


Embedding 模型的入参格式：

{Instruction}{Query}<|endoftext|>
Reranking 模型的入参格式：

<|im_start|>system Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>:{Instruction} 
<Query>: {Query}
<Document>:{Document}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n
相似性打分逻辑：


模型训练
Qwen3 Embedding 系列模型的训练继承了 GTE-Qwen 系列的多阶段训练范式，但针对具体应用场景进行了深度优化。


Embedding 模型训练
采用三阶段训练架构：

第一阶段通过超大规模弱监督数据进行对比学习预训练；

第二阶段基于高质量标注数据进行监督训练；

最终通过模型融合策略融合多个候选模型，以提升整体性能。 这种分阶段训练机制有效平衡了模型的泛化能力与任务适配性。

Reranking 模型训练
基于实验验证结果，作者直接采用高质量标注数据进行监督训练，以提升训练效率。特别需要说明的是，在 Embedding 模型的第一阶段弱监督训练中，作者构建了多任务适配的 Prompt 体系，利用 Qwen3 基础模型的文本生成能力，针对不同任务类型和语言特性，动态生成了一系列弱监督文本对，突破了传统方法依赖社区论坛或开源数据筛选获取弱监督文本对的局限性，实现了大规模弱监督数据的高效生成。其中 SFT 优化 loss 函数定义如下："""
prompt = [{"role": "user", "content": text + "\n请总结上述的文本。"}]
prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

print(prompt)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成文本
with torch.no_grad():
    t1 = time.time()
    outputs = model.generate(**inputs,
                             max_new_tokens=512,
                             eos_token_id=tokenizer.eos_token_id,
                             temperature=0.01)
    t2 = time.time()
    print("耗时：{}s".format(t2-t1))

for i in range(len(outputs)):
    print(tokenizer.decode(outputs[i], skip_special_tokens=False))