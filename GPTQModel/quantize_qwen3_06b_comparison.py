#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-0.6B-Chat模型量化脚本 - GPTQ V1和V2比较
此脚本用于演示如何使用GPTQModel库对Qwen3-0.6B模型（Chat版本）进行量化,
并比较GPTQ V1和V2两种量化方法的效果。
"""

import os
import time
import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

# 设置环境变量以优化CUDA使用
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 模型ID和保存路径
MODEL_ID = "/data/gongoubo/checkpoints/Qwen/Qwen3-0.6B"  # Qwen3-0.6B原生支持指令跟随功能
OUTPUT_DIR = "./quantized_models"
V1_MODEL_PATH = os.path.join(OUTPUT_DIR, "qwen3-0.6b-chat-gptq-v1-4bit-128g")
V2_MODEL_PATH = os.path.join(OUTPUT_DIR, "qwen3-0.6b-chat-gptq-v2-4bit-128g")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_calibration_dataset(tokenizer, nsamples=512, seqlen=1024):
    """准备用于量化的校准数据集"""
    print("正在准备校准数据集...")
    # 使用wikitext-2数据集作为校准数据
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="./cache")
    
    # 过滤掉太短的文本
    traindata = traindata.filter(lambda x: len(x["text"]) >= seqlen)
    
    # 确保不选择超出数据集大小的样本
    actual_samples = min(nsamples, len(traindata))
    if actual_samples < nsamples:
        print(f"警告: 可用样本数量({actual_samples})少于请求的样本数量({nsamples})")
    
    # 选择样本并进行标记化
    samples = [tokenizer(example["text"][:seqlen]) for example in traindata.select(range(actual_samples))]
    
    print(f"已准备{len(samples)}个校准样本")
    return samples


def quantize_model(model_id, output_path, quant_config, calibration_dataset):
    """量化模型并返回量化时间"""
    print(f"\n开始量化模型到 {output_path}...")
    print(f"量化配置: bits={quant_config.bits}, group_size={quant_config.group_size}, v2={quant_config.v2}")
    
    # 加载未量化的模型
    model = GPTQModel.load(model_id, quant_config)
    
    # 开始计时
    start_time = time.time()
    
    # 量化模型
    model.quantize(calibration_dataset, batch_size=1)
    
    # 结束计时
    end_time = time.time()
    quant_time = end_time - start_time
    
    print(f"量化完成，耗时: {quant_time:.2f}秒")
    
    # 保存量化后的模型
    model.save(output_path)
    print(f"已保存量化模型到 {output_path}")
    
    return quant_time


def create_chat_prompt(tokenizer, prompt):
    """为Chat模型创建合适的聊天提示格式"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def evaluate_model(model_path, tokenizer, prompt="解释量子计算的基本原理"):
    """评估量化后的模型性能"""
    print(f"\n加载并评估模型 {model_path}...")
    
    # 加载量化后的模型
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPTQModel.load(model_path, device=device)
    
    # 测试模型是否正常工作 - 使用"你是谁"测试
    print("\n测试模型是否正常工作 (提示: 你是谁)...")
    test_prompt = "你是谁"
    test_chat_prompt = create_chat_prompt(tokenizer, test_prompt)
    test_inputs = tokenizer(test_chat_prompt, return_tensors="pt").to(model.device)
    
    try:
        test_outputs = model.generate(
            **test_inputs,
            max_new_tokens=50,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        test_generated = tokenizer.decode(test_outputs[0][len(test_inputs["input_ids"][0]):], skip_special_tokens=True)
        print(f"测试生成结果:\n{test_generated}")
        print("模型测试通过，可以继续进行评估。")
    except Exception as e:
        print(f"模型测试失败: {str(e)}")
        print("评估过程已终止。")
        return {
            "ppl": float('inf'),
            "gen_speed": 0,
            "generated_text": "",
            "error": str(e)
        }
    
    # 计算困惑度
    from gptqmodel.utils.perplexity import Perplexity
    ppl_calculator = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="validation",
        text_column="text"
    )
    
    print("计算困惑度...")
    ppl_scores = ppl_calculator.calculate(n_ctx=512, n_batch=32)
    avg_ppl = sum(ppl_scores) / len(ppl_scores)
    print(f"平均困惑度: {avg_ppl:.4f}")
    
    # 生成文本 - 使用聊天格式
    print(f"\n生成文本 (提示: {prompt})...")
    
    # 创建符合Chat模型的提示
    chat_prompt = create_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    end_time = time.time()
    
    # 提取生成的回复部分（去掉输入的提示）
    generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    print(f"生成文本:\n{generated_text}")
    print(f"生成耗时: {end_time - start_time:.2f}秒")
    
    # 计算生成速度 (tokens/second)
    num_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
    gen_speed = num_tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0
    print(f"生成速度: {gen_speed:.2f} tokens/second")
    
    return {
        "ppl": avg_ppl,
        "gen_speed": gen_speed,
        "generated_text": generated_text
    }


def main():
    """主函数"""
    print("===== Qwen3-0.6B 量化比较 (GPTQ V1 vs V2) =====")
    
    # 加载分词器
    print(f"\n加载分词器: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 准备校准数据集
    calibration_dataset = prepare_calibration_dataset(tokenizer)
    
    # 定义GPTQ V1量化配置
    quant_config_v1 = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        sym=True,
        true_sequential=True,
        v2=False  # 使用GPTQ V1
    )
    
    # 定义GPTQ V2量化配置
    quant_config_v2 = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,  # 禁用desc_act，可能有助于解决文本乱码问题
        sym=False,       # 使用非对称量化，可能更适合某些模型
        true_sequential=True,
        v2=True,  # 使用GPTQ V2
        v2_alpha=0.5     # 尝试中等alpha值
    )
    
    # 检查并量化模型，对于V1直接根据路径是否存在判断
    if os.path.exists(V1_MODEL_PATH):
        print(f"\n模型 {V1_MODEL_PATH} 已存在，跳过量化步骤")
        v1_time = 0  # 不存在量化时间
    else:
        v1_time = quantize_model(MODEL_ID, V1_MODEL_PATH, quant_config_v1, calibration_dataset)
        
    # 对于V2，由于我们更改了v2_alpha参数，强制重新执行量化
    print(f"\n由于更改了v2_alpha参数，强制重新量化V2模型")
    # 如果模型文件存在，先删除它
    if os.path.exists(V2_MODEL_PATH):
        import shutil
        shutil.rmtree(V2_MODEL_PATH)
        print(f"已删除旧的V2模型: {V2_MODEL_PATH}")
    # 执行量化
    v2_time = quantize_model(MODEL_ID, V2_MODEL_PATH, quant_config_v2, calibration_dataset)
    
    # 评估模型
    v1_results = evaluate_model(V1_MODEL_PATH, tokenizer)
    v2_results = evaluate_model(V2_MODEL_PATH, tokenizer)
    
    # 比较结果
    print("\n===== 量化结果比较 ====")
    print(f"GPTQ V1:")
    print(f"  量化时间: {v1_time:.2f}秒")
    print(f"  困惑度: {v1_results['ppl']:.4f}")
    print(f"  生成速度: {v1_results['gen_speed']:.2f} tokens/second")
    
    print(f"GPTQ V2:")
    print(f"  量化时间: {v2_time:.2f}秒")
    print(f"  困惑度: {v2_results['ppl']:.4f}")
    print(f"  生成速度: {v2_results['gen_speed']:.2f} tokens/second")
    
    # 计算差异
    ppl_diff = v2_results['ppl'] - v1_results['ppl']
    speed_diff = v2_results['gen_speed'] - v1_results['gen_speed']
    time_diff = v2_time - v1_time
    
    print("\n===== 差异分析 ====")
    print(f"困惑度差异: {ppl_diff:.4f} ({'更好' if ppl_diff < 0 else '更差'})")
    print(f"生成速度差异: {speed_diff:.2f} tokens/second ({'更快' if speed_diff > 0 else '更慢'})")
    print(f"量化时间差异: {time_diff:.2f}秒 ({'更快' if time_diff < 0 else '更慢'})")
    
    print("\n===== 结论 ====")
    if ppl_diff < 0:
        print("GPTQ V2在保持相似生成速度的情况下，提供了更好的模型质量（更低的困惑度）。")
    else:
        print("在本次测试中，GPTQ V1表现优于GPTQ V2。")
    
    print(f"\n量化模型已保存至:\n- GPTQ V1: {V1_MODEL_PATH}\n- GPTQ V2: {V2_MODEL_PATH}")


if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 运行主函数
    main()