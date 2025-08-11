import argparse
import time
import json
import os
import torch
import torch.multiprocessing as mp
from contextlib import contextmanager

# --- Web Server Imports ---
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Model & Quantization Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from neural_compressor.torch.quantization import MXQuantConfig, prepare, convert


# =============================================================================
# 1. 定义API的输入输出格式 (类OpenAI)
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str  # The model name
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    max_tokens: Optional[int] = 31000  # Renamed from max_new_tokens for OpenAI compatibility
    stream: Optional[bool] = False  # Streaming is not implemented in this example


# --- Response Models ---
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    # usage: ... # Usage stats can be added if needed


# =============================================================================
# 2. 将模型加载和量化封装成一个函数
# =============================================================================

def load_and_quantize_model(args, device):
    """
    在指定的设备上加载、量化并返回模型和分词器。
    """
    print(f"[{device}] 开始加载模型 {args.model}...")

    # 加载原始模型和分词器
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        #attn_implementation="flash_attention_2"
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 加载以节省内存
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    # 适配 Llama3 等新模型可能没有 pad_token 的情况
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.peft_model_id:
        print(f"[{device}] 应用PEFT adapter: {args.peft_model_id}")
        user_model = PeftModel.from_pretrained(user_model, args.peft_model_id)

    user_model.eval()

    # 将模型移动到目标GPU
    user_model.to(device)
    print(f"[{device}] 模型已移动到 {device}")

    if args.quantize:
        print(f"[{device}] 开始量化模型 (w_dtype={args.w_dtype}, act_dtype={args.act_dtype})...")
        quant_config = MXQuantConfig(w_dtype=args.w_dtype, act_dtype=args.act_dtype, weight_only=args.woq)
        # 注意: prepare 和 convert 应该在模型移动到GPU之后进行
        user_model = prepare(model=user_model, quant_config=quant_config)
        user_model = convert(model=user_model)
        user_model.eval()
        print(f"[{device}] 模型量化完成。")
        user_model.to(device)
        print(f"[{device}] 模型已移动到 {device}")

    user_model = torch.compile(user_model, mode="reduce-overhead", fullgraph=True)
    print(f"[{device}] 模型编译完成。")
    return user_model, tokenizer


# =============================================================================
# 3. 定义每个GPU上运行的服务进程
# =============================================================================

def run_server(rank, args, base_port):
    """
    为单个GPU启动一个FastAPI服务。
    - rank: GPU的索引 (0, 1, 2, ...)
    - args: 命令行参数
    - base_port: 服务的起始端口号
    """
    # 关键步骤: 为当前进程设置可见的GPU
    device = f"cuda:{rank}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # 清理显存缓存
    torch.cuda.empty_cache()

    print(f"[Process {os.getpid()}] 正在为 {device} 初始化服务...")

    # 加载并量化模型
    # device = "cuda:7"
    model, tokenizer = load_and_quantize_model(args, device)

    # 创建FastAPI应用
    app = FastAPI()

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        """
        处理聊天补全请求。
        """
        # 1. 将OpenAI格式的messages转换为模型能理解的单个prompt
        #    Hugging Face的 `apply_chat_template` 是标准做法
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in request.messages],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Error applying chat template: {e}. Falling back to simple concatenation.")
            prompt_text = "\n".join([f"{m.role}: {m.content}" for m in request.messages])

        # 2. Tokenize 输入
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # 3. 模型推理
        print(f"[{device}] 正在生成回复...")
        with torch.no_grad():
            output_sequences = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=True if request.temperature > 0 else False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 4. 解码输出
        # 我们只解码生成的新部分
        generated_text = tokenizer.decode(output_sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        print(f"[{device}] 生成完成: {generated_text[:100]}...")

        # 5. 构建并返回OpenAI格式的响应
        response_message = ChatMessage(role="assistant", content=generated_text)
        choice = ChatCompletionResponseChoice(index=0, message=response_message)

        return ChatCompletionResponse(
            id=f"chatcmpl-{time.time()}",
            model=request.model,
            choices=[choice]
        )

    # 启动Uvicorn服务器
    port = base_port + rank
    print(f"[Process {os.getpid()}] 服务已在 {device} 上启动, 监听 http://0.0.0.0:{port}")
    print(f"API文档地址: http://127.0.0.1:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)


# =============================================================================
# 4. 主程序入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- 模型和量化参数 ---
    parser.add_argument(
        "--model", nargs="?", default="EleutherAI/gpt-j-6b"
    )
    parser.add_argument(
        "--trust_remote_code", default=True,
        help="Transformers parameter: use the external repo")
    parser.add_argument(
        "--revision", default=None,
        help="Transformers parameter: set the model hub commit number")
    parser.add_argument("--quantize", action="store_true")
    # dynamic only now
    parser.add_argument("--w_dtype", type=str, default="int8",
                        choices=["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3", "fp6_e3m2",
                                 "fp6_e2m3", "fp4", "float16", "bfloat12"],
                        help="weight data type")
    parser.add_argument("--act_dtype", type=str, default="int8",
                        choices=["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3", "fp6_e3m2",
                                 "fp6_e2m3", "fp4", "float16", "bfloat12"],
                        help="input activation data type")
    parser.add_argument("--woq", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--iters", default=100, type=int,
                        help="For accuracy measurement only.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="For accuracy measurement only.")
    parser.add_argument("--save_accuracy_path", default=None,
                        help="Save accuracy results path.")
    parser.add_argument("--tasks", nargs="+", default=["lambada_openai"], type=str,
                        help="tasks list for accuracy validation"
                        )
    parser.add_argument("--peft_model_id", type=str, default=None, help="model_name_or_path of peft model")

    # --- 服务部署参数 ---
    parser.add_argument("--num_gpus", type=int, default=8, help="要使用的GPU数量")
    parser.add_argument("--base_port", type=int, default=8000, help="服务的起始端口号")


    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPUs.")

    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"警告: 请求使用 {args.num_gpus} GPUs, 但只有 {available_gpus} 可用。将使用所有可用GPU。")
        args.num_gpus = available_gpus

    # --- 使用 'spawn' 方法启动多进程, 这对于CUDA是必须的 ---
    mp.set_start_method("spawn", force=True)

    processes = []
    for rank in range(args.num_gpus):
        p = mp.Process(target=run_server, args=(rank, args, args.base_port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # 主进程等待所有服务进程结束 (正常情况下它们会一直运行)