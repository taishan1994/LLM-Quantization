import torch
from transformers import AutoTokenizer, Qwen3Config
# 确保下面的导入指向您修改后的文件
# 例如，如果您将上面的代码保存为 `modeling_qwen3_w8a8.py`
from modeling_qwen3_w8a8 import Qwen3ForCausalLM

# 1. 配置模型
# 确保你使用的 config 和你的权重是匹配的
model_path = "/data/gongoubo/checkpoints/Qwen/Qwen3-0___6B_ori" # 或者你的本地模型路径
config = Qwen3Config.from_pretrained(model_path)

# 2. 实例化 W8A8 模型
# 使用 from_config 创建一个空的 W8A8 模型结构
# 注意：这里我们不能使用 from_pretrained，因为它会尝试加载原始的 fp16/bf16 权重
# 并且由于我们更改了模型定义，直接加载会失败。
print("Initializing empty W8A8 model structure...")
# model = Qwen3ForCausalLM(config)

# 3. 加载你的 W8A8 state_dict
# 假设你的 W8A8 权重保存在 "path/to/your/w8a8_weights.pth"
print("Loading W8A8 weights...")
# from safetensors.torch import load_file
# w8a8_state_dict = load_file("/data/gongoubo/checkpoints/Qwen/llmc/Qwen3-0.6B-w8a8-quarot/vllm_quant_model/model.safetensors")
# w8a8_state_dict = load_file("/data/gongoubo/checkpoints/Qwen/Qwen3-0___6B_ori/model.safetensors")
# model.load_state_dict(w8a8_state_dict, strict=True)

model_path = "/data/gongoubo/checkpoints/Qwen/llmc/Qwen3-8B-w8a8-offline/vllm_quant_model/"
model = Qwen3ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("W8A8 weights loaded successfully.")

# 4. 使用 accelerate 进行多卡推理
# device_map="auto" 会自动将模型分片到所有可用的 GPU 上
# 这是实现多卡推理最简单的方法
print("Distributing model across GPUs with accelerate...")
# model.to(torch.bfloat16) # 将模型的非量化部分（如 embedding, layernorm）转为合适的精度

# # 正确的做法：遍历参数，有选择地转换
# with torch.no_grad():
#     for name, param in model.named_parameters():
#         # 如果参数名中包含 "weight_scale"，则跳过它，保持其 float32 类型
#         if "weight_scale" in name:
#             # print(f"  - Skipping {name}, keeping dtype {param.dtype}")
#             continue
#
#         # 对于所有其他参数（如 embeddings, layernorm.weight, bias, 和 lm_head.weight），
#         # 如果它们是浮点数，就将它们转换为 bfloat16
#         if param.is_floating_point():
#             # print(f"  - Converting {name} from {param.dtype} to torch.bfloat16")
#             param.data = param.data.to(torch.bfloat16)
#
# print("Selective conversion finished.")


for k,v in model.named_parameters():
    print(k, v.dtype)

model.eval() # 设置为评估模式
# accelerate 会处理设备放置
# model = model.to("cuda" if torch.cuda.is_available() else "cpu", non_blocking=True) # or use Accelerate's dispatch_model
# 如果你有 Accelerate，更推荐的方式是：
# from accelerate import Accelerator
# accelerator = Accelerator()
# model = accelerator.prepare(model)

# 5. 进行推理
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = [{"role": "user", "content": "你是谁？"}]
prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成文本
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256)

print(tokenizer.decode(outputs[0], skip_special_tokens=False))