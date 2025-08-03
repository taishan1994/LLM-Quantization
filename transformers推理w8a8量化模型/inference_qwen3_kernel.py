import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from transformers import AutoTokenizer, Qwen3Config
# 确保下面的导入指向您修改后的文件
# 例如，如果您将上面的代码保存为 `modeling_qwen3_w8a8.py`
from modeling_qwen3_w8a8_kernel import Qwen3ForCausalLM

# 1. 配置模型
# 确保你使用的 config 和你的权重是匹配的
model_path = "/data/gongoubo/checkpoints/Qwen/llmc/Qwen3-8B-w8a8-offline/vllm_quant_model/" # 或者你的本地模型路径
config = Qwen3Config.from_pretrained(model_path)

model_path = "/data/gongoubo/checkpoints/Qwen/llmc/Qwen3-8B-w8a8-offline/vllm_quant_model/"
model = Qwen3ForCausalLM.from_pretrained(model_path,
                                         torch_dtype=torch.bfloat16,
                                         device_map="auto",
                                         attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained(model_path)


for k,v in model.named_parameters():
    print(k, v.dtype)

model.eval() # 设置为评估模式

# 5. 进行推理
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = [{"role": "user", "content": "你是谁？"}]
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