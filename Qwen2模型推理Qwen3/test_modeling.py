import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# from modeling_qwen3 import Qwen3ForCausalLM
from eval_utils.modeling_qwen2 import Qwen2ForCausalLM

# 1. 准备模型和分词器
model_name = "/data/gongoubo/checkpoints/Qwen/Qwen3-0.6B"
# model_name = "/data/gongoubo/Qwen-1.5-Factory/model_hub/Qwen/Qwen2___5-1___5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config._attn_implementation = "eager"
model = Qwen2ForCausalLM.from_pretrained(model_name, config=config).to("cuda:0")

# 确保模型在评估模式
model.eval()

max_new_tokens = 256

# 2. 准备初始输入
message = [{"role":"user", "content":"你是谁？"}]
message = tokenizer.apply_chat_template(message, tokenize=False, add_special_tokens=True)
input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)

direct_output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.01)
direct_text = tokenizer.decode(direct_output[0])
print(f"直接调用 model.generate 的文本: \n'{direct_text}'")