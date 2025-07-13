import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_name = "/data/gongoubo/checkpoints/Qwen/llmc/Qwen3-8B-w8a8-online/fake_quant_model/"
# model_name = "/data/gongoubo/Qwen-1.5-Factory/model_hub/Qwen/Qwen2___5-1___5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
CONFIG = AutoConfig.from_pretrained(model_name)
# transformers==4.53.0
# from modeling_qwen3 import Qwen3ForCausalLM

process_word_embeddings= False
if CONFIG.tie_word_embeddings:
    CONFIG.tie_word_embeddings = False
    process_word_embeddings = True
from modeling_qwen3_online_llmc import Qwen3ForCausalLM
# from modeling_qwen3_online_r3_r4 import Qwen3ForCausalLM
# from modeling_qwen3 import Qwen3ForCausalLM
model = Qwen3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, config=CONFIG).to("cuda:0")

# model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
# 确保模型在评估模式
model.eval()

# message = [{"role":"user", "content":"你是谁？"}]
# message = tokenizer.apply_chat_template(message, tokenize=False, add_special_tokens=True)

message = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(message, return_tensors="pt")
input_ids = input_ids.to(model.device)
direct_output = model.generate(input_ids, max_new_tokens=256, do_sample=False, temperature=1)
direct_text = tokenizer.decode(direct_output[0])

print(direct_text)