import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import defaultdict
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils import get_balanced_memory

import torch.nn.functional as F

model_name = "/data/gongoubo/Qwen-1.5-Factory/model_hub/Qwen/Qwen2___5-1___5B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
# model.eval().cuda()

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


class AWQLinear(nn.Module):
    def __init__(self, W_q, bias=None, device=None):
        super().__init__()
        self.register_buffer('weight', W_q.to(torch.int8).to(device))  # [out, in]
        self.register_buffer('weight_scale', torch.zeros(W_q.shape[0], 1, dtype=torch.float16, device=device))  # [out, 1]

        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float16).to(device))
        else:
            self.bias = None

    def forward(self, x):  # x: [batch, in]
        W = self.weight * self.weight_scale  # [out, in]
        if self.bias is not None:
            self.bias = self.bias
        out = F.linear(x, W, self.bias)
        return out


from safetensors.torch import load_file
state_dict = load_file("output/qwen2.5-1.5B-Instruct-AWQ/model.safetensors")
state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

linear_layers_to_replace = []
print("识别需要替换的 nn.Linear 层...")
for k, v in state_dict.items(): # 迭代这个在 'meta' device 上的模型
    if "scale" in k:
        # print(k)
        linear_layers_to_replace.append(k)


def get_parent_module(model, full_name):
    """获取模块的父对象与属性名，用于替换子模块"""
    names = full_name.split(".")
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent, names[-1]


def replace_with_awq_linear(model):
    # 注意这里的module不是最终的子模块，比如
    # <class 'torch.nn.modules.activation.SiLU'>
    # <class 'transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm'>
    # <class 'transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm'>
    # <class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>
    # <class 'transformers.models.qwen2.modeling_qwen2.Qwen2Attention'>
    # 当然也有可能会存在nn.Linear
    for name, module in model.named_modules():
        if name + ".weight_scale" in linear_layers_to_replace:
            weight_scale = state_dict[name + ".weight_scale"]

            # 构造 AWQLinear
            awq_layer = AWQLinear(
                W_q=module.weight,
                bias=module.bias,
                device=module.weight.device,
            )

            # 替换模块
            parent, child_name = get_parent_module(model, name)
            setattr(parent, child_name, awq_layer)
            print(f"Replaced {name} with AWQLinear")


replace_with_awq_linear(model)

print(model)

model.load_state_dict(state_dict, assign=True)
print(model.device)
# 推理设备映射（多卡自动分层）
max_memory = get_balanced_memory(model, max_memory={0: "20GiB", 1: "20GiB"}, dtype=torch.float16)

model.tie_weights()
device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["Qwen2DecoderLayer"])

# 将模型分布在多卡上
model = dispatch_model(model, device_map=device_map)

# 推理测试
text = "你好，请用一句话描述你自己"
text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(output[0], skip_special_tokens=False))
