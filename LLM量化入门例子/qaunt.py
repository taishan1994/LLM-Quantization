import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import defaultdict
from accelerate import init_empty_weights
import torch.nn.functional as F

model_name = "/data/gongoubo/Qwen-1.5-Factory/model_hub/Qwen/Qwen2___5-1___5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval().cuda()

# config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Step 1: 注册 Hook，收集输入 activation
input_cache = defaultdict(list)


def register_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(skip in name for skip in ["embed", "lm_head", "norm", "layernorm"]):
                continue

            def hook_fn(module, input, output, layer_name=name):
                input_cache[layer_name].append(input[0].detach().cpu())

            # 这里只会把q,k,v,o,gate,up,down的输入输出提取出来
            module.register_forward_hook(hook_fn)


register_hooks(model)

# Step 2: 推理样本收集 activations
examples = ["你好", "请问你是谁？", "帮我写一段Python代码"]
examples = [tokenizer.apply_chat_template([{"role": "user", "content": ex}], tokenize=False, add_generation_prompt=True)
            for ex in examples]
for text in examples:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)


# Step 3: 精简版 AWQ 量化
def optimize_awq_alpha(W: torch.Tensor, A: torch.Tensor, n_bits=8, iters=200, lr=0.01):
    """
    A: [N, in] - activation
    W: [out, in] - original weight
    """

    W = W.clone().detach()
    A = A.to(W.device)
    out_features = W.shape[0]

    # 初始化 per-channel 对称量化
    max_val = torch.amax(torch.abs(W), dim=1, keepdim=True)  # [out, 1]
    scale = max_val / (2 ** (n_bits - 1) - 1)
    W_q = torch.round(W / scale).clamp(-128, 127).to(torch.int8)
    W_deq = W_q.float() * scale  # [out, in]

    # 原始输出（浮点）
    Y_fp32 = A @ W.T  # [N, out]
    Y_q_base = A @ W_deq.T  # 初始 AWQ 重建输出

    # 可学习的 alpha
    alpha = torch.ones(out_features, 1, device=W.device, requires_grad=True)

    optimizer = torch.optim.Adam([alpha], lr=lr)

    for _ in range(iters):
        optimizer.zero_grad()

        # 当前量化输出：A @ (alpha * W_deq.T)
        Y_pred = A @ (alpha * W_deq).T  # [N, out]

        loss = F.mse_loss(Y_pred, Y_fp32)
        loss.backward()
        optimizer.step()

    alpha = alpha.detach()
    W_awq = W_q.float() * scale * alpha  # 最终模拟量化权重

    return W_awq, W_q, scale, alpha


class AWQLinear(nn.Module):
    def __init__(self, W_q, scale, bias=None):
        super().__init__()
        self.register_buffer('weight', W_q.to(torch.int8))  # [out, in]
        self.register_buffer('weight_scale', scale.to(torch.float32))  # [out, 1]

        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32))
        else:
            self.bias = None

    def forward(self, x):  # x: [batch, in]
        W = self.weight.to(x.device) * self.weight_scale.to(x.device)  # [out, in]
        if self.bias is not None:
            self.bias = self.bias.to(x.device)
        out = F.linear(x, W, self.bias)
        return out


# Step 4: 替换量化后的权重
def apply_awq_quantization(model):
    new_state_dict = {}
    orig_state_dict = dict(model.named_parameters())
    # for k,v in orig_state_dict:
    #     print(k, type(v))
    # 这里的name是从父模块开始，比如：
    # model
    # model.embed_tokens
    # model.layers.0
    # model.layers.0.self_attn
    # model.layers.0.self_attn.q_pro
    for name, module in model.named_modules():
        # print(name, type(module))
        if name + ".weight" in orig_state_dict:
            tname = name + ".weight"
            if any([1 if skip in tname else 0 for skip in ["embed", "lm_head", "norm", "layernorm"]]):
                new_state_dict[tname] = orig_state_dict[tname]
                # 额外存储lm_head
                if "embed" in tname:
                    new_state_dict["lm_head.weight"] = orig_state_dict[tname]
                continue
        elif name + ".bias" in orig_state_dict:
            # bias不被量化
            tname = name + ".bias"
            new_state_dict[tname] = orig_state_dict[tname]
            continue
        else:
            continue

        hiddem_dim = input_cache[name][0].shape[-1]
        for i, t in enumerate(input_cache[name]):
            input_cache[name][i] = t.reshape(-1, hiddem_dim)

        A = torch.cat(input_cache[name], dim=0)  # [N, in]
        #W = module.weight.data.detach().cpu()

        W = module.weight.data.cuda()

        print(W.dtype)

        # 量化 + 激活感知调整
        W_awq, W_q, scale, alpha = optimize_awq_alpha(W, A)

        # 保存量化信息（可选）
        module.weight_scale = scale * alpha

        # 更新权重（模拟量化）
        module.weight.data.copy_(W_q.to(module.weight.device))
        print(f"AWQ quantized: {name} | weight shape: {W.shape}")

        print('attn values sanity check:', torch.allclose(W, W_awq, rtol=0, atol=1e-02))


apply_awq_quantization(model)

# for name, module in model.named_modules():
#     if name in input_cache:
#         print(name, module.weight.shape)
#         print(name, module.weight_scale.shape)
#         print("="*100)


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
        if not hasattr(module, 'weight_scale'):
            continue  # 说明没量化过

        # 构造 AWQLinear
        awq_layer = AWQLinear(
            W_q=module.weight,
            scale=module.weight_scale,
            bias=module.bias
        )

        # 替换模块
        parent, child_name = get_parent_module(model, name)
        setattr(parent, child_name, awq_layer)
        print(f"Replaced {name} with AWQLinear")


replace_with_awq_linear(model)

output_dir = "output/qwen2.5-1.5B-Instruct-AWQ"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

model.config.tie_word_embeddings = False

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

for k,v in model.named_parameters():
    print(k, v.shape)

# 推理测试
text = "你好，请用一句话描述你自己"
text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(output[0], skip_special_tokens=False))
