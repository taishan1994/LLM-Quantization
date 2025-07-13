import torch


state = torch.load('qwen3_0.6b_w8a8', weights_only=False)

w_quantizers = state['w_quantizers']
model = state['model']
for k,v in model.items():
    if k == 'model.layers.0.self_attn.q_proj.weight':
        print(k, v.dtype)
        print(v)
