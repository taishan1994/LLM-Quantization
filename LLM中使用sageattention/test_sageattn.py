import torch
from sageattention import sageattn  # 确保你已经正确安装或实现该模块

# 设置随机种子，方便复现
torch.manual_seed(42)

# 参数配置
batch_size = 2
num_heads = 4
seq_len = 16
head_dim = 64

# 构造输入张量（float16 或 float32，视情况而定）
q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device='cuda:0')
k = torch.randn_like(q)
v = torch.randn_like(q)


# 调用 SAGE Attention
output2 = sageattn(q, k, v, tensor_layout="HND", is_causal=True)
print(output2)
# 输出检查
print(f"输入 shape: {q.shape}")
print(f"输出 shape: {output2.shape}")
print(f"输出数据类型: {output2.dtype}")
print(f"输出是否在 CUDA 上: {output2.is_cuda}")