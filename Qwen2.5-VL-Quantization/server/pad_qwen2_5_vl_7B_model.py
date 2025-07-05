import os
import sys
import torch

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
ld_library = os.environ.get("LD_LIBRARY_PATH", "")
print(ld_library)

import torch.nn.functional as F

new_path = "/opt/hpcx/ucx/lib"
if new_path not in ld_library.split(":"):
    os.environ["LD_LIBRARY_PATH"] = f"{new_path}:{ld_library}" if ld_library else new_path
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForVision2Seq, AutoProcessor
from vision_process import process_vision_info

# model_path = "/ms/FM/checkpoints/Qwen-Zoo/Qwen2___5-VL-7B-Instruct"
model_path = "/ms/FM/tangqie/llmc/save/Qwen2___5-VL-7B-Instruct_sym_w8_a8-dynamic_fix_bias/transformed_model"

# default: Load the model on the available device(s)
#  device_map="auto",
model = AutoModelForVision2Seq.from_pretrained(
    model_path, torch_dtype="auto", trust_remote_code=True
).to("cuda:0")

sd = model.named_parameters()

gsize = 128

new_state = {}
index = {
    "metadata": "",
    "weight_map": "",
}
# visual的gate、up、down是3240，不能被128整除
for i, (k,v) in enumerate(dict(sd).items()):
    print(k, i, v.shape)
    if "visual" in k:
        # 模拟输入
        if "gate_proj" in k or "up_proj" in k:
            if "weight" in k:
                x = torch.ones((1, v.shape[1]))
                pad_size = (int(v.shape[0] // gsize) + 1) * gsize - v.shape[0]
                new_v = F.pad(v, (0, 0, 0, pad_size))
                print(v.shape, new_v.shape)
                new_state[k] = new_v
                print(v, new_v)
            elif "bias" in k:
                pad_size = (int(v.shape[0] // gsize) + 1) * gsize - v.shape[0]
                new_v = F.pad(v, (0, pad_size))
                print(v.shape, new_v.shape)
                new_state[k] = new_v
                print(v, new_v)
        elif "down_proj" in k:
            if "weight" in k:
                pad_size = (int(v.shape[1] // gsize) + 1) * gsize - v.shape[1]
                new_v = F.pad(v, (0, pad_size, 0, 0))
                print(v.shape, new_v.shape)
                new_state[k] = new_v
                print(v, new_v)
            else:
                print(v)
                new_state[k] = v
        else:
            new_state[k] = v
    else:
        new_state[k] = v

# gsize = 128  # 保证维度对齐的倍数
#
# for i, k in enumerate(sd):
#     v = sd[k]
#     print(k, i, v.shape)
#
#     if "visual" in k:
#         if "gate_proj" in k or "up_proj" in k:
#             if "weight" in k:
#                 pad_size = (int(v.shape[0] // gsize) + 1) * gsize - v.shape[0]
#                 # 插入空行（interleaved padding）
#                 interleaved = F.pad(v.unsqueeze(1), (0, 0, 0, 1, 0, 0))  # [N, 2, D]
#                 interleaved = interleaved.reshape(v.shape[0] * 2, v.shape[1])
#                 interleaved = interleaved[:pad_size * 2]
#                 new_v = torch.cat([interleaved, v[pad_size:]], dim=0)
#                 print(v.shape, new_v.shape)
#                 new_state[k] = new_v
#
#             elif "bias" in k:
#                 pad_size = (int(v.shape[0] // gsize) + 1) * gsize - v.shape[0]
#                 interleaved = F.pad(v.unsqueeze(1), (0, 1)).reshape(-1)[:pad_size * 2]
#                 new_v = torch.cat([interleaved, v[pad_size:]], dim=0)
#                 print(v.shape, new_v.shape)
#                 new_state[k] = new_v
#
#         elif "down_proj" in k:
#             if "weight" in k:
#                 pad_size = (int(v.shape[1] // gsize) + 1) * gsize - v.shape[1]
#                 interleaved = F.pad(v.unsqueeze(2), (0, 1))  # [M, N, 2]
#                 interleaved = interleaved.reshape(v.shape[0], -1)[:, :pad_size * 2]
#                 new_v = torch.cat([interleaved, v[:, pad_size:]], dim=1)
#                 print(v.shape, new_v.shape)
#                 new_state[k] = new_v
#             else:
#                 new_state[k] = v
#
#     else:
#         new_state[k] = v


from safetensors.torch import save_file

# # if not os.path.exists("/ms/FM/gongoubo/checkpoints/data/Qwen2___5-VL-7B-Instruct-editQwen2___5-VL-7B-Instruct-edit/"):
# #     os.makedirs("/ms/FM/gongoubo/checkpoints/data/Qwen2___5-VL-7B-Instruct-edit", exist_ok=True)

save_file(new_state, "/ms/FM/gongoubo/checkpoints/data/Qwen2___5-VL-7B-Instruct-edit/model.safetensors")