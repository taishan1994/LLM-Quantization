from safetensors.torch import load_file


path2 = "/data/gongoubo/other_project/EfficientQAT/output/block_ap_models/qwen3-4b-w4g128/"
path = "/data/gongoubo/checkpoints/Qwen/Qwen3-4B-awq-W4A16-asy-G128/"
path = "/data/gongoubo/checkpoints/Qwen/Qwen3-4B-AWQ/"
path = "/data/gongoubo/other_project/EfficientQAT/output/e2e-qp-output/qwen3-4b-w4g128-redpajama-4096/checkpoint-170/"
path = path + "model.safetensors"
# path2 = path2 + "model.safetensors"
state = load_file(path)
# state2 = load_file(path2)

print("="*100)
for k,v in state.items():
    print(k, v.shape)
