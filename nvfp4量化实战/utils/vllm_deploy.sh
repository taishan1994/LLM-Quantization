pip uninstall flash-attn -y

#pip install vllm==0.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# pip install numpy==2.2.0

# https://pypi.tuna.tsinghua.edu.cn/simple
model_path=/nfs/FM/gongoubo/new_project/workflow/checkpoints/qwen3-vl-8b-NVFP4
port=32000

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

CUDA_VISIBLE_DEVICES=0 vllm serve ${model_path} --host 0.0.0.0 --port ${port} --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 12400
