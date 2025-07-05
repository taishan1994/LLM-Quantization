pip install vllm==0.8.5.post1

export CUDA_VISIBLE_DEVICES=0

cp vllm_0.8.5.post1/qwen2_5_vl.py /usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen2_5_vl.py
cp vllm_0.8.5.post1/_custom_ops.py /usr/local/lib/python3.10/dist-packages/vllm/_custom_ops.py

vllm serve /ms/FM/gongoubo/checkpoints/data/Qwen2___5-VL-7B-Instruct-W4A16-include-vit  \
  --tensor_parallel_size 1 \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --enforce-eager
