pip install vllm==0.8.5.post1

export CUDA_VISIBLE_DEVICES=0

vllm serve /ms/FM/gongoubo/checkpoints/data/Qwen2___5-VL-7B-Instruct-W4A16-include-vit  \
  --tensor_parallel_size 1 \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --enforce-eager
