export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=0

vllm serve r1-bf16 \
  --trust-remote-code \
  --port 32000 \
  --host 0.0.0.0 \
  --dtype float16 \
  --tensor-parallel-size 32 \
  --max-model-len 34000 \
  --max-num-batched-tokens 34000 \
  --enforce-eager  \
  --gpu-memory-utilization 0.85