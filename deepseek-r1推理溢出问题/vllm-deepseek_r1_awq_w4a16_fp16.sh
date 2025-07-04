

pip install vllm==0.7.3

cp deepseek_v2.py /usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/deepseek_v2.py

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=0

vllm serve bf16-awq-g64 \
  --trust-remote-code \
  --port 32000 \
  --host 0.0.0.0 \
  --dtype float16 \
  --tensor-parallel-size 8 \
  --max-model-len 34000 \
  --max-num-batched-tokens 34000 \
  --enforce-eager  \
  --gpu-memory-utilization 0.85