# Qwen3-4B-quarot-sym-w4a16/transformed_model-quarot-gptq-W4A16-G128/
# /data/gongoubo/checkpoints/Qwen/Qwen3-4B-gptq-W4A16-G128
# /data/gongoubo/checkpoints/Qwen/Qwen3-4B-quarot-sym-w4a16/transformed_model-quarot-gptq-W4A16-G128/
# /data/gongoubo/other_project/EfficientQAT/output/block_ap_models/qwen3-4b-w4g128
# /data/gongoubo/checkpoints/Qwen/Qwen3-4B-AWQ

CUDA_VISIBLE_DEVICES=6,7 vllm serve /data/gongoubo/checkpoints/Qwen/Qwen3-4B-AWQ \
--served-model-name  qwen3-4B-awq \
--max-model-len 34000 \
--tensor-parallel-size 2 \
--host 0.0.0.0 \
--port 18003 \
--gpu-memory-utilization 0.85 \
--enforce-eager
