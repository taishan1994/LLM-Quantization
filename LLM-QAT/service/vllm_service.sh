# /home/gongoubo/project/LLM-QAT/model_hub/Qwen/Qwen3-4B/models/qwen3-4B-finetuned
# /home/gongoubo/project/LLM-QAT/model_hub/Qwen/Qwen3-4B
# /home/gongoubo/project/LLM-QAT/model_hub/Qwen/Qwen3-4B/models/qwen3-4B-quant-finetuned-qat-rtn-w8a8/vllm_quant_model
# /home/gongoubo/project/LLM-QAT/model_hub/Qwen/Qwen3-4B/models/qwen3-4B-finetuned2
# /home/gongoubo/project/LLM-QAT/model_hub/Qwen/Qwen3-4B/models/qwen3-4B-finetuned2-rtn-w8a8/vllm_quant_model


CUDA_VISIBLE_DEVICES=4,5 vllm serve /home/gongoubo/project/LLM-QAT/model_hub/Qwen/Qwen3-4B/models/qwen3-4B-gen-quant-finetuned-qat-rtn-w8a8/vllm_quant_model \
--served-model-name Qwen3-4B-gen-quant-finetuned2-rtn-w8a8 \
--max-model-len 34000 \
--tensor-parallel-size 2 \
--host 0.0.0.0 \
--port 11778