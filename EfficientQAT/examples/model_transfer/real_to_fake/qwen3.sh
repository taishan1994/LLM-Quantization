# llama-2-7b-w2g64
#CUDA_VISIBLE_DEVICES=0 python -m model_transfer.real_to_fake \
#--model /data/gongoubo/other_project/EfficientQAT/output/block_ap_models/qwen3-4b-w4g128 \
#--save_dir /data/gongoubo/other_project/EfficientQAT/output/block_ap_models/qwen3-4b-w4g128-fake \
#--wbits 4 \
#--group_size 128


CUDA_VISIBLE_DEVICES=0 python -m model_transfer.real_to_fake \
--model /data/gongoubo/other_project/EfficientQAT/output/e2e-qp-output/qwen3-4b-w4g128-redpajama-4096/checkpoint-170 \
--save_dir /data/gongoubo/other_project/EfficientQAT/output/block_ap_models/qwen3-4b-w4g128-e2e-fake \
--wbits 4 \
--group_size 128