cp /data/gongoubo/other_project/EfficientQAT/examples/block_ap/qwen3/modeling_qwen3.py /data/anaconda3/envs/py311/lib/python3.11/site-packages/transformers/models/qwen3/modeling_qwen3.py

CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
--model /data/gongoubo/checkpoints/Qwen/Qwen3-4B  \
--output_dir ./output/block_ap_log/qwen3-4b-w4g128 \
--wbits 4 \
--group_size 128 \
--quant_lr 1e-4 \
--weight_lr 1e-5 \
--save_quant_dir ./output/block_ap_models/qwen3-4b-w4g128 \
--real_quant