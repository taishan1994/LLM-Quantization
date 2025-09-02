export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

#--max_train_samples 4096 \
#--num_train_epochs 1 \
#max_steps=max_train_samples/(per_device_train_batch_size*gradient_accumulation_steps*num_gpus)
#=4096/(1*8*3)=170

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --nnodes=1 main_e2e_qp.py \
    --quant_model_path ./output/block_ap_models/qwen3-4b-w4g128 \
    --model_fanmily qwen3 \
    --wbits 4 \
    --group_size 128 \
    --learning_rate 1e-5 \
    --dataset redpajama \
    --dataset_format pt \
    --output_dir ./output/e2e-qp-output/qwen3-4b-w4g128-redpajama-4096 \
    --do_train True \
    --pt_context_len 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --save_strategy epoch \
    --training_strategy epochs \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --max_train_samples 4096 \
    --max_steps 170 \
    --eval_dataset_size 64 \
    --bf16 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    --preprocessing_num_workers 32