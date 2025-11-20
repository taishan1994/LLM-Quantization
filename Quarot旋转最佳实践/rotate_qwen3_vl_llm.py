import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from qwen_vl_utils import process_vision_info
# from transformers.models.qwen3 import Qwen3ForCausalLM


def inference_normal(model, processor):

    # message = [{"role":"user", "content":"你是谁？"}]
    # message = tokenizer.apply_chat_template(message, tokenize=False, add_special_tokens=True)
    #     message = """<|im_start|>user
    # <|vision_start|><|image_pad|><|vision_end|>你是谁？<|im_end|>
    # <|im_start|>assistant
    # """
    #     input_ids = tokenizer.encode(message, return_tensors="pt")
    #     input_ids = input_ids.to(model.device)
    #     direct_output = model.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    #     direct_text = tokenizer.decode(direct_output[0])

    #     print(direct_text)
    print("========== SAMPLE GENERATION ==============")
    # dispatch_for_generation(model)
    messages = [
        {
            "role": "user",
            "content": [
                # {
                #     "type": "image",
                #     "image": "http://images.cocodataset.org/train2017/000000231895.jpg",
                # },
                {"type": "text", "text": "老婆饼里面为什么没有老婆"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    print(prompt)
    image_inputs, video_inputs = process_vision_info(messages)
    MAX_SEQUENCE_LENGTH=2048
    inputs = processor(
        text=[prompt],
        images=None,
        videos=None,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
    print(processor.decode(output[0], skip_special_tokens=False))
    print("==========================================")


def merge_rmsnorm_in_model(model):
    """
    将模型中所有 LlamaRMSNorm 层的参数合并到其后的线性层中。

    这个函数会遍历模型的所有 decoder layer，并将以下配对的参数进行合并：
    1. input_layernorm -> q_proj, k_proj, v_proj
    2. post_attention_layernorm -> gate_proj, up_proj

    合并后，RMSNorm 层的 weight 会被设置为 1.0，使其在数学上成为一个纯粹的归一化层，
    而其缩放效果已经提前应用到了线性层的权重上。

    Args:
        model (nn.Module): Hugging Face Transformer 模型。
    """
    print("开始合并 RMSNorm 参数...")

    # 检查模型是否包含预期的 decoder layers 结构
    if not hasattr(model, 'language_model') or not hasattr(model.language_model, 'layers'):
        print("警告: 模型结构不符合预期 (如 Llama, Mistral)。跳过合并。")
        return

    for layer in model.language_model.layers:
        # 1. 合并 Attention 块中的 RMSNorm
        # input_layernorm -> q_proj, k_proj, v_proj

        # 获取 RMSNorm 层的 gamma 参数 (weight)
        # 使用 .data 避免梯度追踪
        gamma_attn = layer.input_layernorm.weight.data

        # 获取线性层的权重
        q_weight = layer.self_attn.q_proj.weight.data
        k_weight = layer.self_attn.k_proj.weight.data
        v_weight = layer.self_attn.v_proj.weight.data

        # 执行合并：将 gamma 乘到线性层的权重上
        # PyTorch 的广播机制会自动处理 (out_features, in_features) * (in_features,)
        q_weight *= gamma_attn
        k_weight *= gamma_attn
        v_weight *= gamma_attn

        # 将 RMSNorm 的 gamma 参数重置为 1
        layer.input_layernorm.weight.data.fill_(1.0)

        # 2. 合并 MLP 块中的 RMSNorm
        # post_attention_layernorm -> gate_proj, up_proj

        gamma_mlp = layer.post_attention_layernorm.weight.data

        gate_weight = layer.mlp.gate_proj.weight.data
        up_weight = layer.mlp.up_proj.weight.data

        gate_weight *= gamma_mlp
        up_weight *= gamma_mlp

        layer.post_attention_layernorm.weight.data.fill_(1.0)

    # 还要对lm_head之间的model_norm进行合并
    model_norm_weight = model.language_model.norm.weight.data
    print(model_norm_weight.shape)
    print(model.lm_head.weight.data.shape)
    model.lm_head.weight.data = model.lm_head.weight.data * model_norm_weight
    model.language_model.norm.weight.data.fill_(1.0)
    print("RMSNorm 参数合并完成！")


def fuse_layer_rotation(layer, R_in=None, R_out=None):
    """
    将输入旋转 (R_in) 和输出旋转 (R_out) 合并到线性层。
    W_new = R_out.T @ W @ R_in
    """
    device, dtype = layer.weight.device, layer.weight.dtype
    W = layer.weight.data.to(torch.float32)
    W = W.T
    with torch.no_grad():
        # 融合输入旋转
        if R_in is not None:
            R_in = R_in.to(W.device, torch.float32)
            W = R_in @ W

        # 融合输出旋转
        if R_out is not None:
            R_out = R_out.to(W.device, torch.float32)
            W = W @ R_out

    W = W.T
    layer.weight.data = W.to(device=device, dtype=dtype)


def create_block_diag_from_head_matrix(R_head, num_heads):
    """
    根据每个头的旋转矩阵 R_head，创建一个分块对角矩阵。
    """
    return torch.block_diag(*[R_head for _ in range(num_heads)])

if __name__ == '__main__':
    # 1. 准备模型和分词器
    model_name = "/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-VL-4B-Instruct/Qwen/Qwen3-VL-4B-Instruct"
    # model_name = "/data/gongoubo/Qwen-1.5-Factory/model_hub/Qwen/Qwen2___5-1___5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    CONFIG = AutoConfig.from_pretrained(model_name)
    # from modeling_qwen3 import Qwen3ForCausalLM
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    process_word_embeddings= False
    if CONFIG.tie_word_embeddings:
        CONFIG.tie_word_embeddings = False
        process_word_embeddings = True
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, config=CONFIG).to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    # 确保模型在评估模式
    model.eval()
    #
    inference_normal(model, processor)

    if process_word_embeddings:
        model.lm_head.weight.data = model.language_model.embed_tokens.weight.data.clone()

    merge_rmsnorm_in_model(model)

    # for k,v in model.language_model.named_parameters():
    #     if "norm" in k:
    #         print(k, v)
    #
    inference_normal(model, processor)

    # model.save_pretrained("/nfs/FM/gongoubo/new_project/qwen-vl-quant/rotated_qwen3_vl/Qwen3-VL-4B-Instruct-merge-norm")
    # tokenizer.save_pretrained("/nfs/FM/gongoubo/new_project/qwen-vl-quant/rotated_qwen3_vl/Qwen3-VL-4B-Instruct-merge-norm")

    # print(CONFIG)
    text_config = CONFIG.text_config
    hidden_size = text_config.hidden_size
    head_dim = text_config.head_dim
    num_q_heads = text_config.num_attention_heads
    num_kv_heads = text_config.num_key_value_heads

    print(hidden_size, head_dim, num_q_heads, num_kv_heads)

    from hadamard_utils import (
        get_hadK,
        apply_exact_had_to_linear,
        random_hadamard_matrix,
    )

    R1 = random_hadamard_matrix(hidden_size, device=model.device)

    # 5. 执行旋转合并
    logger.info("开始将旋转矩阵 R1 合并到模型权重中...")

    R1_inv = R1.T

    print(R1 @ R1_inv)

    # 右乘的时候
    # 对与embedding层，直接右乘R即可
    # q,k,v的权重要要左乘以R.T
    # x @ (R.T @ W.T)
    # 对于o
    # x @ W.T @ R
    # x1 @ R1 @ (R1.T @ W_Q.T)
    # x2 @ R1 @ (R1.T @ W_K.T)
    # x1 @ R1 @ (R1.T @ W_V.T @ R2) @ (R2.T @ W_O.T @ R1) @ (R1.T @ R_UP.T) @ (W_DOWN.T @ R1) @ (R1.T @ W_HEAD)

    # embedding层直接右乘即可
    with torch.no_grad():
        device, dtype = model.language_model.embed_tokens.weight.device, model.language_model.embed_tokens.weight.dtype
        W = model.language_model.embed_tokens.weight.data.to(torch.float32)
        R1 = R1.to(model.language_model.embed_tokens.weight.device, dtype=torch.float32)
        new_w = W @ R1
        model.language_model.embed_tokens.weight.data = new_w.to(model.language_model.embed_tokens.weight.dtype)
    logger.info("已合并 Embedding 层。")

    # b) 遍历所有 Transformer Blocks
    for i, layer in enumerate(model.language_model.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        # Wq' = Wq @ (R1_inv).T = Wq @ R1
        fuse_layer_rotation(attn.q_proj, R_in=R1_inv)
        fuse_layer_rotation(attn.k_proj, R_in=R1_inv)

        # Wv' = R2_v_eff @ Wv @ (R1_inv).T = R2_v_eff @ Wv @ R1
        fuse_layer_rotation(attn.v_proj, R_in=R1_inv)

        # Wo' = R1 @ Wo @ (R2_inv).T = R1 @ Wo @ R2_o_eff
        fuse_layer_rotation(attn.o_proj, R_out=R1)

        # -- 前馈网络 (FFN) 的合并 --
        # W' = W @ (R1_inv).T = W @ R1
        fuse_layer_rotation(mlp.gate_proj, R_in=R1_inv)
        fuse_layer_rotation(mlp.up_proj, R_in=R1_inv)

        # W' = R1 @ W
        fuse_layer_rotation(mlp.down_proj, R_out=R1)

        logger.info(f"已合并 Block {i + 1}/{len(model.language_model.layers)} 的权重。")

    # c) 合并输出头 (LM Head)
    # W_head' = W_head @ (R1_inv).T = W_head @ R1
    fuse_layer_rotation(model.lm_head, R_in=R1_inv)
    logger.info("已合并 LM Head。")

    logger.info("旋转合并后模型推理：")

    # for k,v in model.named_parameters():
    #     if "norm" in k:
    #         print(k, v)

    inference_normal(model, processor)
