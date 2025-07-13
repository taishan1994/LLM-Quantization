import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def inference_normal(model, tokenizer):

    # message = [{"role":"user", "content":"你是谁？"}]
    # message = tokenizer.apply_chat_template(message, tokenize=False, add_special_tokens=True)

    message = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(message, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    direct_output = model.generate(input_ids, max_new_tokens=256, do_sample=False, temperature=1)
    direct_text = tokenizer.decode(direct_output[0])

    print(direct_text)


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
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        print("警告: 模型结构不符合预期 (如 Llama, Mistral)。跳过合并。")
        return

    for layer in model.model.layers:
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
    model_norm_weight = model.model.norm.weight.data
    print(model_norm_weight.shape)
    print(model.lm_head.weight.data.shape)
    model.lm_head.weight.data = model.lm_head.weight.data * model_norm_weight
    model.model.norm.weight.data.fill_(1.0)
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
    model_name = "/data/gongoubo/checkpoints/Qwen/Qwen3-0.6B"
    # model_name = "/data/gongoubo/Qwen-1.5-Factory/model_hub/Qwen/Qwen2___5-1___5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    CONFIG = AutoConfig.from_pretrained(model_name)
    # transformers==4.53.0
    # from modeling_qwen3 import Qwen3ForCausalLM

    process_word_embeddings= False
    if CONFIG.tie_word_embeddings:
        CONFIG.tie_word_embeddings = False
        process_word_embeddings = True
    from modeling_qwen3_online_r3_r4 import Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, config=CONFIG).to("cuda:0")

    # model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    # 确保模型在评估模式
    model.eval()
    #
    inference_normal(model, tokenizer)

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    merge_rmsnorm_in_model(model)

    # for k,v in model.named_parameters():
    #     if "norm" in k:
    #         print(k, v)
    #
    inference_normal(model, tokenizer)

    hidden_size = CONFIG.hidden_size
    head_dim = 128
    num_q_heads = CONFIG.num_attention_heads
    num_kv_heads = CONFIG.num_key_value_heads

    from hadamard_utils2 import (
        get_hadK,
        apply_exact_had_to_linear,
        random_hadamard_matrix,
    )

    R1 = random_hadamard_matrix(hidden_size, device=model.device)
    R2 = random_hadamard_matrix(num_q_heads*head_dim, device=model.device)

    R2_head = random_hadamard_matrix(head_dim, device=model.device)

    logger.info(f"已生成旋转矩阵: R1 ({R1.shape}) 和 R2_head ({R2_head.shape})")

    # 4. 根据 R2_head 创建用于 W_v 和 W_o 的有效旋转矩阵 R2_eff
    R2_v_eff = create_block_diag_from_head_matrix(R2_head, num_kv_heads).to(model.device)
    R2_o_eff = create_block_diag_from_head_matrix(R2_head, num_q_heads).to(model.device)

    logger.info(f"已创建分块对角矩阵: R2_v_eff ({R2_v_eff.shape}) 和 R2_o_eff ({R2_o_eff.shape})")

    # 5. 执行旋转合并
    logger.info("开始将旋转矩阵 R1 和 R2 合并到模型权重中...")

    R1_inv = R1.T
    R2_inv = R2.T
    R2_v_eff_inv = R2_v_eff.T
    R2_o_eff_inv = R2_o_eff.T

    print(R1 @ R1_inv)
    print(R2_head @ R2_head.T)

    logger.info("embedding层形状：" + str(model.model.embed_tokens.weight.data.shape))

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
        device, dtype = model.model.embed_tokens.weight.device, model.model.embed_tokens.weight.dtype
        W = model.model.embed_tokens.weight.data.to(torch.float32)
        R1 = R1.to(model.model.embed_tokens.weight.device, dtype=torch.float32)
        new_w = W @ R1
        model.model.embed_tokens.weight.data = new_w.to(model.model.embed_tokens.weight.dtype)
    logger.info("已合并 Embedding 层。")

    # b) 遍历所有 Transformer Blocks
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        # Wq' = Wq @ (R1_inv).T = Wq @ R1
        fuse_layer_rotation(attn.q_proj, R_in=R1_inv)
        fuse_layer_rotation(attn.k_proj, R_in=R1_inv)

        # Wv' = R2_v_eff @ Wv @ (R1_inv).T = R2_v_eff @ Wv @ R1
        # fuse_layer_rotation(attn.v_proj, R_in=R1_inv, R_out=R2_v_eff_inv)
        # fuse_layer_rotation(attn.o_proj, R_in=R2_o_eff, R_out=R1)

        print(attn.v_proj.weight.data.shape, attn.o_proj.weight.data.shape)

        fuse_layer_rotation(attn.v_proj, R_in=R1_inv)
        fuse_layer_rotation(attn.o_proj, R_out=R1)

        apply_exact_had_to_linear(attn.v_proj, had_dim=head_dim, output=True)
        # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
        apply_exact_had_to_linear(attn.o_proj, had_dim=head_dim, output=False)

        # -- 前馈网络 (FFN) 的合并 --
        # W' = W @ (R1_inv).T = W @ R1
        fuse_layer_rotation(mlp.gate_proj, R_in=R1_inv)
        fuse_layer_rotation(mlp.up_proj, R_in=R1_inv)

        # 左乘
        apply_exact_had_to_linear(mlp.down_proj, had_dim=-1, output=False)

        # W' = R1 @ W
        fuse_layer_rotation(mlp.down_proj, R_out=R1)

        logger.info(f"已合并 Block {i + 1}/{len(model.model.layers)} 的权重。")

    # c) 合并输出头 (LM Head)
    # W_head' = W_head @ (R1_inv).T = W_head @ R1
    fuse_layer_rotation(model.lm_head, R_in=R1_inv)
    logger.info("已合并 LM Head。")

    logger.info("旋转合并后模型推理：")
    inference_normal(model, tokenizer)