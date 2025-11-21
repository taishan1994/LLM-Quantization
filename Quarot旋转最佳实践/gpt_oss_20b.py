import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def generate_random_orthogonal_matrix(size, device, dtype=torch.bfloat16):
    logger.info(f"生成 {size}x{size} 的随机正交矩阵 (dtype={dtype})...")
    random_matrix = torch.randn(size, size, device=device)
    q, r = torch.linalg.qr(random_matrix.float())
    return q.to(dtype=dtype)

def inference_normal(model, tokenizer):
    message = [{"role":"user", "content":"你好，请介绍一下你自己。"}]
    message = tokenizer.apply_chat_template(message, tokenize=False, add_special_tokens=True)
    input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)
    logger.info("开始生成...")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        direct_output = model.generate(input_ids, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    direct_text = tokenizer.decode(direct_output[0])
    print("模型输出:\n", direct_text)

def merge_rmsnorm_in_model(model):
    print("开始合并 RMSNorm 参数...")
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'): return
    for layer in model.model.layers:
        gamma_attn = layer.input_layernorm.weight.data
        layer.self_attn.q_proj.weight.data.mul_(gamma_attn)
        layer.self_attn.k_proj.weight.data.mul_(gamma_attn)
        layer.self_attn.v_proj.weight.data.mul_(gamma_attn)
        layer.input_layernorm.weight.data.fill_(1.0)
        gamma_mlp = layer.post_attention_layernorm.weight.data
        if hasattr(layer.mlp, 'router'):
            layer.mlp.router.weight.data.mul_(gamma_mlp)
            layer.mlp.experts.gate_up_proj.data.mul_(gamma_mlp.view(1, -1, 1))
        else:
            layer.mlp.gate_proj.weight.data.mul_(gamma_mlp)
            layer.mlp.up_proj.weight.data.mul_(gamma_mlp)
        layer.post_attention_layernorm.weight.data.fill_(1.0)
    if hasattr(model.model, 'norm'):
        model_norm_weight = model.model.norm.weight.data
        model.lm_head.weight.data.mul_(model_norm_weight)
        model.model.norm.weight.data.fill_(1.0)
    print("RMSNorm 参数合并完成！")

if __name__ == '__main__':
    model_name = "/nfs/FM/gongoubo/checkpoints/gpt-oss-20b-BF16/unsloth/gpt-oss-20b-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    CONFIG = AutoConfig.from_pretrained(model_name)
    process_word_embeddings = False
    if CONFIG.tie_word_embeddings:
        CONFIG.tie_word_embeddings = False
        process_word_embeddings = True
        
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, config=CONFIG, device_map="auto")
    model.eval()
    
    print("========== 原始模型推理 ==========")
    inference_normal(model, tokenizer)

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    merge_rmsnorm_in_model(model)
    
    print("\n========== 合并RMSNorm后推理 ==========")
    inference_normal(model, tokenizer)

    hidden_size = CONFIG.hidden_size
    R = generate_random_orthogonal_matrix(hidden_size, device=model.device, dtype=torch.bfloat16)

    logger.info("开始将旋转矩阵 R 合并到模型权重中...")

    with torch.no_grad():
        # Rule: W_new = W @ R
        model.model.embed_tokens.weight.data @= R
    logger.info("已合并 Embedding 层。")

    for i, layer in enumerate(model.model.layers):
        # --- Attention Block ---
        # Input Layers: W_new = W @ R
        layer.self_attn.q_proj.weight.data @= R
        layer.self_attn.k_proj.weight.data @= R
        layer.self_attn.v_proj.weight.data @= R
        
        # Output Layer: W_new = R.T @ W, b_new = b @ R
        layer.self_attn.o_proj.weight.data = R.T @ layer.self_attn.o_proj.weight.data
        if layer.self_attn.o_proj.bias is not None:
            layer.self_attn.o_proj.bias.data @= R

        # --- MLP Block ---
        if hasattr(layer.mlp, 'router') and hasattr(layer.mlp, 'experts'): # MoE
            # Input Layer (Router): W_new = W @ R
            layer.mlp.router.weight.data @= R
            
            # Input Layer (Experts): W_new = R.T @ W_stored (unsloth layout)
            layer.mlp.experts.gate_up_proj.data = R.T @ layer.mlp.experts.gate_up_proj.data
            
            # Output Layer (Experts): W_new = W_stored @ R (unsloth layout), b_new = b @ R
            layer.mlp.experts.down_proj.data @= R
            if hasattr(layer.mlp.experts, 'down_proj_bias') and layer.mlp.experts.down_proj_bias is not None:
                layer.mlp.experts.down_proj_bias.data @= R
        else: # Dense
            # Input Layers: W_new = W @ R
            layer.mlp.gate_proj.weight.data @= R
            layer.mlp.up_proj.weight.data @= R
            # Output Layer: W_new = R.T @ W, b_new = b @ R
            layer.mlp.down_proj.weight.data = R.T @ layer.mlp.down_proj.weight.data
            if layer.mlp.down_proj.bias is not None:
                layer.mlp.down_proj.bias.data @= R

        if (i + 1) % 5 == 0 or (i + 1) == len(model.model.layers):
            logger.info(f"已合并 Block {i + 1}/{len(model.model.layers)} 的权重。")

    with torch.no_grad():
        # Final Layer: W_new = W @ R
        model.lm_head.weight.data @= R
    logger.info("已合并 LM Head。")

    print("\n========== 旋转合并后模型推理 ==========")
    inference_normal(model, tokenizer)
