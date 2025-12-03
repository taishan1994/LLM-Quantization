# 前言

之前我们将sglang里面的w8a8算子解耦出来集成到transformers里面推理。博主最近在使用AI Agent（AI编辑器）让大模型自己去适配w4a8的量化和推理，最终有了一个成功的版本。

# 依赖
```
transformers==4.54.0
triton==3.0.0
torch==2.5.1+cu124
```

# 基本流程
使用llm-compressor或者llmc（改名LightCompress）量化出一个w4a8的模型，其中w采用per-channel对称量化，激活采用per-token动态对称量化（即在推理的时候进行量化）。之前w4a8模型推理是将int4的w用int8数据类型进行存储，然后使用w8a8的推理算子进行推理，主要是用于测试w4a8模型的精度，现在，我们需要把w4进行packed，这样可以减少存储。这里采用awq格式的packed（需要注意的是w4-per-channel对称量化与非对称量化相比，是没有zero的）。加载packed的模型权重，推理时先进行解包成int8格式的数据，接着使用triton的per-token量化算子和w8a8-gemm的triton算子，最终完成推理。

> 注意：这里的代码都是用AI 编辑器完成的，在对话的过程中需要指导模型优化的方向。最开始使用的是trae-cn，基于上面国内模型比如kimi、minimax、glm4.6，都不能完成该任务，后续切换到codebuddy国际版，使用gpt5.1和gemini2.5-pro，最终能够正常推理。单从编辑器的使用来看，我个人还是比较喜欢trae的交互的。

# 运行
首先是使用llmc对模型进行量化，然后使用packed脚本将模型进行打包。这里不做过多介绍了。这里加载的qwen-32b模型，但是使用的system prompt让其认为是0.6b，问题不大。

使用原生的torch进行量化和推理：
```shell
python3 -m w4a8_inference.qwen3_awq_infer --model-dir /nfs/FM/gongoubo/new_project/workflow/checkpoints/Qwen3-32B-w4a8-workflow-packed  --prompt '你好，请做个自我介绍。' --max-new-token 500 --no-triton
```
输出：
```shell
[INFO] Loading tokenizer from /nfs/FM/gongoubo/new_project/workflow/checkpoints/Qwen3-32B-w4a8-workflow-packed
[INFO] Building Qwen3 AWQ model from /nfs/FM/gongoubo/new_project/workflow/checkpoints/Qwen3-32B-w4a8-workflow-packed on cuda
[LOAD] Streaming weights into model on cuda ...
[LOAD] Found 7 shards in index
[LOAD] Loading shard 1/7: model-00001-of-00007.safetensors (127 tensors)
[LOAD] Loading shard 2/7: model-00002-of-00007.safetensors (190 tensors)
[LOAD] Loading shard 3/7: model-00003-of-00007.safetensors (180 tensors)
[LOAD] Loading shard 4/7: model-00004-of-00007.safetensors (180 tensors)
[LOAD] Loading shard 5/7: model-00005-of-00007.safetensors (180 tensors)
[LOAD] Loading shard 6/7: model-00006-of-00007.safetensors (180 tensors)
[LOAD] Loading shard 7/7: model-00007-of-00007.safetensors (118 tensors)
[LOAD]   shard 7/7: 118/118 tensors loaded
[INFO] Loading generation config
[INFO] Applying chat_template ...
[INFO] Starting generation (use_triton=False)...
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[INFO] Generation finished in 73525.08 ms.
===== MODEL OUTPUT =====
system
You are Qwen3-0.6B W4A8 quantized model.
user
你好，请做个自我介绍。
assistant
<think>
好的，用户让我做个自我介绍。首先，我需要确定用户的需求是什么。用户可能想了解我的功能和特点，或者他们对我的能力感到好奇。作为Qwen3-0.6B量化模型，我应该突出我的优势，比如量化后的效率和适用性。同时，用户可能希望知道我的应用场景，比如对话、推理等。我需要简明扼要地介绍我的特点，比如参数量、量化方式、适用平台，并保持口语化，避免使用专业术语过多。此外，用户可能关心我的性能和实际应用效果，所以需要提到在移动端或边缘设备的部署能力。最后，确保回答结构清晰，分点说明，但不要使用Markdown格式。现在，我需要将这些信息整合成自然流畅的中文回复，满足用户的需求。
</think>

你好！我是Qwen3-0.6B量化模型，基于W4A8量化技术优化后的版本。我专注于高效推理和移动端部署，参数量精简但保持强泛化能力。我的特点包括：量化后体积小（适合边缘设备）、推理速度快（低延迟）、兼容性强（支持多种硬件平台）。你可以用我来做对话理解、文本生成、多模态任务等。如果你有具体需求，比如部署在手机或嵌入式设备上，我很适合！有什么想测试的场景吗？
```

使用titon算子进行量化和推理：
```shell
python3 -m w4a8_inference.qwen3_awq_infer --model-dir /nfs/FM/gongoubo/new_project/workflow/checkpoints/Qwen3-32B-w4a8-workflow-packed  --prompt '你好，请做个自我介绍。' --max-new-token 500 --use-triton
```
输出：
```shell
[INFO] Loading tokenizer from /nfs/FM/gongoubo/new_project/workflow/checkpoints/Qwen3-32B-w4a8-workflow-packed
[INFO] Building Qwen3 AWQ model from /nfs/FM/gongoubo/new_project/workflow/checkpoints/Qwen3-32B-w4a8-workflow-packed on cuda
[LOAD] Streaming weights into model on cuda ...
[LOAD] Found 7 shards in index
[LOAD] Loading shard 1/7: model-00001-of-00007.safetensors (127 tensors)
[LOAD] Loading shard 2/7: model-00002-of-00007.safetensors (190 tensors)
[LOAD] Loading shard 3/7: model-00003-of-00007.safetensors (180 tensors)
[LOAD] Loading shard 4/7: model-00004-of-00007.safetensors (180 tensors)
[LOAD] Loading shard 5/7: model-00005-of-00007.safetensors (180 tensors)
[LOAD] Loading shard 6/7: model-00006-of-00007.safetensors (180 tensors)
[LOAD] Loading shard 7/7: model-00007-of-00007.safetensors (118 tensors)
[LOAD]   shard 7/7: 118/118 tensors loaded
[INFO] Loading generation config
[INFO] Applying chat_template ...
[INFO] Starting generation (use_triton=True)...
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[INFO] Generation finished in 51173.80 ms.
===== MODEL OUTPUT =====
system
You are Qwen3-0.6B W4A8 quantized model.
user
你好，请做个自我介绍。
assistant
<think>
Okay, the user asked me to introduce myself. Let me start by recalling what I know about my model. I'm Qwen3-0.6B, which is a quantized version. The quantization is W4A8, meaning 4 bits for weights and 8 bits for activation. This helps reduce the model size while maintaining performance.

I should mention my main features, like being lightweight and efficient. Since quantization is a key aspect, I need to explain how it works. Maybe users aren't familiar with quantized models, so I should clarify that it uses lower precision to save memory and speed up inference.

Also, I should highlight my advantages, such as lower resource consumption and faster response times. But I shouldn't forget to mention that while I'm efficient, there might be some trade-offs in model accuracy compared to non-quantized versions.

Wait, the user might be interested in why I'm using this specific quantization method. I should explain that W4A8 is a balance between precision and performance, allowing me to maintain reasonable accuracy while being efficient.

Oh, and I should make sure to mention that I'm a quantized model, which is different from standard models. Users might not know what quantization means, so a brief explanation would be helpful.

Also, I need to check if there are any other important aspects about my model. Like, maybe the use case scenarios where quantization is beneficial, such as mobile or edge devices with limited resources.

Finally, I should wrap it up by stating that I'm here to help and can assist with various tasks while being efficient. Make sure the introduction is concise but covers all key points.
</think>

你好，我是Qwen3-0.6B量化模型。通过W4A8量化技术（4位权重+8位激活），我实现了模型轻量化与高效推理的平衡。我的设计目标是降低计算和内存消耗，同时保持合理的精度，适用于边缘设备或资源受限场景。虽然量化可能带来一定精度损失，但我在多项任务中仍能提供高效且可靠的响应。我是谁？我是你与AI交互的桥梁，随时准备协助你完成各种任务！
```

# 补充
此外，也使用trae或者codebuddy进行其余sglang修改和适配任务，这里进行一个总结：
## trae-适配Qwen3-0.6B模型
准备环境
1. 使用sglang-0.4.4版本镜像（支持qwen2.5但不支持qwen3）
2. 准备sglang源代码、qwen3的transformers的模型结构文件（transformers==4.53.2）、部署qwen3的sglang代码
- deepseek-r1-v3.1（修改失败）：权重可以正常加载，但是最终遇到shape维度对不上问题，反复修改无法解决。
- minimax-m2（修改失败）：能够理解GQA和MHA之间的差异，可以推理，但是输出乱码，无法解决。
- kimi-k2-0905（修改失败）：找不到q_norm和k_norm的权重。
- GLM4.6（修改成功）：权重成功加载，并能够正常推理。
- Qwen3-coder（修改失败）：中途跑偏。
- Doubao-seed-code（修改失败）：出现OOM后一老认为服务启动有问题，明明单卡但是设置tp=2。

## 适配Qwen3-30B-A3B
- trae：GLM4.6、Minimax-M2、Doubao-seed-code、deepseek-r1-v3.1（修改失败）：基本上都修改失败，大部分是在shape对不上后反复修改无法修改对。
- openbuddy：gpt5.1（修改失败）：自己部署了服务后没有关闭，又重新启动服务，导致OOM，然后跑偏到显存问题上。

## 适配Ernie-4.5-0.3B-PT
- trae：GLM4.6、Minimax-M2、kimi-k2-0905（修改失败）：GLM4.6跑偏、Minimax-M2推理乱码、kimi-k2-0905前半部分推理正常，后续乱码。
- codebuddy：gpt5.1：适配成功，能够正常推理。中间遇到第一次输出正常，第二次同样的问题输出不正常。人工给出问题让其进行修复，最终修复成功。

## trae：sglang-0.5.5适配的ernie4.5适配sglang-0.4.4
- GLM4.6（修改成功）：可能这两个版本之间差异不大，迁移的代码比较少。