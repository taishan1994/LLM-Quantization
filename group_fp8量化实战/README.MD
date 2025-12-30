# 说明

这里的FP8量化是DeepSeek-FP8格式的量化，即权重Group分块量化，激活动态per-group量化。kernel代码摘自llmc框架（原始的deepseek仓库里面没有权重量化代码）。提供了三种模型的量化，理论上支持任意模型的量化，需要修改量化哪些层。这里我们只量化LLM部分的权重。

使用5090进行部署推理。

- Qwen3-vl模型量化完成后，需要找到Qwen官方的FP8量化权重里面的config.json替换掉我们自己量化的。

- Gemma3量化完成后可以直接部署推理。
- MiniCPM-4.5-V怎么修改config.json都无法正常完成部署，因此采用了vllm推理时进行量化反量化。安装好vllm==0.13.0后将linear.py替换掉vllm路径下的即可。加载权重还是原始的未量化的权重。

当然，你也可以使用llm-compressor库或者LightCompress库进行量化，可能需要适配修改代码，这里没有进行尝试。
