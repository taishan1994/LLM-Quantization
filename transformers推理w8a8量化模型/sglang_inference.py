# launch the offline engine
import asyncio
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from PIL import Image
import requests
import time
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge
from torch.ao.quantization.quantizer.x86_inductor_quantizer import quantizable_ops
from transformers import AutoTokenizer

if is_in_ci():
    import patch
else:
    import nest_asyncio
    nest_asyncio.apply()


if __name__ == '__main__':
    model_path = "/data/gongoubo/checkpoints/Qwen/llmc/Qwen3-8B-w8a8-offline/vllm_quant_model"
    llm = sgl.Engine(model_path=model_path,
                     quantization="w8a8_int8",
                     attention_backend="triton",
                     )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompts = [
        "你是谁？",
    ]

    template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompts = [template.format(i) for i in prompts]

    print(prompts)

    # sampling_params = {"temperature": 0.6, "top_p": 0.95}
    sampling_params = {"temperature": 0.01, "max_new_tokens": 512}

    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t2 = time.time()
    print("耗时：{}s".format(t2-t1))

    for prompt, output in zip(prompts, outputs):
        print(output)
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
