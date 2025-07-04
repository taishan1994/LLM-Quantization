import requests
import json
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import os
import sys

ld_library = os.environ.get("LD_LIBRARY_PATH", "")
print(ld_library)

new_path = "/opt/hpcx/ucx/lib"
if new_path not in ld_library.split(":"):
    os.environ["LD_LIBRARY_PATH"] = f"{new_path}:{ld_library}" if ld_library else new_path
    os.execv(sys.executable, [sys.executable] + sys.argv)

from transformers import AutoTokenizer
from openai import OpenAI

# /ms/FM/gongoubo/checkpoints/DeepSeek-R1-Block-INT8/
# "/ms/FM/checkpoints/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
model_id = "/ms/FM/gongoubo/checkpoints/data/Qwen3/Qwen3-0.6B-quarot-no-quant/gptq_w4a8"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# tokenizer = AutoTokenizer.from_pretrained("/ms/FM/gongoubo/checkpoints/DeepSeek-R1-Block-INT8/")
# tokenizer = AutoTokenizer.from_pretrained("/ms/FM/gongoubo/checkpoints/Qwen3-32B-smoothquant-gptq-w4a8/", use_fast=False)


def inference_requests(message,
                       max_tokens,
                       url,
                       model="/ms/FM/checkpoints/cognitivecomputations/DeepSeek-R1-awq",
                       temperature=0.6,
                       use_chat=False):
    openai_key = "EMPTY"
    openai_api_base = url
    # openai_api_base = "http://192.168.112.17:9000/v1"

    if use_chat:
        message = [{"role": "user", "content": message}]

    # message = [{"role": "user", "content": "1+1等于几？"},
    #            {"role": "assistant", "content": "1+1等于2"}]

    # print(tokenizer.decode([151643, 151669, 105043, 100165, 11319, 151670]))

    client = OpenAI(base_url=url, api_key="None")
    model_name = client.models.list().data[0].id

    if use_chat:
        response = client.chat.completions.create(
            model=model_name,
            messages=message,
            max_tokens=max_tokens,
            top_p=0.95,
            extra_body={
                "top_k":20,
            },
            #frequency_penalty=1,
            #presence_penalty=1,
        )
        print(response.choices[0].message.content)
        return response
    else:
        response = client.completions.create(
            model=model_name,
            prompt=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            extra_body={
                "top_k": 20,
            },
            # frequency_penalty=1,
            # presence_penalty=1,
        )
        print(response.choices[0].text)
        return response


message = "For how many positive integers $n>1$ is it true that $2^{24}$ is a perfect $n^{\\text{th}}$ power?\nPlease reason step by step, and put your final answer within \\boxed{}."

temperature = 0.6
while True:
    response = inference_requests(message,
                                  max_tokens=10000,
                                  url="http://xxx:34000/v1",
                                  model="xxx",
                                  temperature=temperature,
                                  use_chat=True)
    print("=" * 100)


