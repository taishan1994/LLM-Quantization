import json
import time
import requests
import random
from typing import List
import concurrent.futures
import time
import random
from itertools import cycle

from evalscope.utils.logger import get_logger
from evalscope.models import CustomModel

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = get_logger()


class DummyCustomModel(CustomModel):

    def __init__(self,
                 config: dict = {},
                 model_path=None,
                 generation_config=None,
                 **kwargs):
        super(DummyCustomModel, self).__init__(config=config, **kwargs)

        self.sampling_params = generation_config
        # 初始化 LLM（只加载一次到显存里）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(model=model_path,
                  tensor_parallel_size=4,
                  max_model_len=32000,
                  enforce_eager=True)

    def make_request_messages(self, input_item: dict) -> list:
        """
        Make request messages for OpenAI API.
        """
        if input_item.get('messages', None):
            return input_item['messages']

        data: list = input_item['data']
        if isinstance(data[0], tuple):  # for truthful_qa and hellaswag
            query = '\n'.join(''.join(item) for item in data)
            system_prompt = input_item.get('system_prompt', None)
        else:
            query = data[0]
            system_prompt = input_item.get('system_prompt', None)

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': query})

        return messages

    def predict(self, prompts: List[dict], **kwargs):

        original_inputs = kwargs.get('origin_inputs', None)
        infer_cfg = kwargs.get('infer_cfg', None)

        logger.debug(f'** Prompts: {prompts}')
        if original_inputs is not None:
            logger.debug(f'** Original inputs: {original_inputs}')
        if infer_cfg is not None:
            logger.debug(f'** Inference config: {infer_cfg}')

        # Simulate a response based on the prompts
        # Must return a list of dicts with the same format as the OpenAI API.
        responses = []

        messages = [self.make_request_messages(d) for d in original_inputs]
        messages = [self.tokenizer.apply_chat_template(d, tokenize=False, add_generation_prompt=True) for d in messages]
        print(messages[0])
        outputs = self.llm.generate(messages, self.sampling_params)

        for inp,out in zip(messages, outputs):
            res_d = {
                "id": "Qwen3-4B",
                "object": "chat.completion",
                "created": time.time(),
                "model": "Qwen3-4B",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": out.outputs[0].text
                    },
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            responses.append(res_d)
        return responses


generation_config = SamplingParams(
    max_tokens=31000,  # 最大生成token数，建议设置为较大值避免输出截断
    temperature=0.6,  # 采样温度 (qwen 报告推荐值)
    top_p=0.95,  # top-p采样 (qwen 报告推荐值)
    top_k=20,  # top-k采样 (qwen 报告推荐值)
    n=1,  # 每个请求产生的回复数量
)

# 实例化DummyCustomModel
model_path = "/data/gongoubo/checkpoints/Qwen/Qwen3-4B"
dummy_model = DummyCustomModel(generation_config=generation_config,
                               model_path=model_path)


from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model=dummy_model,
    model_id='dummy-model',  # 自定义模型ID
    eval_type='custom',
    datasets=[
        'data_collection',
    ],
    dataset_args={
        'data_collection': {
            'dataset_id': 'outputs/uniform_mixed_data_1000.jsonl',
            'filters': {'remove_until': '</think>'}  # 过滤掉思考的内容
        }
    },
    eval_batch_size=32,
    # generation_config=generation_config,
    timeout = 600000,  # 超时时间
    stream = True,  # 是否使用流式输出
    # limit = 100,  # 设置为100条数据进行测试
    use_cache="outputs/qwen3-4b/",
)

run_task(task_cfg=task_cfg)