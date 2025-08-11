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

logger = get_logger()


def get_chat_completion(
        prompt,
        host: str = "127.0.0.1",
        port: int = 8000,
        model_name: str = "my-quantized-model",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p=1.0,
        top_k=20,
):
    """
    向 FastAPI 服务器发送一个聊天补全请求。

    Args:
        prompt (str): 你向模型提出的问题或指令。
        host (str): 服务器的主机名或IP地址。
        port (int): 服务器的端口号。
        model_name (str): 在请求体中使用的模型名称 (可以是任意字符串)。
        max_tokens (int): 希望模型生成的最大 token 数量。
        temperature (float): 控制生成文本的随机性。

    Returns:
        str: 模型生成的回复文本。
        None: 如果请求失败。
    """
    # 构建请求的 URL
    url = f"http://{host}:{port}/v1/chat/completions"

    # 构建符合 OpenAI 格式的请求体 (payload)
    # 这是一个单轮对话的例子
    payload = {
        "model": model_name,
        "messages": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": False,  # 这个例子中不使用流式传输
    }

    # 设置请求头
    headers = {
        "Content-Type": "application/json"
    }

    try:
        print(f"▶️  正在向 http://{host}:{port} 发送请求...")
        print(" PAYLOAD ".center(40, '='))
        print(json.dumps(payload, indent=2))
        print("=" * 40)

        # 发送 POST 请求
        response = requests.post(url, headers=headers, json=payload)

        # 检查响应状态码，如果不是 2xx，则会抛出异常
        response.raise_for_status()

        # 解析 JSON 响应
        data = response.json()

        # 提取并返回助手的回复
        assistant_reply = data['choices'][0]['message']['content']
        return assistant_reply.strip()

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return None



class DummyCustomModel(CustomModel):

    def __init__(self, config: dict = {}, **kwargs):
        super(DummyCustomModel, self).__init__(config=config, **kwargs)

    def infer_one(self, index, input_item, config, port):
        message = self.make_request_messages(input_item)

        response = get_chat_completion(
            prompt=message,
            host="127.0.0.1",
            port=port,
            model_name="/home/gongoubo/checkpoints/Qwen/Qwen3-4B",
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            max_tokens=config["max_tokens"],
        )

        res_d = {
            "id": config.get('model_id'),
            "object": "chat.completion",
            "created": time.time(),
            "model": config.get('model_id'),
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        return index, res_d

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

        print(self.config)

        print(len(original_inputs))

        # for input_item in original_inputs:
        #     message = self.make_request_messages(input_item)
        #
        #     # You can replace this with actual model inference logic
        #     # For demonstration, we will just return a dummy response
        #     # response = f"Dummy response for prompt: {message}"
        #
        #     print(message)
        #     port = random.choices([11334 + i for i in range(1)])
        #     print(port)
        #
        #     response = get_chat_completion(
        #         prompt=message,
        #         host="127.0.0.1",
        #         port=port[0],
        #         model_name="/home/gongoubo/checkpoints/Qwen/Qwen3-4B",
        #         temperature=self.config["temperature"],
        #         top_p=self.config["top_p"],
        #         top_k=self.config["top_k"],
        #         max_tokens=self.config["max_tokens"],
        #     )
        #     print(response)
        #
        #     res_d = {
        #         "id": self.config.get('model_id'),
        #         "object": "chat.completion",
        #         "created": time.time(),
        #         "model": self.config.get('model_id'),
        #         "system_fingerprint": "fp_44709d6fcb",
        #         "choices": [{
        #             "index": 0,
        #             "message": {
        #                 "role": "assistant",
        #                 "content": response
        #             },
        #         }],
        #         "usage": {
        #             "prompt_tokens": 0,
        #             "completion_tokens": 0,
        #             "total_tokens": 0
        #         }
        #     }
        #
        #     responses.append(res_d)
        #
        # return responses

        ports = [11335, 11336, 11337, 11338]
        ports_cycle = cycle(ports)  # 创建一个无限循环的端口分配器

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(ports)) as executor:
            futures = []
            for idx, input_item in enumerate(original_inputs):
                port = next(ports_cycle)
                futures.append(executor.submit(self.infer_one, idx, input_item, self.config, port))

            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # 按 index 排序，保证输出顺序一致
        results.sort(key=lambda x: x[0])
        responses = [res for idx, res in results]
        return responses


generation_config = {
    'max_tokens': 31000,  # 最大生成token数，建议设置为较大值避免输出截断
    'temperature': 0.6,  # 采样温度 (qwen 报告推荐值)
    'top_p': 0.95,  # top-p采样 (qwen 报告推荐值)
    'top_k': 20,  # top-k采样 (qwen 报告推荐值)
    'n': 1,  # 每个请求产生的回复数量
}

# 实例化DummyCustomModel
dummy_model = DummyCustomModel(config=generation_config)


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
            'dataset_id': 'modelscope/EvalScope-Qwen3-Test',
            'filters': {'remove_until': '</think>'}  # 过滤掉思考的内容
        }
    },
    eval_batch_size=128,
    generation_config=generation_config,
    timeout = 600000,  # 超时时间
    stream = True,  # 是否使用流式输出
    limit = 100,  # 设置为100条数据进行测试
    use_cache="outputs/qwen3_4b_w-fp6_e2m3_a-fp6_e2m3",
)

run_task(task_cfg=task_cfg)