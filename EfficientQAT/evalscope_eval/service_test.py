from openai import OpenAI

class VLLMClient:
    def __init__(self, base_url="http://localhost:8000/v1", api_key="EMPTY"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = self._get_model()
        print(self.model)

    def _get_model(self):
        """自动获取 vLLM 部署的模型名"""
        models = self.client.models.list()
        if len(models.data) == 0:
            raise RuntimeError("未在 vLLM 服务中找到模型，请确认服务是否正常启动。")
        return models.data[0].id   # 默认取第一个模型

    def chat(self, messages, temperature=0.7, max_tokens=1024):
        """调用 chat 接口"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


if __name__ == "__main__":
    client = VLLMClient(base_url="http://192.168.16.6:18000/v1", api_key="EMPTY")

    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "你是谁？"}
    ]

    reply = client.chat(messages)
    print("模型输出：\n", reply)
