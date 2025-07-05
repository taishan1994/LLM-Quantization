import base64
from openai import OpenAI
from PIL import Image

class LLMClient:
    def __init__(self,
                 url,
                 max_tokens,
                 frequency_penalty=0.0,
                 model_name=None,
                 stop=None):
        self.client = OpenAI(api_key="empty", base_url=url, max_retries=4)
        self.max_tokens = max_tokens
        if model_name is None:
            self.model_name = self.client.models.list().data[0].id
        else:
            self.model_name = model_name
        print(self.model_name)
        self.frequency_penalty = frequency_penalty
        self.stop = stop

    def generate(self, image, prompt):
        with open(image, "rb") as f:
            img_data = f.read()
        image_bs64 = base64.b64encode(img_data).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_bs64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },

                    ]
                }
            ],
            temperature=0.0,
            frequency_penalty=self.frequency_penalty,
            max_tokens=self.max_tokens,
            stop=self.stop,
            logprobs=True,
        )
        print(response)
        return response.choices[0].message.content

if __name__ == '__main__':
    url = "http://192.168.112.93:8000/v1"
    max_tokens = 1000
    prompt = "请简单描述下图片。"

    llmclient = LLMClient(url, max_tokens)
    image = "images/test2.png"
    # image = Image.open(image).convert("RGB").tobytes()
    print(llmclient.generate(image, prompt))