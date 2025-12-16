from openai import OpenAI
import base64
import concurrent.futures

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

infer_config = {
    "max_tokens": 1024,
    "temperature": 0.01,
    "top_p": 1,
    "repetition_penalty": 1,
    "top_k": 50,
}

def process_item(vllm_url, image_path):
    """处理单个项目的函数，用于多线程调用"""
    try:
        # 为每个线程创建独立的OpenAI客户端
        thread_client = OpenAI(base_url=vllm_url, api_key="None")
        thread_model = thread_client.models.list().data[0].id
        
        base64_image = encode_image(image_path)
        response = thread_client.chat.completions.create(
            model=thread_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "请描述这张图片。",
                        },
                    ],
                }
            ],
            max_tokens=infer_config["max_tokens"],
            temperature=infer_config["temperature"],
            top_p=infer_config["top_p"],
            frequency_penalty=infer_config["repetition_penalty"],
            extra_body={"top_k": infer_config["top_k"]}
        )
        content = response.choices[0].message.content
        # print(content)
        if "</think>" in content:
            content = content.split("</think>")[-1]
        print(content)
        
        return content
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    vllm_url= "http://192.168.16.24:32000/v1"
    image_path = "/nfs/FM/gongoubo/new_project/workflow/aihub_data/LMUData_v2/images/CCBench/98.jpg"
    process_item(vllm_url, image_path)

