from itertools import cycle

import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# vLLM 部署接口地址
VLLM_SERVERS = [
    "http://14.18.247.55:11778/v1"
]

CLIENTS = [OpenAI(base_url=url, api_key="EMPTY") for url in VLLM_SERVERS]
models = CLIENTS[0].models.list()
MODEL_NAME = models.data[0].id

with open("data/medical_o1_sft_Chinese.json", "r") as fp:
    data = json.load(fp)
data_tmp = []
for i,d in enumerate(data):
    d["id"] = i
    data_tmp.append(d)
data = data_tmp


RESULTS_FILE = "data/medical_o1_sft_Chinese_generate.json"      # 最终结果
CHECKPOINT_FILE = "data/medical_o1_sft_Chinese_generate.jsonl"  # 中间保存 (每行一条 JSON)

def load_checkpoint():
    """读取已完成的数据，返回 dict: id -> record"""
    done = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    done[record["id"]] = record
                except:
                    continue
    return done

def save_checkpoint(record):
    """把新结果追加到 checkpoint 文件"""
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def fetch_response(item, client):
    # print(item)
    """单个请求任务"""
    messages = [
        {"role": "user", "content": item["Question"]}
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
        )
        response_text = resp.choices[0].message.content
    except Exception as e:
        print(e)
        response_text = ""

    record = {
        "id": item["id"],
        "Question": item["Question"],
        "Complex_CoT": "",
        "Response": response_text
    }
    save_checkpoint(record)  # 立即保存
    return record

def main():
    # 读取已完成的数据
    done = load_checkpoint()
    print(f"已完成 {len(done)} 条，将跳过这些。")

    # 筛选未完成的数据
    remaining = [item for item in data if item["id"] not in done]

    results = list(done.values())  # 把已完成的先加进去

    # 轮询分配服务器
    server_cycle = cycle(CLIENTS)

    # 开始并发处理剩余的
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_item = {}
        for item in remaining:
            client = next(server_cycle)  # 轮询取一个服务器
            future = executor.submit(fetch_response, item, client)
            future_to_item[future] = item
        for future in as_completed(future_to_item):
            res = future.result()
            results.append(res)

    # 最后保存成 JSON（合并）
    results = sorted(results, key=lambda x: x["id"])  # 保持顺序
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"全部完成，共 {len(results)} 条，已保存到 {RESULTS_FILE}")

if __name__ == "__main__":
    main()
