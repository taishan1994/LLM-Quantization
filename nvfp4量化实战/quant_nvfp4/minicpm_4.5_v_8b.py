import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datasets import load_dataset
import numpy as np
from PIL import Image
from copy import deepcopy
from transformers import AutoModel, AutoTokenizer, AutoProcessor   

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "/home/gongoubo/checkpoints/OpenBMB/MiniCPM-V-4_5"


# Load model.
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16, device_map="auto") # sdpa or flash_attention_2, no eager
# model = model.eval().cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True) # or openbmb/MiniCPM-o-2_6

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)


DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 8192

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")


# def preprocess_function(example):
#     messgages = []
#     for message in example["messages"]:
#         messgages.append(
#             {
#                 "role": message["role"],
#                 "content": [{"type": "text", "text": message["content"]}],
#             }
#         )

#     return processor.apply_chat_template(
#         messgages,
#         return_tensors="pt",
#         padding=False,
#         truncation=True,
#         max_length=MAX_SEQUENCE_LENGTH,
#         tokenize=True,
#         add_special_tokens=False,
#         return_dict=True,
#         add_generation_prompt=False,
#     )

enable_thinking = False
system_prompt = None

dummy_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)

def preprocess_function2(example):
    msgs = []
    for message in example["messages"]:
        role = message["role"]
        if role == "user":
            msgs.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )
        else:
            msgs.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

    print(msgs)

    inputs = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking)
    print(inputs)
    inputs = processor.tokenizer(inputs, return_tensors="pt", padding="max_length", max_length=MAX_SEQUENCE_LENGTH, truncation=True)
    print(inputs)
    return inputs



ds = ds.map(preprocess_function2, batched=False, remove_columns=ds.column_names)

print(ds)

def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with group-wise quantization
#   * quantize the activations to fp4 with dynamic group activations
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:.*visual.*",
        "re:.*vision.*",
        "re:.*image.*",
        "re:.*resampler.*",
        "re:.*vpm.*",
        "re:.*aligner.*",
    ],
)

# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    dataset=ds,
    data_collator=data_collator,
    trust_remote_code_model=True,
    sequential_targets=["Qwen3DecoderLayer"],
)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
# 创建一个虚拟图像以满足多模态模型的输入要求

# 创建一个标准尺寸的白色图像 (224x224)
dummy_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)

text = "你是谁"
msgs = [{'role': 'user', 'content': [dummy_image, text]}]

# 使用chat方法进行推理，这样可以正确处理多模态输入
response = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
)

print(response)
print("==========================================")


# Save to disk in compressed-tensors format.
SAVE_DIR = "/home/gongoubo/outputs/minicpm-4.5-v-8b-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)