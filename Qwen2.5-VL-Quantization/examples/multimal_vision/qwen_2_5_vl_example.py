import os
import sys
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
ld_library = os.environ.get("LD_LIBRARY_PATH", "")
print(ld_library)

new_path = "/opt/hpcx/ucx/lib"
if new_path not in ld_library.split(":"):
    os.environ["LD_LIBRARY_PATH"] = f"{new_path}:{ld_library}" if ld_library else  new_path
    os.execv(sys.executable, [sys.executable] + sys.argv)

import base64
from io import BytesIO

import torch
from datasets import load_dataset
from vision_process import process_vision_info
from transformers import AutoProcessor

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import (
    TraceableQwen2_5_VLForConditionalGeneration,
)

# Load model.
model_id = "/ms/FM/tangqie/llmc/save/Qwen2___5-VL-7B-Instruct_sym_w8_a8-dynamic_fix_bias/transformed_model"
tokenizer_model_id = "/ms/FM/checkpoints/Qwen-Zoo/Qwen2___5-VL-7B-Instruct"
model = TraceableQwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
).to("cuda:0")
processor = AutoProcessor.from_pretrained(tokenizer_model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "/ms/FM/gongoubo/checkpoints/data/flikr30"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
print("开始加载数据集！！")
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)

print(ds)

# Apply chat template and tokenize inputs.
def preprocess_and_tokenize(example):
    # preprocess
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_qwen},
                {"type": "text", "text": "What does the image show?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # tokenize
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )


ds = ds.map(preprocess_and_tokenize, remove_columns=ds["calibration"].column_names)


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="w8a8",
        sequential_targets=["Qwen2_5_VLDecoderLayer"],
        # ignore=["lm_head", "re:visual.*"],
        ignore=["lm_head"],
    ),
]

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
)

# Confirm generations of the quantized model look sane.
# print("========== SAMPLE GENERATION ==============")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "http://images.cocodataset.org/train2017/000000231895.jpg",
#             },
#             {"type": "text", "text": "Please describe the animal in this image\n"},
#         ],
#     }
# ]
# prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[prompt],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=False,
#     max_length=MAX_SEQUENCE_LENGTH,
#     truncation=True,
#     return_tensors="pt",
# ).to("cuda")
# output = model.generate(**inputs, max_new_tokens=100)
# print(processor.decode(output[0], skip_special_tokens=True))
# print("==========================================")


# Save to disk compressed.
SAVE_DIR =  "/ms/FM/gongoubo/checkpoints/data/Qwen2___5-VL-7B-Instruct"+ "-W8A8-include-vit"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
