import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from datasets import load_dataset
import numpy as np
from PIL import Image
from copy import deepcopy
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer, AutoProcessor   

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "/home/gongoubo/checkpoints/google/gemma-3-4b-it"


# Load model.
model = Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True,  torch_dtype=torch.bfloat16, device_map="auto") # sdpa or flash_attention_2, no eager
# model = model.eval().cuda()

print(model)

for k,v in model.named_parameters():
    print(k)

# import sys
# sys.exit(0)

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

    inputs = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking)
    inputs = processor.tokenizer(inputs, return_tensors="pt", padding="max_length", max_length=MAX_SEQUENCE_LENGTH, truncation=True)

    return inputs



ds = ds.map(preprocess_function2, batched=False, remove_columns=ds.column_names)


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
        "re:model.visual_tower.*",
        "re:model.multi_modal_projector.*",
        "re:.*vision.*",
        "re:.*multi_modal_projector.*",
        "re:.*visual_tower.*",
        "model.multi_modal_projector.*",
        "model.multi_modal_projector.mm_soft_emb_norm",
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
)

print("========== SAMPLE GENERATION ==============")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")
print("==========================================")


# Save to disk in compressed-tensors format.
SAVE_DIR = "/home/gongoubo/outputs/gemma-4b-it-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)