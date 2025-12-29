import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from io import BytesIO
from qwen_vl_utils import process_vision_info
import base64
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "/home/gongoubo/checkpoints/Qwen/Qwen3-VL-32B-Instruct"


# Load model.
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto",device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Oneshot arguments
DATASET_ID = "/home/gongoubo/flikr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)


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


ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)


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
        "re:visual.*",
        "re:model.visual.*",
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
    sequential_targets=["Qwen3VLTextDecoderLayer"]
)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")


# Save to disk in compressed-tensors format.
SAVE_DIR = "/home/gongoubo/outputs/qwen3-vl-32b-NVFP4-flickr"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)