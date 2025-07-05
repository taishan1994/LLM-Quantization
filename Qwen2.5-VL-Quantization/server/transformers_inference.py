import os
import sys
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
ld_library = os.environ.get("LD_LIBRARY_PATH", "")
print(ld_library)

new_path = "/opt/hpcx/ucx/lib"
if new_path not in ld_library.split(":"):
    os.environ["LD_LIBRARY_PATH"] = f"{new_path}:{ld_library}" if ld_library else  new_path
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForVision2Seq, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vision_process import process_vision_info
from safetensors.torch import load_file

# model_path = "/ms/FM/checkpoints/Qwen-Zoo/Qwen2___5-VL-7B-Instruct"

model_path = "/ms/FM/gongoubo/checkpoints/data/Qwen2___5-VL-7B-Instruct-edit"

# default: Load the model on the available device(s)
#  device_map="auto",
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", trust_remote_code=True
).eval().to("cuda:0")


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "images/test2.png",
            },
            {"type": "text", "text": "请简单描述下图片。"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(text)

# print(processor.decode([151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 151652, 151655, 151653, 26987, 25, 18137, 246, 227, 57553, 108704, 8997, 104155, 9370, 109729, 102298, 99604, 109963, 99877, 1773, 90919, 101441, 31905, 20412, 115534, 32948, 33108, 101843, 99888, 32948, 1773, 115534, 32948, 99877, 44063, 52510, 33108, 99298, 17177, 45181, 104155, 9370, 99408, 32948, 90840, 105971, 110365, 1773, 115534, 32948, 101097, 105798, 101971, 101252, 3837, 107835, 104155, 9370, 109729, 105798, 101079, 1773, 101843, 99888, 32948, 99877, 44063, 99298, 17177, 45181, 110365, 90840, 105971, 104155, 9370, 92894, 99659, 1773, 101843, 99888, 32948, 99877, 101047, 99298, 17177, 73670, 105798, 57191, 111228, 101079, 3837, 107835, 104155, 9370, 109729, 8997, 14582, 25, 220, 87752, 104673, 116925, 57218, 45930, 48921, 106188, 94432, 3798, 510, 32, 13, 18137, 253, 100, 99888, 32948, 198, 33, 13, 220, 115534, 32948, 198, 5501, 3293, 279, 4396, 4226, 504, 279, 2606, 3403, 13, 3155, 537, 2924, 894, 4960, 2213, 26, 2550, 279, 4226, 5961, 198, 151645, 198, 151644, 77091, 198]))

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=None,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
