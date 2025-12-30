import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import types
import math
import importlib
import torch
from datasets import load_dataset
import torch.fx.graph_module as fx_graph_module
import numpy as np
from PIL import Image
from copy import deepcopy
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from torch.fx import wrap

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# 禁用 llmcompressor 内部的 accelerate 自动 dispatch，以避免 meta 设备与 cuda 设备混用
import llmcompressor.utils as _lc_utils
import llmcompressor.pipelines.basic.pipeline as _basic_pipeline_module


def _no_dispatch_for_generation(model):
    # 保持模型在当前设备，不做自动切分/下 offload
    return model


_lc_utils.dispatch_for_generation = _no_dispatch_for_generation
_basic_pipeline_module.dispatch_for_generation = _no_dispatch_for_generation
# 覆盖当前模块中的 dispatch_for_generation 引用
dispatch_for_generation = _no_dispatch_for_generation

MODEL_ID = "/home/gongoubo/checkpoints/OpenBMB/MiniCPM-V-4_5"

# Patch torch.fx.graph_module._copy_attr 以避免空 target ('') 导致的 AttributeError
# GraphModule 在构建时会为每个 `call_module` 复制子模块，如果 target 是 ''，其语义等价于
# `get_submodule("") -> self`，不需要再额外复制属性，此时我们直接跳过即可。
_orig_copy_attr = fx_graph_module._copy_attr


def _copy_attr_skip_empty(from_module, to_module, field):
    if field == "":
        # 空字段用于表示 root 本身，不需要在 GraphModule 上新增同名属性
        return
    return _orig_copy_attr(from_module, to_module, field)


fx_graph_module._copy_attr = _copy_attr_skip_empty

# Load model.
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,  # or openbmb/MiniCPM-o-2_6
                                  attn_implementation='sdpa',
                                  torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager

# --- Patch SiglipVisionEmbeddings 以避免 position_ids 落在 meta 设备 ---
try:
    vision_embeddings = getattr(model, "vpm", None)
    if vision_embeddings is not None and hasattr(vision_embeddings, "embeddings"):
        siglip_embeddings = vision_embeddings.embeddings


        def siglip_embeddings_forward_patched(self, pixel_values, patch_attention_mask, tgt_sizes=None):
            batch_size = pixel_values.size(0)

            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
            max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
            boundaries = torch.arange(
                1 / self.num_patches_per_side,
                1.0,
                1 / self.num_patches_per_side,
                device=pixel_values.device,
            )
            position_ids = torch.full(
                size=(
                    batch_size,
                    max_nb_patches_h * max_nb_patches_w,
                ),
                fill_value=0,
                device=pixel_values.device,
                dtype=torch.long,
            )

            for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                if tgt_sizes is not None:
                    nb_patches_h = tgt_sizes[batch_idx][0]
                    nb_patches_w = tgt_sizes[batch_idx][1]
                else:
                    nb_patches_h = p_attn_mask[:, 0].sum()
                    nb_patches_w = p_attn_mask[0].sum()

                fractional_coords_h = torch.arange(
                    0, 1 - 1e-6, 1 / nb_patches_h, device=pixel_values.device
                )
                fractional_coords_w = torch.arange(
                    0, 1 - 1e-6, 1 / nb_patches_w, device=pixel_values.device
                )

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
                # p_attn_mask 可能在 CPU 或 GPU，这里统一到 position_ids 所在设备
                mask_flat = p_attn_mask.view(-1).to(position_ids.device)
                position_ids[batch_idx][mask_flat] = pos_ids

            # 关键改动：统一将 position_ids 放在图像所在设备，而不是 position_embedding.weight 的设备
            embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings


        vision_embeddings.embeddings.forward = types.MethodType(
            siglip_embeddings_forward_patched, siglip_embeddings
        )
except Exception:
    # 如果结构与预期不符（例如未来版本结构调整），保持原实现不变
    pass

# 保存一份原始 forward，用于自定义一个带显式输入名的包装函数，方便 FX 跟踪
_orig_forward = model.forward

# 将原始 forward 暴露到定义 MiniCPMV 类的模块里，方便 autowrap_forward 重新编译时访问
minicpmv_module = importlib.import_module(model.__class__.__module__)
setattr(minicpmv_module, "_orig_forward", _orig_forward)


def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_sizes=None,
        image_bound=None,
        tgt_sizes=None,
        temporal_ids=None,
        position_ids=None,
        **kwargs,
):
    """FX 友好的 MiniCPMV 多模态 forward：构造 data 和 position_ids，然后直接调用底层 LLM。"""
    # 有些上游会额外传 image_sizes，下游 LLM 不需要，直接丢弃
    kwargs.pop("image_sizes", None)

    # 构造默认的 position_ids（始终使用 torch.long，避免在 forward 中访问 dtype）
    if input_ids is not None:
        seq_len = input_ids.shape[-1]
        default_position_ids = torch.arange(
            seq_len, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0).expand_as(input_ids)
    else:
        default_position_ids = None

    # 避免使用 if 语句，让 AutoWrapper 不去抽取这段逻辑
    position_ids = position_ids if position_ids is not None else default_position_ids

    # 构造 MiniCPMV 使用的 data 字典（仅用于视觉路径和 embedding 构造）
    data = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_bound": image_bound,
        "tgt_sizes": tgt_sizes,
        "position_ids": position_ids,
        "temporal_ids": temporal_ids,
    }

    # 通过已经 patch 过的 get_vllm_embedding 走完整的图文路径
    vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)

    # 清理与 LLM 冲突的键，保持与原始 forward 一致
    for key in ["input_ids", "inputs_embeds", "position_ids"]:
        if key in kwargs:
            kwargs.pop(key)

    # attention_mask 等仍然通过 kwargs 传入 Qwen3 LLM
    return self.llm(
        input_ids=None,
        position_ids=position_ids,
        inputs_embeds=vllm_embedding,
        **kwargs,
    )


# 替换模型的 forward，使 SequentialTracer 能看到显式的文本+图像参数，而不是 data/kwargs
model.forward = types.MethodType(forward, model)


@wrap
def _minicpmv_get_vllm_embedding_fx(self, data):
    if 'vision_hidden_states' not in data:
        dtype = self.llm.model.embed_tokens.weight.dtype
        pixel_values_list = data['pixel_values']
        tgt_sizes = data['tgt_sizes']
        # 设备直接沿用 LLM 嵌入权重所在设备（当前脚本中即 cuda:0）
        device = self.llm.model.embed_tokens.weight.device
        temporal_ids = data.get('temporal_ids', None)
        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []
        all_temporal_ids = None

        for pixel_values in pixel_values_list:
            img_cnt.append(len(pixel_values))
            all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

        if temporal_ids is not None:
            all_temporal_ids = []
            for t in temporal_ids:
                all_temporal_ids.extend(t)

        # exist image
        if all_pixel_values:
            tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

            all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                all_pixel_values, batch_first=True, padding_value=0.0
            )
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

            patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
            for i in range(B):
                patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

            vision_batch_size = self.config.vision_batch_size
            all_pixel_values = all_pixel_values.type(dtype)
            if B > vision_batch_size:
                hs = []
                for i in range(0, B, vision_batch_size):
                    start_idx = i
                    end_idx = i + vision_batch_size
                    tmp_hs = self.vpm(
                        all_pixel_values[start_idx:end_idx],
                        patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                        tgt_sizes=tgt_sizes[start_idx:end_idx],
                    ).last_hidden_state
                    hs.append(tmp_hs)
                vision_embedding = torch.cat(hs, dim=0)
            else:
                vision_embedding = self.vpm(
                    all_pixel_values,
                    patch_attention_mask=patch_attn_mask,
                    tgt_sizes=tgt_sizes,
                ).last_hidden_state
            vision_embedding = self.resampler(vision_embedding, tgt_sizes, all_temporal_ids)

            start = 0
            for pixel_values in pixel_values_list:
                cur_cnt = len(pixel_values)
                if cur_cnt > 0:
                    vision_hidden_states.append(vision_embedding[start: start + cur_cnt])
                    start += cur_cnt
                else:
                    vision_hidden_states.append([])
        else:  # no image
            if self.training:
                dummy_image = torch.zeros(
                    (1, 3, 224, 224),
                    device=device,
                    dtype=dtype,
                )
                tgt_sizes = torch.Tensor(
                    [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
                ).type(torch.int32)
                dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
            else:
                dummy_feature = []
            for _ in range(len(pixel_values_list)):
                vision_hidden_states.append(dummy_feature)
    else:
        vision_hidden_states = data['vision_hidden_states']

    if hasattr(self.llm.config, 'scale_emb'):
        vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
    else:
        vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

    vision_hidden_states = [
        i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
        for i in vision_hidden_states
    ]

    bs = len(data['input_ids'])
    device = vllm_embedding.device
    embed_dim = vllm_embedding.shape[-1]

    updated_vllm_embedding = torch.empty_like(vllm_embedding)

    for i in range(bs):
        cur_vs_hs = vision_hidden_states[i]
        cur_vllm_emb = vllm_embedding[i]

        if len(cur_vs_hs) == 0:
            updated_vllm_embedding[i] = cur_vllm_emb
            continue

        cur_image_bound = data['image_bound'][i]

        if len(cur_image_bound) > 0:
            image_indices = torch.cat(
                [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
            ).to(device)

            indices_expanded = image_indices.view(-1, 1).expand(-1, embed_dim)
            vision_features = cur_vs_hs.view(-1, embed_dim)

            updated_emb = cur_vllm_emb.clone()
            vision_features = vision_features.to(cur_vllm_emb.device)

            updated_emb.scatter_(0, indices_expanded, vision_features)
            updated_vllm_embedding[i] = updated_emb
        elif self.training:
            if isinstance(cur_vs_hs, torch.Tensor) and cur_vs_hs.numel() > 0:
                dummy_gradient_term = cur_vs_hs.sum() * 0.0
                updated_vllm_embedding[i] = cur_vllm_emb + dummy_gradient_term
            else:
                updated_vllm_embedding[i] = cur_vllm_emb
        else:
            updated_vllm_embedding[i] = cur_vllm_emb

    vllm_embedding = updated_vllm_embedding

    return vllm_embedding, vision_hidden_states


def get_vllm_embedding_patched(self, data):
    # 在 FX trace 阶段，_minicpmv_get_vllm_embedding_fx 会被当作叶子函数处理，
    # 因此不会在其中迭代 Proxy；在实际校准和推理时，该函数会正常执行多模态逻辑。
    return _minicpmv_get_vllm_embedding_fx(self, data)


# 用 FX 友好的实现覆盖原始 get_vllm_embedding，实现图文路径的量化
model.get_vllm_embedding = types.MethodType(get_vllm_embedding_patched, model)
model = model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)  # or openbmb/MiniCPM-o-2_6

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

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

DATASET_ID = "/home/gongoubo/flikr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)

enable_thinking = False
system_prompt = None


def preprocess_function2(example):
    msgs = [
        {
            "role": "user",
            "content": [example["image"].convert('RGB'), "What does the image show?"],
        }
    ]

    image = None
    batched = False
    sampling = False
    stream = False
    max_slice_nums = None
    use_image_id = None
    temporal_ids = None
    max_inp_length = 16384

    msgs_list = msgs
    images_list = image

    if batched is False:
        images_list, msgs_list = [images_list], [msgs_list]
    else:
        assert images_list is None, "Please integrate image to msgs when using batch inference."
        images_list = [None] * len(msgs_list)
    assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

    prompts_lists = []
    input_images_lists = []
    for image, msgs in zip(images_list, msgs_list):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        copy_msgs = deepcopy(msgs)

        assert len(msgs) > 0, "msgs is empty"
        assert sampling or not stream, "if use stream mode, make sure sampling=True"

        if image is not None and isinstance(copy_msgs[0]["content"], str):
            copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["system", "user", "assistant"]
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        if system_prompt:
            sys_msg = {'role': 'system', 'content': system_prompt}
            copy_msgs = [sys_msg] + copy_msgs

        prompts_lists.append(
            processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=enable_thinking))
        input_images_lists.append(images)

    if enable_thinking:
        prefill_answer = '<think>\n'
    else:
        prefill_answer = ''

    # print(prompts_lists)
    # print(input_images_lists)

    inputs = processor(
        prompts_lists,
        input_images_lists,
        max_slice_nums=max_slice_nums,
        use_image_id=use_image_id,
        temporal_ids=temporal_ids,
        return_tensors="pt",
        max_length=max_inp_length,
        padding="max_length",
    )

    # print(inputs.keys())

    return inputs


ds = ds.map(preprocess_function2, batched=False, remove_columns=ds.column_names)

print(ds)


def data_collator(batch):
    """MiniCPMV 专用 collator，复用官方测试脚本里的逻辑。

    - 对 `input_ids`/`attention_mask` 等按最大长度做 padding；
    - 对 `pixel_values`/`image_bound`/`tgt_sizes`/`temporal_ids` 保持为 list[list[Tensor]] 结构，
      与 `MiniCPMVImageProcessor` 和 `get_vllm_embedding` 的预期一致。
    """
    result = {}
    for key in batch[0].keys():
        # `image_sizes` 在 forward 中不会用到，这里直接丢弃以避免后续 tensors_to_device
        if key == "image_sizes":
            continue
        if key == "pixel_values":
            # 每个样本的 pixel_values 是形如 [new_images]，其中第 0 个是该样本的 patch 列表。
            # 这里将每个 patch 显式转换为 Tensor，但不在此处 stack，保留为 list[list[Tensor]]，
            # 由 MiniCPMV 的 get_vllm_embedding 内部 pad/reshape 处理变长 patch。
            result[key] = [
                [torch.as_tensor(p, dtype=torch.float32) for p in item[key][0]]
                for item in batch
            ]
        elif key == "image_bound":
            # 形如 [tensor(num_images, 2)]，这里统一转成 Tensor
            result[key] = [torch.tensor(item[key][0]) for item in batch]
        elif key == "tgt_sizes":
            # 形如 [tensor(num_patches, 2)]，统一转成 Tensor
            result[key] = [torch.tensor(item[key][0]) for item in batch]
        elif key == "temporal_ids":
            # 原始为嵌套 list[int]，这里转为 Tensor，后续在 resampler 中只会用到其数值
            result[key] = [torch.tensor(item[key][0]) for item in batch]
        else:
            # 对文本类键（input_ids/attention_mask）显式转换为 Tensor，
            # 因为 HuggingFace Dataset 会把它们存成 Python 序列
            if key in ("input_ids", "attention_mask", "position_ids"):
                values = [torch.tensor(item[key]) for item in batch]
            else:
                values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                # 2D 序列（[B, L]）按最大长度做右侧 padding，再拼成 batch
                if len(values[0].shape) == 2:
                    max_len = max(v.size(1) for v in values)
                    padded = []
                    for v in values:
                        pad_len = max_len - v.size(1)
                        if pad_len > 0:
                            if key == "input_ids":
                                pad_value = (
                                    tokenizer.pad_token_id
                                    if tokenizer.pad_token_id is not None
                                    else 0
                                )
                            elif key == "attention_mask":
                                pad_value = False
                            else:
                                pad_value = 0
                            padded_v = torch.nn.functional.pad(
                                v, (0, pad_len), value=pad_value
                            )
                            padded.append(padded_v)
                        else:
                            padded.append(v)
                    result[key] = torch.cat(padded, dim=0)
                else:
                    # 其他张量直接按 batch 维拼接
                    result[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], list):
                result[key] = values
            else:
                result[key] = values
    return result


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
# 这里显式指定使用 `basic` pipeline，避免 sequential pipeline 里的 FX 子图追踪，
# 直接用我们构造好的图文 batch 走整模型 forward 进行校准。
oneshot(
    model=model,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    dataset=ds,
    data_collator=data_collator,
    trust_remote_code_model=True,
    pipeline="basic",
)

# 简单统计一下被附加量化配置的层数，方便确认逐层量化已生效
quantized_modules = []
for name, module in model.named_modules():
    if hasattr(module, "quantization_scheme"):
        quantized_modules.append(name)
        print(name)
print(f"[Debug] Quantized modules count: {len(quantized_modules)}")
# 如需查看具体某几层，可取消下一行注释：
# print("[Debug] First 10 quantized modules:", quantized_modules[:10])

# print("========== SAMPLE GENERATION ==============")
# print("[Debug] Starting quantized chat demo (max_new_tokens=64)...")

# dispatch_for_generation(model)

# # 创建一个虚拟图像以满足多模态模型的输入要求
# # 创建一个标准尺寸的白色图像 (224x224)
# dummy_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)

# text = "你是谁"
# msgs = [{'role': 'user', 'content': [dummy_image, text]}]

# # 使用 chat 方法进行推理，注意这里将 max_new_tokens 缩小到 64，方便快速 sanity check
# response = model.chat(
#     msgs=msgs,
#     tokenizer=tokenizer,
#     max_new_tokens=64,
#     do_sample=False,
# )

# print("[Debug] Chat response:", response)
# print("==========================================")


# Save to disk in compressed-tensors format.
SAVE_DIR = "/home/gongoubo/outputs/minicpm-4.5-v-8b-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)