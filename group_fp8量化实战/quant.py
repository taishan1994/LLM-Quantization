import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import json
from kernel import weight_cast_to_fp8
import shutil
from safetensors.torch import save_file


def quant_all(model, model_path, save_dir, ignore_layers=[]):
    quant_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [
            128,
            128
        ],
        "ignore": []
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = {}

    for k, v in model.named_parameters():

        v = v.cuda()

        if any([1 if ignore in k else 0 for ignore in ignore_layers]):
            state_dict[k] = v.detach().cpu()
            k = k.replace(".weight", "").replace(".bias", "")
            if k not in quant_config["ignore"]:
                quant_config["ignore"].append(k)
            continue
        if "norm" in k or "bias" in k or "embed" in k:
            state_dict[k] = v.detach().cpu()
            continue

        w, s = weight_cast_to_fp8(v)

        weight_name = k
        scale_name = weight_name + "_scale_inv"
        w = w.detach().cpu()
        s = s.detach().cpu()
        state_dict[weight_name] = w
        state_dict[scale_name] = s

    save_file(state_dict, os.path.join(save_dir, "model.safetensors"))

    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)

    config["quantization_config"] = quant_config

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 将 model_path 下除了 *.safetensors 和 config.json 以外的文件拷贝到 save_dir
    for item in os.listdir(model_path):
        src_path = os.path.join(model_path, item)
        dst_path = os.path.join(save_dir, item)
        if os.path.isfile(src_path) and item != "config.json" and "safetensors" not in item:
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


if __name__ == "__main__":

    model_name = "qwen3-vl"

    if model_name == "qwen3_0.6b":
        from transformers import AutoModelForCausalLM

        model_path = "/home/gongoubo/checkpoints/Qwen/Qwen3-0___6B"
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map="auto")
        save_dir = "/home/gongoubo/outputs/qwen3-0.6b-fp8"

        ignore_layers = [
            "lm_head"
        ]

        quant_all(model, model_path, save_dir, ignore_layers)

    elif model_name == "qwen3-vl":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = "/home/gongoubo/checkpoints/Qwen/Qwen3-VL-32B-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        save_dir = "/home/gongoubo/outputs/qwen3-vl-32b-fp8"

        ignore_layers = [
            "lm_head",
            "visual",
        ]

        quant_all(model, model_path, save_dir, ignore_layers)

    elif model_name == "minicpm-4.5-v-8b":
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        model_path = "/home/gongoubo/checkpoints/OpenBMB/MiniCPM-V-4_5"
        model = model = AutoModel.from_pretrained(model_path, trust_remote_code=True,  # or openbmb/MiniCPM-o-2_6
                                                  attn_implementation='sdpa', torch_dtype=torch.bfloat16).cuda()
        save_dir = "/home/gongoubo/outputs/minicpm-4.5-v-8b-fp8"

        ignore_layers = [
            "lm_head",
            "visual",
            "vision",
            "image",
            "resampler",
            "vpm",
            "aligner",
        ]

        quant_all(model, model_path, save_dir, ignore_layers)

    elif model_name == "gemma":
        from transformers import Gemma3ForConditionalGeneration, AutoTokenizer, AutoProcessor

        model_path = "/home/gongoubo/checkpoints/google/gemma-3-4b-it"
        # Load model.
        model = Gemma3ForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True,
                                                               torch_dtype=torch.bfloat16).cuda()
        save_dir = "/home/gongoubo/outputs/gemma-3-4b-it-fp8"

        ignore_layers = [
            "lm_head",
            "model.visual_tower",
            "model.multi_modal_projector",
            "vision",
        ]
        quant_all(model, model_path, save_dir, ignore_layers)