import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Tuple, Union

import torch
from accelerate import (
    init_empty_weights,
)
from accelerate.utils.modeling import set_module_tensor_to_device
from tqdm import tqdm
from transformers import (
    GenerationMixin,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM, GenerationConfig, DynamicCache,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from safetensors.torch import (
    load_file
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import gc


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


class AirLLMBaseModel(GenerationMixin):

    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                                 'layer_prefix': 'model.layers',
                                 'norm': 'model.norm',
                                 'lm_head': 'lm_head', }

    def move_layer_to_device(self, state_dict):
        layers_to_move = set()
        for param_name in state_dict.keys():
            module_name = ".".join(param_name.split('.')[:-1])
            layers_to_move.add(module_name)

        for module_name in layers_to_move:
            try:
                tensor_name = f"{module_name}.weight"
                if tensor_name in state_dict:
                    set_module_tensor_to_device(self.model, tensor_name, self.running_device,
                                                value=state_dict[tensor_name],
                                                dtype=self.running_dtype)
                tensor_name_bias = f"{module_name}.bias"
                if tensor_name_bias in state_dict:
                    set_module_tensor_to_device(self.model, tensor_name_bias, self.running_device,
                                                value=state_dict[tensor_name_bias],
                                                dtype=self.running_dtype)
            except Exception as e:
                print(f"Could not move {module_name}: {e}")
        return layers_to_move

    def set_layers_from_layer_names(self):
        self.layers = []
        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)
        print(f"Total layers to process: {len(self.layers)}")

    def init_model(self):
        if hasattr(self, 'model') and self.model is not None:
            return

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)

        if hasattr(self, 'rotary_emb'):
            self.model.model.rotary_emb = self.rotary_emb

        self.model.eval()
        self.model.tie_weights()
        self.set_layers_from_layer_names()

        for buffer_name, buffer in self.model.named_buffers():
            if "rotary_emb" not in buffer_name:
                set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer,
                                            dtype=self.running_dtype)

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, **kwargs):
        super().__init__()
        self.set_layer_names_dict()
        self.model_local_path = Path(model_local_path_or_repo_id)
        self.running_device = device
        self.device = torch.device(self.running_device)
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)
        self._supports_cache_class = True
        self.generation_config = GenerationConfig.from_pretrained(model_local_path_or_repo_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

        print("Initializing Rotary Embedding on CPU...")
        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        print(f"Moving Rotary Embedding to {self.running_device}...")
        self.rotary_emb.to(self.running_device)

        self.model = None
        self.init_model()

        self.checkpoint_path = self.model_local_path
        model_attr = self.model.model.layers
        layers_count = len(model_attr)

        self.layer_names = [self.layer_names_dict['embed']] + [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in
                                                               range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]
        self.main_input_name = "input_ids"
        self.prefetching = kwargs.get("prefetching", True)
        if self.prefetching:
            self.stream = torch.cuda.Stream()
        print(f"Initialized AirLLM with {len(self.layer_names)} layer names.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_layer_to_cpu(self, layer_name):
        path = self.checkpoint_path
        if not os.path.exists(os.path.join(path, "model.safetensors.index.json")):
            filepath = Path(path) / "model.safetensors"
            if not hasattr(self, '_full_model_state_dict_cpu') or self._full_model_state_dict_cpu is None:
                self._full_model_state_dict_cpu = load_file(filepath, device="cpu")
            state_dict = {k: v for k, v in self._full_model_state_dict_cpu.items() if k.startswith(layer_name)}
        else:
            print(layer_name)
            filepath = Path(path) / "layers_safetensors" / (layer_name + ".safetensors")
            state_dict = load_file(filepath, device="cpu")

        if self.prefetching and torch.cuda.is_available():
            for k in state_dict:
                state_dict[k] = state_dict[k].pin_memory()
        return state_dict

    def run_lm_head(self, layer, seq):
        return layer(seq).to(torch.float32)

    def run_norm(self, layer, seq):
        return layer(seq)

    def can_generate(self):
        return True

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[DynamicCache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            if hasattr(self, '_full_model_state_dict_cpu'):
                del self._full_model_state_dict_cpu
                self._full_model_state_dict_cpu = None
            clean_memory()
        self.init_model()

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            hidden_states_batch = input_ids.to(self.running_device)
        else:
            hidden_states_batch = inputs_embeds.to(self.running_device)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        position_embeddings = None
        # --- THE FINAL, CRITICAL FIX ---
        # Initialize the 4D mask to None. It will be created at the correct time.
        expanded_attn_mask = None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                               desc=f'running layers({self.running_device})',
                                               total=len(self.layers), disable=True):
                if self.prefetching:
                    state_dict = future.result()
                    self.move_layer_to_device(state_dict)
                    if (i + 1) < len(self.layer_names):
                        future = executor.submit(self.load_layer_to_cpu, self.layer_names[i + 1])

                if layer_name == self.layer_names_dict['embed']:
                    if inputs_embeds is None:
                        hidden_states_batch = layer(hidden_states_batch)

                    # --- THE FINAL, CRITICAL FIX (Part 2) ---
                    # Now that hidden_states_batch is a float tensor, create the 4D mask.
                    # This happens only once, right after the embedding lookup.
                    expanded_attn_mask = _prepare_4d_causal_attention_mask(
                        attention_mask,
                        (hidden_states_batch.shape[0], hidden_states_batch.shape[1]),
                        hidden_states_batch,
                        past_key_values.get_seq_length() if past_key_values is not None else 0,
                    )

                    position_embeddings = self.rotary_emb(hidden_states_batch, position_ids)

                elif layer_name == self.layer_names_dict['norm']:
                    hidden_states_batch = self.run_norm(layer, hidden_states_batch)

                elif layer_name == self.layer_names_dict['lm_head']:
                    hidden_states_batch = self.run_lm_head(layer, hidden_states_batch)

                else:  # Decoder Layer
                    if output_hidden_states:
                        all_hidden_states += (hidden_states_batch,)

                    layer_outputs = layer(
                        hidden_states=hidden_states_batch,
                        attention_mask=expanded_attn_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                        cache_position=cache_position,
                    )
                    hidden_states_batch = layer_outputs[0]

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)

                layer.to("meta")

        logits = hidden_states_batch
        if output_hidden_states:
            all_hidden_states += (logits,)

        if not return_dict:
            outputs = (logits, past_key_values, all_hidden_states, all_self_attns)
            return tuple(v for v in outputs if v is not None)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


if __name__ == '__main__':
    #model_local_path_or_repo_id = "/data/gongoubo/Qwen-1.5-Factory/model_hub/Qwen/Qwen2___5-1___5B-Instruct"
    model_local_path_or_repo_id = "/data/gongoubo/checkpoints/Qwen/Qwen3-14B"

    print(f"Loading model from: {model_local_path_or_repo_id}")

    air_model = AirLLMBaseModel(model_local_path_or_repo_id,
                                device="cuda:0",
                                dtype=torch.float16)

    input_text = [
        'What is the capital of United States?',
        '你是谁？'
    ]



    input_text = [air_model.tokenizer.apply_chat_template([{"role":"user","content":t}],
                                                          tokenize=False,
                                                          add_generation_prompt=True)
    for t in input_text]

    print(f"\nInput text: {input_text}")
    input_tokens = air_model.tokenizer(input_text,
                                       return_tensors="pt",
                                       padding=True,
                                       truncation=True,
                                       max_length=128)

    print(f"\nTokenized input: {input_tokens}")

    input_ids = input_tokens['input_ids'].to(air_model.device)
    attention_mask = input_tokens['attention_mask'].to(air_model.device)

    generation_output = air_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=30,
        use_cache=True,
        return_dict_in_generate=True,
        do_sample=False
    )

    print("\n--- Generation Output ---")
    # print(generation_output)

    decoded_text = air_model.tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)

    print("\n--- Decoded Text ---")
    for text in decoded_text:
        print(text)