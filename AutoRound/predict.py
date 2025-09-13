import os
from re import T
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams

prompts = [
    "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95)
model_name = "./tmp_autoround"
llm = LLM(model=model_name, 
tensor_parallel_size=1, 
max_model_len=2048,
enforce_eager=True,
dtype="bfloat16")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")