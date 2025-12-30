# 运行

准备环境：

```shell
cd llm-compressor

pip install -e .
```

量化：注意替换模型以及保存地址。

```shell
python3 quant_nvfp4/qwen3_vl_dense.py
```

部署：注意替换部署模型地址。

```shell
bash utils/vllm_deploy.sh
```

推理：注意替换服务url以及图片路径。

```shell
bash utils/predict_vlm.py
```

注意：

- 这里我们只量化LLM。
- 如果显卡不支持nvfp4的矩阵运算，其会退回到嗲用w4a16：

> (EngineCore_DP0 pid=598) WARNING 12-16 07:44:48 [compressed_tensors.py:490] Current platform does not support cutlass NVFP4. Running CompressedTensorsW4A16Fp4.
>
> (EngineCore_DP0 pid=598) WARNING 12-16 07:45:01 [marlin_utils_fp4.py:128] Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.

4090可以w4a4推理，但是需要：

```
export VLLM_USE_NVFP4_CT_EMULATIONS=1
vllm serve /nfs/FM/gongoubo/new_project/workflow/checkpoints/qwen3-vl-8b-NVFP4 --port 32000 --host 0.0.0.0 --max-model-len 4000
```

- export VLLM_USE_NVFP4_CT_EMULATIONS=1
- 并且不使用cuda_graph：enforce-eager

# 其它说明

quant_nvfp4里面提供了三类模型的量化：Qwen-VL、MiniCPM-4.5-V、Gemma3

量化完成后使用5090进行nvfp4的推理，直接pip安装vllm==0.13.0即可。

量化MiniCPM-4.5-V时，修改了modeling_minicpmv.py里面的forward：

```python
def forward(self, input_ids=None, attention_mask=None):
        kwargs = {'attention_mask': attention_mask}
        return self.llm(
            input_ids=input_ids,
            **kwargs
        )

def forward_v2(self, data, **kwargs):
    if isinstance(data, torch.Tensor):
        attention_mask = torch.ones_like(data, dtype=torch.bool)
        kwargs = {'attention_mask': attention_mask}
        return self.llm(
            input_ids=data,
            **kwargs
        )
    ......
```

**使用图文数据量化的模型一般会比纯文本量化的模型精度要高。**

使用图文数据集来量化Qwen3-VL，需要安装指定版本的transformers：`cd transformers-main && pip install -e .`

>  百度网盘现在贼恶心，上传文件还要限制。阿里云盘分享又有限制，需要的联系我分享。

使用图文数据集量化MiniCPM-4.5-V会报各种各样的错误，这里我已经修改好了，可参考带flickr后缀的代码。
