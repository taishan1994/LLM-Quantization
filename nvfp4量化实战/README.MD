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

