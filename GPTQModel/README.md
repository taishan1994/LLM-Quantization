安装依赖：

```shell
conda create -n py310 python==3.10.0
conda activate py310

pip install -r requirements.txt
```

量化比较：

```shell
bash run_quantize_comparison.sh
```

量化结果：

```shell
===== 量化结果比较 ====
GPTQ V1:
  量化时间: 192.97秒
  困惑度: 126.1760
  生成速度: 17.47 tokens/second
GPTQ V2:
  量化时间: 251.02秒
  困惑度: 121.7291
  生成速度: 17.47 tokens/second
```

以后可以把GPTQ量化换为GPTQV2了。
