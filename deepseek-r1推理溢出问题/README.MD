# 说明

对于多机多卡部署deepseek-r1满血版，需要先使用ray将不同的机器进行整合（请确保所有机器之间或者所有容器之间可以进行免密登录），至少需要4×A800(80G)

```shell
1. 修改ray_start.sh里面的机器地址，列表中第一个为master的ip地址
bash ray_start.sh

2. 修改vllm-deepseek_r1_fp16.sh的ip为master的ip地址
bash vllm-deepseek_r1_fp16.sh
```

如果使用awq模型启动，只需要一台A800(80G)：

```shell
bash vllm-deepseek_r1_awq_w4a16_fp16.sh
```

test_chat.py用于测试部署好的模型。