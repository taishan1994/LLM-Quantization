首先根据neural-compressor安装好相关的环境。
启动量化：
```shell
python quant.py --model /home/gongoubo/checkpoints/Qwen/Qwen3-4B --num_gpus 8 --base_port 11334 --quantize --accuracy --tasks leaderboard --w_dtype fp6_e2m3 --act_dtype fp6_e2m3
```
启动测试：
```shell
python eval.py
```
注意这里面可能需要根据实际情况修改调用api的端口，里面是用了4个port，根据实际起的实例进行修改。