#!/bin/bash

# 设置中文显示
export LANG="zh_CN.UTF-8"
export LC_ALL="zh_CN.UTF-8"

# 设置CUDA可见设备，可根据需要修改
export CUDA_VISIBLE_DEVICES="0"

# 增加pandas显示宽度
export PYTHONIOENCODING=utf-8

# 激活虚拟环境（如果有）
# source /path/to/your/venv/bin/activate

# 确保脚本有执行权限
echo "设置脚本权限..."
chmod +x quantize_qwen3_06b_comparison.py

# 打印运行说明
echo "\n===== Qwen3-0.6B-Chat 量化比较 (GPTQ V1 vs V2) ====="
echo "此脚本将运行量化比较任务，包括以下步骤："
echo "1. 下载Qwen3-0.6B模型（Chat版本）和wikitext-2数据集"
echo "2. 准备校准数据"
echo "3. 使用GPTQ V1量化模型"
echo "4. 使用GPTQ V2量化模型"
echo "5. 评估两种量化方法的性能（困惑度、生成速度等）"
echo "6. 输出比较结果"
echo "\n注意事项："
echo "- 确保您的环境已安装所需依赖 (pip install -r requirements.txt)"
echo "- 量化过程可能需要一定的GPU内存（推荐至少8GB）"
echo "- 整个过程可能需要几分钟到几十分钟不等"
echo "- 结果将保存在 ./quantized_models/ 目录下"
echo "\n按Enter键继续，按Ctrl+C取消..."
read -r

# 运行Python脚本
python quantize_qwen3_06b_comparison.py

# 检查脚本执行结果
if [ $? -eq 0 ]; then
    echo "\n===== 量化比较任务已完成 ===="
    echo "您可以查看生成的量化模型："
    echo "- GPTQ V1: ./quantized_models/qwen3-0.6b-chat-gptq-v1-4bit-128g"
    echo "- GPTQ V2: ./quantized_models/qwen3-0.6b-chat-gptq-v2-4bit-128g"
    echo "\n比较结果已显示在屏幕上。"
else
    echo "\n===== 量化比较任务失败 ===="
    echo "请检查错误信息并尝试解决问题后重新运行。"
    exit 1
fi