#!/bin/bash
set -e

# 定义要安装的机器列表

HOSTS=(
    "192.168.112.86" # head node
    "192.168.112.95"
    "192.168.112.88"
    "192.168.112.9"

)



HEAD_NODE="${HOSTS[0]}" # 使用第一个节点作为head节点

# 记录开始时间
start_time=$(date +%s)
echo "开始安装 vLLM (editable mode) 到多节点: $(date)"

##################### setup pip config #################################

# 创建临时目录存储安装日志
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# 获取当前脚本所在目录的上一级目录（vLLM 代码根目录）
VLLM_ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
echo "vLLM root directory: $VLLM_ROOT_DIR"

##################### setup git config #################################
echo -e "\nSetting up git config safe.directory on all nodes..."
for host in "${HOSTS[@]}"; do
    echo "Setting git config on $host..."
    ssh "$host" "git config --global --add safe.directory $VLLM_ROOT_DIR" || {
        echo "Failed to set git config on $host"
    }
done

# 在每台机器上并行执行pip配置和vllm安装
for host in "${HOSTS[@]}"; do
    {
        echo "Starting setup on $host..."

        # 在editable模式下安装vLLM
        echo "Installing vLLM in editable mode on $host..."
        if ssh "$host" "pip install vllm==0.7.3" >>"$TEMP_DIR/$host.log" 2>&1; then
            ssh "$host" "pip install transformers==4.48.2"
            echo "Successfully installed vLLM in editable mode on $host"
            # 验证安装
            if ssh "$host" "python3 -c 'import vllm; print(f\"Successfully installed vllm {vllm.__version__}\")'" >>"$TEMP_DIR/$host.log" 2>&1; then
                echo "Verified vLLM installation on $host"
                echo "SUCCESS" >"$TEMP_DIR/$host.status"
            else
                echo "Failed to verify vLLM installation on $host"
                echo "FAILED" >"$TEMP_DIR/$host.status"
            fi
        else
            echo "Failed to install vLLM in editable mode on $host"
            echo "FAILED" >"$TEMP_DIR/$host.status"
        fi
    } &
done

# 等待所有后台任务完成
wait

# 汇总安装结果
echo -e "\nInstallation Results:"
echo "--------------------"
for host in "${HOSTS[@]}"; do
    status=$(cat "$TEMP_DIR/$host.status" 2>/dev/null || echo "UNKNOWN")
    echo "$host: $status"
    if [ "$status" = "FAILED" ]; then
        echo "Error log for $host:"
        cat "$TEMP_DIR/$host.log"
    fi
done

# 清理临时文件
rm -rf "$TEMP_DIR"

##################### start ray cluster #################################
echo -e "\nStarting Ray cluster..."

# 首先停止所有节点上可能存在的Ray进程
for host in "${HOSTS[@]}"; do
    echo "Stopping existing Ray processes on $host..."
    ssh "$host" "ray stop --force" || true
done

# 在head节点启动head服务
echo "Starting head node on $HEAD_NODE..."
ssh "$HEAD_NODE" "ray start --head --port=6379 --dashboard-host=0.0.0.0" || {
    echo "Failed to start head node"
    exit 1
}

# 拷贝文件到每台机器
ssh "$HEAD_NODE" "cp deepseek_v2.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py"

# 等待几秒钟确保head节点完全启动
sleep 5

# 在其他节点启动worker
for host in "${HOSTS[@]:1}"; do
    echo "Connecting worker node $host to head node..."
    ssh "$host" "cp deepseek_v2.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py"
    sleep 1s
    ssh "$host" "ray start --address='$HEAD_NODE:6379'" || {
        echo "Failed to connect worker node $host"
    }
    sleep 1s
done

# 记录结束时间并计算总耗时
end_time=$(date +%s)
echo -e "\nRay cluster setup completed. Head node is $HEAD_NODE"
echo "To check cluster status, run: ray status on any node"

echo -e "\n安装完成: $(date)"
duration=$((end_time - start_time))
echo "总安装时间: $((duration / 60)) 分 $((duration % 60)) 秒" 
