#!/bin/bash

# === 配置区域 ===
TARGET_SCRIPT="/root/code/VLMTraffic/src/training/rpo_trainer.sh"
CHECK_INTERVAL=10         # 检查间隔时间（秒）
# ===============

# 1. 前置检查
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到 nvidia-smi 命令。"
    exit 1
fi

if [ ! -x "$TARGET_SCRIPT" ]; then
    echo "错误: 目标脚本 '$TARGET_SCRIPT' 不存在或不可执行。"
    echo "请运行: chmod +x $TARGET_SCRIPT"
    exit 1
fi

echo "======================================================="
echo "开始监控 A800 GPU 状态..."
echo "目标：当检测到 GPU 完全空闲（没有计算进程）时，启动脚本。"
echo "======================================================="

# 无限循环开始监控
while true; do
    # === 核心修改 ===
    # 只查询计算进程 (Compute Apps)。
    # 对于 A800 这种纯计算卡，这是最准确的方法。
    # grep -v '^$' 用于去除可能的空行
    # wc -l 用于统计行数（即进程数）
    compute_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l)
    
    # 这里为了方便后续判断，直接把计算进程数当作总进程数
    total_processes=$compute_processes

    if [ "$total_processes" -eq 0 ]; then
        echo ""
        echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] 检测结果：GPU 当前空闲！"
        echo ">>> 正在启动 $TARGET_SCRIPT ..."
        
        # 执行目标脚本 (阻塞模式，等待执行完毕)
        # "$TARGET_SCRIPT"
        
        # 如果需要后台执行，取消下面这行的注释，并注释掉上面那行
        nohup "$TARGET_SCRIPT" > /root/code/VLMTraffic/rpo_trainer.log 2>&1 &

        echo ">>> $TARGET_SCRIPT 启动动作已执行。"
        echo ">>> 监控任务完成，退出监控脚本。"
        break
    else
        # 这个数字现在应该稳定显示为 4
        echo "[$(date '+%H:%M:%S')] GPU 忙碌中 (当前有 $total_processes 个计算进程)，继续等待空闲..."
        sleep $CHECK_INTERVAL
    fi
done