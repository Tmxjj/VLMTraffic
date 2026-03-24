#!/bin/bash

# --- 路径定义 ---
EVAL_SRC="/root/code/VLMTraffic/data/eval/"
LOG_SRC="/root/code/VLMTraffic/log/"


EVAL_DST="/root/autodl-fs/data/eval"
LOG_DST="/root/autodl-fs/data/log"

# --- 1. 创建目标目录 ---
# mkdir -p 会递归创建目录，如果已存在则跳过
echo "📂 检查并创建目标目录..."
mkdir -p "$EVAL_DST"
mkdir -p "$LOG_DST"

# --- 2. 使用 rsync 同步 Eval 数据 ---
# -a: 归档模式（保留权限、时间戳等）
# -v: 显示详细过程
# -z: 传输时压缩（如果跨网络建议加上，本地路径可选）
# --ignore-existing: 跳过目标目录已存在的文件（模拟你原脚本的 -n）
echo "🚀 开始增量同步 Eval 数据 (仅备份新文件)..."
rsync -av --ignore-existing "$EVAL_SRC" "$EVAL_DST"

# --- 3. 使用 rsync 移动 Log 文件 ---
# --remove-source-files: 传输完成后删除源端文件（实现“移动”效果）
echo "📦 开始移动日志文件 (Move)..."
rsync -av --remove-source-files "$LOG_SRC" "$LOG_DST"

# 注意：rsync --remove-source-files 只删除文件，不删除源端的空目录。
# 如果需要清理空目录，可以取消下面这行的注释：
# find "$LOG_SRC" -type d -empty -delete

echo "✅ 所有操作已完成！"