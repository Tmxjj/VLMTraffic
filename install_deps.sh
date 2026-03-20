#!/bin/bash
# 自动安装系统依赖、VirtualGL、SUMO 和 Python 依赖

# 遇到错误立即停止
set -e 

echo "🔄 [1/4] 正在安装系统基础依赖 (SUMO, Xvfb, OpenGL)..."
sudo apt-get update
sudo apt-get install -y sumo sumo-tools sumo-doc \
    xvfb x11-xserver-utils libgl1-mesa-glx wget

echo "⚙️ [2/4] 设置 SUMO_HOME 环境变量..."
if [ -z "$SUMO_HOME" ]; then
    echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
    export SUMO_HOME=/usr/share/sumo
    echo "SUMO_HOME 已设置为 $SUMO_HOME"
fi

echo "🎮 [3/4] 正在下载并安装 VirtualGL (GPU 加速核心)..."
# 下载 VirtualGL 稳定版
cd ~/code/VLMTraffic
if [ ! -f "virtualgl.deb" ]; then
    echo "未检测到安装包，正在从 SourceForge 下载 VirtualGL..."
    wget -q -O virtualgl.deb https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download
else
    echo "✅ 已检测到 virtualgl.deb，跳过下载步骤！"
fi
wget -q -O virtualgl.deb https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download
sudo dpkg -i virtualgl.deb || sudo apt-get install -f -y
echo "VirtualGL 安装完成！"

echo "🐍 [4/4] 正在安装 Python 依赖..."
# 确保使用的是当前激活的 conda 环境的 pip
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ 环境配置全部完成！"
echo "👉 运行提示: 请使用 xvfb-run 和 vglrun -d egl 启动你的 Python 脚本以获得硬件加速。"