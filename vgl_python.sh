#!/bin/bash
# vgl_python.sh (必须 chmod +x)
# 注意：-d egl 必须在 vglrun 后面，而在 python 的前面
#
# 重置 SHLVL 避免深层 bash 嵌套（conda activate + xvfb-run 多层子 shell 累加）触发警告
export SHLVL=1
xvfb-run -a -s "-screen 0 1920x1080x24" /opt/VirtualGL/bin/vglrun -d egl /home/jyf/anaconda3/envs/VLMTraffic/bin/python "$@"