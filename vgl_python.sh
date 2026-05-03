#!/bin/bash
# vgl_python.sh (必须 chmod +x)
# 注意：-d egl 必须在 vglrun 后面，而在 python 的前面
#
# 用法示例：
#   ./vgl_python.sh --gpu_id 1 src/evaluation/run_eval.py ...
#   VGL_GPU=1 ./vgl_python.sh src/evaluation/run_eval.py ...
#
# 重置 SHLVL 避免深层 bash 嵌套（conda activate + xvfb-run 多层子 shell 累加）触发警告
export SHLVL=1

GPU_ID="${VGL_GPU:-${CUDA_VISIBLE_DEVICES:-}}"
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu_id)
            if [[ $# -lt 2 ]]; then
                echo "[vgl_python.sh] Error: --gpu_id requires a value" >&2
                exit 1
            fi
            GPU_ID="$2"
            shift 2
            ;;
        --gpu_id=*)
            GPU_ID="${1#*=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "${GPU_ID}" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export EGL_VISIBLE_DEVICES="${GPU_ID}"
    export VGL_GPU="${GPU_ID}"
    echo "[vgl_python.sh] Using GPU ${GPU_ID} for EGL/CUDA rendering"
fi

xvfb-run -a -s "-screen 0 1920x1080x24" /opt/VirtualGL/bin/vglrun -d egl /home/jyf/anaconda3/envs/VLMTraffic/bin/python "${ARGS[@]}"
