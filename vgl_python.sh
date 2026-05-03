#!/bin/bash
# vgl_python.sh (必须 chmod +x)
# 注意：-d egl 必须在 vglrun 后面，而在 python 的前面
#
# 用法示例：
#   ./vgl_python.sh --gpu_id 1 src/evaluation/run_eval.py ...
#   VGL_GPU=1 ./vgl_python.sh src/evaluation/run_eval.py ...
# 说明：
#   - 若同时传入 --gpu_id 和环境变量，则以 --gpu_id 为准
#   - 这里同时做两层约束：
#       1. 通过设置 CUDA/EGL 可见设备约束当前进程；
#       2. 通过给 vglrun 传入具体 DRI device 路径约束 VirtualGL 渲染设备。
#   - 这里设置的是“当前进程可见 GPU”，不是给 vglrun 直接传一个物理卡编号
#   - 对当前项目来说，这种方式最稳：渲染和本地 CUDA 都会跟随同一张卡
#
# 重置 SHLVL 避免深层 bash 嵌套（conda activate + xvfb-run 多层子 shell 累加）触发警告
export SHLVL=1

# 优先级：
#   1. 命令行 --gpu_id
#   2. VGL_GPU
#   3. 现有 CUDA_VISIBLE_DEVICES
# 这样设计是为了兼容三种常见场景：
#   1. 批处理脚本显式传参；
#   2. 手工调试时用环境变量指定；
#   3. 外层调度系统已预先限制 CUDA_VISIBLE_DEVICES。
GPU_ID="${VGL_GPU:-${CUDA_VISIBLE_DEVICES:-}}"
ARGS=()
VGL_DEVICE="egl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu_id)
            if [[ $# -lt 2 ]]; then
                echo "[vgl_python.sh] Error: --gpu_id requires a value" >&2
                exit 1
            fi
            # 显式命令行参数优先级最高，覆盖前面从环境变量读取到的值。
            GPU_ID="$2"
            shift 2
            ;;
        --gpu_id=*)
            # 同时兼容 --gpu_id=1 这种等号写法，便于其他脚本自动拼接命令。
            GPU_ID="${1#*=}"
            shift
            ;;
        *)
            # 其他所有参数都原样转发给 Python 主程序。
            ARGS+=("$1")
            shift
            ;;
    esac
done

resolve_vgl_device() {
    local gpu_id="$1"

    # VirtualGL 官方文档说明：
    #   vglrun -d <d> 中的 <d> 可以是 X display/screen，也可以是 DRI device 路径。
    # 对 EGL 后端来说，如果只写 -d egl，VirtualGL 会自己挑一个可用设备，
    # 这正是当前“明明传了 GPU 1 但实际仍跑到 GPU 0”最可能的原因。
    #
    # 这里改为在本地尽量把 GPU 编号解析成一个明确的 DRI render device：
    #   1. 优先读取 /dev/dri/by-path/*-render（更稳，且与 PCI Bus 顺序关联）；
    #   2. 再退化到 /dev/dri/renderD12x 的常见编号规则；
    #   3. 如果都找不到，则退回到 egl，让旧行为继续可用。
    #
    # 由于脚本前面已经设置 CUDA_DEVICE_ORDER=PCI_BUS_ID，若服务器枚举正常，
    # 则这里按 PCI 顺序拿到的第 N 个 render device 通常就对应 nvidia-smi 中的 GPU N。

    if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    if [[ -d /dev/dri/by-path ]]; then
        local -a render_devices=()
        while IFS= read -r path; do
            render_devices+=("$path")
        done < <(find /dev/dri/by-path -maxdepth 1 -type l -name '*-render' | sort)

        if (( gpu_id < ${#render_devices[@]} )); then
            readlink -f "${render_devices[$gpu_id]}"
            return 0
        fi
    fi

    local fallback_render="/dev/dri/renderD$((128 + gpu_id))"
    if [[ -e "${fallback_render}" ]]; then
        echo "${fallback_render}"
        return 0
    fi

    return 1
}

if [[ -n "${GPU_ID}" ]]; then
    # 保持 GPU 编号与 nvidia-smi / PCI Bus 顺序一致，避免编号歧义。
    # 例如传入 --gpu_id 1 后：
    #   CUDA_VISIBLE_DEVICES=1
    # 这表示当前 Python 进程只“看得见”物理 GPU 1。
    # 对进程内部库而言，这张卡随后通常会被重新映射成逻辑 cuda:0。
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    # 约束 EGL 离屏渲染后端只使用目标 GPU，避免渲染落到别的卡上。
    export EGL_VISIBLE_DEVICES="${GPU_ID}"
    # 保留一份项目内部约定的变量，便于下游脚本或调试日志继续读取。
    export VGL_GPU="${GPU_ID}"

    # 对 VirtualGL/EGL 来说，仅设置 CUDA_VISIBLE_DEVICES 往往还不够，
    # 因为 vglrun 自己仍可能默认选到第一个 DRI 设备。
    # 因此这里再尝试把 GPU 编号转换成明确的 DRI device 路径，并传给 vglrun -d。
    resolved_device="$(resolve_vgl_device "${GPU_ID}")"
    if [[ -n "${resolved_device}" ]]; then
        VGL_DEVICE="${resolved_device}"
        echo "[vgl_python.sh] Using GPU ${GPU_ID} via DRI device ${VGL_DEVICE}"
    else
        echo "[vgl_python.sh] Warning: failed to resolve DRI device for GPU ${GPU_ID}, fallback to 'egl'" >&2
        echo "[vgl_python.sh] If rendering still lands on GPU 0, check /dev/dri/by-path and driver mapping on this host." >&2
    fi
    echo "[vgl_python.sh] Using GPU ${GPU_ID} for EGL/CUDA visibility"
fi

# 启动链路说明：
#   xvfb-run        -> 提供无显示器场景下的虚拟显示环境
#   vglrun -d egl   -> 让 OpenGL/EGL 走 GPU 加速而不是纯软件渲染
#   python          -> 真正执行评测逻辑
# 一旦上面的环境变量已经设置好，整条链路都会继承相同的 GPU 可见性配置。
xvfb-run -a -s "-screen 0 1920x1080x24" /opt/VirtualGL/bin/vglrun -d "${VGL_DEVICE}" /home/jyf/anaconda3/envs/VLMTraffic/bin/python "${ARGS[@]}"
