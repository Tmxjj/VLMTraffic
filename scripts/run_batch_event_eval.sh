#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-05-02
 # @Description: 常规 + 泛化 + 事件迁移场景批量评测脚本（基线 + VLM）
 #
 #               路由文件类型：
 #                 normal          — 常规车流（*.rou.xml）
 #                                   覆盖：
 #                                     1) 主实验：JiNan × 3 + Hangzhou × 2
 #                                     2) 拓扑迁移：Songdo / Massy / YMT
 #                                     3) 规模迁移：NewYork（196 路口）
 #                 emergy_bus      — 紧急车辆 + 公交/校车（*_emergy_bus.rou.xml）
 #                 accident_debris — 交通事故 + 路面碎片（*_accident_debris.rou.xml）
 #
 #               评测数据集：
 #                 normal：JiNan × 3 + Hangzhou × 2 + 拓扑迁移 3 场景 + NewYork × 1 = 9 条任务
 #                 event ：JiNan × 1 + Hangzhou × 1 + 拓扑迁移 3 场景 = 5 条任务/事件类型
 #                 合计：9 条 normal + 5 条 emergy_bus + 5 条 accident_debris = 19 条评测任务
 #
 #               口径说明：
 #                 - NewYork 在本脚本中作为“规模迁移验证”场景，只加入 normal 评测。
 #                 - collect_metrics.py 当前的 gen_scale 只统计
 #                   NewYork/anon_28_7_newyork_real_double 常规路由结果。
 #                 - 尽管 data/raw/NewYork/env/ 下已存在事件路由文件，本脚本默认不将其纳入
 #                   event 分支，以避免和当前论文表格/结果收集口径不一致。
 #
 #               输出路径（与 src/evaluation/run_eval.py 一致）：
 #                 data/eval/{dataset}/{route_stem}/{method}/
 #               例：data/eval/JiNan/anon_3_4_jinan_real_emergy_bus/fixed_time/
 #
 #               max_sumo_seconds 自动从路由文件最大 depart 时间动态计算：
 #                 max_sumo_seconds = max_depart + BUFFER_S，上下限 [MIN, CAP]
 #
 #               使用说明：
 #                 0) 环境准备
 #                    - 已激活 VLMTraffic 虚拟环境
 #                    - 若跑 VLM，需先启动兼容 OpenAI Chat Completions 的服务
 #                    - 若跑 VLM，建议提前确认 vgl / SUMO / 渲染依赖正常
 #
 #                 # 仅跑基线（本地，无需 GPU）
 #                 # 会顺序执行：
 #                 #   normal(主实验 + 拓扑迁移 + 规模迁移)
 #                 #   emergy_bus(5个事件基础场景)
 #                 #   accident_debris(5个事件基础场景)
 #                 bash scripts/run_batch_event_eval.sh --baseline-only
 #
 #                 # 仅跑 normal（包含主实验 + gen_topology + gen_scale）
 #                 bash scripts/run_batch_event_eval.sh --baseline-only --event normal
 #
 #                 # 仅跑事件泛化：紧急车辆 + 公交/校车
 #                 bash scripts/run_batch_event_eval.sh --baseline-only --event emergy_bus
 #
 #                 # 仅跑事件泛化：事故 + 占道
 #                 bash scripts/run_batch_event_eval.sh --baseline-only --event accident_debris
 #
 #                 # 仅跑 VLM（远程服务器，需提前启动 vLLM）
 #                 # 默认也会覆盖 normal + 两类 event
 #                 bash scripts/run_batch_event_eval.sh --port 8000 --model_name qwen3-vl-8b-sft-rlvr
 #
 #                 # 指定本机渲染使用 GPU 1
 #                 # 说明：
 #                 #   - 该参数只影响当前评测进程（EGL/VirtualGL 渲染 + 本地 CUDA 模型）
 #                 #   - 若 VLM 是远程 vLLM/OpenAI 兼容服务，服务端实际跑在哪张卡不受此参数影响
 #                 bash scripts/run_batch_event_eval.sh --port 8000 --model_name qwen3-vl-8b-sft-rlvr --gpu_id 1
 #
 #                 # 同时跑基线 + VLM（推荐用于生成完整对比结果）
 #                 bash scripts/run_batch_event_eval.sh --port 8000 --model_name qwen3-vl-8b-sft-rlvr --with-baseline
 #
 #                 # 控制生成温度和最大输出长度
 #                 bash scripts/run_batch_event_eval.sh \
 #                     --port 8000 \
 #                     --model_name qwen3-vl-8b-sft-rlvr \
 #                     --temperature 0.2 \
 #                     --max_new_tokens 1024
 #
 #               评测完成后收集指标：
 #                 # 主实验（JiNan/Hangzhou 常规场景）
 #                 python src/evaluation/collect_metrics.py --type main
 #                 # 拓扑迁移（Songdo/Massy/YMT）
 #                 python src/evaluation/collect_metrics.py --type gen_topology
 #                 # 规模迁移（NewYork）
 #                 python src/evaluation/collect_metrics.py --type gen_scale
 #                 # 事件迁移（5 个基础场景）
 #                 python src/evaluation/collect_metrics.py --type gen_event_emergy_bus
 #                 python src/evaluation/collect_metrics.py --type gen_event_accident_debris
 #
 # @FilePath: /VLMTraffic/scripts/run_batch_event_eval.sh
###

# ─── 默认配置 ──────────────────────────────────────────────────────────────────
# BUFFER_S:
#   在路由文件最大 depart/begin 时间基础上额外增加的仿真缓冲秒数，
#   避免最后一批车辆尚未完成通行就提前结束评测。
# MIN_SUMO_SECONDS:
#   防止短路由文件导致评测时长过短。
# MAX_SUMO_SECONDS_CAP:
#   防止异常路由文件导致评测时长失控。
BUFFER_S=180
MIN_SUMO_SECONDS=300
MAX_SUMO_SECONDS_CAP=3600
LOG_DIR="./log/eval_results"

API_PORT=""
MODEL_NAME=""
TEMPERATURE=""
MAX_NEW_TOKENS=""
# GPU_ID:
#   用于指定“当前这次批量评测任务”绑定哪一张本机 GPU。
#   它会继续透传到：
#     1. vgl_python.sh：在 Python 进程启动前设置 CUDA/EGL 可见设备；
#     2. src/evaluation/run_eval.py：在环境初始化和模型加载前再次显式设置。
#   这样可以同时约束两类资源：
#     1. 离屏 3D 渲染（xvfb + vglrun + EGL）；
#     2. 评测过程中可能加载的本地 CUDA 模型。
#   注意：
#     1. 该参数主要服务于 VLM 分支；
#     2. baseline-only 分支内部已关闭 3D 渲染，通常不会实际消耗 GPU；
#     3. 若 VLM 走远端 API，请求发往哪张远端卡不受这里控制。
GPU_ID=""
BASELINE_ONLY=false
WITH_BASELINE=false
EVENT_FILTER=""   # 可选：normal / emergy_bus / accident_debris（不指定则三类都跑）

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)           API_PORT="$2";       shift ;;
        --model_name)     MODEL_NAME="$2";     shift ;;
        --temperature)    TEMPERATURE="$2";    shift ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift ;;
        # 这里解析用户传入的 GPU 编号，例如：
        #   --gpu_id 0
        #   --gpu_id 1
        # 本脚本不会直接在这里 export CUDA_VISIBLE_DEVICES，
        # 而是把参数继续向下透传给真正负责启动渲染/评测进程的脚本。
        --gpu_id)         GPU_ID="$2";         shift ;;
        --baseline-only)  BASELINE_ONLY=true   ;;
        --with-baseline)  WITH_BASELINE=true   ;;
        --event)          EVENT_FILTER="$2";   shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"
mkdir -p "$LOG_DIR"

# ─── 路由文件条目 ──────────────────────────────────────────────────────────────
# 格式：DATASET|ENV_DIR|BASE_STEM
# BASE_STEM 是不含 .rou.xml 的基础文件名；
# normal 使用 ${BASE_STEM}.rou.xml；
# 事件场景会自动在其后追加 _emergy_bus.rou.xml 或 _accident_debris.rou.xml。
#
# 设计约束：
#   1. NORMAL_ENTRIES 覆盖主实验 + gen_topology + gen_scale；
#   2. EVENT_ENTRIES 覆盖 gen_event 与事件相关消融当前使用的 5 个基础场景；
#   3. NewYork 虽然有事件路由文件，但当前结果收集口径未纳入，因此只放在 NORMAL_ENTRIES。
declare -a NORMAL_ENTRIES
declare -a EVENT_ENTRIES

# JiNan × 3
NORMAL_ENTRIES+=("JiNan|data/raw/JiNan/env|anon_3_4_jinan_real")
NORMAL_ENTRIES+=("JiNan|data/raw/JiNan/env|anon_3_4_jinan_real_2000")
NORMAL_ENTRIES+=("JiNan|data/raw/JiNan/env|anon_3_4_jinan_real_2500")
EVENT_ENTRIES+=("JiNan|data/raw/JiNan/env|anon_3_4_jinan_real")

# Hangzhou × 2
NORMAL_ENTRIES+=("Hangzhou|data/raw/Hangzhou/env|anon_4_4_hangzhou_real")
NORMAL_ENTRIES+=("Hangzhou|data/raw/Hangzhou/env|anon_4_4_hangzhou_real_5816")
EVENT_ENTRIES+=("Hangzhou|data/raw/Hangzhou/env|anon_4_4_hangzhou_real_5816")

# SouthKorea_Songdo × 1
NORMAL_ENTRIES+=("SouthKorea_Songdo|data/raw/SouthKorea_Songdo/env|songdo")
EVENT_ENTRIES+=("SouthKorea_Songdo|data/raw/SouthKorea_Songdo/env|songdo")

# France_Massy × 1
NORMAL_ENTRIES+=("France_Massy|data/raw/France_Massy/env|massy")
EVENT_ENTRIES+=("France_Massy|data/raw/France_Massy/env|massy")

# Hongkong_YMT × 1
NORMAL_ENTRIES+=("Hongkong_YMT|data/raw/Hongkong_YMT/env|YMT")
EVENT_ENTRIES+=("Hongkong_YMT|data/raw/Hongkong_YMT/env|YMT")

# NewYork × 1（规模迁移，只加入 normal 分支）
NORMAL_ENTRIES+=("NewYork|data/raw/NewYork/env|anon_28_7_newyork_real_double")

# 场景类型列表
SCENE_TYPES=("normal" "emergy_bus" "accident_debris")

# ─── 工具函数 ──────────────────────────────────────────────────────────────────

get_max_sumo_seconds() {
    local rou_path="$1"
    # 从路由文件中提取最大出发时刻，动态决定评测总时长。
    # 同时兼容 vehicle / trip / flow 三类定义方式。
    python3 -c "
import xml.etree.ElementTree as ET, sys
try:
    root = ET.parse('${rou_path}').getroot()
    vals = []
    for tag in ('vehicle', 'trip', 'flow'):
        for v in root.findall(tag):
            d = v.get('depart') or v.get('begin')
            try: vals.append(float(d))
            except: pass
    if not vals:
        print(${MIN_SUMO_SECONDS}); sys.exit()
    seconds = int(max(vals) + ${BUFFER_S})
    seconds = max(seconds, ${MIN_SUMO_SECONDS})
    seconds = min(seconds, ${MAX_SUMO_SECONDS_CAP})
    print(seconds)
except Exception:
    print(3600)
" 2>/dev/null
}

cleanup_sumo() { sleep 5; }

get_route_file() {
    local BASE_STEM="$1"
    local SCENE_TYPE="$2"
    # normal 直接使用基础路由；
    # 事件场景在基础 stem 后追加统一后缀。
    if [ "$SCENE_TYPE" = "normal" ]; then
        echo "${BASE_STEM}.rou.xml"
    else
        echo "${BASE_STEM}_${SCENE_TYPE}.rou.xml"
    fi
}

get_entries_for_scene_type() {
    local SCENE_TYPE="$1"
    # normal 返回主实验 + 拓扑迁移 + 规模迁移场景；
    # 事件返回当前论文口径下的 5 个事件基础场景。
    if [ "$SCENE_TYPE" = "normal" ]; then
        printf '%s\n' "${NORMAL_ENTRIES[@]}"
    else
        printf '%s\n' "${EVENT_ENTRIES[@]}"
    fi
}

run_baseline() {
    local DATASET="$1"
    local ROUTE_FILE="$2"    # 仅文件名（不含路径）
    local SCENE_TYPE="$3"
    local METHOD_FLAG="$4"
    local METHOD_NAME="$5"
    local MAX_SUMO_S="$6"

    cleanup_sumo
    echo "  [${METHOD_NAME}] ${DATASET}/${ROUTE_FILE}  sumo_s=${MAX_SUMO_S}"
    python src/evaluation/run_eval.py \
        --scenario         "$DATASET" \
        --log_dir          "$LOG_DIR" \
        --route_file       "$ROUTE_FILE" \
        --scene_type       "$SCENE_TYPE" \
        --max_sumo_seconds "$MAX_SUMO_S" \
        "$METHOD_FLAG"
    [ $? -ne 0 ] && echo "  [WARNING] ${METHOD_NAME} failed: ${DATASET}/${ROUTE_FILE}"
}

run_vlm() {
    local DATASET="$1"
    local ROUTE_FILE="$2"
    local SCENE_TYPE="$3"
    local MAX_SUMO_S="$4"
    local EXTRA="$5"

    cleanup_sumo
    echo "  [VLM] ${DATASET}/${ROUTE_FILE}  sumo_s=${MAX_SUMO_S}"
    # 统一通过 vgl_python.sh 启动，确保：
    #   1. EGL/VirtualGL 渲染链路已启用；
    #   2. 若传入 --gpu_id，则当前进程只看到目标 GPU。
    # GPU 参数传递链路如下：
    #   run_batch_event_eval.sh
    #     -> vgl_python.sh
    #     -> run_eval.py
    # 最终效果是：真正执行评测的 Python 进程以及它创建的渲染环境，
    # 都只会“看到”指定编号的那张物理卡。
    ./vgl_python.sh src/evaluation/run_eval.py \
        --scenario         "$DATASET" \
        --log_dir          "$LOG_DIR" \
        --route_file       "$ROUTE_FILE" \
        --scene_type       "$SCENE_TYPE" \
        --max_sumo_seconds "$MAX_SUMO_S" \
        $EXTRA
    [ $? -ne 0 ] && echo "  [WARNING] VLM failed: ${DATASET}/${ROUTE_FILE}"
}

# ─── 构建 VLM 额外参数 ─────────────────────────────────────────────────────────
EXTRA_VLM_ARGS=""
if [ -n "$API_PORT" ] && [ -n "$MODEL_NAME" ]; then
    EXTRA_VLM_ARGS="--api_url http://localhost:${API_PORT}/v1/chat/completions --model_name ${MODEL_NAME}"
fi
[ -n "$TEMPERATURE"    ] && EXTRA_VLM_ARGS="${EXTRA_VLM_ARGS} --temperature ${TEMPERATURE}"
[ -n "$MAX_NEW_TOKENS" ] && EXTRA_VLM_ARGS="${EXTRA_VLM_ARGS} --max_new_tokens ${MAX_NEW_TOKENS}"
# 如果指定了 GPU_ID，则把它拼进下游 VLM 参数。
# 这里选择“参数透传”而不是“当前 shell 全局 export”的原因是：
#   1. 让 GPU 约束只作用在 VLM 评测子进程，避免污染整个批处理脚本环境；
#   2. 让 vgl_python.sh、run_eval.py 都能明确感知这是一次显式的 GPU 绑定请求；
#   3. 后续排查时，命令行里能直接看到 --gpu_id，行为更直观。
[ -n "$GPU_ID"         ] && EXTRA_VLM_ARGS="${EXTRA_VLM_ARGS} --gpu_id ${GPU_ID}"

# ─── 打印头部信息 ──────────────────────────────────────────────────────────────
echo "================================================================"
echo "  E2ELight 常规 + 泛化 + 事件迁移场景批量评测"
echo "  LOG_DIR: $LOG_DIR"
if [ "$BASELINE_ONLY" = true ]; then
    echo "  模式   : 仅基线 (FixedTime + MaxPressure)"
elif [ -n "$MODEL_NAME" ]; then
    echo "  模式   : VLM [${MODEL_NAME}]$([ "$WITH_BASELINE" = true ] && echo " + 基线")"
fi
[ -n "$GPU_ID"         ] && echo "  GPU_ID: ${GPU_ID}"
[ -n "$TEMPERATURE"    ] && echo "  temperature:    ${TEMPERATURE}"
[ -n "$MAX_NEW_TOKENS" ] && echo "  max_new_tokens: ${MAX_NEW_TOKENS}"
[ -n "$EVENT_FILTER"   ] && echo "  场景过滤: ${EVENT_FILTER}" || echo "  场景类型: normal + emergy_bus + accident_debris"
echo "  normal覆盖: main + gen_topology + gen_scale"
echo "  event覆盖 : gen_event (5个基础场景)"
echo "================================================================"

# ─── 核心评测循环 ──────────────────────────────────────────────────────────────

run_all_baselines() {
    echo ""
    echo "=== [基线] FixedTime + MaxPressure ==="

    for EVENT in "${SCENE_TYPES[@]}"; do
        [ -n "$EVENT_FILTER" ] && [ "$EVENT" != "$EVENT_FILTER" ] && continue

        echo ""
        if [ "$EVENT" = "normal" ]; then
            echo "── Scene: ${EVENT} (main + gen_topology + gen_scale) ───────────"
        else
            echo "── Scene: ${EVENT} (gen_event 5-base-scenes) ───────────────────"
        fi
        while IFS= read -r entry; do
            IFS='|' read -r DATASET ENV_DIR BASE_STEM <<< "$entry"
            ROUTE_FILE="$(get_route_file "$BASE_STEM" "$EVENT")"
            FULL_PATH="${ENV_DIR}/${ROUTE_FILE}"

            if [ ! -f "$FULL_PATH" ]; then
                echo "  [SKIP] 路由文件不存在: ${FULL_PATH}"
                continue
            fi

            MAX_S=$(get_max_sumo_seconds "$FULL_PATH")
            run_baseline "$DATASET" "$ROUTE_FILE" "$EVENT" "--fixed_time"   "FixedTime"   "$MAX_S"
            run_baseline "$DATASET" "$ROUTE_FILE" "$EVENT" "--max_pressure" "MaxPressure" "$MAX_S"
        done < <(get_entries_for_scene_type "$EVENT")
    done
}

run_all_vlm() {
    echo ""
    echo "=== [VLM] ${MODEL_NAME} ==="

    for EVENT in "${SCENE_TYPES[@]}"; do
        [ -n "$EVENT_FILTER" ] && [ "$EVENT" != "$EVENT_FILTER" ] && continue

        echo ""
        if [ "$EVENT" = "normal" ]; then
            echo "── Scene: ${EVENT} (main + gen_topology + gen_scale) ───────────"
        else
            echo "── Scene: ${EVENT} (gen_event 5-base-scenes) ───────────────────"
        fi
        while IFS= read -r entry; do
            IFS='|' read -r DATASET ENV_DIR BASE_STEM <<< "$entry"
            ROUTE_FILE="$(get_route_file "$BASE_STEM" "$EVENT")"
            FULL_PATH="${ENV_DIR}/${ROUTE_FILE}"

            if [ ! -f "$FULL_PATH" ]; then
                echo "  [SKIP] 路由文件不存在: ${FULL_PATH}"
                continue
            fi

            MAX_S=$(get_max_sumo_seconds "$FULL_PATH")
            run_vlm "$DATASET" "$ROUTE_FILE" "$EVENT" "$MAX_S" "$EXTRA_VLM_ARGS"
        done < <(get_entries_for_scene_type "$EVENT")
    done
}

# ─── 执行 ──────────────────────────────────────────────────────────────────────
if [ "$BASELINE_ONLY" = true ]; then
    run_all_baselines
elif [ -n "$MODEL_NAME" ]; then
    [ "$WITH_BASELINE" = true ] && run_all_baselines
    run_all_vlm
else
    echo "请指定运行模式："
    echo "  --baseline-only                                        # 仅基线，覆盖 normal + 两类 event"
    echo "  --port 8000 --model_name <name>                        # 仅 VLM，覆盖 normal + 两类 event"
    echo "  --port 8000 --model_name <name> --with-baseline        # 先基线后VLM"
    echo "  --baseline-only --event normal                         # 仅跑常规场景：main + gen_topology + gen_scale"
    echo "  --baseline-only --event emergy_bus                     # 仅跑事件迁移：紧急车辆 + 公交"
    echo "  --baseline-only --event accident_debris                # 仅跑事件迁移：事故 + 占道"
    exit 1
fi

echo ""
echo "================================================================"
echo "  常规 + 泛化 + 事件迁移评测完成。"
echo "  结果目录: data/eval/{dataset}/{route_stem}/{method}/"
echo "  收集指标:"
echo "    python src/evaluation/collect_metrics.py --type main"
echo "    python src/evaluation/collect_metrics.py --type gen_topology"
echo "    python src/evaluation/collect_metrics.py --type gen_scale"
echo "    python src/evaluation/collect_metrics.py --type gen_event_emergy_bus"
echo "    python src/evaluation/collect_metrics.py --type gen_event_accident_debris"
echo "    python src/evaluation/collect_metrics.py --type abl_train_event_emergy_bus"
echo "    python src/evaluation/collect_metrics.py --type abl_train_event_accident_debris"
echo "    python src/evaluation/collect_metrics.py --type abl_cot"
echo "    python src/evaluation/collect_metrics.py --type abl_bulletin_emergy"
echo "    python src/evaluation/collect_metrics.py --type abl_bulletin_accident"
echo "================================================================"
