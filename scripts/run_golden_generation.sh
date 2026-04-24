#!/bin/bash
###
 # @Author: yufei Ji
 # @Date: 2026-04-24
 # @Description: Golden 数据集生成脚本（单场景 / 单路由文件）
 #
 #   针对指定场景 + 路由文件生成 golden SFT 数据。
 #
 #   核心逻辑：
 #     - 动作空间：phase × duration 全候选（最多 4×6=24 个）
 #     - Warmup：前 warmup_seconds（默认 300s）仅用 FixedTime 推进，不跑 VLM、不保存数据
 #     - Rollout：1步候选动作 + rollout_follow_steps 步 FixedTime（默认2步）
 #     - 仿真推进：Warmup 阶段后用 VLM student action 推进
 #
 #   必填参数：
 #     --scenario    场景键名（e.g., JiNan, Hangzhou）
 #     --route_file  路由文件名（位于 data/raw/{ScenarioName}/env/ 下）
 #
 #   可选参数（不传时读 configs/model_config.py 中 MODEL_CONFIG 的值）：
 #     --port        vLLM 服务端口，拼为 http://localhost:{port}/v1/chat/completions
 #     --model_name  覆盖 model_config.py 中 requests.model_name
 #
 #   使用示例：
 #     # JiNan 真实流量（前300s warmup）：
    #  bash scripts/run_golden_generation.sh \
    #    --scenario JiNan \
    #    --route_file anon_3_4_jinan_real_2000.rou.xml \
 #       --port 8000 --model_name qwen3-vl-8b
 #
 #     # 自定义 warmup 时长与 rollout 步数：
 #     bash scripts/run_golden_generation.sh \
 #       --scenario Hangzhou \
 #       --route_file anon_4_4_hangzhou_real_2000.rou.xml \
 #       --port 8000 --model_name qwen3-vl-8b \
 #       --warmup_seconds 600 \
 #       --max_sumo_seconds 3600 \
 #       --rollout_follow_steps 3
 #
 #   输出路径：
 #     data/sft_dataset/{ScenarioName}/{route_stem}/01_dataset_raw.jsonl
 #     data/sft_dataset/{ScenarioName}/{route_stem}/{jid}/{sumo_step}/  （图像）
 #
 #   日志路径：
 #     log/golden_dataset/{ScenarioKey}/{route_stem}/
###

set -euo pipefail

# ── 项目根目录（脚本所在目录的上一层）────────────────────────────────
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# ── PYTHONPATH：项目根 + src/ ─────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 切换到项目根，保证相对路径（data/、log/、vgl_python.sh）正确解析
cd "${PROJECT_ROOT}"

# ── 默认参数 ──────────────────────────────────────────────────────────
PORT=""           # 不设默认：不传时不拼 --api_url，读 MODEL_CONFIG requests.url
MODEL_NAME=""     # 不设默认：不传时不拼 --model_name，读 MODEL_CONFIG requests.model_name
MAX_SUMO_SECONDS=3600
WARMUP_SECONDS=300
ROLLOUT_FOLLOW_STEPS=2
LOG_DIR="./log/golden_dataset"
SCENARIO_KEY=""
ROUTE_FILE=""

# ── 参数解析 ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)               PORT="$2";                shift 2 ;;
        --model_name)         MODEL_NAME="$2";           shift 2 ;;
        --max_sumo_seconds)   MAX_SUMO_SECONDS="$2";     shift 2 ;;
        --warmup_seconds)     WARMUP_SECONDS="$2";       shift 2 ;;
        --rollout_follow_steps) ROLLOUT_FOLLOW_STEPS="$2"; shift 2 ;;
        --log_dir)            LOG_DIR="$2";              shift 2 ;;
        --scenario)           SCENARIO_KEY="$2";         shift 2 ;;
        --route_file)         ROUTE_FILE="$2";           shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

# ── 必填参数校验 ──────────────────────────────────────────────────────
if [[ -z "${SCENARIO_KEY}" ]]; then
    echo "[ERROR] --scenario 为必填参数，例如: --scenario JiNan"
    exit 1
fi
if [[ -z "${ROUTE_FILE}" ]]; then
    echo "[ERROR] --route_file 为必填参数，例如: --route_file anon_3_4_jinan_real_2000.rou.xml"
    exit 1
fi

# ── 构建可选参数（不传时不拼，让 Python 读 MODEL_CONFIG）────────────
EXTRA_ARGS=()
if [[ -n "${PORT}" ]]; then
    EXTRA_ARGS+=(--api_url "http://localhost:${PORT}/v1/chat/completions")
fi
if [[ -n "${MODEL_NAME}" ]]; then
    EXTRA_ARGS+=(--model_name "${MODEL_NAME}")
fi

echo "════════════════════════════════════════════════════════"
echo "[GOLDEN] 生成配置"
echo "  Project Root         : ${PROJECT_ROOT}"
echo "  Scenario             : ${SCENARIO_KEY}"
echo "  Route File           : ${ROUTE_FILE}"
echo "  API URL              : ${PORT:+http://localhost:${PORT}/v1/chat/completions}${PORT:-（读 MODEL_CONFIG）}"
echo "  Model                : ${MODEL_NAME:-（读 MODEL_CONFIG）}"
echo "  Warmup Seconds       : ${WARMUP_SECONDS}s"
echo "  Max SUMO Seconds     : ${MAX_SUMO_SECONDS}s"
echo "  Rollout Follow Steps : ${ROLLOUT_FOLLOW_STEPS}"
echo "  Log Dir              : ${LOG_DIR}"
echo "════════════════════════════════════════════════════════"

# ── 执行生成（需要 3D 渲染，走 VirtualGL）────────────────────────────
./vgl_python.sh src/dataset/golden_gener/1_golden_generation.py \
    --scenario             "${SCENARIO_KEY}" \
    --route_file           "${ROUTE_FILE}" \
    --max_sumo_seconds     "${MAX_SUMO_SECONDS}" \
    --warmup_seconds       "${WARMUP_SECONDS}" \
    --rollout_follow_steps "${ROLLOUT_FOLLOW_STEPS}" \
    --log_dir              "${LOG_DIR}" \
    "${EXTRA_ARGS[@]}"

STATUS=$?
echo ""
echo "════════════════════════════════════════════════════════"
if [[ ${STATUS} -eq 0 ]]; then
    echo "[GOLDEN] ✓ 完成: ${SCENARIO_KEY} / ${ROUTE_FILE}"
    echo "  数据目录: data/sft_dataset/"
else
    echo "[GOLDEN] ✗ 失败 (exit=${STATUS}): ${SCENARIO_KEY} / ${ROUTE_FILE}"
fi
echo "════════════════════════════════════════════════════════"

exit ${STATUS}
