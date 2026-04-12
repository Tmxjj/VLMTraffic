'''
Author: yufei Ji
Date: 2026-04-11
Description: RLVR 双路可验证奖励函数。

    实现 Simulation-Grounded Dual-Verifiable RLVR 框架中的两类奖励信号：

    r_perc（感知可验证奖励）
    ─────────────────────────
      定义：模型 Scene Understanding 段的车辆预测计数与 SUMO e2 检测器 GT 的匹配度
      计算：r_perc = -clamp(MAE_norm, 0, 1)
              MAE_norm = |pred_total - gt_total| / max(gt_total, 1)
      直接治愈计数幻觉（Counting Hallucination）

    r_env（效率可验证奖励，MaxPressure 代理）
    ─────────────────────────────────────────
      定义：模型 Selection Logic 选择的相位在当前路口的"MaxPressure 压力分"排名
      计算：r_env = pressure[selected_phase] / max(pressure.values() + ε)
              值域 [0, 1]，选最优相位得 1.0，选空相位得 ≈0.0
      注：此为 SUMO rollout ΔATT 的轻量代理，精度略低但无需额外仿真步骤。
          如需切换为真实 rollout 奖励，将 phase_pressure 替换为 phase_att_deltas 即可。

    联合奖励：
      r = α × r_perc + β × r_env，默认 α=0.7, β=0.3

    解析函数：
      parse_predicted_total(response_text) → int
          从模型 Thought 段提取所有预测车辆数之和
      parse_selected_phase(response_text) → int
          从 Action 行提取选择的相位索引

FilePath: /VLMTraffic/src/training/rlvr/rlvr_reward.py
'''

import json
import re
from typing import Any


# ===========================================================================
# 解析函数：从模型输出中提取预测值
# ===========================================================================

def parse_predicted_total(response_text: str) -> int | None:
    """
    从模型的 Scene Understanding 段提取所有预测车辆计数之和。

    支持的格式（来自 prompt_builder.py 的 COT_LANE_TEMPLATES）：
      North Approach: Lane 1(Left-Turn):3, Lane 2(Straight):5, Lane 3(Right-Turn):2
      South Approach: Lane 1(Left-Turn):0, Lane 2(Straight):4, Lane 3(Right-Turn):1

    策略：用正则找到所有 ":<整数>" 模式，求和作为预测总数。

    Args:
        response_text (str): 模型完整输出文本

    Returns:
        int | None: 预测车辆总数，解析失败返回 None
    """
    if not response_text:
        return None

    # 定位 Scene Understanding 段（从 "Scene Understanding:" 到 "Scene Analysis:" 之间）
    su_match = re.search(
        r"Scene Understanding:(.*?)(?:Scene Analysis:|$)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not su_match:
        # 降级：在全文中查找车道计数
        search_text = response_text
    else:
        search_text = su_match.group(1)

    # 匹配 "Lane X(...):<整数>" 中的整数部分
    lane_counts = re.findall(r"Lane\s+\d+[^:]*:\s*(\d+)", search_text, re.IGNORECASE)

    if not lane_counts:
        return None

    try:
        return sum(int(c) for c in lane_counts)
    except (ValueError, TypeError):
        return None


def parse_selected_phase(response_text: str) -> int | None:
    """
    从模型输出中提取最终选择的相位索引。

    支持以下格式：
      Action: 1
      Action: The Selected Phase Index, e.g., 2
      Conclusion: Phase 3
      Action:[2]

    Args:
        response_text (str): 模型完整输出文本

    Returns:
        int | None: 相位索引（0-based），解析失败返回 None
    """
    if not response_text:
        return None

    # 优先匹配 "Action:" 行
    action_match = re.search(
        r"Action\s*[:\-]\s*.*?(\d+)",
        response_text,
        re.IGNORECASE,
    )
    if action_match:
        return int(action_match.group(1))

    # 降级匹配 "Conclusion: Phase X"
    conclusion_match = re.search(
        r"Conclusion\s*[:\-]\s*Phase\s+(\d+)",
        response_text,
        re.IGNORECASE,
    )
    if conclusion_match:
        return int(conclusion_match.group(1))

    return None


# ===========================================================================
# 奖励计算函数
# ===========================================================================

def compute_r_perc(
    pred_total: int | None,
    gt_total: float,
    max_penalty: float = 1.0,
) -> float:
    """
    计算感知可验证奖励 r_perc。

    公式：
        MAE_norm = |pred_total - gt_total| / max(gt_total, 1)
        r_perc   = -clamp(MAE_norm, 0, max_penalty)

    特殊情况：
      - pred_total 解析失败（None）：返回 -max_penalty（最大惩罚）
      - gt_total == 0 且 pred_total == 0：返回 0.0（零流量场景正确）
      - gt_total == 0 且 pred_total > 0：返回 -clamp(pred_total, 0, max_penalty)

    Args:
        pred_total  (int | None): 模型预测的车辆总数
        gt_total    (float)     : SUMO GT 车辆总数
        max_penalty (float)     : 惩罚上限（默认 1.0）

    Returns:
        float: r_perc ∈ [-max_penalty, 0.0]，越接近 0 表示预测越准确
    """
    # 解析失败：施加最大惩罚
    if pred_total is None:
        return -max_penalty

    gt_total = max(float(gt_total), 0.0)

    if gt_total == 0.0:
        # 全零流量场景：预测为 0 则正确，否则按预测量惩罚
        error_norm = float(pred_total) / max(pred_total, 1.0) if pred_total > 0 else 0.0
    else:
        error_norm = abs(float(pred_total) - gt_total) / gt_total

    return -min(error_norm, max_penalty)


def compute_r_env(
    selected_phase: int | None,
    phase_pressure: dict,
    epsilon: float = 1e-6,
) -> float:
    """
    计算效率可验证奖励 r_env（MaxPressure 压力代理）。

    公式：
        r_env = pressure[selected_phase] / (max_pressure + ε)

    选择压力最大相位得分 ≈ 1.0，选择空相位得分 ≈ 0.0。
    若所有相位压力均为 0（零流量），返回 0.5（中性奖励，不鼓励也不惩罚）。

    Args:
        selected_phase (int | None) : 模型选择的相位索引（None 表示解析失败）
        phase_pressure (dict)       : {phase_str: pressure_float}，来自数据集
        epsilon        (float)      : 防除零极小值

    Returns:
        float: r_env ∈ [0.0, 1.0]
    """
    if selected_phase is None:
        return 0.0  # 无法解析相位，给予最低效率奖励

    # 将 phase_pressure 的 key 统一转为 int（JSONL 可能序列化为字符串）
    pressure_int: dict[int, float] = {
        int(k): float(v) for k, v in phase_pressure.items()
    }

    max_pressure = max(pressure_int.values(), default=0.0)

    # 零流量场景：所有相位压力均为 0
    if max_pressure < epsilon:
        return 0.5

    phase_score = pressure_int.get(selected_phase, 0.0)
    return phase_score / (max_pressure + epsilon)


def compute_rlvr_reward(
    response_text: str,
    gt_total: float,
    phase_pressure: dict,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """
    计算单条模型输出的 RLVR 联合奖励。

    r = α × r_perc + β × r_env

    最终奖励值域：
      r ∈ [-α, α + β] = [-0.7, 1.0]（使用默认参数）

    Args:
        response_text (str)   : 模型完整输出文本
        gt_total      (float) : SUMO GT 车辆总数（来自数据集 gt_total_all 字段）
        phase_pressure (dict) : 各相位 MaxPressure 分值（来自数据集 phase_pressure 字段）
        alpha         (float) : r_perc 权重（感知准确性）
        beta          (float) : r_env 权重（决策效率）

    Returns:
        float: 联合奖励值
    """
    # 解析模型输出
    pred_total = parse_predicted_total(response_text)
    selected_phase = parse_selected_phase(response_text)

    # 计算双路奖励
    r_perc = compute_r_perc(pred_total, gt_total)
    r_env = compute_r_env(selected_phase, phase_pressure)

    return alpha * r_perc + beta * r_env


# ===========================================================================
# TRL GRPOTrainer 兼容的批量奖励函数
# ===========================================================================

def batch_rlvr_reward_fn(
    completions: list[str],
    gt_total_all: list[Any],
    phase_pressure: list[Any],
    alpha: float = 0.7,
    beta: float = 0.3,
    **kwargs,
) -> list[float]:
    """
    批量 RLVR 奖励函数，满足 TRL GRPOTrainer 的 reward_funcs 接口规范。

    TRL 调用约定：
        def reward_fn(completions, **kwargs) -> list[float]
    其中 kwargs 由数据集中的同名列自动注入（需在 GRPOConfig 中配置 reward_fn_kwargs_dataset_columns）。

    Args:
        completions   (list[str])       : G 个模型生成响应（已解码为字符串）
        gt_total_all  (list[Any])       : 对应 batch 的 GT 车辆总数（float 或 JSON 字符串）
        phase_pressure (list[Any])      : 对应 batch 的相位压力字典（dict 或 JSON 字符串）
        alpha         (float)           : r_perc 权重，默认 0.7
        beta          (float)           : r_env 权重，默认 0.3
        **kwargs                        : TRL 传入的其他数据集字段（忽略）

    Returns:
        list[float]: 长度为 len(completions) 的奖励列表
    """
    rewards = []
    for completion, gt_raw, pressure_raw in zip(completions, gt_total_all, phase_pressure):
        # --- 反序列化 GT 总数 ---
        try:
            gt_total = float(gt_raw) if not isinstance(gt_raw, str) else float(gt_raw)
        except (ValueError, TypeError):
            gt_total = 0.0

        # --- 反序列化相位压力 ---
        if isinstance(pressure_raw, dict):
            phase_pressure_dict = pressure_raw
        elif isinstance(pressure_raw, str):
            try:
                phase_pressure_dict = json.loads(pressure_raw)
            except (json.JSONDecodeError, TypeError):
                phase_pressure_dict = {}
        else:
            phase_pressure_dict = {}

        # --- 计算联合奖励 ---
        reward = compute_rlvr_reward(
            response_text=completion,
            gt_total=gt_total,
            phase_pressure=phase_pressure_dict,
            alpha=alpha,
            beta=beta,
        )
        rewards.append(reward)

    return rewards


# ===========================================================================
# 单元测试（本地快速验证）
# ===========================================================================

if __name__ == "__main__":
    # 模拟一条正确的模型输出
    _mock_correct = """
Thought: [
Scene Understanding:
- Lane Analysis (Mandatory):
North Approach: Lane 1(Left-Turn):3, Lane 2(Straight):5, Lane 3(Right-Turn):2
South Approach: Lane 1(Left-Turn):1, Lane 2(Straight):6, Lane 3(Right-Turn):2
East Approach: Lane 1(Left-Turn):2, Lane 2(Straight):4, Lane 3(Right-Turn):1
West Approach: Lane 1(Left-Turn):0, Lane 2(Straight):2, Lane 3(Right-Turn):1
]
Action: 1
"""
    # 模拟一条有幻觉的输出（预测值远高于 GT）
    _mock_hallucination = """
Thought: [
Scene Understanding:
North Approach: Lane 1(Left-Turn):15, Lane 2(Straight):20, Lane 3(Right-Turn):18
South Approach: Lane 1(Left-Turn):12, Lane 2(Straight):22, Lane 3(Right-Turn):14
East Approach: Lane 1(Left-Turn):11, Lane 2(Straight):19, Lane 3(Right-Turn):16
West Approach: Lane 1(Left-Turn):13, Lane 2(Straight):21, Lane 3(Right-Turn):17
]
Action: 3
"""
    _mock_gt_total = 29.0  # 实际 GT 总数
    _mock_pressure = {"0": 11.0, "1": 14.0, "2": 5.0, "3": 3.0}  # Phase 1 最优

    r_correct = compute_rlvr_reward(
        _mock_correct, _mock_gt_total, _mock_pressure
    )
    r_hallucination = compute_rlvr_reward(
        _mock_hallucination, _mock_gt_total, _mock_pressure
    )

    print("=== RLVR 奖励单元测试 ===")
    print(f"正确输出（预测≈GT，选最优Phase 1）: r = {r_correct:.4f}")
    print(f"幻觉输出（严重高估计数，选次优Phase 3）: r = {r_hallucination:.4f}")
    assert r_correct > r_hallucination, "正确输出奖励应高于幻觉输出"
    print("✅ 测试通过：奖励排序符合预期（正确 > 幻觉）")
