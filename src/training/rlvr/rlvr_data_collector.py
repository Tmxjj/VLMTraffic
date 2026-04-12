'''
Author: yufei Ji
Date: 2026-04-11
Description: RLVR 训练数据收集脚本。

    在 SUMO 仿真环境中运行，为每一个决策步骤收集以下四类信息，
    并保存为 JSONL 格式供离线 GRPO 训练使用：

    1. BEV 图像路径（相对于项目根目录）
    2. GT 车辆计数：每个 movement 的真实排队车辆数（来自 SUMO e2 检测器 jam_length_vehicle）
    3. 相位压力分数：每个相位的 MaxPressure 分值（GT队列加权求和，用作 r_env 代理）
    4. 元信息：场景、步骤号、路口 ID、当前相位等

    数据样本格式（每行 JSON）：
    {
        "sample_id"           : "JiNan_anon_3_4_jinan_real_step42_intersection_1_1",
        "scenario_key"        : "JiNan",
        "route_file"          : "anon_3_4_jinan_real.rou.xml",
        "junction_id"         : "intersection_1_1",
        "current_phase_id"    : 2,
        "image_path"          : "data/rlvr_dataset/JiNan/.../bev.png",
        "gt_vehicle_counts"   : {"North_total": 8, "South_total": 3, ...},
        "phase_pressure"      : {"0": 11.0, "1": 4.0, "2": 8.5, "3": 2.0},
        "optimal_phase"       : 0
    }

运行方式（仿真机器，需激活 VLMTraffic 环境）：
    python src/training/rlvr/rlvr_data_collector.py \\
        --scenario JiNan \\
        --route_file anon_3_4_jinan_real.rou.xml \\
        --max_steps 120 \\
        --output_dir data/rlvr_dataset

FilePath: /VLMTraffic/src/training/rlvr/rlvr_data_collector.py
'''

import argparse
import copy
import json
import os
import re
import sys
import time

import cv2
from loguru import logger

# 将项目根目录加入 PYTHONPATH
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from utils.make_tsc_env import make_env
from utils.tools import create_folder, save_to_json
from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from configs.prompt_builder import PromptBuilder
from scripts.add_lane_watermarks import add_lane_watermarks
from utils.tools import convert_rgb_to_bgr


# ===========================================================================
# 工具函数
# ===========================================================================

def _build_gt_counts_per_direction(tls_state: dict) -> dict:
    """
    从 render_json 的 TLS 状态字段中，将 jam_length_vehicle（e2检测器GT值）
    按方向（North/South/East/West）聚合为总计数。

    SUMO movement_id 格式形如 "edgeId--direction"，其中 direction 为
    字母缩写（l=左转, s=直行, r=右转）。边 ID 本身不携带方向信息，
    因此这里通过 phase2movements 的结构来区分方向。

    Args:
        tls_state (dict): render_json['tls'][jid]，包含：
            - movement_ids       : List[str]，各 movement 的 ID
            - jam_length_vehicle : List[float]，与 movement_ids 对齐的排队车辆数
            - phase2movements    : Dict[int, List[str]]，每个相位包含的 movements

    Returns:
        dict: {
            "gt_per_movement": {movement_id: count, ...},  # movement 级别GT
            "phase_pressure"  : {phase_idx: sum_count, ...},  # 各相位压力分值
            "optimal_phase"   : int  # 压力最大的相位索引
        }
    """
    movement_ids = tls_state.get("movement_ids", [])
    jam_veh = tls_state.get("jam_length_vehicle", [])
    phase2movements = tls_state.get("phase2movements", {})

    # 构建 movement → GT计数 映射
    movement_queue: dict[str, float] = {
        mid: float(cnt)
        for mid, cnt in zip(movement_ids, jam_veh)
    }

    # 计算每个相位的压力（所属 movements 的 GT 计数之和）
    phase_pressure: dict[str, float] = {}
    for phase_idx, movements in phase2movements.items():
        phase_pressure[str(phase_idx)] = sum(
            movement_queue.get(mid, 0.0) for mid in movements
        )

    # 确定最优相位（压力最大相位，若全为 0 则默认 phase 0）
    if any(v > 0 for v in phase_pressure.values()):
        optimal_phase = int(
            max(phase_pressure, key=lambda k: phase_pressure[k])
        )
    else:
        optimal_phase = 0

    return {
        "gt_per_movement": movement_queue,
        "phase_pressure": phase_pressure,
        "optimal_phase": optimal_phase,
    }


def _compute_direction_totals(movement_queue: dict[str, float]) -> dict[str, float]:
    """
    将 movement 级别的 GT 计数按方向汇总。

    由于 SUMO movement_id 格式（如 "29257863#2--l"）不直接携带 N/S/E/W 信息，
    这里退而其次统计以下聚合量，可与模型输出的 Scene Understanding 进行方向对比：
      - total_all : 路口全部 movement 的 GT 总计数
      - total_per_movement: 保留原始每 movement 的计数（供精细对比使用）

    Args:
        movement_queue (dict): {movement_id: count}

    Returns:
        dict: {"total_all": float, "per_movement": dict}
    """
    total_all = sum(movement_queue.values())
    return {
        "total_all": total_all,
        "per_movement": movement_queue,
    }


# ===========================================================================
# 数据收集主类
# ===========================================================================

class RLVRDataCollector:
    """
    RLVR 训练数据收集器。

    在仿真环境中以"纯观察"方式运行（使用 MaxPressure 做决策以保证合理轨迹），
    同时在每一步收集 GRPO 训练所需的多模态数据：
      - BEV 图像
      - SUMO GT 车辆计数（e2检测器 jam_length_vehicle）
      - 各相位 MaxPressure 分值（用作 r_env 代理）

    数据最终保存为 JSONL 格式，上传至 GPU 训练服务器后即可直接用于 GRPO 训练。
    """

    def __init__(
        self,
        scenario_key: str = "JiNan",
        route_file: str = None,
        output_dir: str = "data/rlvr_dataset",
        max_steps: int = 120,
    ):
        """
        Args:
            scenario_key (str): 场景键名（如 "JiNan"、"Hangzhou"）
            route_file   (str): SUMO 路由文件名（如 "anon_3_4_jinan_real.rou.xml"）
            output_dir   (str): 数据集输出目录
            max_steps    (int): 最大决策步数
        """
        self.scenario_key = scenario_key
        self.route_file = route_file
        self.max_steps = max_steps

        # 场景配置
        self.scenario_config = SCENARIO_CONFIGS.get(scenario_key)
        if not self.scenario_config:
            raise ValueError(f"场景 '{scenario_key}' 未在 SCENARIO_CONFIGS 中配置")

        self.scenario_name = self.scenario_config["SCENARIO_NAME"]
        self.junction_name = self.scenario_config["JUNCTION_NAME"]
        self.is_multi_agent = isinstance(self.junction_name, list)

        # 路由文件名（去掉 .xml 后缀，保留 .rou）
        route_stem = (
            os.path.splitext(os.path.basename(route_file))[0]
            if route_file else "default"
        )

        # 输出路径：data/rlvr_dataset/<ScenarioName>/<route_stem>/
        self.output_dir = os.path.join(
            _PROJECT_ROOT, output_dir, self.scenario_name, route_stem
        )
        create_folder(self.output_dir)

        # JSONL 数据文件路径
        self.jsonl_path = os.path.join(self.output_dir, "rlvr_samples.jsonl")

        # 初始化日志
        log_dir = os.path.join(self.output_dir, "logs")
        create_folder(log_dir)
        set_logger(log_dir, terminal_log_level="INFO")

        # 初始化仿真环境（与 vlm_decision.py 保持相同配置）
        self._init_env(route_stem)

        # 统计计数
        self.total_samples = 0

    def _init_env(self, route_stem: str):
        """初始化 SUMO/TransSimHub 仿真环境"""
        path_convert = get_abs_path(__file__)

        base_sumo_cfg = path_convert(
            f"data/raw/{self.scenario_name}/{self.scenario_config['NETFILE']}.sumocfg"
        )
        scenario_glb_dir = path_convert(
            f"data/raw/{self.scenario_name}/3d_assets/"
        )

        # 输出文件（仅用于 SUMO 内部统计，不需要分析）
        eval_dir = os.path.join(self.output_dir, "sumo_eval")
        create_folder(eval_dir)

        tls_add = [
            path_convert(f"data/raw/{self.scenario_name}/add/e2.add.xml"),
            path_convert(f"data/raw/{self.scenario_name}/add/tls_programs.add.xml"),
        ]

        # 若指定了路由文件，动态替换 sumocfg 中的路由路径
        sumo_cfg = base_sumo_cfg
        if self.route_file:
            with open(base_sumo_cfg, "r") as f:
                cfg_content = f.read()
            new_route_path = f"./env/{os.path.basename(self.route_file)}"
            cfg_content = re.sub(
                r'<route-files value="[^"]+"/>',
                f'<route-files value="{new_route_path}"/>',
                cfg_content,
                count=1,
            )
            self._temp_cfg = os.path.join(
                os.path.dirname(base_sumo_cfg), f"temp_rlvr_{route_stem}.sumocfg"
            )
            with open(self._temp_cfg, "w") as f:
                f.write(cfg_content)
            sumo_cfg = self._temp_cfg

        env_params = {
            "tls_id": self.junction_name,
            "number_phases": self.scenario_config["PHASE_NUMBER"],
            "sumo_cfg": sumo_cfg,
            "scenario_glb_dir": scenario_glb_dir,
            "trip_info": os.path.join(eval_dir, "tripinfo.xml"),
            "statistic_output": os.path.join(eval_dir, "statistic_output.xml"),
            "summary": os.path.join(eval_dir, "summary.txt"),
            "queue_output": os.path.join(eval_dir, "queue_output.xml"),
            "tls_state_add": tls_add,
            "renderer_cfg": self.scenario_config.get("RENDERER_CFG"),
            "sensor_cfg": self.scenario_config.get("SENSOR_CFG"),
            "tshub_env_cfg": TSHUB_ENV_CONFIG,
        }
        self.env = make_env(**env_params)()
        logger.info(f"[RLVR-Collect] 仿真环境初始化完成: {self.scenario_name}")

    def _save_bev_image(
        self, sensor_datas: dict, jid: str, step_dir: str
    ) -> str | None:
        """
        保存 BEV 图像（含车道水印），返回相对于项目根目录的路径。

        Args:
            sensor_datas (dict): infos['3d_data']
            jid          (str) : 路口 ID
            step_dir     (str) : 当前步骤输出目录

        Returns:
            str | None: 图像相对路径（保存成功）或 None（无图像数据）
        """
        sensor_imgs = sensor_datas.get("image", {})
        aircraft_jid = f"aircraft_{jid}"

        if not sensor_imgs or aircraft_jid not in sensor_imgs:
            return None

        junction_img = sensor_imgs[aircraft_jid].get("aircraft_all")
        if junction_img is None:
            return None

        try:
            # 保存原始 BEV 图像
            raw_path = os.path.join(step_dir, f"{aircraft_jid}_raw.png")
            cv2.imwrite(raw_path, convert_rgb_to_bgr(junction_img))

            # 添加车道水印（与训练数据格式一致）
            wm_path = os.path.join(step_dir, f"{aircraft_jid}_bev.png")
            add_lane_watermarks(raw_path, wm_path)

            # 返回相对于项目根目录的路径（方便跨机器使用）
            rel_path = os.path.relpath(wm_path, _PROJECT_ROOT)
            return rel_path
        except Exception as exc:
            logger.warning(f"[RLVR-Collect] BEV 图像保存失败 ({jid}): {exc}")
            return None

    def _make_decision_maxpressure(
        self, render_json: dict, jid: str
    ) -> int:
        """
        使用 MaxPressure 算法（基于 jam_length_vehicle）为当前路口生成决策动作。
        数据收集阶段使用 MaxPressure 保证轨迹质量，同时获取可靠的相位压力分布。

        Args:
            render_json (dict): 当前仿真步骤的原始状态字典
            jid         (str) : 路口 ID

        Returns:
            int: 选择的相位索引
        """
        tls_state = render_json.get("tls", {}).get(jid, {})
        movement_ids = tls_state.get("movement_ids", [])
        jam_veh = tls_state.get("jam_length_vehicle", [])
        phase2movements = tls_state.get("phase2movements", {})

        if not (movement_ids and jam_veh and phase2movements):
            # 数据缺失时使用循环 fallback
            fallback = self._step_count % self.scenario_config.get("PHASE_NUMBER", 4)
            logger.warning(f"[RLVR-Collect] {jid} 缺少 TLS 状态，fallback → Phase {fallback}")
            return fallback

        movement_queue = {mid: float(q) for mid, q in zip(movement_ids, jam_veh)}
        phase_pressure = {
            p_idx: sum(movement_queue.get(mid, 0.0) for mid in mvts)
            for p_idx, mvts in phase2movements.items()
        }
        return int(max(phase_pressure, key=phase_pressure.get))

    def collect(self):
        """
        运行仿真并收集 RLVR 训练数据。

        收集流程：
          1. 仿真环境 warm-up（第一步获取初始图像）
          2. 主循环：
             a. 对每个路口执行 MaxPressure 决策
             b. 收集 BEV 图像 + GT 计数 + 相位压力
             c. 将数据样本追加写入 JSONL 文件
          3. 仿真结束后汇报统计信息
        """
        logger.info(
            f"[RLVR-Collect] 开始数据收集 | 场景={self.scenario_name} | "
            f"路由={self.route_file} | 最大步数={self.max_steps}"
        )
        junctions = (
            self.junction_name if self.is_multi_agent else [self.junction_name]
        )

        # --- Warm-up 步骤 ---
        init_action = {jid: 0 for jid in junctions} if self.is_multi_agent else 0
        obs, _info = self.env.reset()
        obs, rewards, truncated, dones, infos, render_json = self.env.step(init_action)
        self._step_count = 0

        # 打开 JSONL 文件（追加模式）
        jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

        try:
            while True:
                # 终止条件检查
                if dones or truncated:
                    logger.info("[RLVR-Collect] 仿真 episode 结束")
                    break
                if self._step_count >= self.max_steps:
                    logger.info(f"[RLVR-Collect] 达到最大步数 {self.max_steps}")
                    break

                # 创建当前步骤的输出目录
                step_dir = os.path.join(self.output_dir, f"step_{self._step_count:04d}")
                os.makedirs(step_dir, exist_ok=True)
                save_to_json(render_json, os.path.join(step_dir, "render.json"))

                sensor_datas = infos.get("3d_data", {})
                action_dict = {}

                for jid in junctions:
                    tls_state = render_json.get("tls", {}).get(jid, {})
                    current_phase_id = tls_state.get("this_phase_index", 0)

                    # --- (1) 保存 BEV 图像 ---
                    jid_step_dir = os.path.join(step_dir, jid)
                    os.makedirs(jid_step_dir, exist_ok=True)
                    image_rel_path = self._save_bev_image(
                        sensor_datas, jid, jid_step_dir
                    )

                    # --- (2) 提取 GT 计数与相位压力 ---
                    gt_result = _build_gt_counts_per_direction(tls_state)

                    # --- (3) MaxPressure 决策（保证轨迹合理性）---
                    action = self._make_decision_maxpressure(render_json, jid)
                    action_dict[jid] = action

                    # --- (4) 构建样本字典并写入 JSONL ---
                    if image_rel_path is not None:
                        # 构建与 SFT/DPO 数据集兼容的 prompt 消息格式
                        prompt_text = PromptBuilder.build_decision_prompt(
                            current_phase_id=current_phase_id,
                            scenario_name=self.scenario_key,
                        )
                        prompt_messages = [
                            {
                                "role": "user",
                                "content": [
                                    # 图像占位符（训练时由 processor 动态填充）
                                    {"type": "image", "image": image_rel_path},
                                    {"type": "text",  "text": prompt_text},
                                ],
                            }
                        ]

                        # 直接方向总计（供奖励计算对齐模型输出格式）
                        direction_totals = _compute_direction_totals(
                            gt_result["gt_per_movement"]
                        )

                        sample = {
                            "sample_id": (
                                f"{self.scenario_key}_"
                                f"{os.path.splitext(os.path.basename(self.route_file or 'default'))[0]}"
                                f"_step{self._step_count:04d}_{jid}"
                            ),
                            "scenario_key": self.scenario_key,
                            "route_file": self.route_file or "",
                            "junction_id": jid,
                            "step": self._step_count,
                            "current_phase_id": current_phase_id,
                            "image_path": image_rel_path,
                            "prompt": prompt_messages,
                            # GT 计数（movement 级别 + 汇总）
                            "gt_per_movement": gt_result["gt_per_movement"],
                            "gt_total_all": direction_totals["total_all"],
                            # 相位压力分值（用作 r_env 代理）
                            "phase_pressure": gt_result["phase_pressure"],
                            "optimal_phase": gt_result["optimal_phase"],
                            # MaxPressure 实际选择的动作
                            "collector_action": action,
                        }
                        jsonl_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        self.total_samples += 1

                # --- 环境推进 ---
                final_action = action_dict if self.is_multi_agent else action_dict.get(
                    self.junction_name, 0
                )
                obs, rewards, truncated, dones, infos, render_json = self.env.step(
                    final_action
                )
                self._step_count += 1

        finally:
            jsonl_file.close()
            self.env.close()
            # 清理临时 SUMO 配置文件
            if hasattr(self, "_temp_cfg") and os.path.exists(self._temp_cfg):
                os.remove(self._temp_cfg)

        logger.info(
            f"[RLVR-Collect] 数据收集完成 | 总样本数={self.total_samples} | "
            f"保存路径={self.jsonl_path}"
        )
        return self.jsonl_path


# ===========================================================================
# 入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RLVR 训练数据收集脚本：从 SUMO 仿真采集 BEV + GT计数 + 相位压力"
    )
    parser.add_argument(
        "--scenario", type=str, default="JiNan",
        help="场景键名（如 JiNan、Hangzhou）"
    )
    parser.add_argument(
        "--route_file", type=str, default="anon_3_4_jinan_real.rou.xml",
        help="SUMO 路由文件名"
    )
    parser.add_argument(
        "--max_steps", type=int, default=120,
        help="最大决策步数（1h=120）"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/rlvr_dataset",
        help="数据集输出目录（相对于项目根目录）"
    )
    args = parser.parse_args()

    collector = RLVRDataCollector(
        scenario_key=args.scenario,
        route_file=args.route_file,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
    saved_path = collector.collect()
    print(f"\n[完成] RLVR 数据已保存至: {saved_path}")


if __name__ == "__main__":
    main()
