'''
Author: yufei Ji
Date: 2026-04-24
Description: Golden 数据集生成脚本（适配联合动作空间 + VLM student 轨迹 + 多步 rollout）

核心设计：
  1. 动作空间：(phase_id, duration) 笛卡尔积，枚举全部候选（最多 4×6=24 个）
  2. Rollout 策略：第1步执行候选动作，后续2步使用 FixedTime 保证公平比较
  3. 仿真推进：用 VLM student action 推进真实仿真（student 轨迹），而非 golden action
  4. 异步决策：与 run_eval.py 一致，仅对 can_perform_action=True 的路口触发决策+rollout
  5. 视觉输入：4张进口道停止线视图（{jid}_N/E/S/W），不含上游图
  6. 数据保存：JSONL，含 vlm_response / vlm_action / best_action / all_rollout_rewards 等字段

输出路径：data/sft_dataset/{scenario_name}/{route_stem}/01_dataset_raw.jsonl
图像路径：data/sft_dataset/{scenario_name}/{route_stem}/{jid}/{sumo_step}/

bug：libsumo 模式下，rollout_env 与 self.env 共享同一个 libsumo 实例（libsumo 不支持同进程内多实例）。因此：
  - rollout_env.reset() 会关闭并重启 libsumo → 覆盖了主环境的SUMO 进程                                                 
  - 每次 rollout_env.load_state() 回滚 SUMO 状态 →主环境的统计数据也被回滚                                   
  - 最终 self.env.close() 时 SUMO                            
  写出的统计数据，实际上是最后一次 load_state时的快照状态（接近仿真初始时刻），所以所有指标为 0         
                                                    
  这是 libsumo 单进程单实例限制导致的两个环境互相干扰的经典问题。解决方案是：要么改用traci（多进程），要么放弃双环境设计、用单环境 + load_state自我回滚。

  暂不处理，gloden 生成阶段主要关注决策和数据质量，统计数据的准确性相对次要（且后续评估阶段会用单环境正常统计）。在使用和评估阶段，均使用单环境设计，不受此问题影响。                                        
'''

import os
import re
import sys
import cv2
import json
import time
import copy
from collections import deque
from typing import Dict, List, Optional, Tuple

from loguru import logger

# --- 路径设置 ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
from add_lane_watermarks import add_lane_watermarks

from tshub.utils.init_log import set_logger
from src.utils.make_tsc_env import make_env
from src.utils.tools import create_folder, append_response_to_file, convert_rgb_to_bgr
from src.utils.tsc_env.tsc_wrapper import GREEN_DURATION_CANDIDATES, FIXED_TIME_GREEN_DURATION
from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from configs.prompt_builder import PromptBuilder
from src.inference.vlm_agent import VLMAgent
from src.utils.event_bulletin import EventBulletin


class GoldenGenerator:
    """
    Golden 数据集生成器。

    运行逻辑：
      - 对每个 can_perform_action=True 的路口，枚举所有 (phase, duration) 候选
      - 每个候选执行 1 步目标动作 + rollout_follow_steps 步 FixedTime，记录累计 reward
      - 选出 best_action，同时跑 VLM 推理获取 student action
      - 用 VLM action 推进真实仿真（student 轨迹）
      - 将样本写入 JSONL
    """

    def __init__(
        self,
        scenario_key: str = "JiNan",
        log_dir: str = "./log/golden_dataset",
        route_file: Optional[str] = None,
        rollout_follow_steps: int = 2,
        api_url: Optional[str] = None,
        model_name_override: Optional[str] = None,
    ):
        self.scenario_key = scenario_key
        self.rollout_follow_steps = rollout_follow_steps

        # --- 1. 加载场景配置 ---
        self.scenario_config = SCENARIO_CONFIGS.get(scenario_key)
        if not self.scenario_config:
            raise ValueError(f"Scenario '{scenario_key}' not found in SCENARIO_CONFIGS")

        self.scenario_name = self.scenario_config["SCENARIO_NAME"]
        self.junction_name = self.scenario_config["JUNCTION_NAME"]
        self.num_phases = self.scenario_config["PHASE_NUMBER"]
        self.approach_dirs = self.scenario_config.get("APPROACH_DIRS", ["N", "E", "S", "W"])

        # 多路口支持
        self.junctions = (
            self.junction_name if isinstance(self.junction_name, list) else [self.junction_name]
        )
        self.is_multi_agent = isinstance(self.junction_name, list)

        # --- 2. 路径设置 ---
        base_sumo_cfg = os.path.join(
            _PROJECT_ROOT, "data", "raw", self.scenario_name,
            f"{self.scenario_config['NETFILE']}.sumocfg"
        )
        scenario_glb_dir = os.path.join(
            _PROJECT_ROOT, "data", "raw", self.scenario_name, "3d_assets"
        )
        tls_add = [
            os.path.join(_PROJECT_ROOT, "data", "raw", self.scenario_name, "add", "e2.add.xml"),
            os.path.join(_PROJECT_ROOT, "data", "raw", self.scenario_name, "add", "tls_programs.add.xml"),
        ]

        # 处理路由文件
        sumo_cfg = base_sumo_cfg
        self.temp_cfg_path = None
        if route_file:
            route_file_name = os.path.basename(route_file)
            try:
                with open(base_sumo_cfg, 'r') as f:
                    cfg_content = f.read()
                new_route_path = f"./env/{route_file_name}"
                cfg_content = re.sub(
                    r'<route-files value="[^"]+"/>',
                    f'<route-files value="{new_route_path}"/>',
                    cfg_content, count=1
                )
                route_stem = os.path.splitext(route_file_name)[0]
                if route_stem.endswith(".rou"):
                    route_stem = route_stem[:-4]
                self.temp_cfg_path = os.path.join(
                    os.path.dirname(base_sumo_cfg), f"temp_golden_{route_stem}.sumocfg"
                )
                with open(self.temp_cfg_path, 'w') as f:
                    f.write(cfg_content)
                sumo_cfg = self.temp_cfg_path
            except Exception as e:
                logger.error(f"[GOLDEN] 路由文件替换失败: {e}")
                raise
        else:
            try:
                with open(base_sumo_cfg, 'r') as f:
                    cfg_content = f.read()
                match = re.search(r'<route-files value="([^"]+)"/>', cfg_content)
                route_file_name = os.path.basename(match.group(1)) if match else "default.rou.xml"
            except Exception:
                route_file_name = "default.rou.xml"
            route_stem = route_file_name
            if route_stem.endswith(".rou.xml"):
                route_stem = route_stem[:-8]
            elif route_stem.endswith(".xml"):
                route_stem = route_stem[:-4]

        self.route_stem = route_stem

        # 日志目录
        self.logger_path = os.path.join(log_dir, self.scenario_key, route_stem)
        create_folder(self.logger_path)
        set_logger(self.logger_path, terminal_log_level='INFO')
        logger.info(f"[GOLDEN] 日志目录: {self.logger_path}")

        # 输出目录（图像 + JSONL）
        self.output_dir = os.path.join(
            _PROJECT_ROOT, "data", "sft_dataset", self.scenario_name, route_stem
        )
        create_folder(self.output_dir)
        logger.info(f"[GOLDEN] 数据输出目录: {self.output_dir}")

        trip_info        = os.path.join(self.output_dir, "tripinfo_golden.out.xml")
        statistic_output = os.path.join(self.output_dir, "statistic_output_golden.xml")
        summary          = os.path.join(self.output_dir, "summary_golden.txt")
        queue_output     = os.path.join(self.output_dir, "queue_output_golden.xml")

        self.env_params = {
            'tls_id':           self.junction_name,
            'number_phases':    self.num_phases,
            'sumo_cfg':         sumo_cfg,
            'scenario_glb_dir': scenario_glb_dir,
            'trip_info':        trip_info,
            'statistic_output': statistic_output,
            'summary':          summary,
            'queue_output':     queue_output,
            'tls_state_add':    tls_add,
            'renderer_cfg':     self.scenario_config.get("RENDERER_CFG"),
            'sensor_cfg':       self.scenario_config.get("SENSOR_CFG"),
            'tshub_env_cfg':    TSHUB_ENV_CONFIG,
        }

        # --- 3. 初始化仿真环境 ---
        logger.info(f"[GOLDEN] 初始化仿真环境: {self.scenario_name} ...")
        try:
            self.env = make_env(**self.env_params)()
        except Exception as e:
            logger.critical(f"[GOLDEN] 环境初始化失败: {e}")
            raise

        # rollout 用环境（禁用 3D 渲染以节省资源）
        rollout_env_params = copy.deepcopy(self.env_params)
        rollout_renderer_cfg = copy.deepcopy(self.env_params.get('renderer_cfg') or {})
        rollout_renderer_cfg['is_render'] = False
        rollout_env_params['renderer_cfg'] = rollout_renderer_cfg
        rollout_env_params['sensor_cfg'] = None
        # rollout 不需要记录统计文件，避免覆盖主环境输出
        rollout_env_params['trip_info'] = None
        rollout_env_params['statistic_output'] = None
        rollout_env_params['summary'] = None
        rollout_env_params['queue_output'] = None

        logger.info("[GOLDEN] 初始化 rollout 环境（禁用渲染）...")
        try:
            self.rollout_env = make_env(**rollout_env_params)()
        except Exception as e:
            logger.critical(f"[GOLDEN] Rollout 环境初始化失败: {e}")
            raise

        # --- 4. 初始化 VLM Agent ---
        logger.info("[GOLDEN] 初始化 VLM Agent...")
        try:
            agent_kwargs = {}
            if api_url:
                agent_kwargs["url"] = api_url
            if model_name_override:
                agent_kwargs["model_name"] = model_name_override
            self.agent = VLMAgent(**agent_kwargs)
        except Exception as e:
            logger.critical(f"[GOLDEN] VLM Agent 初始化失败: {e}")
            raise

        # --- 5. 初始化 EventBulletin（上下游协同广播板）---
        topology = self.scenario_config.get("TOPOLOGY", {})
        self.bulletin = EventBulletin(topology=topology)

        # 状态文件路径（SUMO checkpoint）
        self._state_file = os.path.join(self.output_dir, "_temp_state.xml")

        logger.info(
            f"[GOLDEN] 初始化完成 | 场景={self.scenario_key} | 路由={route_stem} | "
            f"相位数={self.num_phases} | rollout_follow_steps={self.rollout_follow_steps} | "
            f"候选数={self.num_phases * len(GREEN_DURATION_CANDIDATES)}"
        )

    def __del__(self):
        for attr in ('env', 'rollout_env'):
            env_obj = getattr(self, attr, None)
            if env_obj is not None:
                try:
                    env_obj.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 辅助：图像采集
    # ------------------------------------------------------------------

    def _collect_images(
        self, jid: str, sensor_imgs: dict, step_dir: str
    ) -> List[str]:
        """采集路口 jid 的进口道停止线视图，叠加水印后写盘。
        返回成功保存的图像路径列表（按 approach_dirs 顺序）。
        """
        image_paths = []
        for d in self.approach_dirs:
            element_id = f'{jid}_{d}'
            img_data = None
            if element_id in sensor_imgs:
                img_data = sensor_imgs[element_id].get('junction_front_all')

            if img_data is None:
                logger.debug(f"[GOLDEN] 无图像数据: {element_id}")
                continue

            img_path = os.path.join(step_dir, f"{element_id}.png")
            try:
                cv2.imwrite(img_path, convert_rgb_to_bgr(img_data))
                try:
                    add_lane_watermarks(
                        input_path=img_path,
                        output_path=img_path,
                        scenario_name=self.scenario_key,
                    )
                except Exception as wm_err:
                    logger.warning(f"[GOLDEN] 水印叠加失败 {element_id}: {wm_err}")
                image_paths.append(img_path)
            except Exception as e:
                logger.warning(f"[GOLDEN] 图像保存失败 {element_id}: {e}")

        return image_paths

    # ------------------------------------------------------------------
    # 辅助：保存 / 恢复 wrapper 内部状态
    # ------------------------------------------------------------------

    @staticmethod
    def _save_wrapper_state(env) -> dict:
        return {
            "states": list(env.states),
            "occupancy_elements": list(env.occupancy.elements),
        }

    @staticmethod
    def _restore_wrapper_state(env, saved: dict):
        env.states = deque(saved["states"], maxlen=env.states.maxlen)
        env.occupancy.clear_elements()
        env.occupancy.elements = list(saved["occupancy_elements"])

    # ------------------------------------------------------------------
    # 核心：多步 rollout（候选动作 + FixedTime 后续步）
    # ------------------------------------------------------------------

    def _run_candidate_rollout(
        self,
        state_file: str,
        wrapper_state: dict,
        candidate_action,
    ) -> float:
        """
        从保存的 checkpoint 出发：
          步骤1：执行 candidate_action（目标动作）
          步骤2~N：执行 FixedTime（相位+1，时长 FIXED_TIME_GREEN_DURATION）
        返回累计折扣 reward（来自 env.step 内的 compute_rollout_q_value）。

        candidate_action 格式与 env.step 一致（单路口 dict / 多路口 dict of dict）。
        """
        # 关闭 rollout_env 渲染（已在 __init__ 禁用，此处双重保证）
        try:
            self.rollout_env.unwrapped.tsc_env.is_render = False
        except AttributeError:
            pass

        # --- 加载 checkpoint ---
        try:
            self.rollout_env.unwrapped.load_state(state_file)
            self._restore_wrapper_state(self.rollout_env, wrapper_state)
        except Exception as e:
            logger.error(f"[ROLLOUT] 加载 state 失败: {e}")
            return float('-inf')

        total_reward = 0.0
        discount = 1.0
        gamma = 0.95

        # --- 步骤 1：目标候选动作 ---
        try:
            _, reward, truncated, dones, infos, render_json = self.rollout_env.step(candidate_action)
            if isinstance(reward, dict):
                step_r = sum(reward.values())
            else:
                step_r = float(reward)
            total_reward += discount * step_r
            discount *= gamma
        except Exception as e:
            logger.error(f"[ROLLOUT] 候选动作执行失败: {e}")
            return float('-inf')

        # --- 步骤 2~N：FixedTime 后续步 ---
        for follow_i in range(self.rollout_follow_steps):
            if dones or truncated:
                break
            # FixedTime：相位顺序轮转
            if self.is_multi_agent:
                fixed_action = {
                    jid: {
                        'phase_id': (
                            render_json.get('tls', {}).get(jid, {}).get('this_phase_index', 0) + 1
                        ) % self.num_phases,
                        'duration': FIXED_TIME_GREEN_DURATION,
                    }
                    for jid in self.junctions
                }
            else:
                cur_phase = render_json.get('tls', {}).get(self.junction_name, {}).get('this_phase_index', 0)
                fixed_action = {
                    'phase_id': (cur_phase + 1) % self.num_phases,
                    'duration': FIXED_TIME_GREEN_DURATION,
                }
            try:
                _, reward, truncated, dones, infos, render_json = self.rollout_env.step(fixed_action)
                if isinstance(reward, dict):
                    step_r = sum(reward.values())
                else:
                    step_r = float(reward)
                total_reward += discount * step_r
                discount *= gamma
            except Exception as e:
                logger.warning(f"[ROLLOUT] FixedTime 步 {follow_i + 1} 失败: {e}")
                break

        return total_reward

    # ------------------------------------------------------------------
    # 核心：枚举所有候选，返回 {key: reward} 字典
    # ------------------------------------------------------------------

    def _evaluate_all_candidates(
        self, state_file: str, wrapper_state: dict, current_phases: Dict[str, int]
    ) -> Dict[str, Dict[str, float]]:
        """
        对全部 (phase_id, duration) 候选执行 rollout，返回：
          {jid: {"phase_dur": reward, ...}}  （多路口统一返回，方便后续 per-jid 解析）

        key 格式：f"{phase_id}_{duration}"
        """
        results: Dict[str, Dict[str, float]] = {jid: {} for jid in self.junctions}

        total_candidates = self.num_phases * len(GREEN_DURATION_CANDIDATES)
        logger.info(
            f"[ROLLOUT] ═══ 开始枚举 {total_candidates} 个候选 "
            f"({self.num_phases} 相位 × {len(GREEN_DURATION_CANDIDATES)} 时长) ═══"
        )

        for phase_id in range(self.num_phases):
            for duration in GREEN_DURATION_CANDIDATES:
                key = f"{phase_id}_{duration}"

                # 构建候选动作
                if self.is_multi_agent:
                    candidate_action = {
                        jid: {'phase_id': phase_id, 'duration': duration}
                        for jid in self.junctions
                    }
                else:
                    candidate_action = {'phase_id': phase_id, 'duration': duration}

                reward = self._run_candidate_rollout(state_file, wrapper_state, candidate_action)

                # 多路口时 reward 已是全局标量（sum），统一记录到每个 jid
                for jid in self.junctions:
                    results[jid][key] = reward

                logger.debug(
                    f"[ROLLOUT] phase={phase_id} | duration={duration}s | reward={reward:.4f}"
                )

        logger.info(f"[ROLLOUT] ═══ 枚举完成 ═══")
        return results

    # ------------------------------------------------------------------
    # 核心：VLM 推理
    # ------------------------------------------------------------------

    def _run_vlm_inference(
        self, jid: str, image_paths: List[str], cur_phase: int,
        coord_ctx: str = "",
    ) -> Tuple[str, dict, Optional[str]]:
        """
        对单个路口执行 VLM 推理，返回 (vlm_response, vlm_action_dict, vlm_raw_thought)。
        vlm_action_dict = {'phase_id': int, 'duration': int}

        coord_ctx: 来自 EventBulletin 的上游协同上下文（可为空串）。
        若 VLM 推理失败或解析失败，返回 FixedTime 降级动作。
        """
        prompt = PromptBuilder.build_decision_prompt(
            current_phase_id=cur_phase,
            scenario_name=self.scenario_key,
            neighbor_messages=coord_ctx,
        )
        default_action = {
            'phase_id': (cur_phase + 1) % self.num_phases,
            'duration': FIXED_TIME_GREEN_DURATION,
        }

        try:
            vlm_response, latency, parsed_action_tuple, thought = self.agent.get_decision(
                image_paths, prompt
            )
        except Exception as e:
            logger.error(f"[GOLDEN] VLM 推理异常 {jid}: {e}")
            return "ERROR", default_action, None

        if vlm_response == "ERROR" or not vlm_response:
            logger.warning(f"[GOLDEN] VLM 返回 ERROR，降级为 FixedTime | {jid}")
            return vlm_response, default_action, thought

        # parsed_action_tuple 来自 VLMAgent._parse_action，格式 (phase_id, duration)
        p_id, raw_dur = parsed_action_tuple
        if not 0 <= p_id < self.num_phases:
            logger.warning(
                f"[GOLDEN] VLM 输出非法 phase_id={p_id}（合法 [0,{self.num_phases-1}]），"
                f"降级为 FixedTime | {jid}"
            )
            return vlm_response, default_action, thought

        # 吸附 duration 到候选集
        actual_dur = min(GREEN_DURATION_CANDIDATES, key=lambda x: abs(x - raw_dur))
        if raw_dur not in GREEN_DURATION_CANDIDATES:
            logger.warning(
                f"[GOLDEN] VLM duration {raw_dur}s 不在候选集，已吸附至 {actual_dur}s | {jid}"
            )

        vlm_action = {'phase_id': p_id, 'duration': actual_dur}
        logger.info(
            f"[GOLDEN] VLM 决策 | {jid} | phase={p_id} | duration={actual_dur}s | "
            f"latency={latency:.2f}s"
        )
        return vlm_response, vlm_action, thought

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def generate(self, max_sumo_seconds: int = 3600, warmup_seconds: int = 300):
        """
        Golden 数据生成主循环。

        参数：
            max_sumo_seconds: SUMO 仿真总时间上限（秒），默认 3600s
            warmup_seconds:   前 N 秒仅用 FixedTime 推进，不跑 VLM、不保存数据，默认 300s
        """
        logger.info(
            f"[GOLDEN] ════ 开始生成 | 场景={self.scenario_key} | "
            f"warmup={warmup_seconds}s | max_sumo={max_sumo_seconds}s ════"
        )
        dataset_file = os.path.join(self.output_dir, "01_dataset_raw.jsonl")
        sample_count = 0

        # 重置主环境，获取初始状态
        self.env.reset()
        dones, truncated = False, False
        sumo_sim_step = 0.0

        # rollout_env 也需要 reset 以建立 SUMO TraCI 连接，
        # 后续每次 rollout 通过 load_state 恢复到检查点，而非 reset
        logger.info("[GOLDEN] 初始化 rollout 环境 TraCI 连接（reset）...")
        try:
            self.rollout_env.reset()
        except Exception as e:
            logger.critical(f"[GOLDEN] rollout_env reset 失败: {e}")
            self._cleanup()
            return

        # 热身步
        logger.info("[GOLDEN] 执行热身步...")
        init_action = {jid: 0 for jid in self.junctions} if self.is_multi_agent else 0
        try:
            obs, rewards, truncated, dones, infos, render_json = self.env.step(init_action)
            sumo_sim_step = float(infos.get('step_time', 0))
        except Exception as e:
            logger.critical(f"[GOLDEN] 热身步失败: {e}")
            self._cleanup()
            return

        # ── Warmup 阶段：FixedTime 推进，不跑 VLM，不保存数据 ────────────
        if warmup_seconds > 0:
            logger.info(f"[GOLDEN] ── Warmup 阶段（FixedTime，前 {warmup_seconds}s）──")
        while not (dones or truncated) and sumo_sim_step < warmup_seconds:
            if self.is_multi_agent:
                warmup_action = {
                    jid: {
                        'phase_id': (
                            render_json.get('tls', {}).get(jid, {}).get('this_phase_index', 0) + 1
                        ) % self.num_phases,
                        'duration': FIXED_TIME_GREEN_DURATION,
                    }
                    for jid in self.junctions
                }
            else:
                cur_p = render_json.get('tls', {}).get(self.junctions[0], {}).get('this_phase_index', 0)
                warmup_action = {
                    'phase_id': (cur_p + 1) % self.num_phases,
                    'duration': FIXED_TIME_GREEN_DURATION,
                }
            try:
                obs, rewards, truncated, dones, infos, render_json = self.env.step(warmup_action)
                sumo_sim_step = float(infos.get('step_time', sumo_sim_step))
            except Exception as e:
                logger.error(f"[GOLDEN] Warmup 阶段 env.step 失败: {e}")
                self._cleanup()
                return
        if warmup_seconds > 0:
            logger.info(
                f"[GOLDEN] ── Warmup 结束，sumo_t={sumo_sim_step:.0f}s，进入 VLM 阶段 ──"
            )

        # ── 主循环（VLM 推理 + rollout + 数据保存）────────────────────────
        while not (dones or truncated) and sumo_sim_step < max_sumo_seconds:
            logger.info(
                f"[GOLDEN] ══════ sumo_t={sumo_sim_step:.0f}s ══════"
            )

            # 清理过期事件通知（基于 SUMO 时间，与 run_eval.py 保持一致）
            self.bulletin.tick(sumo_sim_step)

            sensor_datas = infos.get('3d_data', {})
            sensor_imgs  = sensor_datas.get('image', {})

            # 确定本轮需要决策的路口
            if self.is_multi_agent:
                deciding_jids = [
                    jid for jid in self.junctions
                    if render_json.get('tls', {}).get(jid, {}).get('can_perform_action', False)
                ]
            else:
                deciding_jids = self.junctions

            if not deciding_jids:
                logger.warning("[GOLDEN] 无路口需要决策，直接推进仿真")
                try:
                    final_action = (
                        {jid: {'phase_id': 0, 'duration': FIXED_TIME_GREEN_DURATION} for jid in self.junctions}
                        if self.is_multi_agent
                        else {'phase_id': 0, 'duration': FIXED_TIME_GREEN_DURATION}
                    )
                    obs, rewards, truncated, dones, infos, render_json = self.env.step(final_action)
                    sumo_sim_step = float(infos.get('step_time', sumo_sim_step))
                except Exception as e:
                    logger.error(f"[GOLDEN] env.step 失败: {e}")
                    break
                continue

            # ── 阶段1：采集图像 ──────────────────────────────────────────
            per_jid_images: Dict[str, List[str]] = {}
            per_jid_cur_phase: Dict[str, int] = {}

            for jid in deciding_jids:
                cur_phase = render_json.get('tls', {}).get(jid, {}).get('this_phase_index', 0)
                per_jid_cur_phase[jid] = cur_phase

                jid_step_dir = os.path.join(
                    self.output_dir, jid, str(int(sumo_sim_step))
                )
                os.makedirs(jid_step_dir, exist_ok=True)

                image_paths = self._collect_images(jid, sensor_imgs, jid_step_dir)
                per_jid_images[jid] = image_paths

                if not image_paths:
                    logger.warning(f"[GOLDEN] {jid}: 无可用图像，本步跳过")

            # ── 阶段2：保存 SUMO checkpoint ──────────────────────────────
            self.env.unwrapped.save_state(self._state_file)
            main_wrapper_state = self._save_wrapper_state(self.env)

            # ── 阶段3：枚举 rollout，求 best_action ──────────────────────
            current_phases_map = per_jid_cur_phase
            all_rollout_rewards = self._evaluate_all_candidates(
                self._state_file, main_wrapper_state, current_phases_map
            )

            # ── 阶段4：恢复主环境 checkpoint（rollout 不影响主环境）────────
            self.env.unwrapped.load_state(self._state_file)
            self._restore_wrapper_state(self.env, main_wrapper_state)

            # ── 阶段5：VLM 推理 + 构建 student action ────────────────────
            last_action: Dict[str, dict] = {}

            for jid in deciding_jids:
                image_paths = per_jid_images.get(jid, [])
                cur_phase   = per_jid_cur_phase.get(jid, 0)

                if not image_paths:
                    # 无图像：降级 FixedTime
                    fallback_action = {
                        'phase_id': (cur_phase + 1) % self.num_phases,
                        'duration': FIXED_TIME_GREEN_DURATION,
                    }
                    last_action[jid] = fallback_action
                    logger.warning(f"[GOLDEN] {jid}: 无图像，VLM 推理跳过，使用 FixedTime")
                    continue

                # 获取上游协同上下文（与 run_eval.py 保持一致）
                coord_ctx = self.bulletin.get_context(jid, sumo_sim_step)
                if coord_ctx:
                    logger.info(
                        f"[Bulletin][注入] {jid} | sumo_t={sumo_sim_step:.0f}s | 注入上游协同通知"
                    )

                vlm_response, vlm_action, vlm_thought = self._run_vlm_inference(
                    jid, image_paths, cur_phase, coord_ctx=coord_ctx
                )
                last_action[jid] = vlm_action

                # 广播事件通知给邻居路口（多路口模式下才有意义）
                if self.is_multi_agent and vlm_response not in ("ERROR", ""):
                    self.bulletin.broadcast(
                        from_jid=jid,
                        vlm_response=vlm_response,
                        green_duration=vlm_action['duration'],
                        current_sumo_step=sumo_sim_step,
                    )

                # ── 阶段6：确定 best_action，写入样本 ────────────────────
                metrics = all_rollout_rewards.get(jid, {})
                if not metrics:
                    logger.warning(f"[GOLDEN] {jid}: rollout metrics 为空，跳过样本保存")
                    continue

                best_key = max(metrics, key=metrics.get)
                best_reward = metrics[best_key]
                best_phase_id, best_duration = map(int, best_key.split("_"))
                best_action_dict = {'phase_id': best_phase_id, 'duration': best_duration}

                vlm_key = f"{vlm_action['phase_id']}_{vlm_action['duration']}"
                label = "accepted" if vlm_key == best_key else "rejected"

                sample = {
                    "scenario":              self.scenario_key,
                    "junction_id":           jid,
                    "sumo_step":             int(sumo_sim_step),
                    "current_phase_id":      int(cur_phase),
                    "image_paths":           [os.path.abspath(p) for p in image_paths],
                    "prompt":                PromptBuilder.build_decision_prompt(
                        current_phase_id=cur_phase,
                        scenario_name=self.scenario_key,
                    ),
                    "vlm_response":          vlm_response,
                    "vlm_action":            vlm_action,
                    "vlm_thought":           vlm_thought,
                    "best_action":           best_action_dict,
                    "best_reward":           float(best_reward),
                    "all_rollout_rewards":   {k: float(v) for k, v in metrics.items()},
                    "label":                 label,
                    "rollout_follow_steps":  self.rollout_follow_steps,
                }

                append_response_to_file(dataset_file, json.dumps(sample, ensure_ascii=False))
                sample_count += 1
                logger.info(
                    f"[GOLDEN] 样本保存 #{sample_count} | {jid} | sumo_t={sumo_sim_step:.0f}s | "
                    f"VLM({vlm_action['phase_id']},{vlm_action['duration']}s) vs "
                    f"BEST({best_phase_id},{best_duration}s) → {label}"
                )

            # ── 阶段7：用 VLM student action 推进真实仿真 ─────────────────
            # 非决策路口沿用上次的 last_action（从 render_json 读取当前相位保持不变）
            for jid in self.junctions:
                if jid not in last_action:
                    last_action[jid] = {
                        'phase_id': render_json.get('tls', {}).get(jid, {}).get('this_phase_index', 0),
                        'duration': FIXED_TIME_GREEN_DURATION,
                    }

            final_action = (
                last_action if self.is_multi_agent
                else last_action.get(self.junctions[0], {'phase_id': 0, 'duration': FIXED_TIME_GREEN_DURATION})
            )

            try:
                obs, rewards, truncated, dones, infos, render_json = self.env.step(final_action)
                sumo_sim_step = float(infos.get('step_time', sumo_sim_step))
            except Exception as e:
                logger.error(f"[GOLDEN] env.step 失败 @ sumo_t={sumo_sim_step:.0f}s: {e}")
                break

        # ── 结束 ─────────────────────────────────────────────────────────
        logger.info(
            f"[GOLDEN] ════ 生成完成 | 共保存 {sample_count} 条样本 | "
            f"数据文件: {dataset_file} ════"
        )
        if self.is_multi_agent:
            self.bulletin.log_topology()
        self._cleanup()

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def _cleanup(self):
        if os.path.exists(self._state_file):
            try:
                os.remove(self._state_file)
            except OSError:
                pass
        if self.temp_cfg_path and os.path.exists(self.temp_cfg_path):
            try:
                os.remove(self.temp_cfg_path)
                logger.info(f"[GOLDEN] 已删除临时配置: {self.temp_cfg_path}")
            except OSError as e:
                logger.warning(f"[GOLDEN] 删除临时配置失败: {e}")
        for attr in ('env', 'rollout_env'):
            env_obj = getattr(self, attr, None)
            if env_obj is not None:
                try:
                    env_obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)


# ──────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Golden 数据集生成（联合动作空间 + VLM student 轨迹 + 多步 rollout）"
    )
    parser.add_argument(
        "--scenario", "-sc", type=str, default="Hongkong_YMT",
        help="场景键名 (e.g., JiNan, Hangzhou, Hongkong_YMT, SouthKorea_Songdo, France_Massy)"
    )
    parser.add_argument(
        "--route_file", "-r", type=str, default="YMT_emergy.rou.xml",
        help=".rou.xml 路由文件名（SUMO 将在 env/ 下查找）"
    )
    parser.add_argument(
        "--max_sumo_seconds", "-n", type=int, default=800,
        help="最大 SUMO 仿真时间（秒），默认 3600s"
    )
    parser.add_argument(
        "--warmup_seconds", "-w", type=int, default=30,
        help="Warmup 阶段时长（秒），该时段仅 FixedTime 推进，不跑 VLM 也不保存数据，默认 300s"
    )
    parser.add_argument(
        "--rollout_follow_steps", "-rfs", type=int, default=0,
        help="候选动作之后额外执行的 FixedTime 步数（用于多步 rollout 评估），默认 2"
    )
    parser.add_argument(
        "--log_dir", "-l", type=str, default="./log/golden_dataset",
        help="日志输出目录"
    )
    parser.add_argument(
        "--api_url", type=str, default=None,
        help="（可选）覆盖 model_config.py requests.url，不传则读 MODEL_CONFIG"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="（可选）覆盖 model_config.py requests.model_name，不传则读 MODEL_CONFIG"
    )

    args = parser.parse_args()

    generator = GoldenGenerator(
        scenario_key=args.scenario,
        log_dir=args.log_dir,
        route_file=args.route_file,
        rollout_follow_steps=args.rollout_follow_steps,
        api_url=args.api_url,
        model_name_override=args.model_name,
    )
    generator.generate(
        max_sumo_seconds=args.max_sumo_seconds,
        warmup_seconds=args.warmup_seconds,
    )
