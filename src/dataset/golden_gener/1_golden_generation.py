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

Rollout 架构说明（多进程独立 worker）：
  libsumo 单进程单实例，不支持同进程内多个独立 SUMO 实例。原双环境设计存在三个致命 bug：
    Bug1: rollout_env.reset() 会杀死主环境 SUMO 进程并重启，导致两个 wrapper 共享同一实例，
          信号灯 Python 对象状态与 SUMO 实际状态完全错位。
    Bug2: rollout_env.load_state() 等价于对共享 SUMO 执行 loadState，反复回滚主环境仿真时钟，
          车辆被 SUMO 删除但统计缓冲区仍有记录，导致大量"幽灵车辆"滞留路网。
    Bug3: 每次 rollout 的 set_next_phases() 调用（setPhase/setPhaseDuration）直接作用于
          共享信号灯，污染主环境 TLS Python 对象的 next_action_time 等字段。

    1、去掉 rollout_env，改为单环境自我 save/load：
    rollout 前：self.env.save_state() + _save_wrapper_state()
    rollout 中：self.env.load_state() + _restore_wrapper_state() → step() → 记录 reward
    rollout 后：self.env.load_state() + _restore_wrapper_state() 恢复主轨迹
    主轨迹继续用 VLM action 推进，统计数据完整连续，无幽灵车辆。
    2、修复：主进程只保留真实仿真；rollout 改为多个独立 worker 进程并行评估：
    rollout 前：主进程 self.env.save_state() + _save_wrapper_state()
    rollout 中：每个 worker 从同一个 checkpoint loadState + restore_wrapper_state，
                只评估自己负责的一个候选动作，并返回 reward
    rollout 后：主进程不需要 loadState 回滚，直接继续真实轨迹
    由于 worker 永久关闭渲染，且主/worker SUMO 实例彼此独立，因此不会再污染主轨迹。

输出路径：data/sft_dataset/{scenario_name}/{route_stem}/01_dataset_raw.jsonl
图像路径：data/sft_dataset/{scenario_name}/{route_stem}/{jid}/{sumo_step}/
'''

import os
import re
import sys
import cv2
import json
import time
import copy
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

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
from src.utils.tools import create_folder, append_response_to_file, convert_rgb_to_bgr, save_to_json
from src.utils.tsc_env.tsc_wrapper import GREEN_DURATION_CANDIDATES, FIXED_TIME_GREEN_DURATION
from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from configs.prompt_builder import PromptBuilder
from src.inference.vlm_agent import VLMAgent
from src.utils.event_bulletin import EventBulletin


# ----------------------------------------------------------------------
# Rollout Worker 全局变量
# ----------------------------------------------------------------------
# 说明：
#   1. ProcessPoolExecutor 的每个子进程在 initializer 中只初始化一次 env；
#   2. 后续该子进程收到多个候选动作任务时，都会复用自己的独立 env；
#   3. 由于 libsumo 不能在同一进程内多开多个独立实例，因此这里必须依赖多进程隔离。
_ROLLOUT_WORKER_ENV = None
_ROLLOUT_WORKER_CFG: Dict[str, Any] = {}


def _get_tls_builder_from_env(env):
    """获取 TLS builder。

    这里单独抽成模块级函数，原因是：
      - 主进程和 rollout 子进程都要调用；
      - ProcessPoolExecutor 的任务函数必须位于模块顶层，不能依赖实例方法闭包。
    """
    return env.unwrapped.tsc_env.tshub_env.scene_objects.get('tls')


def _save_wrapper_state_for_env(env) -> dict:
    """保存 Python wrapper 侧状态。

    只保存 SUMO checkpoint 还不够，因为以下信息只存在 Python 侧：
      - TSC wrapper 的 states / occupancy 缓冲区；
      - tls_action 的 phase_index / next_action_time / yellow 状态；
      - can_perform_action 等异步决策时序字段。
    若不恢复这些字段，子进程虽然能 loadState 回到同一 SUMO 时刻，
    但 Python 侧信号灯对象仍可能与 SUMO 实际状态错位。
    """
    tls_action_states = {}
    try:
        tls_builder = _get_tls_builder_from_env(env)
        if tls_builder is not None:
            for jid, tl in tls_builder.traffic_lights.items():
                a = tl.tls_action
                tls_action_states[jid] = {
                    "phase_index":             a.phase_index,
                    "next_action_time":        a.next_action_time,
                    "is_yellow":               a.is_yellow,
                    "yellow_end_time":         getattr(a, 'yellow_end_time', 0.0),
                    "pending_green_duration":  getattr(a, '_pending_green_duration', a.delta_time),
                    "can_perform_action":      tl.can_perform_action,
                }
        if not tls_action_states:
            logger.warning("[GOLDEN] _save_wrapper_state: tls_action_states 为空，TLS 状态未保存")
    except Exception as e:
        logger.warning(f"[GOLDEN] 保存 tls_action 状态失败: {e}")
    return {
        "states": list(env.states),
        "occupancy_elements": list(env.occupancy.elements),
        "tls_action_states": tls_action_states,
    }


def _restore_wrapper_state_for_env(env, saved: dict):
    """恢复 Python wrapper 侧状态。"""
    env.states = deque(saved["states"], maxlen=env.states.maxlen)
    env.occupancy.clear_elements()
    env.occupancy.elements = list(saved["occupancy_elements"])
    tls_action_states = saved.get("tls_action_states", {})
    if not tls_action_states:
        logger.warning("[GOLDEN] _restore_wrapper_state: tls_action_states 为空，TLS 状态未恢复")
        return
    try:
        tls_builder = _get_tls_builder_from_env(env)
        if tls_builder is not None:
            for jid, state in tls_action_states.items():
                tl = tls_builder.traffic_lights[jid]
                a = tl.tls_action
                a.phase_index             = state["phase_index"]
                a.next_action_time        = state["next_action_time"]
                a.is_yellow               = state["is_yellow"]
                a.yellow_end_time         = state.get("yellow_end_time", 0.0)
                a._pending_green_duration = state["pending_green_duration"]
                tl.can_perform_action     = state["can_perform_action"]
    except Exception as e:
        logger.warning(f"[GOLDEN] 恢复 tls_action 状态失败: {e}")


def _build_rollout_worker_env_params(
    env_params: dict,
    worker_root: str,
    worker_slot: int,
) -> dict:
    """为 rollout worker 生成独立 env 参数。

    设计目标：
      1. 关闭 3D 渲染，只跑物理仿真与 reward；
      2. 每个 worker 写入自己的临时输出目录，避免多个 SUMO 实例抢同一个文件；
      3. 保持 sumocfg / route / net 等主仿真配置完全一致，确保 checkpoint 可被正确 load。
    """
    worker_env_params = copy.deepcopy(env_params)
    worker_dir = os.path.join(worker_root, f"worker_{worker_slot}")
    os.makedirs(worker_dir, exist_ok=True)

    worker_env_params["trip_info"] = os.path.join(worker_dir, "tripinfo_rollout.out.xml")
    worker_env_params["statistic_output"] = os.path.join(worker_dir, "statistic_output_rollout.xml")
    worker_env_params["summary"] = os.path.join(worker_dir, "summary_rollout.txt")
    worker_env_params["queue_output"] = os.path.join(worker_dir, "queue_output_rollout.xml")

    renderer_cfg = copy.deepcopy(worker_env_params.get("renderer_cfg") or {})
    renderer_cfg["is_render"] = False
    worker_env_params["renderer_cfg"] = renderer_cfg
    return worker_env_params


def _rollout_worker_initializer(
    env_params: dict,
    worker_root: str,
    worker_log_root: str,
    rollout_follow_steps: int,
    num_phases: int,
    junction_name,
    junctions: List[str],
    is_multi_agent: bool,
):
    """子进程初始化函数。

    每个 worker 进程只在启动时执行一次：
      - 构造自己的独立 env；
      - 永久关闭渲染；
      - 缓存 rollout 所需的静态配置。
    """
    global _ROLLOUT_WORKER_ENV, _ROLLOUT_WORKER_CFG

    proc = mp.current_process()
    identity = tuple(getattr(proc, "_identity", ()) or ())
    worker_slot = identity[0] if identity else os.getpid()

    # 每个子进程独立初始化日志。
    # 注意：当前 rollout 使用的是 spawn 启动方式，子进程不会继承主进程已配置好的
    # loguru file handler，因此必须在子进程里重新调用 set_logger()。
    # 这里故意把日志目录按 worker 拆开，保证主进程与各个 worker 的日志完全隔离。
    worker_log_dir = os.path.join(worker_log_root, f"worker_{worker_slot}")
    os.makedirs(worker_log_dir, exist_ok=True)
    set_logger(worker_log_dir, terminal_log_level='INFO')

    local_env_params = _build_rollout_worker_env_params(env_params, worker_root, worker_slot)

    _ROLLOUT_WORKER_ENV = make_env(**local_env_params)()
    _ROLLOUT_WORKER_ENV.reset()
    _ROLLOUT_WORKER_CFG = {
        "rollout_follow_steps": rollout_follow_steps,
        "num_phases": num_phases,
        "junction_name": junction_name,
        "junctions": list(junctions),
        "is_multi_agent": is_multi_agent,
    }
    logger.info(
        f"[ROLLOUT-WORKER] 初始化完成 | pid={os.getpid()} | slot={worker_slot} | "
        f"is_render={_ROLLOUT_WORKER_ENV.unwrapped.tsc_env.is_render}"
    )


def _run_rollout_task(task: dict) -> Tuple[str, float]:
    """子进程执行单个候选动作 rollout。

    输入 task 只包含一个候选动作，因此天然满足“一个 worker 任务只跑一个动作”。
    但同一个 worker 进程在未来可能继续处理新的动作任务，因此任务开始时必须：
      1. load 主进程提供的 checkpoint；
      2. restore 对应的 Python wrapper 状态；
    这样无论该 worker 之前执行过什么动作，都能回到同一个基准状态。
    """
    global _ROLLOUT_WORKER_ENV, _ROLLOUT_WORKER_CFG

    env = _ROLLOUT_WORKER_ENV
    if env is None:
        raise RuntimeError("rollout worker env 未初始化")

    key = task["key"]
    target_jid = task["target_jid"]
    candidate_action = task["candidate_action"]
    state_file = task["state_file"]
    wrapper_state = task["wrapper_state"]

    env.unwrapped.load_state(state_file)
    _restore_wrapper_state_for_env(env, wrapper_state)

    total_reward = 0.0
    discount = 1.0
    gamma = 0.95

    _, reward, truncated, dones, infos, render_json = env.step(candidate_action)
    if isinstance(reward, dict):
        step_r = float(reward.get(target_jid, float('-inf')))
    else:
        step_r = float(reward)
    total_reward += discount * step_r
    discount *= gamma

    for _ in range(_ROLLOUT_WORKER_CFG["rollout_follow_steps"]):
        if dones or truncated:
            break

        if _ROLLOUT_WORKER_CFG["is_multi_agent"]:
            fixed_action = {
                jid: {
                    "phase_id": (
                        render_json.get("tls", {}).get(jid, {}).get("this_phase_index", 0) + 1
                    ) % _ROLLOUT_WORKER_CFG["num_phases"],
                    "duration": FIXED_TIME_GREEN_DURATION,
                }
                for jid in _ROLLOUT_WORKER_CFG["junctions"]
            }
        else:
            cur_phase = render_json.get("tls", {}).get(
                _ROLLOUT_WORKER_CFG["junction_name"], {}
            ).get("this_phase_index", 0)
            fixed_action = {
                "phase_id": (cur_phase + 1) % _ROLLOUT_WORKER_CFG["num_phases"],
                "duration": FIXED_TIME_GREEN_DURATION,
            }

        _, reward, truncated, dones, infos, render_json = env.step(fixed_action)
        if isinstance(reward, dict):
            step_r = float(reward.get(target_jid, float('-inf')))
        else:
            step_r = float(reward)
        total_reward += discount * step_r
        discount *= gamma

    return key, float(total_reward)


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
        is_rollout: bool = True,
        rollout_follow_steps: int = 2,
        rollout_num_workers: Optional[int] = None,
        api_url: Optional[str] = None,
        model_name_override: Optional[str] = None,
    ):
        self.scenario_key = scenario_key
        self.rollout_follow_steps = rollout_follow_steps
        self.rollout_num_workers = rollout_num_workers

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
        log_session = set_logger(self.logger_path, terminal_log_level='INFO')
        # set_logger() 会在 self.logger_path 下再创建一层带时间戳的会话目录。
        # 后续如果还要创建子进程日志目录，必须使用这层“真实运行目录”，否则日志会散落到父目录里。
        self.logger_path = log_session["log_dir"]
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

        # --- 3. 初始化主仿真环境（真实轨迹）---
        logger.info(f"[GOLDEN] 初始化仿真环境: {self.scenario_name} ...")
        try:
            self.env = make_env(**self.env_params)()
        except Exception as e:
            logger.critical(f"[GOLDEN] 环境初始化失败: {e}")
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
        self._rollout_worker_root = os.path.join(self.output_dir, "_rollout_workers")
        self._rollout_worker_log_root = os.path.join(self.logger_path, "_rollout_workers")
        create_folder(self._rollout_worker_root)
        create_folder(self._rollout_worker_log_root)
        self._rollout_executor: Optional[ProcessPoolExecutor] = None

        logger.info(
            f"[GOLDEN] 初始化完成 | 场景={self.scenario_key} | 路由={route_stem} | "
            f"相位数={self.num_phases} | rollout_follow_steps={self.rollout_follow_steps} | "
            f"候选数={self.num_phases * len(GREEN_DURATION_CANDIDATES)}"
        )

        self.is_rollout = is_rollout
        if self.is_rollout:
            self._init_rollout_executor()

    def __del__(self):
        self._shutdown_rollout_executor()
        env_obj = getattr(self, 'env', None)
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
        保存加水印前后两版图像：
          - {element_id}_no_watermark.png：原始图像（未加水印）
          - {element_id}.png：加水印后的图像（输入模型）
        返回成功保存的加水印后图像路径列表（按 approach_dirs 顺序）。
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

            # 路径配置
            img_path_no_watermark = os.path.join(step_dir, f"{element_id}_no_watermark.png")
            img_path_with_watermark = os.path.join(step_dir, f"{element_id}.png")
            
            try:
                # 1. 保存原始图像（无水印）
                cv2.imwrite(img_path_no_watermark, convert_rgb_to_bgr(img_data))
                logger.debug(f"[GOLDEN] 原始图像已保存: {img_path_no_watermark}")
                
                # 2. 保存副本并添加水印
                cv2.imwrite(img_path_with_watermark, convert_rgb_to_bgr(img_data))
                try:
                    add_lane_watermarks(
                        input_path=img_path_with_watermark,
                        output_path=img_path_with_watermark,
                        scenario_name=self.scenario_key,
                    )
                    logger.debug(f"[GOLDEN] 水印叠加成功: {img_path_with_watermark}")
                except Exception as wm_err:
                    logger.warning(f"[GOLDEN] 水印叠加失败 {element_id}: {wm_err}")
                
                # 3. 将加水印后的图像路径加入返回列表（供模型输入）
                image_paths.append(img_path_with_watermark)
            except Exception as e:
                logger.warning(f"[GOLDEN] 图像保存失败 {element_id}: {e}")

        return image_paths

    # ------------------------------------------------------------------
    # 辅助：保存 / 恢复 wrapper 内部状态
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tls_builder(env):
        return _get_tls_builder_from_env(env)

    @staticmethod
    def _save_wrapper_state(env) -> dict:
        return _save_wrapper_state_for_env(env)

    @staticmethod
    def _restore_wrapper_state(env, saved: dict):
        _restore_wrapper_state_for_env(env, saved)

    def _init_rollout_executor(self):
        """初始化 rollout 进程池。

        采用 spawn 而不是 fork，原因：
          - libsumo / Panda3D / traci 等底层对象不适合通过 fork 复制；
          - spawn 会让子进程从干净的 Python 解释器启动，再自行构建独立 env，
            隔离性最强，也最符合当前 rollout 需求。
        """
        if self._rollout_executor is not None:
            return

        total_candidates = self.num_phases * len(GREEN_DURATION_CANDIDATES)
        max_workers = self.rollout_num_workers or total_candidates
        max_workers = max(1, min(max_workers, total_candidates))

        mp_ctx = mp.get_context("spawn")
        self._rollout_executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_ctx,
            initializer=_rollout_worker_initializer,
            initargs=(
                self.env_params,
                self._rollout_worker_root,
                self._rollout_worker_log_root,
                self.rollout_follow_steps,
                self.num_phases,
                self.junction_name,
                self.junctions,
                self.is_multi_agent,
            ),
        )
        logger.info(
            f"[ROLLOUT] 进程池已初始化 | workers={max_workers} | "
            f"每个 worker 独立持有一套无渲染 SUMO env"
        )

    def _shutdown_rollout_executor(self):
        """关闭 rollout 进程池。"""
        if self._rollout_executor is not None:
            try:
                self._rollout_executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                logger.warning(f"[ROLLOUT] 关闭进程池失败: {e}")
            self._rollout_executor = None

    # ------------------------------------------------------------------
    # 核心：多 worker 并行 rollout
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 核心：枚举所有候选，返回 {key: reward} 字典
    # ------------------------------------------------------------------

    def _evaluate_all_candidates(
        self,
        state_file: str,
        wrapper_state: dict,
        deciding_jids: List[str],
        render_json: dict,
        last_action: Dict[str, dict],
    ) -> Dict[str, Dict[str, float]]:
        """
        对当前需要决策的路口分别执行 rollout，返回：
          {jid: {"phase_dur": reward, ...}}

        key 格式：f"{phase_id}_{duration}"

        并行策略：
          - 每个 (target_jid, candidate_action) 封装成一个独立 task；
          - task 被提交到独立 rollout worker 进程；
          - worker 收到 task 后，先 load 主进程 checkpoint，再恢复 wrapper_state，
            然后只评估这个 target_jid 的 reward；
          - 对于其它同时 can_perform_action=True 的路口，采用 FixedTime 作为 rollout baseline，
            避免把它们也一起枚举成组合动作空间；
          - 因此动作之间的隔离边界是“进程独立 + 每任务从同一 checkpoint 回滚”。
        """
        results: Dict[str, Dict[str, float]] = {jid: {} for jid in deciding_jids}
        if self._rollout_executor is None:
            raise RuntimeError("rollout 进程池尚未初始化")

        total_candidates = len(deciding_jids) * self.num_phases * len(GREEN_DURATION_CANDIDATES)
        logger.info(
            f"[ROLLOUT] ═══ 开始枚举 {total_candidates} 个候选 "
            f"({len(deciding_jids)} 路口 × {self.num_phases} 相位 × {len(GREEN_DURATION_CANDIDATES)} 时长) ═══"
        )

        future_to_key = {}
        # 先准备一份全量 baseline action：
        #   - 非决策路口：沿用上一次真实仿真中的 last_action
        #   - 本轮决策路口：先用 FixedTime 作为默认 rollout 基线
        # 然后为每个 target_jid 覆盖成自己的候选动作。
        baseline_actions = copy.deepcopy(last_action)
        for jid in self.junctions:
            if jid not in baseline_actions:
                cur_phase = render_json.get("tls", {}).get(jid, {}).get("this_phase_index", 0)
                baseline_actions[jid] = {
                    "phase_id": cur_phase,
                    "duration": FIXED_TIME_GREEN_DURATION,
                }

        for jid in deciding_jids:
            cur_phase = render_json.get("tls", {}).get(jid, {}).get("this_phase_index", 0)
            baseline_actions[jid] = {
                "phase_id": (cur_phase + 1) % self.num_phases,
                "duration": FIXED_TIME_GREEN_DURATION,
            }

        for target_jid in deciding_jids:
            for phase_id in range(self.num_phases):
                for duration in GREEN_DURATION_CANDIDATES:
                    key = f"{phase_id}_{duration}"
                    if self.is_multi_agent:
                        candidate_action = copy.deepcopy(baseline_actions)
                        candidate_action[target_jid] = {
                            "phase_id": phase_id,
                            "duration": duration,
                        }
                    else:
                        candidate_action = {"phase_id": phase_id, "duration": duration}

                    task = {
                        "key": key,
                        "target_jid": target_jid,
                        "state_file": state_file,
                        "wrapper_state": wrapper_state,
                        "candidate_action": candidate_action,
                    }
                    future = self._rollout_executor.submit(_run_rollout_task, task)
                    future_to_key[future] = (target_jid, key)

        for future in as_completed(future_to_key):
            target_jid, key = future_to_key[future]
            try:
                finished_key, reward = future.result()
            except Exception as e:
                logger.error(f"[ROLLOUT] {target_jid} 候选动作 {key} 执行失败: {e}")
                finished_key, reward = key, float('-inf')

            results[target_jid][finished_key] = reward

            logger.debug(
                f"[ROLLOUT] target={target_jid} | candidate={finished_key} | reward={reward:.4f}"
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

        # 重置环境，获取初始状态（单环境，无需额外 rollout_env）
        self.env.reset()
        dones, truncated = False, False
        sumo_sim_step = 0.0
        # 异步多路口模式下，需要像 run_eval.py 一样维护“跨时间步持久化”的 last_action。
        # 否则每一轮循环都重建 last_action，会让非决策路口丢失自己的持续控制动作。
        last_action: Dict[str, dict] = {
            jid: {'phase_id': 0, 'duration': FIXED_TIME_GREEN_DURATION}
            for jid in self.junctions
        }

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
                        last_action
                        if self.is_multi_agent
                        else last_action.get(self.junctions[0], {'phase_id': 0, 'duration': FIXED_TIME_GREEN_DURATION})
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

            # ── 阶段3：保存 checkpoint + rollout 枚举 best_action ────────
            # 诊断：记录 checkpoint 时刻的 SUMO 实际车辆数
            try:
                tshub_env_diag = self.env.unwrapped.tsc_env.tshub_env
                veh_count_before = len(tshub_env_diag.sumo.vehicle.getIDList())
                logger.info(
                    f"[Golden] rollout 前 SUMO 车辆数={veh_count_before}"
                    f" | sumo_t={sumo_sim_step:.0f}s"
                )
            except Exception:
                veh_count_before = -1
            if self.is_rollout:
                # 主进程只负责存档，真正的 rollout 在独立 worker 进程里完成
                self.env.unwrapped.save_state(self._state_file)
                main_wrapper_state = self._save_wrapper_state(self.env)

                all_rollout_rewards = self._evaluate_all_candidates(
                    self._state_file,
                    main_wrapper_state,
                    deciding_jids,
                    render_json,
                    last_action,
                )

            # ── 阶段4：VLM 推理 + 构建 student action ──────────────────────
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

                # fixed time
                # vlm_action =  {
                #     'phase_id': (cur_phase + 1) % self.num_phases,
                #     'duration': 35,
                # }
                # vlm_response = 'Thought: [\nA. Scene Understanding:\n- Lane Analysis:\nN(Major): L1(S): Long (5+ vehicles stopped at stop line)\nS(Major): L2(S): Medium (4-7 vehicles stopped at stop line)\nW(Minor): L1(L): Short (1-3 vehicles stopped at stop line)\n- Phase Mapping:\nPhase 0 (Major Road): OverallPressure: High | CriticalQueue: Long\nTie-Breaker: None\n\nB. Scene Analysis:\n- Event Recognition: Public Bus (Transit) detected at South Approach L2, affects Phase 0\n- Neighboring Messages: Inactive\n- Condition Assessment: Special\n\nC. Adaptive Reasoning:\nImpact Analysis: The public bus, while a transit vehicle, is a large vehicle that requires more space and time to clear the intersection, increasing the discharge time for Phase 0. It does not represent an emergency or crash event, so it does not trigger emergency preemption or capacity reduction.\nPhase Reasoning: Although Phase 0 has the highest pressure, the presence of the bus necessitates a longer green duration to ensure safe and complete discharge. Phase 1 (Minor Road) has low pressure and is not affected by the bus, making it a secondary option. However, since Phase 0 is the major road and has a critical queue, it must be prioritized for the green phase.\nDuration Reasoning: The CriticalQueue for Phase 0 is Long, requiring a longer duration to ensure complete discharge. The bus further increases the discharge time, so a longer duration is justified.\nBroadcast Notice: Public Bus - Increased discharge time required for upstream major road\n]\nAction: {"phase": 0, "duration": 35}'
                # vlm_thought = None
                
                last_action[jid] = vlm_action

                # 广播事件通知给邻居路口（多路口模式下才有意义）
                if self.is_multi_agent and vlm_response not in ("ERROR", ""):
                    self.bulletin.broadcast(
                        from_jid=jid,
                        vlm_response=vlm_response,
                        green_duration=vlm_action['duration'],
                        current_sumo_step=sumo_sim_step,
                    )

                # ── 阶段5b：确定 best_action，写入样本 ───────────────────
                if self.is_rollout:
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
                else:
                    best_action_dict = vlm_action  # 无 rollout 时直接用 VLM action 作为 best_action
                    best_reward = 0
                    metrics = {}
                    best_phase_id,best_duration = None,None
                    label = "no rollout"  # 无 rollout 时默认 VLM action 即为 best
                
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
            # 这里直接传全量 last_action。
            # 对于异步多路口场景，非决策路口会沿用上一轮的动作配置，由 TransSimHub 在内部忽略/延续。
            final_action = (
                last_action if self.is_multi_agent
                else last_action.get(self.junctions[0], {'phase_id': 0, 'duration': FIXED_TIME_GREEN_DURATION})
            )

            try:
                obs, rewards, truncated, dones, infos, render_json = self.env.step(final_action)
                sumo_sim_step = float(infos.get('step_time', sumo_sim_step))
                # 诊断：记录主轨迹每步后的实际车辆数
                try:
                    _tenv = self.env.unwrapped.tsc_env.tshub_env
                    _nveh = len(_tenv.sumo.vehicle.getIDList())
                    logger.info(f"[Golden] 主轨迹 env.step 后: sumo_t={sumo_sim_step:.0f}s | 车辆数={_nveh}")
                except Exception:
                    pass
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
        self._shutdown_rollout_executor()
        if os.path.exists(self._state_file):
            try:
                os.remove(self._state_file)
            except OSError:
                pass
        if os.path.exists(self._rollout_worker_root):
            try:
                shutil.rmtree(self._rollout_worker_root)
            except OSError as e:
                logger.warning(f"[GOLDEN] 删除 rollout worker 临时目录失败: {e}")
        if self.temp_cfg_path and os.path.exists(self.temp_cfg_path):
            try:
                os.remove(self.temp_cfg_path)
                logger.info(f"[GOLDEN] 已删除临时配置: {self.temp_cfg_path}")
            except OSError as e:
                logger.warning(f"[GOLDEN] 删除临时配置失败: {e}")
        env_obj = getattr(self, 'env', None)
        if env_obj is not None:
            try:
                env_obj.close()
            except Exception:
                pass
            self.env = None


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
        "--route_file", "-r", type=str, default="data/raw/Hongkong_YMT/env/YMT_bus.rou.xml",
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
        "--is_rollout",  type=bool, default=True,
        help="是否执行 rollout 评估候选动作，默认 True（执行）"
    )
    parser.add_argument(
        "--rollout_follow_steps", "-rfs", type=int, default=1,
        help="候选动作之后额外执行的 FixedTime 步数（用于多步 rollout 评估），默认 2"
    )
    parser.add_argument(
        "--rollout_num_workers", "-rnw", type=int, default=None,
        help="rollout 并行 worker 数量；不传则默认为候选动作数（即一个候选动作一个 worker 任务）"
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
        rollout_num_workers=args.rollout_num_workers,
        is_rollout=args.is_rollout,
        api_url=args.api_url,
        model_name_override=args.model_name,
    )
    generator.generate(
        max_sumo_seconds=args.max_sumo_seconds,
        warmup_seconds=args.warmup_seconds,
    )
