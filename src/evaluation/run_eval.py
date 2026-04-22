'''
Author: yufei Ji
Date: 2026-04-20
Description: 端到端 LVLM 交通信号控制评测主脚本，支持 VLM / FixedTime / MaxPressure 三种模式。

异步决策框架：
  - 每次 env.step() 返回时（任一路口 can_perform_action=True），
    仅对 can_perform_action=True 的路口进行 VLM 推理和图像渲染。
  - 不需要决策的路口沿用上次动作（由 TransSimHub choose_next_phase_with_duration 处理相位计时）。
  - 终止条件以 SUMO 仿真时间（秒）为准：sumo_sim_step >= max_sumo_seconds。

图像输出路径：
  data/eval/{scenario_key}/{route_stem}/{model_name}/{intersection_id}/{sumo_step}/

上下游协同（EventBulletin）：
  - 拓扑由 configs/scenairo_config.py 的 TOPOLOGY 字段静态配置。
  - TTL = green_duration 秒（以 SUMO 仿真时间为单位）。
  - 广播触发条件：VLM CoT 中检测到 "Final Condition: Special"。
'''
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import re
import copy
import cv2
import json
import time
import argparse
from typing import Dict, List

from loguru import logger
from tshub.utils.init_log import set_logger
from utils.make_tsc_env import make_env
from utils.tools import save_to_json, create_folder, write_response_to_file, convert_rgb_to_bgr
from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from configs.model_config import MODEL_CONFIG
from inference.vlm_agent import VLMAgent
from configs.prompt_builder import PromptBuilder
from src.utils.event_bulletin import EventBulletin
from src.utils.tsc_env.tsc_wrapper import GREEN_DURATION_CANDIDATES, FIXED_TIME_GREEN_DURATION

# 合法 scene_type 列表（normal_triple 用于 NewYork 高密度流量变体）
VALID_SCENE_TYPES = [
    "normal", "emergency", "bus", "accident", "debris", "pedestrian", "normal_triple"
]


class Evaluator:
    """
    端到端评测主类，支持三种运行模式：
      - VLM 模式（默认）：调用视觉语言模型进行信号决策（异步，各路口独立绿灯时长）
      - FixedTime 模式（--fixed_time）：固定配时基线（27s+3s黄灯=30s整步）
      - MaxPressure 模式（--max_pressure）：排队最大压力基线

    输出路径：data/eval/{scenario_key}/{route_stem}/{model_name}/{intersection_id}/{sumo_step}/
    """

    def __init__(self, scenario_key="JiNan", log_dir="./log/eval_results", route_file=None,
                 batch_size=12, use_fixed_time=False, use_max_pressure=False,
                 api_url=None, model_name_override=None, scene_type="normal"):
        self.scenario_key = scenario_key
        self.log_dir = log_dir
        self.use_fixed_time = use_fixed_time
        self.use_max_pressure = use_max_pressure
        self.api_url = api_url
        self.model_name_override = model_name_override
        self.scene_type = scene_type

        # --- 1. 加载场景配置 ---
        self.scenario_config = SCENARIO_CONFIGS.get(scenario_key)
        if not self.scenario_config:
            logger.error(f"[EVAL] Scenario '{scenario_key}' not found in SCENARIO_CONFIGS")
            raise ValueError(f"Scenario '{scenario_key}' not found in SCENARIO_CONFIGS")

        self.scenario_name = self.scenario_config["SCENARIO_NAME"]
        self.junction_name = self.scenario_config["JUNCTION_NAME"]
        num_junctions = len(self.junction_name) if isinstance(self.junction_name, list) else 1
        self.batch_size = min(batch_size, num_junctions)
        logger.info(
            f"[EVAL] Concurrency set to {self.batch_size} "
            f"(Requested: {batch_size}, Junctions: {num_junctions})"
        )

        # 根据运行模式确定 model_name（用于目录命名）
        if self.use_max_pressure:
            model_name = "max_pressure"
        elif self.use_fixed_time:
            model_name = "fixed_time"
        else:
            model_name = (
                self.model_name_override
                or MODEL_CONFIG.get(
                    MODEL_CONFIG.get("api_type", "local_model"), {}
                ).get("model_name", "N/A")
            )

        # SUMO 文件路径
        base_sumo_cfg = os.path.join(
            _PROJECT_ROOT, "data", "raw", self.scenario_name,
            f"{self.scenario_config['NETFILE']}.sumocfg"
        )
        scenario_glb_dir = os.path.join(_PROJECT_ROOT, "data", "raw", self.scenario_name, "3d_assets")

        # 处理自定义路由文件
        sumo_cfg = base_sumo_cfg
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
                self.temp_cfg_path = os.path.join(
                    os.path.dirname(base_sumo_cfg), f"temp_{route_stem}.sumocfg"
                )
                with open(self.temp_cfg_path, 'w') as f:
                    f.write(cfg_content)
                sumo_cfg = self.temp_cfg_path
                logger.info(f"[EVAL] Temporary SUMO config created: {sumo_cfg}")
            except Exception as e:
                logger.error(f"[EVAL] Failed to create temp SUMO config: {e}")
        else:
            try:
                with open(base_sumo_cfg, 'r') as f:
                    cfg_content = f.read()
                match = re.search(r'<route-files value="([^"]+)"/>', cfg_content)
                route_file_name = os.path.basename(match.group(1)) if match else "default_route.rou.xml"
            except Exception:
                route_file_name = "default_route.rou.xml"

        route_stem = route_file_name
        if route_stem.endswith(".rou.xml"):
            route_stem = route_stem[:-8]
        elif route_stem.endswith(".xml"):
            route_stem = route_stem[:-4]

        # 日志路径
        self.logger_path = os.path.join(self.log_dir, self.scenario_key, route_stem, model_name)
        create_folder(self.logger_path)
        set_logger(self.logger_path, terminal_log_level='INFO')
        logger.info(f"[EVAL] Log dir: {self.logger_path}")

        # 输出根目录：data/eval/{scenario_key}/{route_stem}/{model_name}/
        # 图像将存储在：{output_folder}/{intersection_id}/{sumo_step}/
        self.output_folder = os.path.join(
            _PROJECT_ROOT, "data", "eval", self.scenario_key, route_stem, model_name
        )
        create_folder(self.output_folder)

        trip_info        = os.path.join(self.output_folder, "tripinfo.out.xml")
        statistic_output = os.path.join(self.output_folder, "statistic_output.xml")
        summary          = os.path.join(self.output_folder, "summary.txt")
        queue_output     = os.path.join(self.output_folder, "queue_output.xml")

        if not os.path.exists(scenario_glb_dir):
            logger.warning(f"[EVAL] 3D assets dir not found: {scenario_glb_dir}")

        tls_add = [
            os.path.join(_PROJECT_ROOT, "data", "raw", self.scenario_name, "add", "e2.add.xml"),
            os.path.join(_PROJECT_ROOT, "data", "raw", self.scenario_name, "add", "tls_programs.add.xml"),
        ]

        self.env_params = {
            'tls_id':           self.junction_name,
            'number_phases':    self.scenario_config["PHASE_NUMBER"],
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

        # MaxPressure / FixedTime 不需要图像，禁用 3D 渲染节省资源
        if self.use_max_pressure:
            logger.info("[EVAL] MaxPressure mode: disabling 3D rendering.")
            mp_cfg = copy.deepcopy(self.env_params.get('renderer_cfg') or {})
            mp_cfg['is_render'] = False
            self.env_params['renderer_cfg'] = mp_cfg
            self.env_params['sensor_cfg'] = None
        elif self.use_fixed_time:
            logger.info("[EVAL] FixedTime mode: disabling 3D rendering.")
            ft_cfg = copy.deepcopy(self.env_params.get('renderer_cfg') or {})
            ft_cfg['is_render'] = False
            self.env_params['renderer_cfg'] = ft_cfg
            self.env_params['sensor_cfg'] = None

        # 本场景的进口道方向列表（T字路口等非标准路口可能少于4个方向）
        self.approach_dirs = self.scenario_config.get("APPROACH_DIRS", ['N', 'E', 'S', 'W'])
        logger.info(f"[EVAL] Approach directions: {self.approach_dirs}")

        self._log_configurations()

        # --- 2. 初始化仿真环境 ---
        try:
            logger.info(f"[EVAL] Initializing environment: {self.scenario_name} ...")
            self.env = make_env(**self.env_params)()
        except Exception as e:
            logger.critical(f"[EVAL] Failed to create environment: {e}")
            raise

        # --- 3. 初始化 VLM Agent ---
        if self.use_max_pressure or self.use_fixed_time:
            logger.info(
                f"[EVAL] Running in "
                f"{'MaxPressure' if self.use_max_pressure else 'FixedTime'} mode. VLM skipped."
            )
            self.agent = None
        else:
            try:
                logger.info("[EVAL] Initializing VLM Agent...")
                agent_kwargs = {}
                if self.api_url:
                    agent_kwargs["url"] = self.api_url
                if self.model_name_override:
                    agent_kwargs["model_name"] = self.model_name_override
                self.agent = VLMAgent(batch_size=self.batch_size, **agent_kwargs)
            except Exception as e:
                logger.critical(f"[EVAL] Failed to initialize VLM Agent: {e}")
                raise

        # 上下游协同广播板（拓扑由场景配置静态注入）
        topology = self.scenario_config.get("TOPOLOGY", {})
        self.bulletin = EventBulletin(topology=topology)

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except (Exception, SystemExit):
                pass

    def _log_configurations(self):
        logger.info(f"[CFG] === Scenario ({self.scenario_key}) ===")
        logger.info(f"[CFG] {json.dumps(self.scenario_config, indent=2, sort_keys=True, default=str)}")
        logger.info(f"[CFG] === Env Params ===")
        logger.info(f"[CFG] {json.dumps(self.env_params, indent=2, sort_keys=True, default=str)}")
        logger.info(f"[CFG] === Model Config ===")
        logger.info(f"[CFG] {json.dumps(MODEL_CONFIG, indent=2, sort_keys=True, default=str)}")
        logger.info("[CFG] ==========================================")

    def _collect_images(self, jid: str, sensor_imgs: dict, step_dir: str,
                        approach_dirs: List[str]) -> List[str]:
        """采集路口 jid 的多视角图像，按 approach_dirs 指定的方向采集进口道和上游各一张。

        图像命名格式：{element_id}.png
        返回已成功保存的图像路径列表（不含 None），顺序为：
          各方向进口道停止线视图（按 approach_dirs 顺序）→ 各方向上游视图（按 approach_dirs 顺序）
        若全部方向均无图像数据则返回空列表。
        """
        image_paths = []

        for prefix in [jid, f'upstream_{jid}']:
            for d in approach_dirs:
                element_id = f'{prefix}_{d}'
                img_data = None
                if element_id in sensor_imgs:
                    img_data = sensor_imgs[element_id].get('junction_front_all')

                if img_data is not None:
                    img_path = os.path.join(step_dir, f"{element_id}.png")
                    try:
                        cv2.imwrite(img_path, convert_rgb_to_bgr(img_data))
                        image_paths.append(img_path)
                    except Exception as e:
                        logger.warning(f"[EVAL] 图像保存失败 {element_id}: {e}")
                else:
                    logger.debug(f"[EVAL] 无图像数据: {element_id}")

        return image_paths

    @staticmethod
    def _parse_phase_duration(vlm_resp: str):
        """从 VLM 响应中解析 (phase_id, green_duration) 二元组。

        支持格式：
          Action: phase=1, duration=25
          Action: phase=1,duration=25
        兼容旧格式：Action: 1（duration 默认 25s）
        返回 (int, int) 或 None（解析失败）。
        """
        if not vlm_resp or vlm_resp == "ERROR":
            return None
        match = re.search(
            r"Action:?\s*phase\s*=\s*(\d+)[,\s]+duration\s*=\s*(\d+)",
            vlm_resp, re.IGNORECASE
        )
        if match:
            return int(match.group(1)), int(match.group(2))
        match_legacy = re.search(r"Action:?\s*\[?(\d+)\]?", vlm_resp, re.IGNORECASE)
        if match_legacy:
            return int(match_legacy.group(1)), 25
        return None

    def run_eval(self, max_sumo_seconds: int = 3600):
        """
        异步决策评测主循环。

        运行逻辑：
          1. env.step() 返回时，由 tsc_wrapper 保证至少一个路口 can_perform_action=True
          2. 从 render_json 中读取各路口 can_perform_action，确定 deciding_jids
          3. 仅对 deciding_jids 渲染图像、构建 Prompt、执行 VLM 推理
          4. 非决策路口的 last_action 保持不变，由 TransSimHub 内部处理相位计时
          5. 以 SUMO 仿真时间（秒）为终止条件
        """
        logger.info(
            f"[EVAL] 异步评测启动 | output={self.output_folder} | "
            f"max_sumo={max_sumo_seconds}s"
        )

        obs, _ = self.env.reset()
        dones, truncated = False, False
        sumo_sim_step = 0.0
        wall_time_start = time.time()

        junctions = (
            self.junction_name if isinstance(self.junction_name, list)
            else [self.junction_name]
        )
        is_multi_agent = isinstance(self.junction_name, list)
        num_phases = self.scenario_config.get("PHASE_NUMBER", 4)

        # 各路口上一次执行的动作；非决策路口沿用此值
        last_action: Dict[str, dict] = {
            jid: {'phase_id': 0, 'duration': FIXED_TIME_GREEN_DURATION}
            for jid in junctions
        }

        # 热身步：推进仿真获取初始状态与图像
        logger.info("[EVAL] 执行热身步...")
        init_action = {jid: 0 for jid in junctions} if is_multi_agent else 0
        try:
            obs, rewards, truncated, dones, infos, render_json = self.env.step(init_action)
            sumo_sim_step = float(infos.get('step_time', 0))
        except Exception as e:
            logger.critical(f"[EVAL] 热身步失败: {e}")
            return

        # ── 主循环 ────────────────────────────────────────────────────────────
        while not (dones or truncated) and sumo_sim_step < max_sumo_seconds:

            # 清理过期事件通知（基于 SUMO 时间）
            self.bulletin.tick(sumo_sim_step)

            sensor_datas = infos.get('3d_data', {})
            sensor_imgs  = sensor_datas.get('image', {})

            # 确定本轮需要决策的路口（can_perform_action=True）
            if is_multi_agent:
                deciding_jids = [
                    jid for jid in junctions
                    if render_json.get('tls', {}).get(jid, {}).get('can_perform_action', False)
                ]
            else:
                deciding_jids = junctions

            if not deciding_jids:
                # 理论上 tsc_wrapper 已保证至少一个，此处为防御性处理
                logger.warning(
                    f"[EVAL] sumo_t={sumo_sim_step:.0f}s: 无路口需要决策，直接推进仿真"
                )
            else:
                logger.info(
                    f"[EVAL] ══════ sumo_t={sumo_sim_step:.0f}s | "
                    f"决策路口 ({len(deciding_jids)}/{len(junctions)}): "
                    f"{deciding_jids} ══════"
                )

            # VLM 模式下的推理任务队列（批量收集后统一推理）
            inference_tasks = []

            for jid in deciding_jids:
                # 图像目录：{output_folder}/{jid}/{sumo_step}/
                jid_step_dir = os.path.join(self.output_folder, jid, f"{int(sumo_sim_step)}")
                os.makedirs(jid_step_dir, exist_ok=True)

                # ── MaxPressure ──
                if self.use_max_pressure:
                    tls_state     = render_json.get('tls', {}).get(jid, {})
                    movement_ids  = tls_state.get('movement_ids', [])
                    jam_veh       = tls_state.get('jam_length_vehicle', [])
                    phase2mov     = tls_state.get('phase2movements', {})
                    if movement_ids and jam_veh and phase2mov:
                        mov_q = {mid: ql for mid, ql in zip(movement_ids, jam_veh)}
                        pressure = {
                            ph: sum(mov_q.get(mid, 0) for mid in mvs)
                            for ph, mvs in phase2mov.items()
                        }
                        mp_phase = max(pressure, key=pressure.get)
                        last_action[jid] = {'phase_id': mp_phase, 'duration': FIXED_TIME_GREEN_DURATION}
                        logger.info(
                            f"[MaxPressure] {jid} | pressure={pressure} | "
                            f"phase={mp_phase} | dur={FIXED_TIME_GREEN_DURATION}s"
                        )
                    else:
                        logger.warning(f"[MaxPressure] {jid}: 数据缺失，沿用上次动作")
                    continue

                # ── FixedTime ──
                if self.use_fixed_time:
                    cur_phase = render_json.get('tls', {}).get(jid, {}).get('this_phase_index', 0)
                    next_phase = (cur_phase + 1) % num_phases
                    last_action[jid] = {'phase_id': next_phase, 'duration': FIXED_TIME_GREEN_DURATION}
                    logger.info(
                        f"[FixedTime] {jid} | phase={next_phase} | dur={FIXED_TIME_GREEN_DURATION}s"
                    )
                    continue

                # ── VLM 模式 ──
                cur_phase = render_json.get('tls', {}).get(jid, {}).get('this_phase_index', 0)
                coord_ctx = self.bulletin.get_context(jid, sumo_sim_step)
                if coord_ctx:
                    logger.info(
                        f"[Bulletin][注入] {jid} | sumo_t={sumo_sim_step:.0f}s | 注入上游协同通知"
                    )

                image_paths = self._collect_images(
                    jid, sensor_imgs, jid_step_dir, self.approach_dirs
                )
                if image_paths:
                    prompt = PromptBuilder.build_decision_prompt(
                        current_phase_id=cur_phase,
                        scenario_name=self.scenario_key,
                        coordination_context=coord_ctx,
                        available_dirs=self.approach_dirs,
                    )
                    inference_tasks.append((jid, image_paths, prompt, cur_phase, jid_step_dir))
                else:
                    logger.warning(f"[EVAL] {jid}: 无可用图像，沿用上次动作")

            # ── Batch VLM 推理 ──
            if inference_tasks:
                for i in range(0, len(inference_tasks), self.batch_size):
                    batch      = inference_tasks[i:i + self.batch_size]
                    b_jids     = [t[0] for t in batch]
                    b_imgs     = [t[1] for t in batch]
                    b_prompts  = [t[2] for t in batch]
                    b_cur_phs  = [t[3] for t in batch]
                    b_dirs     = [t[4] for t in batch]

                    results = self.agent.get_batch_decision(b_imgs, b_prompts)

                    for jid, step_dir, cur_phase, (vlm_resp, latency, _, thought) in zip(b_jids, b_dirs, b_cur_phs, results):
                        # 保存 VLM 响应文本
                        gt_counts = {}
                        if 'bev_lane_vehicle_counts' in sensor_datas:
                            gt_counts = sensor_datas['bev_lane_vehicle_counts'].get(
                                f'aircraft_{jid}', {}
                            )
                        resp_content = vlm_resp
                        if thought:
                            resp_content += f"\n\n[Thinking Process]\n{thought}"
                        resp_content += (
                            f"\n\n[GT Vehicle Counts]\n"
                            f"{json.dumps(gt_counts, ensure_ascii=False, indent=4)}"
                        )
                        write_response_to_file(
                            file_path=os.path.join(step_dir, "response.txt"),
                            content=resp_content
                        )

                        # 解析并校验动作
                        parsed = self._parse_phase_duration(vlm_resp)
                        if parsed is not None:
                            p_id, raw_dur = parsed
                            actual_dur = min(GREEN_DURATION_CANDIDATES, key=lambda x: abs(x - raw_dur))
                            if raw_dur not in GREEN_DURATION_CANDIDATES:
                                logger.warning(
                                    f"[VLM] {jid} | VLM输出时长 {raw_dur}s 不在候选集 "
                                    f"{GREEN_DURATION_CANDIDATES}，已吸附至 {actual_dur}s"
                                )
                            # 防御性校验：确保吸附结果合法
                            assert actual_dur in GREEN_DURATION_CANDIDATES, \
                                f"吸附后时长 {actual_dur}s 仍不在候选集"
                            last_action[jid] = {'phase_id': p_id, 'duration': actual_dur}
                            logger.info(
                                f"[VLM] {jid} | phase={p_id} | duration={actual_dur}s | "
                                f"latency={latency:.2f}s | sumo_t={sumo_sim_step:.0f}s"
                            )
                            # 广播事件通知给邻居路口
                            if is_multi_agent:
                                self.bulletin.broadcast(
                                    from_jid=jid,
                                    vlm_response=vlm_resp,
                                    green_duration=actual_dur,
                                    current_sumo_step=sumo_sim_step,
                                )
                        else:
                            next_phase = (cur_phase + 1) % num_phases
                            last_action[jid] = {'phase_id': next_phase, 'duration': FIXED_TIME_GREEN_DURATION}
                            logger.warning(
                                f"[EVAL] VLM解析失败 | {jid} | "
                                f"resp={str(vlm_resp)[:80]} | 降级为 FixedTime (phase={next_phase}, duration={FIXED_TIME_GREEN_DURATION}s)"
                            )

            # 保存本轮 render_json
            if deciding_jids:
                render_dir = os.path.join(self.output_folder, "render_info")
                os.makedirs(render_dir, exist_ok=True)
                save_to_json(render_json, os.path.join(render_dir, f'render_{int(sumo_sim_step)}.json'))

            # ── 推进仿真（传全量 last_action，非决策路口动作由 TransSimHub 忽略）──
            final_action = last_action if is_multi_agent else last_action.get(self.junction_name, 0)
            try:
                obs, rewards, truncated, dones, infos, render_json = self.env.step(final_action)
                sumo_sim_step = float(infos.get('step_time', sumo_sim_step))
            except Exception as e:
                logger.error(f"[EVAL] env.step 失败 @ sumo_t={sumo_sim_step:.0f}s: {e}")
                break

        # ── 评测结束 ──
        if is_multi_agent:
            self.bulletin.log_topology()

        wall_elapsed = time.time() - wall_time_start
        logger.info(
            f"[EVAL] 评测完成 | 耗时={wall_elapsed:.1f}s | "
            f"最终 SUMO 时间={sumo_sim_step:.0f}s | "
            f"结束原因={'done/truncated' if (dones or truncated) else 'max_sumo_reached'}"
        )

        if hasattr(self, 'temp_cfg_path') and os.path.exists(self.temp_cfg_path):
            try:
                os.remove(self.temp_cfg_path)
                logger.info(f"[EVAL] 已删除临时配置: {self.temp_cfg_path}")
            except OSError as e:
                logger.warning(f"[EVAL] 删除临时配置失败: {e}")

        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="交通信号控制评测主入口（异步决策，支持 VLM / FixedTime / MaxPressure）"
    )
    parser.add_argument("--scenario",    "-sc", type=str, default="France_Massy",
                        help="场景键名 (e.g., JiNan, Hangzhou, Hongkong_YMT, SouthKorea_Songdo, France_Massy)")
    parser.add_argument("--log_dir",     "-l",  type=str, default="./log/eval_results",
                        help="日志输出目录")
    parser.add_argument("--route_file",  "-r",  type=str, default=None,
                        help=".rou.xml 路由文件名（SUMO 将在 env/ 下查找）")
    parser.add_argument("--max_sumo_seconds",   "-n",  type=int, default=3600,
                        help="最大 SUMO 仿真时间（秒），默认 3600s = 1 小时")
    parser.add_argument("--scene_type",  "-st", type=str, default="normal",
                        choices=VALID_SCENE_TYPES,
                        help="场景类型（default: normal）")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fixed_time",   action="store_true", help="固定配时基线")
    mode_group.add_argument("--max_pressure", action="store_true", help="MaxPressure 基线")

    parser.add_argument("--api_url",    type=str, default=None, help="覆盖 model_config 中的 api_url")
    parser.add_argument("--model_name", type=str, default=None, help="覆盖 model_config 中的 model_name")

    args = parser.parse_args()

    evaluator = Evaluator(
        scenario_key=args.scenario,
        log_dir=args.log_dir,
        route_file=args.route_file,
        use_fixed_time=args.fixed_time,
        use_max_pressure=args.max_pressure,
        api_url=args.api_url,
        model_name_override=args.model_name,
        scene_type=args.scene_type,
    )
    evaluator.run_eval(max_sumo_seconds=args.max_sumo_seconds)
