'''
Author: yufei Ji
Date: 2026-04-20
Description: 端到端 LVLM 交通信号控制评测主脚本，支持 VLM / FixedTime / MaxPressure 三种模式。
             从项目根目录的 vlm_decision.py 重构迁入 src/evaluation/。
             新增 --scene_type 参数，输出路径统一为：
               data/eval/{dataset}/{route_file_name}/{method}/
             scene_type 可选: normal | emergency | bus | accident | debris | pedestrian | normal_triple

             上下游协同机制（EventBulletin）：
               - 每个路口 VLM 决策后，若检测到 Special 事件，将事件描述广播至下游路口
               - 拓扑关系从 infos['vehicle_next_tls'] 自动推断（基于车辆 next_tls 字段）
               - 事件通知的过期时间 = 广播时选定的绿灯时长（秒）/ 30（决策步间隔），向上取整
               - 下游路口在构建 Prompt 时，若有活跃通知则注入"6. Upstream Coordination Context"章节
FilePath: /VLMTraffic/src/evaluation/run_eval.py
'''
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import re
import copy
import math
import cv2
import json
import time
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger
from tshub.utils.init_log import set_logger
from utils.make_tsc_env import make_env
from utils.tools import save_to_json, create_folder, write_response_to_file, convert_rgb_to_bgr
from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from configs.prompt_builder import PromptBuilder
from src.utils.event_bulletin import EventBulletin


class Evaluator:
    """
    端到端评测主类，支持三种运行模式：
      - VLM 模式（默认）：调用视觉语言模型进行信号决策
      - FixedTime 模式（--fixed_time）：固定配时，传统基线
      - MaxPressure 模式（--max_pressure）：排队最大压力基线

    输出路径：data/eval/{scenario_key}/{scene_type}/{model_name}/
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
            logger.error(f"[EVAL] Scenario {scenario_key} not found in SCENARIO_CONFIGS")
            raise ValueError(f"Scenario {scenario_key} not found in SCENARIO_CONFIGS")

        self.scenario_name = self.scenario_config["SCENARIO_NAME"]
        self.junction_name = self.scenario_config["JUNCTION_NAME"]
        num_junctions = len(self.junction_name) if isinstance(self.junction_name, list) else 1
        self.batch_size = min(batch_size, num_junctions)
        logger.info(f"[EVAL] Concurrency set to {self.batch_size} (Requested: {batch_size}, Max Junctions: {num_junctions})")

        # 根据运行模式确定 model_name（用于目录命名）
        if self.use_max_pressure:
            model_name = "max_pressure"
        elif self.use_fixed_time:
            model_name = "fixed_time"
        else:
            model_name = (self.model_name_override
                          or MODEL_CONFIG.get(MODEL_CONFIG.get("api_type", "local_model"), {}).get("model_name", "N/A"))

        # 各 SUMO 文件路径（均以项目根目录为基准）
        base_sumo_cfg = os.path.join(
            _PROJECT_ROOT, "data", "raw", self.scenario_name,
            f"{self.scenario_config['NETFILE']}.sumocfg"
        )
        scenario_glb_dir = os.path.join(_PROJECT_ROOT, "data", "raw", self.scenario_name, "3d_assets")

        # 处理自定义路由文件：生成指向该路由的临时 sumocfg
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
                logger.info(f"[EVAL] Created temporary SUMO config: {sumo_cfg}")
            except Exception as e:
                logger.error(f"[EVAL] Failed to modify SUMO config for route {route_file}: {e}")
        else:
            # 未传入则自动解析 sumocfg 中的默认路由文件
            try:
                with open(base_sumo_cfg, 'r') as f:
                    cfg_content = f.read()
                match = re.search(r'<route-files value="([^"]+)"/>', cfg_content)
                if match:
                    route_file_name = os.path.basename(match.group(1))
                else:
                    route_file_name = "default_route.rou.xml"
            except Exception:
                route_file_name = "default_route.rou.xml"
                
        # 提取去掉后缀的 route_file_name 作为目录层级
        route_stem = route_file_name
        if route_stem.endswith(".rou.xml"):
            route_stem = route_stem[:-8]
        elif route_stem.endswith(".xml"):
            route_stem = route_stem[:-4]

        # 日志路径
        self.logger_path = os.path.join(self.log_dir, self.scenario_key, route_stem, model_name)
        create_folder(self.logger_path)
        set_logger(self.logger_path, terminal_log_level='INFO')
        logger.info(f"[EVAL] Logging initialized at {self.logger_path}")

        # 输出路径：data/eval/{scenario_key}/{route_stem}/{model_name}/
        self.output_folder = os.path.join(
            _PROJECT_ROOT, "data", "eval", self.scenario_key, route_stem, model_name, ""
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
            'tls_id': self.junction_name,
            'number_phases': self.scenario_config["PHASE_NUMBER"],
            'sumo_cfg': sumo_cfg,
            'scenario_glb_dir': scenario_glb_dir,
            'trip_info': trip_info,
            'statistic_output': statistic_output,
            'summary': summary,
            'queue_output': queue_output,
            'tls_state_add': tls_add,
            'renderer_cfg': self.scenario_config.get("RENDERER_CFG"),
            'sensor_cfg': self.scenario_config.get("SENSOR_CFG"),
            'tshub_env_cfg': TSHUB_ENV_CONFIG,
        }

        # MaxPressure / FixedTime 不需要图像，禁用 3D 渲染和传感器以节省资源
        if self.use_max_pressure:
            logger.info("[EVAL] MaxPressure mode: disabling 3D rendering and sensors.")
            mp_cfg = copy.deepcopy(self.env_params.get('renderer_cfg') or {})
            mp_cfg['is_render'] = False
            self.env_params['renderer_cfg'] = mp_cfg
            self.env_params['sensor_cfg'] = None
        elif self.use_fixed_time:
            logger.info("[EVAL] FixedTime mode: disabling 3D rendering and sensors.")
            ft_cfg = copy.deepcopy(self.env_params.get('renderer_cfg') or {})
            ft_cfg['is_render'] = False
            self.env_params['renderer_cfg'] = ft_cfg
            self.env_params['sensor_cfg'] = None

        self._log_configurations()

        # --- 2. 初始化仿真环境 ---
        try:
            logger.info(f"[EVAL] Initializing Environment for {self.scenario_name}...")
            self.env = make_env(**self.env_params)()
        except Exception as e:
            logger.critical(f"[EVAL] Failed to create environment: {e}")
            raise e

        # --- 3. 初始化 VLM Agent ---
        if self.use_max_pressure or self.use_fixed_time:
            logger.info(f"[EVAL] Running in {'MaxPressure' if self.use_max_pressure else 'FixedTime'} mode. VLM skipped.")
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
                raise e

        # 上下游协同广播板（仅 VLM 多路口模式下有意义），并注入静态拓扑数据
        topology = self.scenario_config.get("TOPOLOGY", {})
        self.bulletin = EventBulletin(topology=topology)

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except (Exception, SystemExit):
                pass

    def _log_configurations(self):
        logger.info(f"[CFG] === Scenario Configuration ({self.scenario_key}) ===")
        logger.info(f"[CFG] {json.dumps(self.scenario_config, indent=2, sort_keys=True, default=str)}")
        logger.info(f"[CFG] === Environment Parameters ===")
        logger.info(f"[CFG] {json.dumps(self.env_params, indent=2, sort_keys=True, default=str)}")
        logger.info(f"[CFG] === Model Configuration ===")
        logger.info(f"[CFG] {json.dumps(MODEL_CONFIG, indent=2, sort_keys=True, default=str)}")
        logger.info("[CFG] ===========================================")

    # 进口道方向顺序（与 scene_sync 中 _DIRECTION_SHORT 一致：北=0，顺时针）
    _APPROACH_DIRS = ['N', 'E', 'S', 'W']

    def _collect_8_images(self, jid: str, sensor_imgs: dict, step_dir: str):
        """采集该路口的8张图像：4张进口道（停止线处）+ 4张上游道路。

        命名约定（来自 scene_sync.py）：
          进口道：sensor_key = junction_front_all_{jid}_{dir}，e.g. junction_front_all_J1_N
          上游道：sensor_key = junction_front_all_upstream_{jid}_{dir}，e.g. junction_front_all_upstream_J1_N

        返回有序图像路径列表（N/E/S/W 进口道 → N/E/S/W 上游），缺失位置用 None 填充。
        若全部缺失返回空列表。
        """
        image_paths = []
        any_found = False

        for category, prefix in [('approach', jid), ('upstream', f'upstream_{jid}')]:
            for d in self._APPROACH_DIRS:
                element_id = f'{prefix}_{d}'
                sensor_key = f'junction_front_all_{element_id}'
                img_data = None
                # sensor_imgs 结构：{element_id: {sensor_type: ndarray}}
                if element_id in sensor_imgs:
                    img_data = sensor_imgs[element_id].get('junction_front_all')

                if img_data is not None:
                    img_path = os.path.join(step_dir, f"{element_id}.png")
                    try:
                        cv2.imwrite(img_path, convert_rgb_to_bgr(img_data))
                        image_paths.append(img_path)
                        any_found = True
                    except Exception as e:
                        logger.warning(f"[EVAL] 保存图像失败 {element_id}: {e}")
                        image_paths.append(None)
                else:
                    logger.debug(f"[EVAL] 无图像数据: {element_id}")
                    image_paths.append(None)

        return image_paths if any_found else []

    @staticmethod
    def _parse_phase_duration(vlm_resp: str):
        """从 VLM 响应中解析 (phase_id, green_duration) 二元组。

        支持格式：
          Action: phase=1, duration=25
          Action: phase=1,duration=25
          Action: phase=1 duration=25
        返回 (int, int) 或 None（解析失败时）。
        """
        if not vlm_resp or vlm_resp == "ERROR":
            return None
        match = re.search(
            r"Action:?\s*phase\s*=\s*(\d+)[,\s]+duration\s*=\s*(\d+)",
            vlm_resp, re.IGNORECASE
        )
        if match:
            return int(match.group(1)), int(match.group(2))
        # 兼容旧格式：Action: 1（只有相位，duration 取默认值 25s）
        match_legacy = re.search(r"Action:?\s*\[?(\d+)\]?", vlm_resp, re.IGNORECASE)
        if match_legacy:
            return int(match_legacy.group(1)), 25
        return None

    def run_eval(self, max_decision_step=10):
        logger.info(f"[EVAL] Start Evaluation Loop. Output folder: {self.output_folder}")

        obs, _info = self.env.reset()

        dones, truncated = False, False
        decision_step = 0
        sumo_sim_step = 0
        current_time = time.time()

        junctions = self.junction_name if isinstance(self.junction_name, list) else [self.junction_name]
        is_multi_agent = isinstance(self.junction_name, list)

        # 热身步：执行一次默认动作获取初始状态/图像
        logger.info("[EVAL] Executing Warm-up Step to get initial state...")
        init_action = {jid: 0 for jid in junctions} if is_multi_agent else 0
        try:
            obs, rewards, truncated, dones, infos, render_json = self.env.step(init_action)
            sumo_sim_step = infos.get('step_time', -1)
        except Exception as e:
            logger.critical(f"[EVAL] Warm-up step failed: {e}")
            return

        from src.utils.tsc_env.tsc_wrapper import GREEN_DURATION_CANDIDATES, FIXED_TIME_GREEN_DURATION

        while True:
            if dones or truncated:
                logger.info(f"[EVAL] Episode finished. Dones: {dones}, Truncated: {truncated}")
                break
            if decision_step >= max_decision_step:
                logger.info(f"[EVAL] Reached maximum decision steps: {max_decision_step}. Ending evaluation.")
                break

            # ── 步骤日志头 ──
            logger.info(
                f"[EVAL] ══════ Decision Step {decision_step} | SUMO time={sumo_sim_step}s ══════"
            )

            try:
                _step_dir = os.path.join(self.output_folder, f"step_{decision_step}")
                os.makedirs(_step_dir, exist_ok=True)
                _render_json_file = os.path.join(_step_dir, 'render.json')
            except OSError as e:
                logger.error(f"[EVAL] Failed to create step directory {_step_dir}: {e}")
                break

            save_to_json(render_json, _render_json_file)

            # ── 广播板：清理过期通知 ──
            self.bulletin.tick(decision_step)
            
            action_dict = {}
            inference_tasks = []
            # 记录本步每个路口最终选定的绿灯时长，用于广播 TTL 计算
            step_green_durations: Dict[str, int] = {}

            sensor_datas = infos.get('3d_data', {})
            sensor_imgs  = sensor_datas.get('image', {})

            num_phases = self.scenario_config.get("PHASE_NUMBER", 4)
            fallback_phase = decision_step % num_phases
            # 固定配时/MaxPressure：27s 绿灯 + 3s 黄灯 = 30s 整步
            # VLM 回退（解析失败）：候选集中位值 25s
            fallback_duration = FIXED_TIME_GREEN_DURATION if (self.use_fixed_time or self.use_max_pressure) else 25
            fallback_action = {'phase_id': fallback_phase, 'duration': fallback_duration}

            for jid in junctions:
                action_dict[jid] = fallback_action
                step_green_durations[jid] = fallback_duration

                # ── MaxPressure ──
                if self.use_max_pressure:
                    tls_state = render_json.get('tls', {}).get(jid, {})
                    movement_ids    = tls_state.get('movement_ids', [])
                    jam_veh         = tls_state.get('jam_length_vehicle', [])
                    phase2movements = tls_state.get('phase2movements', {})

                    if movement_ids and jam_veh and phase2movements:
                        movement_queue = {mid: ql for mid, ql in zip(movement_ids, jam_veh)}
                        phase_pressure = {
                            phase_idx: sum(movement_queue.get(mid, 0) for mid in movements)
                            for phase_idx, movements in phase2movements.items()
                        }
                        mp_phase = max(phase_pressure, key=phase_pressure.get)
                        # MaxPressure 固定使用 FIXED_TIME_GREEN_DURATION（27s+3s黄灯=30s整步）
                        action_dict[jid] = {'phase_id': mp_phase, 'duration': FIXED_TIME_GREEN_DURATION}
                        step_green_durations[jid] = FIXED_TIME_GREEN_DURATION
                        logger.info(
                            f"[MaxPressure] {jid} | pressure={phase_pressure} | "
                            f"selected phase={mp_phase} | duration={FIXED_TIME_GREEN_DURATION}s"
                        )
                    else:
                        logger.warning(
                            f"[MaxPressure] jam_length_vehicle 数据缺失 ({jid})，使用 fallback"
                        )
                    continue

                # ── VLM / FixedTime：读取协同 Context，采集图像，构建 Prompt ──
                current_phase_id = 0
                if 'tls' in render_json and jid in render_json['tls']:
                    current_phase_id = render_json['tls'][jid]['this_phase_index']

                # 读取来自上游路口的协同事件通知
                coord_ctx = self.bulletin.get_context(jid, decision_step)
                if coord_ctx:
                    logger.info(
                        f"[Bulletin][注入] {jid} 收到上游协同通知，已注入 Prompt:\n{coord_ctx}"
                    )

                image_paths = self._collect_8_images(jid, sensor_imgs, _step_dir)

                if image_paths:
                    prompt = PromptBuilder.build_decision_prompt(
                        current_phase_id=current_phase_id,
                        scenario_name=self.scenario_key,
                        coordination_context=coord_ctx,
                    )
                    inference_tasks.append((jid, image_paths, prompt, current_phase_id))
                else:
                    logger.warning(f"[EVAL] {jid}: 无可用图像，使用 fallback 动作")

            # ── Batch 推理 ──
            if inference_tasks:
                for i in range(0, len(inference_tasks), self.batch_size):
                    batch_tasks = inference_tasks[i:i + self.batch_size]
                    b_jids    = [t[0] for t in batch_tasks]
                    b_imgs    = [t[1] for t in batch_tasks]
                    b_prompts = [t[2] for t in batch_tasks]
                    b_phases  = [t[3] for t in batch_tasks]

                    if self.use_fixed_time:
                        results = [("ERROR", 0.0, None, None)] * len(b_imgs)
                    else:
                        results = self.agent.get_batch_decision(b_imgs, b_prompts)

                    for jid, phase_id, (vlm_resp, latency, decided_action, thought) in zip(b_jids, b_phases, results):
                        gt_vehicle_counts = {}
                        aircraft_id = f'aircraft_{jid}'
                        if 'bev_lane_vehicle_counts' in sensor_datas and aircraft_id in sensor_datas['bev_lane_vehicle_counts']:
                            gt_vehicle_counts = sensor_datas['bev_lane_vehicle_counts'][aircraft_id]

                        _resp_file = os.path.join(_step_dir, f"response_{jid}.txt")
                        content_to_save = vlm_resp + (f"\n\n[Thinking Process]\n{thought}" if thought else "")
                        content_to_save += f"\n\n[GT Vehicle Counts]\n{json.dumps(gt_vehicle_counts, ensure_ascii=False, indent=4)}"
                        write_response_to_file(file_path=_resp_file, content=content_to_save)

                        # 解析动作：phase=X, duration=Y
                        parsed = self._parse_phase_duration(vlm_resp)
                        if vlm_resp != "ERROR" and parsed is not None:
                            p_id, raw_dur = parsed
                            # 吸附到候选集中最近的合法时长
                            actual_dur = min(GREEN_DURATION_CANDIDATES, key=lambda x: abs(x - raw_dur))
                            # 校验：若 VLM 输出值不在候选集，记录警告（吸附后仍保证合法）
                            if raw_dur not in GREEN_DURATION_CANDIDATES:
                                logger.warning(
                                    f"[VLM] {jid} | VLM 输出绿灯时长 {raw_dur}s 不在候选集 "
                                    f"{GREEN_DURATION_CANDIDATES}，已吸附至最近合法值 {actual_dur}s"
                                )
                            # 二次校验：确保吸附结果合法（防御性断言）
                            assert actual_dur in GREEN_DURATION_CANDIDATES, \
                                f"吸附后时长 {actual_dur}s 仍不在候选集 {GREEN_DURATION_CANDIDATES}"
                            # 使用实际秒数（不再传 duration_idx）
                            action_dict[jid] = {'phase_id': p_id, 'duration': actual_dur}
                            step_green_durations[jid] = actual_dur

                            logger.info(
                                f"[VLM] {jid} | phase={p_id} | duration={actual_dur}s | "
                                f"latency={latency:.2f}s | sumo_t={sumo_sim_step}s"
                            )

                            # ── 广播：若为 Special 事件则向邻近路口发送通知 ──
                            # 传入拓扑在初始化时已指定，因此只做简单的参数传递
                            if is_multi_agent and not self.use_fixed_time:
                                self.bulletin.broadcast(
                                    from_jid=jid,
                                    vlm_response=vlm_resp,
                                    green_duration=actual_dur,
                                    current_step=decision_step
                                )
                        else:
                            log_prefix = "FixedTime" if self.use_fixed_time else "VLM解析失败"
                            logger.warning(
                                f"[EVAL] {log_prefix} | {jid} | resp={vlm_resp[:60]}... | "
                                f"使用 fallback: {fallback_action}"
                            )

            # ── 环境推进 ──
            final_action = action_dict if is_multi_agent else action_dict.get(self.junction_name, 0)
            try:
                obs, rewards, truncated, dones, infos, render_json = self.env.step(final_action)
                sumo_sim_step = infos.get('step_time', -1)
            except Exception as e:
                logger.error(f"[EVAL] Environment step failed at decision_step={decision_step}: {e}")
                break

            decision_step += 1

        # 评测结束后打印最终拓扑汇总
        if is_multi_agent:
            logger.info("[Bulletin][最终拓扑汇总]")


        total_time = time.time() - current_time
        logger.info(f"[EVAL] Evaluation completed in {total_time:.2f} seconds.")

        if hasattr(self, 'temp_cfg_path') and os.path.exists(self.temp_cfg_path):
            try:
                os.remove(self.temp_cfg_path)
                logger.info(f"[EVAL] Removed temporary config: {self.temp_cfg_path}")
            except OSError as e:
                logger.warning(f"[EVAL] Failed to remove temporary config: {e}")

        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交通信号控制评测主入口（支持 VLM / FixedTime / MaxPressure）")
    parser.add_argument("--scenario",   "-sc", type=str, default="JiNan",
                        help="场景键名 (e.g., JiNan, Hangzhou, Hongkong_YMT)")
    parser.add_argument("--log_dir",    "-l",  type=str, default="./log/eval_results",
                        help="日志输出目录")
    parser.add_argument("--route_file", "-r",  type=str, default=None,
                        help=".rou.xml 路由文件名（仅文件名，SUMO 将在 env/ 下查找）")
    parser.add_argument("--max_steps",  "-n",  type=int, default=120,
                        help="最大决策步数 (1h≈120)")
    parser.add_argument("--scene_type", "-st", type=str, default="normal",
                        choices=VALID_SCENE_TYPES,
                        help="场景类型，决定输出路径的中间层目录（default: normal）")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fixed_time",   action="store_true",
                            help="固定配时模式（传统基线）")
    mode_group.add_argument("--max_pressure", action="store_true",
                            help="MaxPressure 模式（对比基线）")

    parser.add_argument("--api_url",    type=str, default=None,
                        help="覆盖 model_config 中的 api_url")
    parser.add_argument("--model_name", type=str, default=None,
                        help="覆盖 model_config 中的 model_name")

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
    evaluator.run_eval(max_decision_step=args.max_steps)
