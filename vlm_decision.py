'''
Author: yufei Ji
Date: 2026-01-12 16:49:26
LastEditTime: 2026-04-13 22:08:28
Description: 端到端 LVLM 交通信号控制评测主脚本，支持 VLM / FixedTime / MaxPressure 三种模式
FilePath: /VLMTraffic/vlm_decision.py
'''
import os
import re
import copy
import cv2
import json
import time
from loguru import logger
import sys

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from utils.make_tsc_env import make_env
from utils.tools import save_to_json, create_folder, append_response_to_file, convert_rgb_to_bgr, write_response_to_file
from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from configs.model_config import MODEL_CONFIG
from src.inference.vlm_agent import VLMAgent
from configs.prompt_builder import PromptBuilder
from src.evaluation.metrics import MetricsCalculator
from scripts.add_lane_watermarks import add_lane_watermarks

import argparse
import shutil

class Evaluator:
    """
    端到端评测主类，支持三种运行模式：
      - VLM 模式（默认）：调用视觉语言模型进行信号决策
      - FixedTime 模式（--fixed_time）：固定配时，用于传统基线对比
      - MaxPressure 模式（--max_pressure）：基于相位占有率的 MaxPressure 算法基线
    """
    def __init__(self, scenario_key="JiNan", log_dir="./log/eval_results", route_file=None,
                 batch_size=12, use_fixed_time=False, use_max_pressure=False,
                 api_url=None, model_name_override=None):
        self.scenario_key = scenario_key
        self.log_dir = log_dir
        self.use_fixed_time = use_fixed_time
        self.use_max_pressure = use_max_pressure
        self.api_url = api_url
        self.model_name_override = model_name_override

        # --- 1. Load Configurations ---
        path_convert = get_abs_path(__file__)
        self.scenario_config = SCENARIO_CONFIGS.get(scenario_key)
        if not self.scenario_config:
            logger.error(f"[EVAL] Scenario {scenario_key} not found in SCENARIO_CONFIGS")
            raise ValueError(f"Scenario {scenario_key} not found in SCENARIO_CONFIGS")

        self.scenario_name = self.scenario_config["SCENARIO_NAME"]
        self.junction_name = self.scenario_config["JUNCTION_NAME"]
        # 处理交叉口列表长度，约束 batch_size
        num_junctions = len(self.junction_name) if isinstance(self.junction_name, list) else 1
        self.batch_size = min(batch_size, num_junctions)
        logger.info(f"[EVAL] Concurrency set to {self.batch_size} (Requested: {batch_size}, Max Junctions: {num_junctions})")

        # Route File Handling：根据运行模式确定 model_name（用于日志目录和 CSV 写入）
        if self.use_max_pressure:
            model_name = "max_pressure"
        elif self.use_fixed_time:
            model_name = "fixed_time"
        else:
            model_name = self.model_name_override if self.model_name_override else MODEL_CONFIG.get(MODEL_CONFIG.get("api_type", "local_model"), {}).get("model_name", "N/A")
            
        route_name = "default"
        if route_file:
            route_name = os.path.splitext(os.path.basename(route_file))[0]
        
        # Update Log Path
        self.logger_path = os.path.join(self.log_dir, self.scenario_key, route_name, model_name)
        create_folder(self.logger_path) # Create log folder
        set_logger(self.logger_path, terminal_log_level='INFO')
        
        logger.info(f"[EVAL] Logging initialized at {self.logger_path}")
        
        # Determine file paths
        base_sumo_cfg = path_convert(f"data/raw/{self.scenario_name}/{self.scenario_config['NETFILE']}.sumocfg")
        scenario_glb_dir = path_convert(f"data/raw/{self.scenario_name}/3d_assets/")
        
        # Output Folder Structure: data/eval/{Scenario}/{RouteName}/{Model_name}
        self.output_folder = path_convert(f"data/eval/{self.scenario_name}/{route_name}/{model_name}/")
        create_folder(self.output_folder)

        trip_info = os.path.join(self.output_folder, "tripinfo.out.xml")
        statistic_output = os.path.join(self.output_folder, "statistic_output.xml")
        summary = os.path.join(self.output_folder, "summary.txt")
        queue_output = os.path.join(self.output_folder, "queue_output.xml")
        
        # Handle Custom Route File
        sumo_cfg = base_sumo_cfg
        if route_file:
            try:
                # We need to create a temporary sumocfg pointing to the new route file
                with open(base_sumo_cfg, 'r') as f:
                    cfg_content = f.read()
                
                new_route_path = f"./env/{os.path.basename(route_file)}" # Assuming route file is in the env folder relative to sumocfg
                cfg_content = re.sub(r'<route-files value="[^"]+"/>', f'<route-files value="{new_route_path}"/>', cfg_content, count=1)
                
                # Save temp config to the same directory as the original to preserve relative paths
                self.temp_cfg_path = os.path.join(os.path.dirname(base_sumo_cfg), f"temp_{route_name}.sumocfg")
                with open(self.temp_cfg_path, 'w') as f:
                    f.write(cfg_content)
                
                sumo_cfg = self.temp_cfg_path
                logger.info(f"[EVAL] Created temporary SUMO config with route {route_file}: {sumo_cfg}")
            except Exception as e:
                logger.error(f"[EVAL] Failed to modify SUMO config for route {route_file}: {e}")
                pass 

        # Checking file existence
        if not os.path.exists(scenario_glb_dir):
             logger.warning(f"[EVAL] Directory not found: {scenario_glb_dir}. Evaluation might fail.")
        
        tls_add = [
            path_convert(f"data/raw/{self.scenario_name}/add/e2.add.xml"),
            path_convert(f"data/raw/{self.scenario_name}/add/tls_programs.add.xml")
        ]

        # Prepare Environment Parameters
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

        # --- MaxPressure 专项优化：禁用 3D 渲染和传感器，大幅降低仿真开销 ---
        # MaxPressure / FixedTime 不依赖 BEV 图像，禁用 3D 渲染和传感器以节省资源
        if self.use_max_pressure:
            logger.info("[EVAL] MaxPressure mode: disabling 3D rendering and sensors for speed.")
            mp_renderer_cfg = copy.deepcopy(self.env_params.get('renderer_cfg') or {})
            mp_renderer_cfg['is_render'] = False          # 关闭 3D 渲染
            self.env_params['renderer_cfg'] = mp_renderer_cfg
            self.env_params['sensor_cfg'] = None          # 关闭所有传感器（不需要图像）
        elif self.use_fixed_time:
            # FixedTime 同样不使用 BEV 图像，禁用渲染可避免不必要的 GPU 开销
            logger.info("[EVAL] FixedTime mode: disabling 3D rendering and sensors for speed.")
            ft_renderer_cfg = copy.deepcopy(self.env_params.get('renderer_cfg') or {})
            ft_renderer_cfg['is_render'] = False
            self.env_params['renderer_cfg'] = ft_renderer_cfg
            self.env_params['sensor_cfg'] = None

        # 保存所有配置参数到日志
        self._log_configurations()

        # --- 2. Initialize Environment ---
        try:
            logger.info(f"[EVAL] Initializing Environment for {self.scenario_name}...")
            self.env = make_env(**self.env_params)()
        except Exception as e:
            logger.critical(f"[EVAL] Failed to create environment: {e}")
            raise e

        # --- 3. Initialize VLM Agent ---
        if self.use_max_pressure:
            logger.info("[EVAL] Running in MAX PRESSURE mode. VLM Agent initialization bypassed.")
            self.agent = None
        elif self.use_fixed_time:
            logger.info("[EVAL] Running in FIXED TIME mode. VLM Agent initialization bypassed.")
            self.agent = None
        else:
            try:
                # Automatically loads from MODEL_CONFIG inside VLMAgent
                logger.info(f"[EVAL] Initializing VLM Agent...")
                agent_kwargs = {}
                if self.api_url:
                    agent_kwargs["url"] = self.api_url
                if self.model_name_override:
                    agent_kwargs["model_name"] = self.model_name_override
                    
                self.agent = VLMAgent(batch_size=self.batch_size, **agent_kwargs) 
            except Exception as e:
                 logger.critical(f"[EVAL] Failed to initialize VLM Agent: {e}")
                 raise e

        self.metrics_calc = MetricsCalculator()

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except (Exception, SystemExit):
                pass

    def _log_configurations(self):
        """Helper to log all configurations with [CFG] tag"""
        logger.info(f"[CFG] === Scenario Configuration ({self.scenario_key}) ===")
        logger.info(f"[CFG] {json.dumps(self.scenario_config, indent=2, sort_keys=True, default=str)}")
        
        logger.info(f"[CFG] === Environment Parameters ===")
        logger.info(f"[CFG] {json.dumps(self.env_params, indent=2, sort_keys=True, default=str)}")
        
        logger.info(f"[CFG] === Model Configuration ===")
        logger.info(f"[CFG] {json.dumps(MODEL_CONFIG, indent=2, sort_keys=True, default=str)}")
        
        logger.info(f"[CFG] ===========================================")

    def run_eval(self, max_decision_step=10):
        
        logger.info(f"[EVAL] Start Evaluation Loop. Output folder: {self.output_folder}")
        
        # Simulation with environment
        self.metrics_calc.reset()
        obs, _info = self.env.reset()
        
        dones, truncated = False, False
        decision_step = 0
        sumo_sim_step = 0

        current_time = time.time()
        
        # Identify Junctions (Single or List)
        junctions = self.junction_name if isinstance(self.junction_name, list) else [self.junction_name]
        is_multi_agent = isinstance(self.junction_name, list)

        # 1. Warm-up Step: Execute one step with default actions to get initial state/images
        logger.info("[EVAL] Executing Warm-up Step to get initial state...")
        
        # Construct initial action
        if is_multi_agent:
            init_action = {jid: 0 for jid in junctions}
        else:
            init_action = 0
            
        try:
            # First step to get initial observation (images)
            obs, rewards, truncated, dones, infos, render_json = self.env.step(init_action)
            sumo_sim_step = infos.get('step_time', -1)
        except Exception as e:
            logger.critical(f"[EVAL] Warm-up step failed: {e}")
            return # Exit evaluation

        while True:
            # Check exit conditions
            if dones or truncated:
                logger.info(f"[EVAL] Episode finished. Dones: {dones}, Truncated: {truncated}")
                break
            if decision_step >= max_decision_step:
                logger.info(f"[EVAL] Reached maximum decision steps: {max_decision_step}. Ending evaluation.")
                break
            
            # Create Step Directory
            try:
                _step_dir = os.path.join(self.output_folder, f"step_{decision_step}")
                os.makedirs(_step_dir, exist_ok=True)
                _render_json_file = os.path.join(_step_dir, 'render.json')
            except OSError as e:
                logger.error(f"[EVAL] Failed to create step directory {_step_dir}: {e}")
                break

            # Save current vehicle/traffic state
            save_to_json(render_json, _render_json_file)

            # --- 阶段 1: 数据收集与降级处理 ---
            action_dict = {}
            inference_tasks = [] # 记录需要送入大模型的任务: (jid, img_path, prompt, phase_id)

            sensor_datas = infos.get('3d_data', {})
            sensor_imgs = sensor_datas.get('image', {})

            # 预先设置默认动作（Fallback），后续成功的推理会将其覆盖
            num_phases = self.scenario_config.get("PHASE_NUMBER", 4)
            fallback_action = decision_step % num_phases

            for jid in junctions:
                action_dict[jid] = fallback_action  # 默认填充 fallback

                # ── MaxPressure 模式：用感应线圈排队车辆数（jam_length_vehicle）计算相位压力 ──
                # 注意：这里使用 render_json（上一步的 TLS 状态）而非 infos['phase_occ']（占有率）
                # jam_length_vehicle 是 e2 检测器上报的排队车辆数，对应真实场景感应线圈能测到的量，
                # 比 last_step_occupancy（连续时间占有率）更贴近实际部署条件。
                if self.use_max_pressure:
                    tls_state = render_json.get('tls', {}).get(jid, {})
                    movement_ids   = tls_state.get('movement_ids', [])
                    jam_veh        = tls_state.get('jam_length_vehicle', [])
                    phase2movements = tls_state.get('phase2movements', {})

                    if movement_ids and jam_veh and phase2movements:
                        # movement → 排队车辆数 的映射
                        movement_queue = {mid: ql for mid, ql in zip(movement_ids, jam_veh)}
                        # 每个相位的压力 = 该相位所有 movement 的排队车辆数之和
                        phase_pressure = {
                            phase_idx: sum(movement_queue.get(mid, 0) for mid in movements)
                            for phase_idx, movements in phase2movements.items()
                        }
                        mp_action = max(phase_pressure, key=phase_pressure.get)
                        action_dict[jid] = mp_action
                        logger.info(
                            f"[MaxPressure] Step {decision_step} | {jid} | "
                            f"pressure={phase_pressure} | selected phase={mp_action}"
                        )
                    else:
                        logger.warning(
                            f"[MaxPressure] jam_length_vehicle 数据缺失 ({jid} step {decision_step})，使用 fallback。"
                        )
                    continue  # MaxPressure 不需要图像，直接进入下一个路口

                # 1. Get Phase Info
                current_phase_id = 0
                if 'tls' in render_json and jid in render_json['tls']:
                    current_phase_id = render_json['tls'][jid]['this_phase_index']

                # 2. Get BEV Image
                bev_image_path = None
                aircraft_jid = f'aircraft_{jid}'
                if sensor_imgs and aircraft_jid in sensor_imgs:
                    try:
                        junction_img_data = sensor_imgs[aircraft_jid].get('aircraft_all')
                        if junction_img_data is not None:
                            raw_bev_path = os.path.join(_step_dir, f"{aircraft_jid}_bev_raw.png")
                            cv2.imwrite(raw_bev_path, convert_rgb_to_bgr(junction_img_data))
                            # 为所有场景添加车道数字水印，具体布局由 scenario_name 自动选择
                            bev_image_path = os.path.join(_step_dir, f"{aircraft_jid}_bev_watermarked.png")
                            add_lane_watermarks(raw_bev_path, bev_image_path,
                                                scenario_name=self.scenario_key)
                    except Exception as e:
                        logger.warning(f"[EVAL] Failed to save/watermark image for {aircraft_jid}: {e}")

                # 3. 如果图像准备就绪，加入待推理队列
                if bev_image_path:
                    prompt = PromptBuilder.build_decision_prompt(
                        current_phase_id=current_phase_id,
                        scenario_name=self.scenario_key
                    )
                    inference_tasks.append((jid, bev_image_path, prompt, current_phase_id))
                else:
                    logger.warning(f"[EVAL] No BEV image available for {jid}, using fallback.")

            # --- 阶段 2: 执行 Batch 推理 ---
            if inference_tasks:
                # 按照约束好的 self.batch_size 切分任务并推理
                for i in range(0, len(inference_tasks), self.batch_size):
                    batch_tasks = inference_tasks[i:i + self.batch_size]
                    b_jids = [t[0] for t in batch_tasks]
                    b_imgs = [t[1] for t in batch_tasks]
                    b_prompts = [t[2] for t in batch_tasks]
                    b_phases = [t[3] for t in batch_tasks]

                    # 调用批量推理接口 或 fixed_time mock 结果
                    if self.use_fixed_time:
                        results = [("ERROR", 0.0, 0, None)] * len(b_imgs)
                    else:
                        results = self.agent.get_batch_decision(b_imgs, b_prompts)

                    # --- 阶段 3: 结果分发与持久化 ---
                    for jid, phase_id, (vlm_resp, latency, decided_action, thought) in zip(b_jids, b_phases, results):
                        # 获取当前路口的 gt_vehicle_counts
                        gt_vehicle_counts = {}
                        aircraft_id = f'aircraft_{jid}'
                        if 'bev_lane_vehicle_counts' in sensor_datas and aircraft_id in sensor_datas['bev_lane_vehicle_counts']:
                            gt_vehicle_counts = sensor_datas['bev_lane_vehicle_counts'][aircraft_id]
                            
                        # 存结果
                        _resp_file = os.path.join(_step_dir, f"response_{jid}.txt")
                        content_to_save = vlm_resp + (f"\n\n[Thinking Process]\n{thought}" if thought else "")
                        content_to_save += f"\n\n[GT Vehicle Counts]\n{json.dumps(gt_vehicle_counts, ensure_ascii=False, indent=4)}"
                        write_response_to_file(file_path=_resp_file, content=content_to_save)

                        # 验证结果
                        match = re.search(r"Action:?\s*\[?(\d+)\]?", vlm_resp, re.IGNORECASE)
                        if vlm_resp != "ERROR" and match:
                            logger.info(f"[EVAL] Step: {decision_step} | Sumo_time {sumo_sim_step} | JID: {jid} | Phase: {phase_id} | Action: {decided_action} | Latency: {latency:.2f}s")
                            action_dict[jid] = decided_action
                        else:
                            log_prefix = "Fixed Time Enforced" if self.use_fixed_time else "VLM Invalid"
                            logger.warning(f"[EVAL] {log_prefix} for {jid} (Resp: {vlm_resp[:30]}...). Fallback: Action {fallback_action}")

            # --- Environment Step ---
            # Prepare final action payload
            if is_multi_agent:
                final_action = action_dict
            else:
                final_action = action_dict.get(self.junction_name, 0)

            try:
                obs, rewards, truncated, dones, infos, render_json = self.env.step(final_action)
                sumo_sim_step = infos.get('step_time', -1)
            except Exception as e:
                 logger.error(f"[EVAL] Environment step failed at {decision_step}: {e}")
                 break
            
            decision_step += 1

        total_time = time.time() - current_time
        logger.info(f"[EVAL] Evaluation completed in {total_time:.2f} seconds.")

        if hasattr(self, 'temp_cfg_path') and os.path.exists(self.temp_cfg_path):
            try:
                os.remove(self.temp_cfg_path)
                logger.info(f"[EVAL] Removed temporary config file: {self.temp_cfg_path}")
            except OSError as e:
                logger.warning(f"[EVAL] Failed to remove temporary config file {self.temp_cfg_path}: {e}")
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM-based Traffic Signal Control Evaluation.")
    parser.add_argument("--scenario",    type=str, default="Hongkong_YMT",
                        help="场景键名 (e.g., JiNan, Hangzhou, Hongkong_YMT)")
    parser.add_argument("--log_dir",     type=str, default="./log/eval_results",
                        help="日志和评测输出目录")
    parser.add_argument("--route_file",  type=str, default="YMT.rou.xml",
                        help="使用的 .rou.xml 路由文件名")
    parser.add_argument("--max_steps",   type=int, default=120,
                        help="最大决策步数 (1h=120, 24h=2880)")

    # 运行模式开关（三选一，默认 VLM 模式）
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fixed_time",    action="store_true",
                            help="固定配时模式，跳过 VLM 推理（传统基线）")
    mode_group.add_argument("--max_pressure",  action="store_true",
                            help="MaxPressure 模式，基于相位占有率最大化压力选相（对比基线）")

    # 允许从命令行覆盖 API 端口和模型名（VLM 模式专用）
    parser.add_argument("--api_url",     type=str, default=None,
                        help="覆盖 model_config 中的 api_url")
    parser.add_argument("--model_name",  type=str, default=None,
                        help="覆盖 model_config 中的 model_name")

    args = parser.parse_args()
    # args.fixed_time = True

    evaluator = Evaluator(
        scenario_key=args.scenario,
        log_dir=args.log_dir,
        route_file=args.route_file,
        use_fixed_time=args.fixed_time,
        use_max_pressure=args.max_pressure,
        api_url=args.api_url,
        model_name_override=args.model_name,
    )
    evaluator.run_eval(max_decision_step=args.max_steps)