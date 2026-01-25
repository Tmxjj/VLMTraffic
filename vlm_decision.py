'''
Author: yufei Ji
Date: 2026-01-12 16:49:26
LastEditTime: 2026-01-25 19:54:31
Description: this script is used to 
FilePath: /VLMTraffic/vlm_decision.py
'''
import os
import re
import cv2
import json
import time
from loguru import logger

# 修复 OpenGL 版本报错 (必须放在最前面)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

from pyvirtualdisplay import Display
# 启动虚拟显示器
display = Display(visible=0, size=(800, 600))
display.start()

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

class Evaluator:
    """
    Runs the end-to-end evaluation loop.
    """
    def __init__(self, scenario_key="JiNan", log_dir="./log/eval_results"):
        self.scenario_key = scenario_key
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 假设 evaluator 是入口，需要初始化
        self.logger_path = os.path.join(self.log_dir, self.scenario_key)
        set_logger(self.logger_path, terminal_log_level='INFO')
        
        logger.info(f"[EVAL] Logging initialized at {self.logger_path}")
        
        # --- 1. Load Configurations ---
        path_convert = get_abs_path(__file__) 
        self.scenario_config = SCENARIO_CONFIGS.get(scenario_key)
        if not self.scenario_config:
            logger.error(f"[EVAL] Scenario {scenario_key} not found in SCENARIO_CONFIGS")
            raise ValueError(f"Scenario {scenario_key} not found in SCENARIO_CONFIGS")

        self.scenario_name = self.scenario_config["SCENARIO_NAME"]
        self.junction_name = self.scenario_config["JUNCTION_NAME"]
        
        # Determine file paths
        sumo_cfg = path_convert(f"data/raw/{self.scenario_name}/{self.scenario_config['NETFILE']}.sumocfg")
        scenario_glb_dir = path_convert(f"data/raw/{self.scenario_name}/3d_assets/")
        trip_info = path_convert(f"data/test/{self.scenario_name}/tripinfo.out.xml")
        statistic_output = path_convert(f"data/eval/{self.scenario_name}/statistic_output.xml")
        summary = path_convert(f"data/eval/{self.scenario_name}/summary.txt")
        queue_output = path_convert(f"data/eval/{self.scenario_name}/queue_output.xml")
        
        # Checking file existence
        required_files = [sumo_cfg, scenario_glb_dir, trip_info]
        for f in required_files:
            if not os.path.exists(f):
                logger.warning(f"[EVAL] File/Directory not found: {f}. Evaluation might fail.")
        
        # Ensure output folder exists (for consistent file structures)
        self.output_folder = path_convert(f"data/eval/{self.scenario_name}/")
        create_folder(self.output_folder)
        
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
        try:
            # Automatically loads from MODEL_CONFIG inside VLMAgent
            logger.info(f"[EVAL] Initializing VLM Agent...")
            self.agent = VLMAgent() 
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
        # Hide complex objects or long paths if needed, here we just Dump
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

            # --- Decision Making Loop (Multi-Junction) ---
            action_dict = {}
            
            # Get sensor data from previous step (or warmup)
            sensor_datas = infos.get('3d_data', {})
            sensor_imgs = sensor_datas.get('image', {})
            
            for jid in junctions:
                jid_action = 0 # Default action
                
                # 1. Get Phase Info
                current_phase_id = 0
                try:
                    if 'tls' in render_json and jid in render_json['tls']:
                         current_phase_id = render_json['tls'][jid]['this_phase_index']
                    else:
                         logger.warning(f"[EVAL] Phase info missing for {jid}, using 0")
                except Exception as e:
                     logger.warning(f"[EVAL] Error extracting phase for {jid}: {e}")

                # 2. Get BEV Image
                bev_image_path = None
                aircraft_jid = f'aircraft_{jid}'
                if sensor_imgs and aircraft_jid in sensor_imgs:
                    # Logic adapted from online_bev_render.py
                    # Assuming dictionary structure: sensor_imgs[jid]['aircraft_all']
                    try:
                        junction_img_data = sensor_imgs[aircraft_jid].get('aircraft_all')
                        if junction_img_data is not None:
                            bev_image_path = os.path.join(_step_dir, f"{aircraft_jid}_bev.jpg")
                            cv2.imwrite(bev_image_path, convert_rgb_to_bgr(junction_img_data))
                    except Exception as e:
                         logger.warning(f"[EVAL] Failed to save image for {aircraft_jid}: {e}")

                # 3. VLM Agent Decision
                if bev_image_path:
                    try:
                        prompt = PromptBuilder.build_decision_prompt(
                            current_phase_id=current_phase_id, 
                        )
                        # VLM Agent Inference
                        vlm_response, latency, decided_action = self.agent.get_decision(bev_image_path, prompt)
                        
                        # Save Response per Junction
                        _resp_file = os.path.join(_step_dir, f"response_{jid}.txt")
                        write_response_to_file(file_path=_resp_file, content=vlm_response)
                        
                        logger.info(f"[EVAL] Step: {decision_step} | JID: {jid} | Phase: {current_phase_id} | Action: {decided_action} | Latency: {latency:.2f}s")
                        jid_action = decided_action
                        
                    except Exception as e:
                        logger.error(f"[EVAL] VLM Inference failed for {jid} at step {decision_step}: {e}")
                        jid_action = 0 # Fallback
                else:
                    logger.warning(f"[EVAL] No BEV image available for {jid}, skipping VLM")
                
                # Collect Action
                action_dict[jid] = jid_action

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
        
        # BUG:self.env.close() 时，libsumo 会触发底层的 C++ 清理逻辑以关闭仿真。
        # 由于 libsumo 和你的 Python 脚本在同一个进程中，底层的退出或崩溃会直接导致整个 Python 脚本立即终止
        # 方案1: 多进程隔离 (复杂)  -—>代码可读性差 ❌ 
        # 方案2: 使用traci，而不是 libsumo ，但由于sumo版本问题，暂时无法切换 ❌
        # 方案3: 后续单独计算指标 不写在日志中 不随evalutor.py运行✅
        self.env.close()

   
if __name__ == "__main__":
    evaluator = Evaluator(scenario_key="JiNan", log_dir="./log/eval_results")
    evaluator.run_eval(max_decision_step=10)

