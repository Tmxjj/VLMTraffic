'''
Author: yufei Ji
Date: 2026-01-12 16:49:26
LastEditTime: 2026-01-14 22:20:07
Description: this script is used to 
FilePath: /VLMTraffic/src/evaluation/evaluator.py
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
from utils.tools import save_to_json, create_folder, append_response_to_file, convert_rgb_to_bgr
from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from configs.model_config import MODEL_CONFIG
from inference.vlm_agent import VLMAgent
from configs.prompt_builder import PromptBuilder
from metrics import MetricsCalculator

class Evaluator:
    """
    Runs the end-to-end evaluation loop.
    """
    def __init__(self, scenario_key="Hongkong_YMT", log_dir="./log/eval_results"):
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
        sumo_cfg = path_convert(f"../../data/raw/{self.scenario_name}/{self.scenario_config['NETFILE']}.sumocfg")
        scenario_glb_dir = path_convert(f"../../data/raw/{self.scenario_name}/3d_assets/")
        trip_info = path_convert(f"../../data/test/{self.scenario_name}/tripinfo.out.xml")
        
        # Checking file existence
        required_files = [sumo_cfg, scenario_glb_dir, trip_info]
        for f in required_files:
            if not os.path.exists(f):
                logger.warning(f"[EVAL] File/Directory not found: {f}. Evaluation might fail.")
        
        # Ensure output folder exists (for consistent file structures)
        self.output_folder = path_convert(f"../../data/eval/{self.scenario_name}/")
        create_folder(self.output_folder)
        
        tls_add = [
            path_convert(f"../../data/raw/{self.scenario_name}/add/e2.add.xml"),
            path_convert(f"../../data/raw/{self.scenario_name}/add/tls_programs.add.xml")
        ]

        # Prepare Environment Parameters
        self.env_params = {
            'tls_id': self.junction_name,
            'number_phases': self.scenario_config["PHASE_NUMBER"],
            'sumo_cfg': sumo_cfg,
            'scenario_glb_dir': scenario_glb_dir,
            'trip_info': trip_info,
            'tls_state_add': tls_add,
            'use_gui': False, # Forced False for evaluation
            'renderer_cfg': self.scenario_config.get("RENDERER_CFG"),
            'sensor_cfg': self.scenario_config.get("SENSOR_CFG"),
            'tshub_env_cfg': TSHUB_ENV_CONFIG,
        }

        # [新增] 保存所有配置参数到日志
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

    def run_eval(self, max_decsion_step=10):
        
        logger.info(f"[EVAL] Start Evaluation Loop. Output folder: {self.output_folder}")
        
        # Simulation with environment
        decsion_step = 0 
        BEV_image_path = None # BEV图像路径
        current_phase_id = None # 初始相位为None，标记未初始化

            
        self.metrics_calc.reset()
        obs, _info = self.env.reset()
        
        dones, truncated = False, False
        decsion_step = 0
        sumo_sim_step = 0

        current_time = time.time()

        while True:
            
            action = 0 # 初始动作
            
            # Create Step Directory
            try:
                _step_dir = os.path.join(self.output_folder, f"step_{decsion_step}")
                os.makedirs(_step_dir, exist_ok=True)
                _render_json_file = os.path.join(_step_dir, 'render.json') # 渲染数据
                _response_txt_file = os.path.join(_step_dir, 'response.txt') # LLM 回复
            except OSError as e:
                logger.error(f"[EVAL] Failed to create step directory {_step_dir}: {e}")
                break

            if BEV_image_path is None or current_phase_id is None:
                logger.debug(f"[EVAL] Step {decsion_step}: Initializing / Warm-up step.")
                
                # 初始化步骤，获取初始BEV图像和相位
                try:
                    obs, rewards, truncated, dones, infos, render_json = self.env.step(action)
                except Exception as e:
                     logger.error(f"[EVAL] Environment step failed at init: {e}")
                     break
                     
                sumo_sim_step = infos.get('step_time', -1)
                
                # 保存车辆 JSON 数据
                save_to_json(render_json, _render_json_file)

                # 保存图片数据 
                sensor_datas = infos.get('3d_data', {})
                sensor_data_imgs = sensor_datas.get('image') # 获得图片数据
                
                if sensor_data_imgs:
                    # 空中 BEV 视角（aircraft_all）
                    aircraft_sensor = sensor_data_imgs.get('junction_cam_1', {})
                    if aircraft_sensor:
                        aircraft_img = aircraft_sensor.get('aircraft_all')
                        if aircraft_img is not None:
                            BEV_image_path = os.path.join(_step_dir, "./bev_aircraft.jpg")
                            try:
                                cv2.imwrite(BEV_image_path, convert_rgb_to_bgr(aircraft_img))
                            except Exception as e:
                                logger.warning(f"[EVAL] Failed to save BEV image at init: {e}")
                                BEV_image_path = None # Reset if failed

                # 获取当前相位
                try:
                     current_phase_id = render_json['tls'][self.junction_name]['this_phase_index']
                except KeyError as e:
                     logger.warning(f"[EVAL] Failed to extract phase info at init: {e}. Defaulting to 0.")
                     current_phase_id = 0
                
                decsion_step += 1

            else:
                # VLM决策
                try:
                    prompt = PromptBuilder.build_decision_prompt(
                        current_phase_id=current_phase_id, 
                    )
                    vlm_response, latency, action = self.agent.get_decision(BEV_image_path, prompt)
                    
                    append_response_to_file(file_path=_response_txt_file, content=vlm_response)
                    logger.info(f"[EVAL] RL Decision | Step: {decsion_step} | Sumo: {sumo_sim_step} | Phase: {current_phase_id} | Action: {action} | Latency: {latency:.2f}s")
                    
                except Exception as e:
                    logger.error(f"[EVAL] VLM Inference failed at step {decsion_step}: {e}. Skipping step with default action 0.")
                    vlm_response = "Error"
                    action = 0 # Fallback
       

                # Simulation
                try:
                    obs, rewards, truncated, dones, infos, render_json = self.env.step(action)
                except Exception as e:
                     logger.error(f"[EVAL] Environment step failed at {decsion_step}: {e}")
                     break
                
                decsion_step += 1
                sumo_sim_step = infos.get('step_time', -1)

                # 更新 BEV_image_path 和 current_phase_id
                save_to_json(render_json, _render_json_file)

                # 保存图片数据 
                sensor_datas = infos.get('3d_data', {})
                sensor_data_imgs = sensor_datas.get('image') 
                if sensor_data_imgs:
                    aircraft_sensor = sensor_data_imgs.get('junction_cam_1', {})
                    if aircraft_sensor:
                        aircraft_img = aircraft_sensor.get('aircraft_all')
                        if aircraft_img is not None:
                            BEV_image_path = os.path.join(_step_dir, "./bev_aircraft.jpg")
                            try:
                                cv2.imwrite(BEV_image_path, convert_rgb_to_bgr(aircraft_img))
                            except Exception as e:
                                 logger.warning(f"[EVAL] Failed to save BEV image at step {decsion_step}: {e}")
                
                # 获取当前相位
                try:
                     current_phase_id = render_json['tls'][self.junction_name]['this_phase_index']
                except KeyError:
                     logger.warning(f"[EVAL] Failed to extract phase info at step {decsion_step}")
                     current_phase_id = 0 # Fallback

               
            # 结束条件：任意环境完成
            if dones or truncated:
                logger.info(f"[EVAL] Episode finished. Dones: {dones}, Truncated: {truncated}")
                break
            if decsion_step >= max_decsion_step:
                logger.info(f"[EVAL] Reached maximum decision steps: {max_decsion_step}. Ending evaluation.")
                break
        
        self.env.close()
        total_time = time.time() - current_time
        logger.info(f"[EVAL] Evaluation completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    evaluator = Evaluator(scenario_key="Hongkong_YMT", log_dir="./log/eval_results")
    evaluator.run_eval(max_decsion_step=10)
