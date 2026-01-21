'''
Author: yufei Ji
Date: 2026-01-19
Description: Generate golden dataset using VLM and simulation rollouts
FilePath: /VLMTraffic/src/dataset/golden_generation.py
'''
import os
import re
import cv2
import json
import time
import copy
import traci
import shutil
from loguru import logger
import numpy as np
from collections import deque

# 修复 OpenGL 版本报错 (必须放在最前面)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

from pyvirtualdisplay import Display
# 启动虚拟显示器
try:
    display = Display(visible=0, size=(800, 600))
    display.start()
except Exception as e:
    logger.warning(f"Failed to start Display: {e}")

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from utils.make_tsc_env import make_env
from utils.tools import save_to_json, create_folder, append_response_to_file, convert_rgb_to_bgr, write_response_to_file
from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from src.inference.vlm_agent import VLMAgent
from configs.prompt_builder import PromptBuilder

class GoldenGenerator:
    def __init__(self, scenario_key="Hongkong_YMT", log_dir="./log/golden_dataset"):
        self.scenario_key = scenario_key
        self.log_dir = log_dir
        create_folder(self.log_dir)
        
        self.logger_path = os.path.join(self.log_dir, "generation.log")
        set_logger(self.logger_path, terminal_log_level='INFO')
        
        logger.info(f"[GOLDEN] Logging initialized at {self.logger_path}")
        
        # --- 1. Load Configurations ---
        path_convert = get_abs_path(__file__) 
        self.scenario_config = SCENARIO_CONFIGS.get(scenario_key)
        if not self.scenario_config:
            raise ValueError(f"Scenario {scenario_key} not found in SCENARIO_CONFIGS")

        self.scenario_name = self.scenario_config["SCENARIO_NAME"]
        self.junction_name = self.scenario_config["JUNCTION_NAME"]
        
        # Determine file paths
        sumo_cfg = path_convert(f"../../data/raw/{self.scenario_name}/{self.scenario_config['NETFILE']}.sumocfg")
        scenario_glb_dir = path_convert(f"../../data/raw/{self.scenario_name}/3d_assets/")
        trip_info = path_convert(f"../../data/test/{self.scenario_name}/tripinfo_golden.out.xml")
        statistic_output = path_convert(f"../../data/eval/{self.scenario_name}/statistic_output_golden.xml")
        summary = path_convert(f"../../data/eval/{self.scenario_name}/summary_golden.txt")
        queue_output = path_convert(f"../../data/eval/{self.scenario_name}/queue_output_golden.xml")
        
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
            'statistic_output': statistic_output,
            'summary': summary,
            'queue_output': queue_output,
            'tls_state_add': tls_add,
            'renderer_cfg': self.scenario_config.get("RENDERER_CFG"),
            'sensor_cfg': self.scenario_config.get("SENSOR_CFG"),
            'tshub_env_cfg': TSHUB_ENV_CONFIG,
            # 'use_gui': False # Force False for automated generation
        }

        self.output_dir = path_convert(f"data/golden_dataset/{self.scenario_name}/")

        # --- 2. Initialize Environment ---
        try:
            logger.info(f"[GOLDEN] Initializing Environment for {self.scenario_name}...")
            self.env_creator = make_env(**self.env_params)
            self.env = self.env_creator() 
            # self.env is TSCEnvWrapper
            
            # Locate SUMO connection
            # self.sumo = self.env.env.tsc_env.tshub_env.sumo # No longer needed, using env methods
            
        except Exception as e:
            logger.critical(f"[GOLDEN] Failed to create environment: {e}")
            raise e
        
        # --- 3. Initialize VLM Agent ---
        try:
            logger.info(f"[GOLDEN] Initializing VLM Agent...")
            self.agent = VLMAgent() 
        except Exception as e:
             logger.critical(f"[GOLDEN] Failed to initialize VLM Agent: {e}")
             raise e

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except (Exception, SystemExit):
                pass

    def _save_wrapper_state(self):
        """Save internal state of TSCEnvWrapper"""
        # TSCEnvWrapper has: self.states (deque), self.occupancy (OccupancyList)
        wrapper_state = {
            "states": list(self.env.states),
            "occupancy_elements": list(self.env.occupancy.elements)
        }
        return wrapper_state

    def _restore_wrapper_state(self, saved_state):
        """Restore internal state of TSCEnvWrapper"""
        self.env.states = deque(saved_state["states"], maxlen=self.env.states.maxlen)
        self.env.occupancy.clear_elements()
        self.env.occupancy.elements = list(saved_state["occupancy_elements"])

    def run_rollout(self, start_state_file, action):
        """
        Loads the state, executes an action by stepping the environment, returns the metric.
        """
        # Disable rendering during rollout to avoid errors and improve performance
        original_render_state = self.env.unwrapped.tsc_env.is_render
        self.env.unwrapped.tsc_env.is_render = False

        try:
            # 1. Load SUMO state
            try:
                self.env.unwrapped.load_state(start_state_file)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return 99999

            # 2. Execute Action
            # Note: self.env.step(action) advances by delta_time until next decision point
            # It calculates reward (negative waiting time)
            # TODO：设计这个reward
            # try:
            obs, rewards, truncated, dones, infos, render_json = self.env.step(action)
                # metric: we want to minimize queue/waiting time.
                # Wrapper returns reward = - total_waiting_time
                # So metric = -reward (positive waiting time)
            metric = -rewards
                
            # except Exception as e:
            #     logger.error(f"Rollout failed: {e}")
            #     metric = 99999
        finally:
            # Always restore rendering state
            self.env.unwrapped.tsc_env.is_render = original_render_state

        return metric

    def generate(self, max_steps=20):
        """
        Main loop to generate golden data.
        """
        logger.info(f"[GOLDEN] Starting generation...")
        
        state_file = os.path.abspath(os.path.join(self.output_dir, "temp_state.xml"))

        # Simulation state initialization
        self.env.reset()
        
        dones, truncated = False, False
        decision_step = 0
        
        # State variables
        BEV_image_path = None 
        current_phase_id = None 

        while True:
            action = 0 # Default action for init
            
            # Check for termination
            if dones or truncated:
                logger.info(f"[GOLDEN] Simulation finished. Dones: {dones}, Truncated: {truncated}")
                break
            if decision_step >= max_steps:
                logger.info(f"[GOLDEN] Reached maximum decision steps: {max_steps}.")
                break

            # Logic Branch: Init vs Decision
            if BEV_image_path is None or current_phase_id is None:
                logger.debug(f"[GOLDEN] Step {decision_step}: Initializing / Warm-up step.")
                
                try:
                    obs, rewards, truncated, dones, infos, render_json = self.env.step(action)
                except Exception as e:
                    logger.error(f"[GOLDEN] Environment step failed at init: {e}")
                    break
                
            else:
                # --- Decision Step ---
                
                # 1. VLM Reasoning (Student)
                vlm_action_idx = -1
                vlm_thought = ""
                prompt = PromptBuilder.build_decision_prompt(current_phase_id=current_phase_id)
                
                try:
                    vlm_thought, _, vlm_action_idx = self.agent.get_decision(BEV_image_path, prompt)
                except Exception as e:
                    logger.warning(f"[GOLDEN] VLM failed: {e}")

                # 2. Rollout (Teacher)
                self.env.unwrapped.save_state(state_file)
                wrapper_state_backup = self._save_wrapper_state()
                
                possible_actions = range(self.scenario_config["PHASE_NUMBER"])
                
                best_action = -1
                best_metric = float('inf')
                action_metrics = {}
                
                for action_candidate in possible_actions:
                    # Restore for each candidate
                    self.env.unwrapped.load_state(state_file)
                    self._restore_wrapper_state(wrapper_state_backup)
                    
                    metric = self.run_rollout(state_file, action_candidate)
                    action_metrics[str(action_candidate)] = float(metric)
                    
                    if metric < best_metric:
                        best_metric = metric
                        best_action = action_candidate
                
                # 3. Restore to State BEFORE Rollout to continue simulation
                self.env.unwrapped.load_state(state_file)
                self._restore_wrapper_state(wrapper_state_backup)

                # 4. Labeling & Saving
                label = "accepted" if int(vlm_action_idx) == int(best_action) else "rejected"
                
                sample = {
                    "image_path": os.path.abspath(BEV_image_path),
                    "current_phase": int(current_phase_id),
                    "prompt": prompt,
                    "vlm_output_raw": vlm_thought,
                    "vlm_action": int(vlm_action_idx),
                    "optimal_action": int(best_action),
                    "label": label,
                    "metric_val": float(best_metric),
                    "all_metrics": action_metrics,
                    "scenario": self.scenario_key,
                    "step": decision_step
                }
                
                sample_file = os.path.join(self.output_dir, "dataset.jsonl")
                append_response_to_file(sample_file, json.dumps(sample))
                logger.info(f"Step {decision_step}: VLM({vlm_action_idx}) vs Golden({best_action}) -> {label}")
                
                # 5. Advance Real Simulation (Expert Policy)
                action = vlm_action_idx
                try:
                    # 重新开启渲染，确保主循环画面正常
                    self.env.unwrapped.tsc_env.is_render = True
                    obs, rewards, truncated, dones, infos, render_json = self.env.step(action)
                except Exception as e:
                    logger.error(f"[GOLDEN] Environment step failed at {decision_step}: {e}")
                    break

            # Post-Step Processing (Extract Info for NEXT Step)
            
            # Update counters
            decision_step += 1
            
            # 1. Save Image
            sensor_datas = infos.get('3d_data', {})
            sensor_data_imgs = sensor_datas.get('image')
            
            if sensor_data_imgs:
                aircraft_sensor = sensor_data_imgs.get('junction_cam_1', {})
                if aircraft_sensor:
                    aircraft_img = aircraft_sensor.get('aircraft_all')
                    if aircraft_img is not None:
                        img_filename = f"step{decision_step}.jpg"
                        BEV_image_path = os.path.join(self.output_dir, img_filename)
                        try:
                            cv2.imwrite(BEV_image_path, convert_rgb_to_bgr(aircraft_img))
                        except Exception as e:
                            logger.warning(f"Failed to save image: {e}")
                            BEV_image_path = None
            
            # 2. Extract Phase ID
            try:
                current_phase_id = render_json['tls'][self.junction_name]['this_phase_index']
            except KeyError:
                logger.warning(f"[GOLDEN] Failed to extract phase info at step {decision_step}. Defaulting to 0.")
                current_phase_id = 0
        
        # Cleanup
        if os.path.exists(state_file):
            os.remove(state_file)
        self.env.close()
        logger.info("[GOLDEN] Generation complete.")

if __name__ == "__main__":
    generator = GoldenGenerator(scenario_key="Hongkong_YMT", log_dir="./data/golden_dataset")
    generator.generate(max_steps=10)
