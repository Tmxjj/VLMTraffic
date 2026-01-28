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
        
        self.logger_path = os.path.join(self.log_dir, self.scenario_key)
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
        # input
        sumo_cfg = path_convert(f"../../data/raw/{self.scenario_name}/{self.scenario_config['NETFILE']}.sumocfg")
        scenario_glb_dir = path_convert(f"../../data/raw/{self.scenario_name}/3d_assets/")
        tls_add = [
            path_convert(f"../../data/raw/{self.scenario_name}/add/e2.add.xml"),
            path_convert(f"../../data/raw/{self.scenario_name}/add/tls_programs.add.xml")
        ]
        # output
        self.output_dir = path_convert(f"../../data/sft_dataset/{self.scenario_name}/")
        create_folder(self.output_dir)

        trip_info = path_convert(f"../../data/sft_dataset/{self.scenario_name}/tripinfo_golden.out.xml")
        statistic_output = path_convert(f"../../data/sft_dataset/{self.scenario_name}/statistic_output_golden.xml")
        summary = path_convert(f"../../data/sft_dataset/{self.scenario_name}/summary_golden.txt")
        queue_output = path_convert(f"../../data/sft_dataset/{self.scenario_name}/queue_output_golden.xml")

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

        
        
        # Multi-intersection support
        self.junctions = self.junction_name if isinstance(self.junction_name, list) else [self.junction_name]
        self.is_multi_agent = isinstance(self.junction_name, list)

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

    def run_rollout(self, start_state_file, action, target_jid=None):
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
               
        finally:
            # Always restore rendering state
            self.env.unwrapped.tsc_env.is_render = original_render_state

        return rewards

    def generate(self, max_decision_step=20):
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
        bev_images = {} # {jid: image_path}
        current_phases = {} # {jid: phase_index}
        
        # Initial Warm-up Step
        logger.debug(f"[GOLDEN] Executing Warm-up Step...")
        if self.is_multi_agent:
            init_action = {jid: 0 for jid in self.junctions}
        else:
            init_action = 0
            
        try:
            obs, rewards, truncated, dones, infos, render_json = self.env.step(init_action)
        except Exception as e:
            logger.error(f"[GOLDEN] Warm-up failed: {e}")
            return

        while True:
            # Check for termination
            if dones or truncated:
                logger.info(f"[GOLDEN] Simulation finished. Dones: {dones}, Truncated: {truncated}")
                break
            if decision_step >= max_decision_step:
                logger.info(f"[GOLDEN] Reached maximum decision steps: {max_decision_step}.")
                break
            
            # Create Step Directory
            _step_dir = os.path.join(self.output_dir, f"step_{decision_step}")
            create_folder(_step_dir)
            
            # 1. Process Sensor Data (Images & Phases) from Previous Step
            sensor_datas = infos.get('3d_data', {})
            sensor_imgs = sensor_datas.get('image', {})
            
            for jid in self.junctions:
                # Extract Phase
                try:
                    current_phases[jid] = render_json['tls'][jid]['this_phase_index']
                except KeyError:
                    current_phases[jid] = 0
                
                # Extract Image
                
                aircraft_key = f'aircraft_{jid}'
                # Fallback for old single-agent configs if needed, but assuming multi-agent format
                
                img_data = None
                if sensor_imgs:
                    if aircraft_key in sensor_imgs:
                        img_data = sensor_imgs[aircraft_key].get('aircraft_all')
                
                if img_data is not None:
                    img_path = os.path.join(_step_dir, f"{jid}_bev.jpg")
                    cv2.imwrite(img_path, convert_rgb_to_bgr(img_data))
                    bev_images[jid] = img_path
                else:
                    bev_images[jid] = None

            # 2. VLM Inference Loop (Collect all VLM decisions first)
            vlm_results = {} # {jid: {action, thought, ...}}
            
            for jid in self.junctions:
                img_path = bev_images.get(jid)
                phase_id = current_phases.get(jid, 0)
                
                if img_path:
                    prompt = PromptBuilder.build_decision_prompt(current_phase_id=phase_id, scenario_name=self.scenario_key)
                    try:
                        vlm_response, _, vlm_action_idx, native_thought = self.agent.get_decision(img_path, prompt)
                        
                        vlm_results[jid] = {
                            "action": int(vlm_action_idx),
                            "Think_Process": native_thought,
                            "prompt": prompt,
                            "img_path": img_path,
                            "phase": phase_id,
                            "response_raw": vlm_response,
                            "success": True
                        }
                    except Exception as e:
                        logger.warning(f"[GOLDEN] VLM failed for {jid}: {e}")
                        vlm_results[jid] = {"action": 0, "Think_Process": "Error", "img_path": img_path, "phase": phase_id, "success": False}
                else:
                    vlm_results[jid] = {"action": 0, "Think_Process": "No Image", "img_path": None, "phase": phase_id, "success": False}

            # 3. Golden Rollouts (Parallel Evaluation across Junctions)
            # Instead of N x Phases rollouts, we perform Phases rollouts.
            # In each rollout, ALL agents take action 'p'. 
            # We assume local independence for the immediate reward calculation.
            
            # Save Base State ONCE
            self.env.unwrapped.save_state(state_file)

            # Save Checkpoint (every 5 steps)
            if decision_step % 5 == 0:
                state_dir = os.path.join(self.output_dir, "state")
                create_folder(state_dir)
                sim_step = infos.get('step_time', 0)
                # Use copy instead of saving again for efficiency
                ckpt_path = os.path.join(state_dir, f"state_sim_{sim_step}_decsion_step_{decision_step}.xml")
                shutil.copy(state_file, ckpt_path)
                logger.info(f"[GOLDEN] Checkpoint saved: {ckpt_path}")

            wrapper_state_backup = self._save_wrapper_state()
            
            possible_actions = range(self.scenario_config["PHASE_NUMBER"])
            
            # Data structure to hold metrics: {jid: {action_str: metric}}
            all_junction_metrics = {jid: {} for jid in self.junctions}
            
            logger.info(f"[SIM]———————————————————————— rollout start {decision_step}————————————————————————")

            for action_candidate in possible_actions:
                # Construct Joint Action: Broadcast candidate to all agents
                # This tests "What if everyone does action X?"
                # While not testing all combinatorial pairs, it is efficient (O(Phases)) and valid under local independence.
                if self.is_multi_agent:
                    current_rollout_action = {jid: action_candidate for jid in self.junctions}
                else:
                    current_rollout_action = action_candidate
                
                # Restore & Rollout
                self.env.unwrapped.load_state(state_file)
                self._restore_wrapper_state(wrapper_state_backup)
                
                rewards = self.run_rollout(state_file, current_rollout_action)
                
                # Process rewards
                if rewards is not None:
                    if isinstance(rewards, dict):
                        for jid, r in rewards.items():
                            if jid in all_junction_metrics:
                                # Metric = Reward (Maximize Reward)
                                all_junction_metrics[jid][str(action_candidate)] = float(r)
                    else:
                        # Single agent scalar case
                        jid = self.junctions[0]
                        all_junction_metrics[jid][str(action_candidate)] = float(rewards)
            logger.info(f"[SIM]———————————————————————— rollout end ————————————————————————")
            # 4. Process Results & Save Data
            for jid in self.junctions:
                # Check VLM Success
                vlm_info = vlm_results[jid]
                if not vlm_info.get("success", False):
                    continue

                # If invalid image, skip
                if bev_images.get(jid) is None:
                    continue
                
                metrics = all_junction_metrics.get(jid, {})
                if not metrics:
                    logger.warning(f"No metrics for {jid}")
                    continue
                    
                # Find Best Action
                best_action = max(metrics, key=metrics.get) # Maximize metric (reward)
                best_metric = metrics[best_action]
                best_action = int(best_action)
                
                # Compare with VLM
                vlm_info = vlm_results[jid]
                label = "accepted" if int(vlm_info['action']) == best_action else "rejected"
                
                sample = {
                    "image_path": os.path.abspath(vlm_info['img_path']) if vlm_info['img_path'] else "",
                    "current_phase": int(vlm_info['phase']),
                    "prompt": vlm_info.get('prompt', ""),
                    "vlm_think_process": vlm_info['Think_Process'],
                    "vlm_action": int(vlm_info['action']),
                    'vlm_response_raw': vlm_info.get('response_raw', ""),
                    "optimal_action": best_action,
                    "label": label,
                    "metric_val": float(best_metric),
                    "all_metrics": metrics,
                    "scenario": self.scenario_key,
                    "junction_id": jid,
                    "step": decision_step
                }
                
                sample_file = os.path.join(self.output_dir, "dataset.jsonl")
                append_response_to_file(sample_file, json.dumps(sample, indent=4))
                logger.info(f"[GOLDEN] Step {decision_step} | JID {jid}: VLM({vlm_info['action']}) vs Golden({best_action}) -> {label}")

            # 5. Advance Real Simulation (Following VLM/Student Policy)
            self.env.unwrapped.load_state(state_file)
            self._restore_wrapper_state(wrapper_state_backup)

            final_action = {k: v['action'] for k, v in vlm_results.items()} if self.is_multi_agent else vlm_results[self.junctions[0]]['action']
            
            try:
                self.env.unwrapped.tsc_env.is_render = True
                obs, rewards, truncated, dones, infos, render_json = self.env.step(final_action)
            except Exception as e:
                logger.error(f"[GOLDEN] Environment step failed at {decision_step}: {e}")
                break

            decision_step += 1
        
        # Cleanup
        if os.path.exists(state_file):
            os.remove(state_file)
        self.env.close()
        logger.info("[GOLDEN] Generation complete.")

if __name__ == "__main__":
    generator = GoldenGenerator(scenario_key="JiNan_test", log_dir="./log/golden_dataset")
    generator.generate(max_decision_step=10)
