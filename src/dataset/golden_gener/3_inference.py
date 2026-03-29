"""
Author: AI Assistant
Date: 2026-03-07
Description: Step 3 - Automate VLM inference based on Ground Truth Lane Analysis. 
This script takes the annotated JSONL, extracts the corrected lane analysis,
builds the new Step 3 prompt, and runs the VLM inference to generate cleaner CoT and Actions.
FilePath: src/dataset/step3_inference.py
"""

import os
import json
import re
import argparse
import sys
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from tshub.utils.init_log import set_logger

# Add project root to sys.path to easily import configs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.prompt_builder_golden_step3 import PromptBuilderStep3
from src.inference.vlm_agent import VLMAgent

def extract_gt_lane_analysis(corrected_raw_text):
    """
    Extracts the previously injected Ground Truth Lane Analysis block from the annotated vlm_response_raw.
    """
    pattern = re.compile(r"- Lane Analysis \(Mandatory\):\s*\n(.*?)(?=\s*(?:- Phase Mapping:|Scene Analysis:))", re.DOTALL)
    match = re.search(pattern, corrected_raw_text)
    if match:
        return match.group(1).strip()
    return None

class Step3Inferencer:
    def __init__(self, input_file, output_file=None):
        self.input_file = input_file
        if output_file is None:
            self.output_file = os.path.join(os.path.dirname(input_file), "03_dataset_reviewed.jsonl")
        else:
            self.output_file = output_file
            
        logger.info(f"Initializing VLMAgent for Step 3...")
        self.agent = VLMAgent() # Uses parameters defined in configs/model_config.py

    def _load_existing_output_index(self):
        existing_index = {}
        if not os.path.exists(self.output_file):
            return existing_index

        try:
            with open(self.output_file, 'r', encoding='utf-8') as fin:
                existing_content = fin.read()
            existing_chunks = re.split(r'\n-----\n+', existing_content)
            for chunk in existing_chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    record = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                key = (record.get("junction_id"), record.get("step"))
                if key[0] is None or key[1] is None:
                    continue
                existing_index[key] = record.get("step3_vlm_response_raw")
        except Exception as e:
            logger.warning(f"Failed to load existing output index from {self.output_file}: {e}")

        logger.info(f"Loaded {len(existing_index)} existing (junction_id, step) records from output.")
        return existing_index
        
    def process(self):
        logger.info(f"Starting Step 3 Inference processing on {self.input_file}")
        
        if not os.path.exists(self.input_file):
            logger.error(f"Input file not found: {self.input_file}")
            return
            
        with open(self.input_file, 'r', encoding='utf-8') as fin:
            content = fin.read()
            
        # Objects separated by "-----" from Step 1/2
        chunks = re.split(r'\n-----\n+', content)

        existing_index = self._load_existing_output_index()
        
        processed_count = 0
        success_count = 0
        
        with open(self.output_file, 'a', encoding='utf-8') as fout:
            for chunk in tqdm(chunks, desc="Processing Step 3 Inference"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                    
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error, skipping chunk: {e}")
                    continue

                sample_key = (data.get("junction_id"), data.get("step"))
                if sample_key[0] is not None and sample_key[1] is not None and sample_key in existing_index:
                    existing_response = existing_index[sample_key]
                    if str(existing_response).strip().upper() != "ERROR":
                        logger.info(
                            f"Skip sample due to existing response in output: junction_id={sample_key[0]}, step={sample_key[1]}"
                        )
                        continue
                
                # Check for required fields
                scenario = data.get("scenario")
                current_phase = data.get("current_phase")
                image_path = data.get("image_path")
                corrected_vlm_raw = data.get("step_2_vlm_response_raw") 
                
                if not all([scenario, current_phase is not None, image_path, corrected_vlm_raw]):
                    fout.write(json.dumps(data, indent=4) + "\n-----\n\n\n\n\n")
                    continue
                    
                # Extract GT Lane Analysis from the corrected block
                gt_lane_analysis = extract_gt_lane_analysis(corrected_vlm_raw)
                
                if not gt_lane_analysis:
                    logger.warning(f"Could not extract GT Lane Analysis for junction {data.get('junction_id')}, step {data.get('step')}.")
                    fout.write(json.dumps(data, indent=4) + "\n-----\n\n\n\n\n")
                    continue
                    
                # Build new prompt
                step3_prompt = PromptBuilderStep3.build_step3_prompt(
                    current_phase_id=current_phase, 
                    scenario_name=scenario, 
                    gt_lane_analysis=gt_lane_analysis
                )
                
                # Run VLM inference
                try:
                    logger.debug(f"Requesting VLM Decision for {image_path}")
                    response, _, vlm_action_idx, native_thought = self.agent.get_decision(image_path, step3_prompt)
                    
                    # Insert GT Lane Analysis into response
                    if "- Phase Mapping:" in response:
                        replacement = f"- Lane Analysis (Mandatory):\n{gt_lane_analysis}\n- Phase Mapping:"
                        response = response.replace("- Phase Mapping:", replacement, 1)
                    
                    # Update fields with the newly generated step 3 response
                    data["step3_prompt"] = step3_prompt
                    data["step3_vlm_response_raw"] = response
                    data["step3_vlm_action"] = vlm_action_idx
            
                    # Evaluate if step 3 succeeded in picking the optimal action
                    optimal_action = data.get("optimal_action")
                    if optimal_action is not None and data["step3_vlm_action"] != -1:
                        data["step3_label"] = "accepted" if data["step3_vlm_action"] == optimal_action else "rejected"
                        
                    success_count += 1
                except Exception as e:
                    logger.error(f"VLM Inference failed for {image_path}: {e}")
                    data["step3_vlm_response_raw"] = "ERROR"
                    data["step3_error"] = str(e)
                
                fout.write(json.dumps(data, indent=4) + "\n-----\n\n\n\n\n")
                processed_count += 1
                
        logger.info(f"Step 3 Processing Complete. Total Processed: {processed_count}. Success Output: {success_count}.")
        logger.info(f"Results saved to {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Step 3 VLM Inference using GT annotated data")
    parser.add_argument("--scenario", type=str, default="JiNan", help="Scenario name (e.g. JiNan)")
    parser.add_argument("--route_file", type=str, default="anon_3_4_jinan_real_incidents.rou", help="Route file name")
    parser.add_argument("--jsonl", type=str, default=None, help="Path to input dataset_annotated.jsonl. If not provided, inferred from scenario and route_file.")
    parser.add_argument("--output", type=str, default=None, help="Path to output jsonl")
    args = parser.parse_args()
    
    # Infer paths if not explicitly provided
    if args.jsonl is None:
        args.jsonl = f"data/sft_dataset/{args.scenario}/{args.route_file}/02_dataset_auto_annotated.jsonl"
        
    route_name = os.path.basename(args.route_file) if args.route_file else "default"
    log_dir = os.path.join(".", "log", "golden_dataset", args.scenario, route_name)
    os.makedirs(log_dir, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_path = os.path.join(log_dir, f"Step3-Inference-{current_time}.log")
    
    set_logger(logger_path, terminal_log_level='INFO')
    logger.info(f"Logging initialized at {logger_path}")
    
    inferencer = Step3Inferencer(args.jsonl, args.output)
    inferencer.process()
