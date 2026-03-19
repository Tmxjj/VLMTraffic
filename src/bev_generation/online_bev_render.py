'''
Author: yufei Ji
Date: 2026-01-12 16:48:24
LastEditTime: 2026-03-16 10:46:50
Description: this script is used to generate BEV images from 3D TSC env
FilePath: /VLMTraffic/src/bev_generation/online_bev_render.py
'''

import os
import sys
import time

# 【 EGL 无头模式】：
os.environ['EGL_VISIBLE_DEVICES'] = '0' # 指定使用显卡 0
os.environ.pop('DISPLAY', None)
os.environ.pop('WAYLAND_DISPLAY', None)

# 3D TSC ENV
import re
import cv2
import numpy as np
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from utils.make_tsc_env import make_env
from utils.tools import save_to_json, create_folder, append_response_to_file

def convert_rgb_to_bgr(image):
    # Convert an RGB image to BGR
    return image[:, :, ::-1]

path_convert = get_abs_path(__file__)

# 全局变量
scenario_key = "JiNan" # Hongkong_YMT, SouthKorea_Songdo, France_Massy，Hangzhou，NewYork，JiNan
set_logger(path_convert(f'../../log/{scenario_key}/'))

config = SCENARIO_CONFIGS.get(scenario_key) # 获取特定场景的配置
SCENARIO_NAME = config["SCENARIO_NAME"] # 场景名称
NETFILE = config["NETFILE"] # sumocfg 文件, 加载 eval 文件
JUNCTION_NAME = config["JUNCTION_NAME"] # sumo net 对应的路口 ID
PHASE_NUMBER = config["PHASE_NUMBER"] # 绿灯相位数量
SENSOR_INDEX_2_PHASE_INDEX = config["SENSOR_INDEX_2_PHASE_INDEX"] # 传感器与 Traffic Phase 的对应关系
RENDERER_CFG = config.get("RENDERER_CFG") # 渲染器配置
SENSOR_CFG = config.get("SENSOR_CFG") # 传感器配置

if __name__ == '__main__':
    # #########
    # Init Env
    # #########
    sumo_cfg = path_convert(f"../../data/raw/{SCENARIO_NAME}/{NETFILE}.sumocfg")
    scenario_glb_dir = path_convert(f"../../data/raw/{SCENARIO_NAME}/3d_assets/")
    trip_info = path_convert(f"../../data/test/{SCENARIO_NAME}/tripinfo.out.xml")
    create_folder(path_convert(f"../../data/test/{SCENARIO_NAME}/"))
    tls_add = [
        path_convert(f'../../data/raw/{SCENARIO_NAME}/add/e2.add.xml'), # 探测器
        path_convert(f'../../data/raw/{SCENARIO_NAME}/add/tls_programs.add.xml'), # 信号灯
    ]
    params = {
        'tls_id':JUNCTION_NAME,
        'number_phases':PHASE_NUMBER,
        'sumo_cfg':sumo_cfg,
        'scenario_glb_dir': scenario_glb_dir, # 场景 3D 素材
        'trip_info': trip_info, # 车辆统计信息
        'tls_state_add': tls_add, # 信号灯策略
        'renderer_cfg': RENDERER_CFG,
        'sensor_cfg': SENSOR_CFG,
        'tshub_env_cfg': TSHUB_ENV_CONFIG,
    }
    
    print("Initialize Environment...")
    init_start = time.perf_counter()
    env = make_env(**params)()
    print(f"✅ Environment initialized in {time.perf_counter() - init_start:.4f} seconds.\n")

    # Simulation with environment
    obs, _info = env.reset()
    time_step = 0

    # --- Profiling Variables ---
    prof_env_step = 0.0
    prof_json_save = 0.0
    prof_img_save = 0.0
    prof_loop_total = 0.0

    while True:
        loop_start = time.perf_counter()
        
        # 固定控制动作，仅为驱动仿真以获取渲染图
        if isinstance(JUNCTION_NAME, list):
            env_action = {jid: 0 for jid in JUNCTION_NAME}
        else:
            env_action = 0

        # 本步交互
        t0 = time.perf_counter()
        obs, rewards, truncated, dones, infos, render_json = env.step(env_action)
        prof_env_step += time.perf_counter() - t0

        time_step += 1
        _save_folder = path_convert(f"../../data/test/{SCENARIO_NAME}/{time_step}/")
        create_folder(_save_folder) # 每次交互存储对话
        _veh_json_file = os.path.join(_save_folder, 'data.json') # 车辆数据
        _response_txt_file = os.path.join(_save_folder, 'response.txt') # LLM 回复

        # ##############################
        # 获得并保存传感器的数据 & 车辆 JSON
        # ##############################
        sensor_datas = infos['3d_data']

        # 保存车辆 JSON 数据，用于后续离线渲染
        t1 = time.perf_counter()
        save_to_json(render_json, _veh_json_file)
        prof_json_save += time.perf_counter() - t1

        # 保存图片数据
        t2 = time.perf_counter()
        sensor_data_imgs = sensor_datas['image'] # 获得图片数据
        if sensor_data_imgs is not None:
            # Handle list of junctions or single junction
            junctions = JUNCTION_NAME if isinstance(JUNCTION_NAME, list) else [JUNCTION_NAME]
            
            for jid in sensor_data_imgs:
                # aircraft_all
                bev_junction_path = os.path.join(_save_folder, f"{jid}.png")
                junction_img = sensor_data_imgs[jid].get('aircraft_all')
                if junction_img is not None:
                    cv2.imwrite(bev_junction_path, convert_rgb_to_bgr(junction_img))
                # junction_front_all
                front_junction_path = os.path.join(_save_folder, f"{jid}_front.png")
                front_img = sensor_data_imgs[jid].get('junction_front_all')
                if front_img is not None:
                    cv2.imwrite(front_junction_path, convert_rgb_to_bgr(front_img))
        
        prof_img_save += time.perf_counter() - t2
        prof_loop_total += time.perf_counter() - loop_start

        # 结束条件：任意环境完成
        if dones or truncated:
            break
        if time_step >= 10: # 安全上限，避免死循环
            break # 

    # === 性能分析输出 (不写入 set_logger) ===
    print("\n" + "="*60)
    print("🕒 Online BEV Render Performance Profiling Summary")
    print("="*60)
    print(f"Total Steps Completed : {time_step}")
    if time_step > 0:
        print(f"Total Loop Time       : {prof_loop_total:.4f} s (Avg: {prof_loop_total/time_step:.4f} s/step)")
        print(f"  - Env Step (Sim+3D) : {prof_env_step:.4f} s (Avg: {prof_env_step/time_step:.4f} s/step) [{prof_env_step/prof_loop_total*100:.1f}%]")
        print(f"  - JSON Data Save    : {prof_json_save:.4f} s (Avg: {prof_json_save/time_step:.4f} s/step) [{prof_json_save/prof_loop_total*100:.1f}%]")
        print(f"  - Image Save (cv2)  : {prof_img_save:.4f} s (Avg: {prof_img_save/time_step:.4f} s/step) [{prof_img_save/prof_loop_total*100:.1f}%]")
        other_overhead = prof_loop_total - prof_env_step - prof_json_save - prof_img_save
        print(f"  - Other Overhead    : {other_overhead:.4f} s (e.g., folder IO, logic) [{other_overhead/prof_loop_total*100:.1f}%]")
    print("="*60 + "\n")

    env.close()