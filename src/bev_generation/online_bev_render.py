'''
Author: yufei Ji
Date: 2026-01-12 16:48:24
LastEditTime: 2026-01-21 22:07:15
Description: this script is used to generate BEV images from 3D TSC env
FilePath: /VLMTraffic/src/bev_generation/online_bev_render.py
'''

import os
import sys

# 1. 修复 OpenGL 版本报错 (必须放在最前面)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

from pyvirtualdisplay import Display
# 启动虚拟显示器
display = Display(visible=0, size=(800, 600))
display.start()

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
scenario_key = "Hangzhou" # Hongkong_YMT, SouthKorea_Songdo, France_Massy
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
    env = make_env(**params)()

    # Simulation with environment
    obs, _info = env.reset()
    time_step = 0

    while True:
        # 固定控制动作，仅为驱动仿真以获取渲染图
        env_action = 0

        # 本步交互
        obs, rewards, truncated, dones, infos, render_json = env.step(env_action)

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
        save_to_json(render_json, _veh_json_file)

        # 保存图片数据
        sensor_data_imgs = sensor_datas['image'] # 获得图片数据
        if sensor_data_imgs is not None:
            for phase_index in range(PHASE_NUMBER):
                image_path = os.path.join(_save_folder, f"./{phase_index}.jpg") # 保存的图像数据
                camera_data = sensor_data_imgs[f"{JUNCTION_NAME}_{phase_index}"]['junction_front_all']
                cv2.imwrite(image_path, convert_rgb_to_bgr(camera_data))
            
            # 空中 BEV 视角（aircraft_all）
            aircraft_img = sensor_data_imgs['junction_cam_1'].get('aircraft_all')
            if aircraft_img is not None:
                bev_aircraft_path = os.path.join(_save_folder, "./bev_aircraft.jpg")
                cv2.imwrite(bev_aircraft_path, convert_rgb_to_bgr(aircraft_img))

        # 结束条件：任意环境完成
        if dones or truncated:
            break

    env.close()