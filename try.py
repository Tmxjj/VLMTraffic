'''
Author: Maonan Wang
Date: 2025-04-23 15:13:54
LastEditTime: 2026-01-14 19:49:40
LastEditors: Please set LastEditors
Description: VLMLight
'''
# AI Agents
import os
import json
from qwen_agent.utils.output_beautify import typewriter_print
from utils.tsc_agent.llm_agents import (
    scene_understanding_agent, # 图像理解 Agent
    scene_analysis_agent, # 场景分析 Agent
    ConcernCaseAgent,
    mode_select_agent,
    llm_cfg
)

# 3D TSC ENV
import re
import cv2
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from utils.make_tsc_env import make_env
from _config import SCENARIO_CONFIGS
from utils.tools import (
    save_to_json, 
    create_folder,
    append_response_to_file,
)

def convert_rgb_to_bgr(image):
    # Convert an RGB image to BGR
    return image[:, :, ::-1]

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def extract_action(response):
    """将回复中的文本转换为 Traffic Phase 进行下发
    """
    match = re.search(r'\d+', response)
    if match:
        return np.array([int(match.group())])
    raise ValueError("No number found in the given string.")


# 全局变量
scenario_key = "Hongkong_YMT" # Hongkong_YMT, SouthKorea_Songdo, France_Massy
config = SCENARIO_CONFIGS.get(scenario_key) # 获取特定场景的配置
SCENARIO_NAME = config["SCENARIO_NAME"] # 场景名称
NETFILE = config["NETFILE"] # sumocfg 文件, 加载 eval 文件
JUNCTION_NAME = config["JUNCTION_NAME"] # sumo net 对应的路口 ID
PHASE_NUMBER = config["PHASE_NUMBER"] # 绿灯相位数量
SENSOR_INDEX_2_PHASE_INDEX = config["SENSOR_INDEX_2_PHASE_INDEX"] # 传感器与 Traffic Phase 的对应关系

# Init Agents
concer_case_decision_agent = ConcernCaseAgent(
    name='concer case decision agent',
    description=(
        'Your task is to make decisions when special vehicles (such as police cars, ambulances, or fire trucks) approach the crossing, '
        'prioritizing their passage while maintaining overall traffic order.'
    ),
    llm_cfg=llm_cfg,
    phase_num=PHASE_NUMBER, # 当前路口存在的相位数量
) 

if __name__ == '__main__':
    # #########
    # Init Env
    # #########
    sumo_cfg = path_convert(f"../sim_envs/{SCENARIO_NAME}/{NETFILE}.sumocfg")
    scenario_glb_dir = path_convert(f"../sim_envs/{SCENARIO_NAME}/3d_assets/")
    trip_info = path_convert(f"./results/{SCENARIO_NAME}/tripinfo.out.xml")
    create_folder(path_convert(f"./results/{SCENARIO_NAME}/"))
    tls_add = [
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/e2.add.xml'), # 探测器
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/tls_programs.add.xml'), # 信号灯
    ]
    params = {
        'tls_id':JUNCTION_NAME,
        'num_seconds':600,
        'number_phases':PHASE_NUMBER,
        'sumo_cfg':sumo_cfg,
        'scenario_glb_dir': scenario_glb_dir, # 场景 3D 素材
        'trip_info': trip_info, # 车辆统计信息
        'tls_state_add': tls_add, # 信号灯策略
        'use_gui':True,
    }
    env = SubprocVecEnv([make_env(**params) for _ in range(1)])
    env = VecNormalize.load(load_path=path_convert(f'../rl_tsc/results/{SCENARIO_NAME}/models/last_vec_normalize.pkl'), venv=env)
    env.training = False # 测试的时候不要更新
    env.norm_reward = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert(f'../rl_tsc/results/{SCENARIO_NAME}/models/last_rl_model.zip')
    model = PPO.load(model_path, env=env, device=device)

    # Simulation with environment
    obs = env.reset()
    sensor_datas = None # 初始传感器数据为空, 
    dones = False # 默认是 False
    time_step = 0

    while not dones:
        rl_action, _state = model.predict(obs, deterministic=True) # RL 获得的动作

        # ##########
        # 新建文件夹 (存储每一个 step 的信息)
        # ##########
        time_step += 1
        _save_folder = path_convert(f"./{SCENARIO_NAME}/{time_step}/")
        create_folder(_save_folder) # 每次交互存储对话
        _veh_json_file = os.path.join(_save_folder, 'data.json') # 车辆数据
        _response_txt_file = os.path.join(_save_folder, 'response.txt') # LLM 回复

        if sensor_datas is None:
            obs, rewards, dones, infos = env.step(rl_action)

            # ##############################
            # 获得并保存传感器的数据 & 车辆 JSON
            # ##############################
            sensor_datas = infos[0]['3d_data']

            # 保存 3D 场景数据
            vehicle_elements = sensor_datas['veh_elements'] # 车辆数据
            save_to_json(vehicle_elements, _veh_json_file)

            # 保存图片数据
            sensor_data = sensor_datas['image'] # 获得图片数据
            for phase_index in range(PHASE_NUMBER):
                image_path = os.path.join(_save_folder, f"./{phase_index}.jpg") # 保存的图像数据
                camera_data = sensor_data[f"{JUNCTION_NAME}_{phase_index}"]['junction_front_all']
                cv2.imwrite(image_path, convert_rgb_to_bgr(camera_data))
                
        else:
            # (1) 保存传感器数据; (2) 场景图片理解; (3) 将图像询问 agents; (3) 使用这里的 decision 与环境交互

            # ##############################
            # 获得并保存传感器的数据 & 车辆 JSON
            # ##############################
            sensor_datas = infos[0]['3d_data']

            # 保存 3D 场景数据
            vehicle_elements = sensor_datas['veh_elements'] # 车辆数据
            save_to_json(vehicle_elements, _veh_json_file)

            # 保存图片数据
            sensor_data = sensor_datas['image'] # 获得图片数据
            for phase_index in range(PHASE_NUMBER):
                image_path = os.path.join(_save_folder, f"./{phase_index}.jpg") # 保存的图像数据
                camera_data = sensor_data[f"{JUNCTION_NAME}_{phase_index}"]['junction_front_all']
                cv2.imwrite(image_path, convert_rgb_to_bgr(camera_data))
            
            # ###############
            # (2) 场景图片理解
            # ###############
            junction_mem = {} # 分别记录多个路口的信息 (一个路口不同方向的信息)
            for scene_index in range(PHASE_NUMBER):
                messages = [] # 对话的历史信息, 这里不同方向是独立的
                image_path = os.path.join(_save_folder, f"./{scene_index}.jpg") # 保存的图像数据
                # 构造多模态输入
                content = [
                    {
                        "text": (
                            "The image shows a traffic intersection from a surveillance camera;" 
                            "describe the current road conditions, including the level of congestion and whether any special vehicles (such as police cars, ambulances, or fire trucks) are present—only identify a vehicle as special if it has clear markings and is approaching the intersection, "
                            "otherwise classify it as a regular vehicle; disregard vehicles that are too far from the intersection or not heading toward it."
                        )
                    }
                ]
                content.append({'image': image_path})
                messages.append({
                    'role': 'user',
                    'content': content  # 多模态的输入
                })  # Append the user query to the chat history.

                # AI Agents 对图片进行解析
                response = []
                response_plain_text = ''
                print(f'-->Scene Understand Agent response for Scene-{scene_index}:')
                for response in scene_understanding_agent.run(messages=messages):
                    response_plain_text = typewriter_print(response, response_plain_text)
                response_plain_text = f'-->Scene Understand Agent response for Scene-{scene_index}:\n' + response_plain_text
                append_response_to_file(file_path=_response_txt_file, content=response_plain_text)
                print('\n---------\n')
                # Append the bot responses to the chat history.
                junction_mem[scene_index] = response[0]['content'] # 每个方向的信息

            # ##################
            # (3) 场景信息分析
            # ##################
            # 组合成每一个 traffic phase 的信息
            junction_description = f"The traffic signal you control has {PHASE_NUMBER} Traffic Phases:\n"
            for scene_index, scene_text in junction_mem.items():
                traffic_phase_index = SENSOR_INDEX_2_PHASE_INDEX[scene_index]
                junction_description += f"• Phase-{traffic_phase_index}: {scene_text}\n"
            
            # 构建询问的信息 --> 总结路口情况, 不按照 traffic phase 总结
            messages = []
            messages.append({
                'role': 'user',
                'content': (
                    "Analyze the entire junction description below to determine ONE of these two conditions:\n"
                    "1. 'Special traffic condition' - ONLY if at least one clearly confirmed emergency vehicle (police/ambulance/fire truck) is present\n"
                    "2. 'Normal traffic condition' - if NO confirmed emergency vehicles are present;\n"
                    "Rules:\n"
                    "- Report ONLY the final condition ('Special traffic condition' or 'Normal traffic condition')\n"
                    "- Consider ONLY visually confirmed emergency vehicles (ignore suspected/unidentified special vehicles)\n"
                    "- Ignore all other factors: congestion levels, regular vehicles, traffic phases\n"
                    "- Default to 'Normal traffic condition' if uncertain\n"
                    f"Junction description: \n{junction_description}"
                )
            })

            response = []
            response_plain_text = ''
            print(f'-->Scene Analysis Agent response:')
            for response in scene_analysis_agent.run(messages=messages):
                response_plain_text = typewriter_print(response, response_plain_text)
            response_plain_text = f'-->Scene Analysis Agent response:\n' + response_plain_text
            append_response_to_file(file_path=_response_txt_file, content=response_plain_text)

            # ###################
            # (4) 快慢分支的选择
            # ###################
            junction_summary = response[0]['content'] # 路口场景总结
            messages = []
            messages.append({
                'role': 'user',
                'content': f"Based on the current status of the junction, select the which agent to use. Now the junction is under \n{junction_summary}."
            })
            response = []
            response_plain_text = ''
            print(f'\n-->Mode Select Agent response:')
            for response in mode_select_agent.run(messages=messages):
                response_plain_text = typewriter_print(response, response_plain_text)
            response_plain_text = f'-->Mode Select Agent response:\n' + response_plain_text
            append_response_to_file(file_path=_response_txt_file, content=response_plain_text)

            # ##############
            # (5) 决策分支
            # ##############
            if 'emergency' in response[-1]['content'].lower():
                messages = []
                messages.append({
                    'role': 'user',
                    'content': f"Based on the current status of each Traffic Phase, please make the decision. \n{junction_description}."
                })
                response = []
                response_plain_text = ''
                print(f'\n-->Concern Case Agent response:')
                for response in concer_case_decision_agent.run(messages=messages):
                    response_plain_text = typewriter_print(response, response_plain_text)
                response_plain_text = f'-->Mode Select Agent response:\n' + response_plain_text
                append_response_to_file(file_path=_response_txt_file, content=response_plain_text)

                # 将 text 转换为 action
                response_dict = json.loads(response[-1]['content'])
                decision, explanation = response_dict['decision'], response_dict['explanation']
                action = extract_action(decision) # 决策转换为动作
            else:
                action = rl_action
                print(f'\n-->Normal Case Agent response:\nRL: {rl_action}')
                
            obs, rewards, dones, infos = env.step(action)

    env.close()