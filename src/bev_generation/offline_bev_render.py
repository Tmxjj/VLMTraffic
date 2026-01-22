'''
Author: yufei Ji
Description: 离线 BEV 渲染器 (安全封装版 + 依赖检测)
'''

import os
import cv2
import json
import sys
import warnings # [新增] 用于发出警告
from pyvirtualdisplay import Display

# 1. 环境设置
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

class OfflineBEVGenerator:
    def __init__(self, scenario_key, scenario_glb_dir, sensor_cfg, renderer_cfg):
        """
        初始化纯渲染环境
        """
        # ==================================================================
        # [安全检查] 检测 Panda3D 是否被提前导入
        # ==================================================================
        if 'panda3d.core' in sys.modules or 'direct.showbase.ShowBase' in sys.modules:
            warning_msg = (
                "\n"
                "CRITICAL WARNING: Panda3D modules detected in memory BEFORE Virtual Display initialization!\n"
            )
            print(f"\033[91m{warning_msg}\033[0m") # 使用红色字体打印
            raise RuntimeError("Panda3D imported too early. See warning above.")

      
        print("Initializing Virtual Display inside class...")
        self.display = Display(visible=0, size=(800, 600))
        self.display.start()
        print(f"Virtual Display started on {os.environ.get('DISPLAY')}")

      
        try:
            from tshub.tshub_env3d.vis3d_renderer.tshub_render import TSHubRenderer
        except ImportError as e:
            # 捕获导入错误，提示路径问题
            raise ImportError(f"Failed to import TSHubRenderer. Ensure TSHub is in PYTHONPATH. Error: {e}")

        # 优先使用 renderer_cfg 中的参数
        _preset = renderer_cfg.get('preset', '720P')
        _resolution = renderer_cfg.get('resolution', 1.0)
        _vehicle_model = renderer_cfg.get('vehicle_model', 'low') 
        _render_mode = renderer_cfg.get('render_mode', 'offscreen')

        # 实例化 TSHubRenderer (Panda3D Init 发生在此刻)
        self.renderer = TSHubRenderer(
            simid="offline_renderer",
            scenario_glb_dir=scenario_glb_dir,
            sensor_config=sensor_cfg,
            preset=_preset,
            resolution=_resolution,
            vehicle_model=_vehicle_model, 
            render_mode=_render_mode,
        )
        
        # 初始化渲染器场景
        dummy_state = {'vehicle': {}, 'tls': {}, 'aircraft': {}}
        self.renderer.reset(dummy_state)
        
        self.renderer._showbase_instance.taskMgr.add(
            self.renderer.dummyTask, "dummyTask"
        )

    def render_step(self, full_data_json, save_folder=None):
        # 组装 State 字典
        current_state = {
            'vehicle': full_data_json.get('vehicle', {}),
            'tls': full_data_json.get('tls', {}),
            'aircraft': full_data_json.get('aircraft', {}) 
        }

        # 调用渲染器的 Step
        sensor_data = self.renderer.step(current_state, should_count_vehicles=False)
        
        # 提取并保存图片 (遍历所有相机，自动处理多路口)
        if save_folder:
            if sensor_data:
                for jid, junction_img_dict in sensor_data.items():
                    # jid 通常是 "junction_cam_{tls_id}"
                    junction_img = junction_img_dict.get('aircraft_all')
                    
                    if junction_img is not None:
                        save_path = os.path.join(save_folder, f"{jid}_offline.jpg")
                        aircraft_img = junction_img[:, :, ::-1] # RGB -> BGR
                        cv2.imwrite(save_path, aircraft_img)
            else:
                 print(f"Warning: No images generated for step.")
        
        return sensor_data

    def close(self):
        self.renderer.destroy()
        self.display.stop()

if __name__ == '__main__':
    # 这里的 import 是安全的，因为 __name__ == '__main__' 只有直接运行此脚本时才执行
    from configs.scenairo_config import SCENARIO_CONFIGS
    from tshub.utils.get_abs_path import get_abs_path

    path_convert = get_abs_path(__file__)
    scenario_key = "Hangzhou" 
    
    config = SCENARIO_CONFIGS.get(scenario_key)
    SCENARIO_NAME = config["SCENARIO_NAME"]
    JUNCTION_NAME = config["JUNCTION_NAME"]
    RENDERER_CFG = config.get("RENDERER_CFG", {})
    SENSOR_CFG = config.get("SENSOR_CFG", {})

    # 仿照 tsc_env.py 的逻辑格式化 sensor_cfg
    # { 'aircraft': { 'aircraft_{tid}': { ... } } }
    formatted_sensor_cfg = {}
    if 'aircraft' in SENSOR_CFG:
        aircraft_cfg = SENSOR_CFG['aircraft']
        tls_ids = JUNCTION_NAME if isinstance(JUNCTION_NAME, list) else [JUNCTION_NAME]
        
        aircraft_sensors_map = {f'aircraft_{tid}': aircraft_cfg for tid in tls_ids}
        formatted_sensor_cfg['aircraft'] = aircraft_sensors_map

    scenario_glb_dir = path_convert(f"../../data/raw/{SCENARIO_NAME}/3d_assets/")
    data_root_folder = path_convert(f"../../data/test/{SCENARIO_NAME}/") 
    
    print(f"Start Offline Rendering for Scenario: {SCENARIO_NAME}")

    renderer = OfflineBEVGenerator(
        scenario_key=scenario_key,
        scenario_glb_dir=scenario_glb_dir,
        sensor_cfg=formatted_sensor_cfg,
        renderer_cfg=RENDERER_CFG
    )

    if not os.path.exists(data_root_folder):
        print(f"Error: Data folder {data_root_folder} does not exist.")
    else:
        step_folders = [
            int(f) for f in os.listdir(data_root_folder) 
            if os.path.isdir(os.path.join(data_root_folder, f)) and f.isdigit()
        ]
        step_folders.sort()

        for time_step in step_folders:
            json_path = os.path.join(data_root_folder, str(time_step), 'data.json')
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        full_data = json.load(f)
                    
                    # 修改：传入 folder 路径，render_step 内部决定文件名
                    save_step_folder = os.path.join(data_root_folder, str(time_step))
                    renderer.render_step(full_data, save_folder=save_step_folder)
                    
                except Exception as e:
                    print(f"Error processing step {time_step}: {e}")
            else:
                print(f"Skipping step {time_step}: data.json not found.")
        
    renderer.close()
    print("Offline Rendering Completed.")