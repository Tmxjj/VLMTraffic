'''
@Author: WANG Maonan
@Date: 2023-09-04 20:43:53
@Description: 信号灯控制环境 (3D)
LastEditTime: 2026-01-13 10:44:21
'''
import gymnasium as gym

from typing import List, Dict, Optional
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment

class TSC3DEnvironment(gym.Env):
    def __init__(self, 
                 sumo_cfg:str, scenario_glb_dir:str,
                 num_seconds:int, tls_ids:List[str], 
                 tls_action_type:str, use_gui:bool=False, 
                 trip_info:str=None, tls_state_add:List=None,
                 renderer_cfg: Optional[dict] = None, sensor_cfg: Optional[dict] = None,
                 is_render: bool = True, #是否渲染
                ) -> None:
        super().__init__()

        # Renderer configuration defaults
        _preset = '720P'
        _resolution = 1.0
        _vehicle_model = 'high'
        _render_mode = 'offscreen'
        _should_count_vehicles = True
        _debuger_print_node = False
        _debuger_spin_camera = False
        if renderer_cfg:
            _preset = renderer_cfg.get('preset', _preset)
            _resolution = renderer_cfg.get('resolution', _resolution)
            _vehicle_model = renderer_cfg.get('vehicle_model', _vehicle_model)
            _render_mode = renderer_cfg.get('render_mode', _render_mode)
            _should_count_vehicles = renderer_cfg.get('should_count_vehicles', _should_count_vehicles)
            _debuger_print_node = renderer_cfg.get('debuger_print_node', _debuger_print_node)
            _debuger_spin_camera = renderer_cfg.get('debuger_spin_camera', _debuger_spin_camera)

        # Sensor configuration defaults
        tls_sensor_types = ['junction_front_all']
        tls_camera_height = 15
        aircraft_cfg = {
            'junction_cam_1': {
                'sensor_types': ['aircraft_all'],
                'height': 50.0,
            }
        }
        if sensor_cfg:
            if 'tls' in sensor_cfg:
                tls_sensor_types = sensor_cfg['tls'].get('sensor_types', tls_sensor_types)
                tls_camera_height = sensor_cfg['tls'].get('tls_camera_height', tls_camera_height)
            if 'aircraft' in sensor_cfg and isinstance(sensor_cfg['aircraft'], dict):
                aircraft_cfg = sensor_cfg['aircraft']

        tls_sensors_map = {tid: {
            'sensor_types': tls_sensor_types,
            'tls_camera_height': tls_camera_height,
        } for tid in tls_ids}

        self.tsc_env = Tshub3DEnvironment(
            sumo_cfg=sumo_cfg,
            is_aircraft_builder_initialized=False, 
            is_vehicle_builder_initialized=True, # 用于获得 vehicle 的 waiting time 来计算 reward
            is_traffic_light_builder_initialized=True,
            tls_ids=tls_ids, 
            trip_info=trip_info, # 输出 tripinfo
            tls_state_add=tls_state_add, # 输出信号灯变化
            num_seconds=num_seconds,
            tls_action_type=tls_action_type,
            use_gui=use_gui,
            is_libsumo=(not use_gui), # 如果不开界面, 就是用 libsumo
            # 用于 TSHubRenderer 渲染的参数 （TransSimHub/tshub/tshub_env3d/vis3d_renderer/tshub_render.py）
            preset = _preset, 
            resolution = _resolution,
            vehicle_model = _vehicle_model, # 车辆加载模型, low 或是 high
            scenario_glb_dir=scenario_glb_dir, # 场景 3D 素材
            render_mode=_render_mode, # 如果设置了 use_render_pipeline, 此时只能是 onscreen
            should_count_vehicles=_should_count_vehicles,
            debuger_print_node=_debuger_print_node,
            debuger_spin_camera=_debuger_spin_camera,
            sensor_config={
                'tls': tls_sensors_map,
                'aircraft': aircraft_cfg,
            },
            is_render = is_render, # 是否渲染
        )

    def reset(self):
        state_infos = self.tsc_env.reset()
        return state_infos
        
    def step(self, action:Dict[str, Dict[str, int]]):
        action = {'tls': action} # 这里只控制 tls 即可
        states, rewards, infos, dones, sensor_data = self.tsc_env.step(action)
        truncated = dones

        return states, rewards, truncated, dones, infos, sensor_data
    
    def close(self) -> None:
        self.tsc_env.close()