'''
@Author: WANG Maonan
@Date: 2023-09-04 20:43:53
@Description: 信号灯控制环境 (3D)
LastEditTime: 2026-01-22 21:51:37
'''
import gymnasium as gym

from typing import List, Dict, Optional
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment

class TSC3DEnvironment(gym.Env):
    def __init__(self, 
                 sumo_cfg:str, scenario_glb_dir:str, tls_ids:List[str], 
                 tls_action_type:str,
                 trip_info:str=None, statistic_output:str=None, 
                 summary:str=None, queue_output:str=None,
                 tls_state_add:List=None,
                 renderer_cfg: Optional[dict] = None, sensor_cfg: Optional[dict] = None,
                 tshub_env_cfg: Optional[dict] = None, # New param
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
        _is_render = True
        if renderer_cfg:
            _preset = renderer_cfg.get('preset', _preset)
            _resolution = renderer_cfg.get('resolution', _resolution)
            _vehicle_model = renderer_cfg.get('vehicle_model', _vehicle_model)
            _render_mode = renderer_cfg.get('render_mode', _render_mode)
            _should_count_vehicles = renderer_cfg.get('should_count_vehicles', _should_count_vehicles)
            _debuger_print_node = renderer_cfg.get('debuger_print_node', _debuger_print_node)
            _debuger_spin_camera = renderer_cfg.get('debuger_spin_camera', _debuger_spin_camera)
            _is_render = renderer_cfg.get('is_render', _is_render)
            _is_every_frame = renderer_cfg.get('is_every_frame', False)

        # Sensor configuration defaults 
        # tls_sensor_types = ['junction_front_all']
        # tls_camera_height = 15
        # aircraft_cfg = {
        #     'junction_cam_1': {
        #         'sensor_types': ['aircraft_all'],
        #         'height': 50.0,
        #     }
        # }

        if sensor_cfg:
            if 'tls' in sensor_cfg:
                tls_sensor_types = sensor_cfg['tls'].get('sensor_types', tls_sensor_types)
                tls_camera_height = sensor_cfg['tls'].get('tls_camera_height', tls_camera_height)
            if 'aircraft' in sensor_cfg and isinstance(sensor_cfg['aircraft'], dict):
                aircraft_cfg = sensor_cfg['aircraft']
                

        # tls_sensors_map = {tid: {
        #     'sensor_types': tls_sensor_types,
        #     'tls_camera_height': tls_camera_height,
        # } for tid in tls_ids}
        aircraft_sensors_map = {f'aircraft_{tid}':aircraft_cfg for tid in tls_ids}

        # Load default TSHub config if provided, else empty dict (will use method defaults if not passed)
        if tshub_env_cfg is None:
            from configs.env_config import TSHUB_ENV_CONFIG
            tshub_env_cfg = TSHUB_ENV_CONFIG.copy()

        self.tsc_env = Tshub3DEnvironment(
            # TshubEnvironment 的参数 (与 SUMO 交互)
            # 1、由脚本调用时传入的参数
            sumo_cfg=sumo_cfg,
            tls_ids=tls_ids, 
            trip_info=trip_info, # Passed from args 
            tls_action_type=tls_action_type, # Passed from args
            tls_state_add=tls_state_add, # Passed from args
            statistic_output= statistic_output,
            summary=summary,
            queue_output=queue_output,

            # 2、由 env_config 提供的参数
            is_map_builder_initialized=tshub_env_cfg.get('is_map_builder_initialized', False),
            is_vehicle_builder_initialized=tshub_env_cfg.get('is_vehicle_builder_initialized', True),
            is_aircraft_builder_initialized=tshub_env_cfg.get('is_aircraft_builder_initialized', True),
            is_traffic_light_builder_initialized=tshub_env_cfg.get('is_traffic_light_builder_initialized', True),
            is_person_builder_initialized=tshub_env_cfg.get('is_person_builder_initialized', True),
            poly_file=tshub_env_cfg.get('poly_file', None),
            osm_file=tshub_env_cfg.get('osm_file', None),
            radio_map_files=tshub_env_cfg.get('radio_map_files', None),
            aircraft_inits=tshub_env_cfg.get('aircraft_inits', None),
            vehicle_action_type=tshub_env_cfg.get('vehicle_action_type', 'lane'),
            hightlight=tshub_env_cfg.get('hightlight', False),
            delta_time=tshub_env_cfg.get('delta_time', 5),
            net_file=tshub_env_cfg.get('net_file', None),
            route_file=tshub_env_cfg.get('route_file', None),
            begin_time=tshub_env_cfg.get('begin_time', 0),
            num_seconds=tshub_env_cfg.get('num_seconds', 20000),
            max_depart_delay=tshub_env_cfg.get('max_depart_delay', 100000),
            time_to_teleport=tshub_env_cfg.get('time_to_teleport', -1),
            sumo_seed=tshub_env_cfg.get('sumo_seed', 'random'),
            tripinfo_output_unfinished=tshub_env_cfg.get('tripinfo_output_unfinished', True),
            collision_action=tshub_env_cfg.get('collision_action', None),
            remote_port=tshub_env_cfg.get('remote_port', None),
            num_clients=tshub_env_cfg.get('num_clients', 1),
            use_gui=tshub_env_cfg.get('use_gui', False),
            is_libsumo=tshub_env_cfg.get('is_libsumo',(not tshub_env_cfg.get('use_gui', False))), # Derived
            
            # 用于 TSHubRenderer 渲染的参数 （TransSimHub/tshub/tshub_env3d/vis3d_renderer/tshub_render.py）
            preset = _preset, 
            resolution = _resolution,
            vehicle_model = _vehicle_model, # 车辆加载模型, low 或是 high
            scenario_glb_dir=scenario_glb_dir, # 场景 3D 素材
            render_mode=_render_mode, # 如果设置了 use_render_pipeline, 此时只能是 onscreen
            should_count_vehicles=_should_count_vehicles,
            debuger_print_node=_debuger_print_node,
            debuger_spin_camera=_debuger_spin_camera,
            is_render = _is_render, # 是否渲染
            is_every_frame = _is_every_frame, # 是否每一帧都渲染

            # 传感器配置
            sensor_config={
                # 'tls': tls_sensors_map,
                'aircraft': aircraft_sensors_map,
            },
        )

    def reset(self):
        state_infos = self.tsc_env.reset()
        return state_infos
        
    def step(self, action:Dict[str, Dict[str, int]]):
        action = {'tls': action} # 这里只控制 tls 即可
        states, rewards, infos, dones, sensor_data = self.tsc_env.step(action)
        truncated = dones

        return states, rewards, truncated, dones, infos, sensor_data
    
    def save_state(self, state_file: str) -> None:
        """保存 SUMO 仿真状态"""
        self.tsc_env.tshub_env._save_state(state_file)

    def load_state(self, state_file: str) -> None:
        """加载 SUMO 仿真状态"""
        self.tsc_env.tshub_env._load_state(state_file)

    def close(self) -> None:
        self.tsc_env.close()