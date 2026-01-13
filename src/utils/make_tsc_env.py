'''
@Author: WANG Maonan
@Date: 2023-09-08 17:45:54
@Description: 创建 TSC Env + Wrapper
LastEditTime: 2026-01-13 16:47:50
'''
import gymnasium as gym
from typing import List, Optional
from utils.tsc_env.tsc_env import TSC3DEnvironment
from utils.tsc_env.tsc_wrapper import TSCEnvWrapper

def make_env(
        tls_id:str, number_phases:int, 
        sumo_cfg:str, scenario_glb_dir:str, use_gui:bool,
        trip_info:str=None, tls_state_add:List=None,
        renderer_cfg: Optional[dict] = None, sensor_cfg: Optional[dict] = None,
        tshub_env_cfg: Optional[dict] = None,
        is_render: bool = True, #是否渲染
    ):
    def _init() -> gym.Env: 
        tsc_scenario = TSC3DEnvironment(
            sumo_cfg=sumo_cfg, 
            scenario_glb_dir=scenario_glb_dir,
            trip_info=trip_info,
            tls_state_add=tls_state_add,
            tls_ids=[tls_id], 
            tls_action_type='choose_next_phase',
            use_gui=use_gui,
            renderer_cfg=renderer_cfg,
            sensor_cfg=sensor_cfg,
            tshub_env_cfg=tshub_env_cfg,
        )
        tsc_wrapper = TSCEnvWrapper(tsc_scenario, tls_id=tls_id, number_phases=number_phases)
        return tsc_wrapper
    
    return _init