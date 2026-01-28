'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: 处理 TSCHub ENV 中的 state, reward (处理后的 state 作为 RL 的输入)
+ state: 5 个时刻的每一个 movement 的 queue length
+ reward: 路口总的 waiting time
LastEditTime: 2026-01-28 11:26:52
'''
import numpy as np
import gymnasium as gym
from gymnasium.core import Env
from collections import deque
import copy
from typing import Any, SupportsFloat, Tuple, Dict, List, Union

class OccupancyList:
    def __init__(self) -> None:
        self.elements = []

    def add_element(self, element) -> None:
        # 支持 dict (多路口) 或 list (单路口)
        if isinstance(element, list) or isinstance(element, dict):
            # 这里对于 dict 类型，不做太严格的 float 检查，假设上游保证正确性
            self.elements.append(element)
        else:
            raise TypeError("添加的元素必须是列表或字典类型")

    def clear_elements(self) -> None:
        self.elements = []

    def calculate_average(self) -> Any:
        """计算一段时间的平均 occupancy
        如果 elements 是 list of lists (单路口): 返回一个平均后的 list
        如果 elements 是 list of dicts (多路口): 返回一个 dict，key 为 tls_id, value 为该路口的平均 occupancy list
        """
        if not self.elements:
            return None
            
        first_elem = self.elements[0]
        
        # Case 1: Single Junction (List of Lists)
        if isinstance(first_elem, list):
            arr = np.array(self.elements)
            averages = np.mean(arr, axis=0, dtype=np.float32)/100
            self.clear_elements()
            return averages.tolist()
            
        # Case 2: Multi Junction (List of Dicts)
        elif isinstance(first_elem, dict):
            # self.elements = [ {'J1': [0,0,..], 'J2': [..]}, {'J1': [..], ...} ]
            # 需要转置为: {'J1': [[0,0,..], [..]], 'J2': ...}
            tls_ids = first_elem.keys()
            result = {}
            for tid in tls_ids:
                # 提取该路口的所有 step 数据
                tls_data = [step_data[tid] for step_data in self.elements]
                arr = np.array(tls_data)
                avg = np.mean(arr, axis=0, dtype=np.float32)/100
                result[tid] = avg.tolist()
            
            self.clear_elements()
            return result
        
        return None


class TSCEnvWrapper(gym.Wrapper):
    """TSC Env Wrapper for single junction or multiple junctions
    """
    def __init__(self, env: Env, tls_id: Union[str, List[str]], number_phases:int, max_states:int=5) -> None:
        super().__init__(env)
        self.tls_id = tls_id # 单路口 id (str) 或 多路口 id 列表 (List[str])
        self.is_multi_agent = isinstance(tls_id, list)
        self.number_phases = number_phases # 路口相位数量, 设置 action space 大小
        
        # states 初始化为 deque
        # 如果是多智能体，states 存储的是 dict 序列
        self.states = deque([self._get_initial_state()] * max_states, maxlen=max_states) 
        
        self.movement_ids = None # 单路口用
        self.phase2movements = None # 单路口用
        
        # 多路口用：存储每个路口的静态信息
        self.multi_movement_ids = {} 
        self.multi_phase2movements = {}
        
        self.occupancy = OccupancyList()
    
    def _get_initial_state(self) -> Union[List[int], Dict[str, List[int]]]:
        if self.is_multi_agent:
            return {tid: [0]*12 for tid in self.tls_id}
        else:
            return [0]*12
    
    def get_state(self):
        # 如果是多智能体，返回 list of dicts (时间序列)
        # 很多 RL 库可能需要把这个展平，或者特定格式。这里维持原始结构。
        if self.is_multi_agent:
            return list(self.states)
        return np.array(self.states, dtype=np.float32)
    
    @property
    def action_space(self):
        if self.is_multi_agent:
            # 多智能体 Action Space 通常是 Dict 或 Tuple
            return gym.spaces.Dict({
                tid: gym.spaces.Discrete(self.number_phases) for tid in self.tls_id
            })
        return gym.spaces.Discrete(self.number_phases)
    
    @property
    def observation_space(self):
        if self.is_multi_agent:
             return gym.spaces.Dict({
                tid: gym.spaces.Box(low=0, high=1, shape=(5, 12)) for tid in self.tls_id
            })
        
        obs_space = gym.spaces.Box(
            low=np.zeros((5,12)),
            high=np.ones((5,12)),
            shape=(5,12)
        ) 
        return obs_space
    
    # Wrapper
    def state_wrapper(self, state):
        """返回当前每个 movement 的 occupancy
        """
        if self.is_multi_agent:
            occupancy = {}
            # 只有当所有路口都可以执行动作时，才返回 True (同步决策)
            # 或者，通常只要有一个能执行就行，但这取决于具体逻辑。
            # 这里沿用原逻辑：取第一个路口的标志，或者做且运算。
            # 假设所有路口步调一致（delta_time 相同）
            can_perform_action = True 
            
            for tid in self.tls_id:
                tls_state = state['tls'][tid]
                occupancy[tid] = tls_state['last_step_occupancy']
                if not tls_state['can_perform_action']:
                    can_perform_action = False
            
            return occupancy, can_perform_action
        else:
            occupancy = state['tls'][self.tls_id]['last_step_occupancy'] # 返回一个长度为 12 的列表
            can_perform_action = state['tls'][self.tls_id]['can_perform_action']
            return occupancy, can_perform_action
    
    def reward_wrapper(self, states) -> float:
        """返回整个路口的排队长度的平均值
        """
        # 注意：原代码这里的 reward 是基于 waiting_time 的，但 compute_rollout_q_value 似乎用了传入的 rewards_list
        # 这部分如果多智能体需要，可能要拆分 reward
        total_waiting_time = 0
        for _, veh_info in states['vehicle'].items():
            total_waiting_time += veh_info['waiting_time']
        return -total_waiting_time
    
    def info_wrapper(self, infos, occupancy):
        """在 info 中加入每个 phase 的占有率
        """
        if self.is_multi_agent:
            phase_occ_all = {}
            for tid in self.tls_id:
                tid_occ = occupancy[tid] # list of 12
                tid_movements = self.multi_movement_ids[tid]
                tid_p2m = self.multi_phase2movements[tid]
                
                movement_occ = {key: value for key, value in zip(tid_movements, tid_occ)}
                phase_occ = {}
                for phase_index, phase_movements in tid_p2m.items():
                    phase_occ[phase_index] = sum([movement_occ[phase] for phase in phase_movements])
                phase_occ_all[tid] = phase_occ
            
            infos['phase_occ'] = phase_occ_all
        else:
            movement_occ = {key: value for key, value in zip(self.movement_ids, occupancy)}
            phase_occ = {}
            for phase_index, phase_movements in self.phase2movements.items():
                phase_occ[phase_index] = sum([movement_occ[phase] for phase in phase_movements])
            
            infos['phase_occ'] = phase_occ
        return infos

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        state =  self.env.reset()
        
        if self.is_multi_agent:
            for tid in self.tls_id:
                self.multi_movement_ids[tid] = state['tls'][tid]['movement_ids']
                self.multi_phase2movements[tid] = state['tls'][tid]['phase2movements']
        else:
            self.movement_ids = state['tls'][self.tls_id]['movement_ids']
            self.phase2movements = state['tls'][self.tls_id]['phase2movements']
            
        # 处理路口动态信息
        occupancy, _ = self.state_wrapper(state=state)
        self.states.append(occupancy)
        state = self.get_state()
        return state, {'step_time':0}
    
    def compute_rollout_q_value(self, rewards_list:Union[List[float], Dict[str, List[float]]], gamma:float=0.95) -> Union[float, Dict[str, float]]:
        """计算一段时间内的 rollout q value
        """    
        if self.is_multi_agent:
            # rewards_list 是 {J1: [r1, r2, ...], J2: [r1, r2, ...] ...}
            if not rewards_list:
                return {tid: 0.0 for tid in self.tls_id}
                
            final_rewards = {tid: 0.0 for tid in self.tls_id}
            for tid in self.tls_id:
                q_val = 0.0
                # 获取该路口的时间序列奖励
                r_series = rewards_list.get(tid, [])
                for i, r_t in enumerate(r_series):
                    discount = gamma ** (i + 1)
                    q_val += discount * r_t
                final_rewards[tid] = q_val
            return final_rewards

        else:   
            rollout_q_value = 0.0
            if rewards_list:
                for i, r_t in enumerate(rewards_list):
                    # enumerate i 从 0 开始，对应时刻 t = i + 1
                    if isinstance(r_t, dict):
                        r_t = r_t.get(self.tls_id, 0)
                    discount = gamma ** (i + 1)
                    rollout_q_value += discount * r_t
            return rollout_q_value

    def step(self, action: Union[int, Dict[str, int]]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        can_perform_action = False
        if self.is_multi_agent:
            rewards_list = {tid: [] for tid in self.tls_id}
        else:
            rewards_list = []
        
        # NOTE: can_perform_action 当前仿真时间 (sim_step) 等于 预定的下一次动作时间 (sim_step+delta_time) 时，该标志位变为 True。
        while not can_perform_action:
            if self.is_multi_agent:
                # 如果已经是 dict 形式的 action {J1: 0, J2: 1}，直接传给 super().step
                # super().step 期望的是 {tls_id: action, ...}
                step_action = action 
            else:
                step_action = {self.tls_id: action} # 构建单路口 action 的动作
            
            states, rewards, truncated, dones, infos, sensor_data = super().step(step_action) # 与环境交互
            
            # wrapper 处理
            occupancy, can_perform_action = self.state_wrapper(state=states) 
            
            # 记录每一时刻的数据
            self.occupancy.add_element(occupancy)
            
            if self.is_multi_agent:
                for tid, r in rewards.items():
                    if tid in rewards_list:
                        rewards_list[tid].append(r)
            else:
                rewards_list.append(rewards)
        
        # 处理好的时序的 state
        render_json = states.copy()
        if 'vehicle' in render_json and isinstance(render_json['vehicle'], dict):
            render_json['vehicle'] = {k: v.copy() for k, v in render_json['vehicle'].items()}
            for vehicle_info in render_json['vehicle'].values():
                if 'next_tls' in vehicle_info:
                    del vehicle_info['next_tls']

        avg_occupancy = self.occupancy.calculate_average()
        # reward 1、车辆端，累计车辆的waiting time，无法表示单个路口车辆的waiting time  2、路口端 通过检测器计算排队和速度
        rewards = self.compute_rollout_q_value(rewards_list=rewards_list) # 路口端 计算一段时间的 reward（基于排队和速度）
        # rewards = self.reward_wrapper(states=states) # 车辆端 计算 vehicle waiting time
        infos = self.info_wrapper(infos, occupancy=avg_occupancy) # info 里面包含每个 phase 的排队
        infos['3d_data'] = sensor_data # info 包含传感器数据 (摄像机数据, 车辆位置)
        self.states.append(avg_occupancy)
        state = self.get_state() # 得到 state

        return state, rewards, truncated, dones, infos, render_json    
    def close(self) -> None:
        return super().close()