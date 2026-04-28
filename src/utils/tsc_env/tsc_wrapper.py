'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: 处理 TSCHub ENV 中的 state, reward (处理后的 state 作为 RL 的输入)
+ state: 5 个时刻的每一个 movement 的 queue length
+ reward: 路口总的 waiting time
LastEditTime: 2026-04-27 17:58:01
'''
import numpy as np
import gymnasium as gym
from gymnasium.core import Env
from collections import deque
import copy
from typing import Any, SupportsFloat, Tuple, Dict, List, Union

# 与 choose_next_phase_with_duration 一致的绿灯候选集（VLM 可选时长，单位：秒）
GREEN_DURATION_CANDIDATES = [15, 20, 25, 30, 35,40]

# 固定配时/MaxPressure 基线的绿灯时长：27s 绿灯 + 3s 黄灯 = 30s 整步，与决策步间隔对齐
FIXED_TIME_GREEN_DURATION = 27

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

        # {tls_id: {movement_key: avg_max_speed_m_s}}，reset() 时从 SUMO 读取并缓存
        # 用于 reward_wrapper() 中对 mean_speed 归一化
        self._lane_max_speed: Dict[str, Dict[str, float]] = {}

        # reward 权重（occupancy 惩罚 + speed 奖励）
        self._w_occ: float = 0.7
        self._w_spd: float = 0.3
        # 归一化 fallback：当 lane maxSpeed 查询失败时使用 50 km/h
        self._fallback_max_speed: float = 13.9
    
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
        # 联合动作空间：phase_id × duration_idx
        # phase_id    ∈ [0, number_phases)
        # duration_idx ∈ [0, len(GREEN_DURATION_CANDIDATES))，映射到实际秒数
        single_space = gym.spaces.Dict({
            'phase_id':    gym.spaces.Discrete(self.number_phases),
            'duration_idx': gym.spaces.Discrete(len(GREEN_DURATION_CANDIDATES))
        })
        if self.is_multi_agent:
            return gym.spaces.Dict({tid: single_space for tid in self.tls_id})
        return single_space
    
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
        """返回当前每个 movement 的 occupancy。

        多路口异步模式：任意一个路口 can_perform_action=True 即返回 True，
        由上层（run_eval.py）根据各路口的 can_perform_action 字段决定哪些路口需要决策。
        """
        if self.is_multi_agent:
            occupancy = {}
            # 异步决策：任一路口可执行动作即退出仿真内循环，交由上层按路口过滤
            can_perform_action = False
            for tid in self.tls_id:
                tls_state = state['tls'][tid]
                occupancy[tid] = tls_state['last_step_occupancy']
                if tls_state['can_perform_action']:
                    can_perform_action = True
            return occupancy, can_perform_action
        else:
            occupancy = state['tls'][self.tls_id]['last_step_occupancy']
            can_perform_action = state['tls'][self.tls_id]['can_perform_action']
            return occupancy, can_perform_action
    
    def reward_wrapper(self, states) -> Union[float, Dict[str, float]]:
        """截面 reward：在 can_perform_action=True 时刻读取 e2 快照，计算归一化指标。

        reward = -w_occ * norm_occ + w_spd * norm_spd
          norm_occ：有效 movement（occ > 1%）的占用率均值 / 100，范围 [0, 1]
          norm_spd：有效 movement（speed >= 0）的 (speed / maxSpeed) 均值，范围 [0, 1]

        多路口时返回 {tls_id: reward}，单路口时返回 float。
        """
        def _compute_single(tls_id: str, tls_info: dict) -> float:
            occupancies = tls_info.get('last_step_occupancy', [])
            mean_speeds  = tls_info.get('last_step_mean_speed', [])
            max_speeds   = self._lane_max_speed.get(tls_id, {})

            # 有效 movement 的 occupancy（过滤空车道 occ <= 1%）
            valid_occ = [occ for occ in occupancies if occ > 1.0]
            norm_occ  = (sum(valid_occ) / len(valid_occ) / 100.0) if valid_occ else 0.0

            # 有效 movement 的 speed（过滤 speed == -1 空车道）
            movement_ids_list = self.multi_movement_ids.get(tls_id) or (
                self.movement_ids if not self.is_multi_agent else []
            )
            norm_spd_list = []
            for i, spd in enumerate(mean_speeds):
                if spd < 0:
                    continue
                mv_key = movement_ids_list[i] if i < len(movement_ids_list) else None
                lim = max_speeds.get(mv_key, self._fallback_max_speed) if mv_key else self._fallback_max_speed
                lim = lim if lim > 0 else self._fallback_max_speed
                norm_spd_list.append(min(spd / lim, 1.0))  # 限速归一化，上限截断到 1
            norm_spd = (sum(norm_spd_list) / len(norm_spd_list)) if norm_spd_list else 0.0

            return float(-self._w_occ * norm_occ + self._w_spd * norm_spd)

        tls_obs = states.get('tls', {})
        if self.is_multi_agent:
            return {tid: _compute_single(tid, tls_obs.get(tid, {})) for tid in self.tls_id}
        else:
            return _compute_single(self.tls_id, tls_obs.get(self.tls_id, {}))
    
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

    def _build_lane_max_speed(self, tls_id: str, state: dict) -> Dict[str, float]:
        """从 SUMO 查询每个 movement 关联 lane 的 maxSpeed，取均值缓存。
        查询失败时 fallback 到 self._fallback_max_speed。
        """
        movement_lane_ids: dict = state['tls'][tls_id].get('movement_lane_ids', {})
        sumo_conn = getattr(self.env.unwrapped, 'sumo', None)
        result: Dict[str, float] = {}
        for mv_key, lane_ids in movement_lane_ids.items():
            speeds = []
            for lane_id in (lane_ids or []):
                try:
                    spd = sumo_conn.lane.getMaxSpeed(lane_id) if sumo_conn else 0.0
                    if spd > 0:
                        speeds.append(spd)
                except Exception:
                    pass
            result[mv_key] = (sum(speeds) / len(speeds)) if speeds else self._fallback_max_speed
        return result

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        state = self.env.reset()

        if self.is_multi_agent:
            for tid in self.tls_id:
                self.multi_movement_ids[tid]   = state['tls'][tid]['movement_ids']
                self.multi_phase2movements[tid] = state['tls'][tid]['phase2movements']
                self._lane_max_speed[tid]       = self._build_lane_max_speed(tid, state)
        else:
            self.movement_ids   = state['tls'][self.tls_id]['movement_ids']
            self.phase2movements = state['tls'][self.tls_id]['phase2movements']
            self._lane_max_speed[self.tls_id] = self._build_lane_max_speed(self.tls_id, state)

        # 处理路口动态信息
        occupancy, _ = self.state_wrapper(state=state)
        self.states.append(occupancy)
        state = self.get_state()
        return state, {'step_time': 0}

    @staticmethod
    def _decode_action(action_input) -> tuple:
        """将上层传入的动作解码为 (phase_id, green_duration) 元组。

        支持四种输入格式（兼容新旧代码）：
          1. Dict {'phase_id': int, 'duration': int}     → 推荐格式，直接使用实际绿灯秒数
          2. Dict {'phase_id': int, 'duration_idx': int} → 索引格式，转换后使用（向后兼容）
          3. Tuple/List (phase_id, green_duration)        → 直接使用
          4. int                                          → 旧格式，duration 取 FIXED_TIME_GREEN_DURATION
        """
        if isinstance(action_input, dict):
            phase_id = int(action_input['phase_id'])
            if 'duration' in action_input:
                # 推荐格式：直接传入实际绿灯秒数，无需索引转换
                duration = int(action_input['duration'])
            elif 'duration_idx' in action_input:
                # 向后兼容：索引 → 实际秒数
                duration = GREEN_DURATION_CANDIDATES[int(action_input['duration_idx'])]
            else:
                duration = FIXED_TIME_GREEN_DURATION
        elif isinstance(action_input, (tuple, list)) and len(action_input) == 2:
            phase_id, duration = int(action_input[0]), int(action_input[1])
        else:
            phase_id = int(action_input)
            duration = FIXED_TIME_GREEN_DURATION  # 旧格式兼容：27s+3s黄灯=30s整步
        return phase_id, duration

    def step(self, action: Union[int, Dict, Tuple]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        can_perform_action = False

        # 将上层动作统一解码为底层可用格式
        # 多路口：{jid: action_input} → {jid: (phase_id, duration)}
        # 单路口：action_input → (phase_id, duration)
        if self.is_multi_agent:
            decoded_action = {tid: self._decode_action(action[tid]) for tid in self.tls_id}
        else:
            decoded_action = self._decode_action(action)

        # NOTE: can_perform_action 当前仿真时间 (sim_step) 等于 预定的下一次动作时间 (sim_step+delta_time) 时，该标志位变为 True。
        while not can_perform_action:
            if self.is_multi_agent:
                step_action = decoded_action
            else:
                step_action = {self.tls_id: decoded_action}

            states, _, truncated, dones, infos, sensor_data = super().step(step_action)

            occupancy, can_perform_action = self.state_wrapper(state=states)
            self.occupancy.add_element(occupancy)

        # 循环结束 → can_perform_action=True 截面时刻，读取一次 e2 快照计算 reward
        render_json = states.copy()

        rewards = self.reward_wrapper(states)
        avg_occupancy = self.occupancy.calculate_average()
        infos = self.info_wrapper(infos, occupancy=avg_occupancy)
        infos['3d_data'] = sensor_data
        self.states.append(avg_occupancy)
        state = self.get_state()

        return state, rewards, truncated, dones, infos, render_json
    def close(self) -> None:
        return super().close()