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
    """缓存一个决策窗口内的 occupancy，并在窗口结束时求平均。

    这里的“决策窗口”不是单个 SUMO step，而是从一次控制动作下发后，
    SUMO 连续推进到下一个 can_perform_action=True 的整段时间。

    例如 choose_next_phase_with_duration 模式下，如果本次动作选择
    27s 绿灯 + 3s 黄灯，那么这 30s 内每个仿真步采集到的 occupancy
    会先暂存在 self.elements 中；到下一个可决策时刻，再对这 30s 的
    occupancy 求平均，形成历史状态窗口 self.states 中的一帧。

    数据形态：
      - 单路口：self.elements 是 List[List[float]]
        例如 [[occ_m0, occ_m1, ...], [occ_m0, occ_m1, ...], ...]
      - 多路口：self.elements 是 List[Dict[str, List[float]]]
        例如 [{'J1': [...], 'J2': [...]}, {'J1': [...], 'J2': [...]}, ...]
    """
    def __init__(self) -> None:
        # 临时缓存一个决策窗口内每个仿真 step 的 occupancy。
        # 每次 calculate_average() 后会被清空，避免跨决策窗口混合。
        self.elements = []

    def add_element(self, element) -> None:
        # 支持 dict (多路口) 或 list (单路口)
        if isinstance(element, list) or isinstance(element, dict):
            # 单路口 element: List[float]，长度为该路口真实 movement 数。
            # 多路口 element: Dict[tls_id, List[float]]，每个路口可有不同 movement 数。
            # 这里不逐项校验 float 类型，默认上游 state_wrapper 已保证结构正确。
            self.elements.append(element)
        else:
            raise TypeError("添加的元素必须是列表或字典类型")

    def clear_elements(self) -> None:
        # 清空当前决策窗口缓存；reset 时也会调用，避免上一个 episode 残留数据。
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
            # arr shape: (窗口内仿真步数, movement 数)
            # axis=0 表示对时间维求平均，得到每个 movement 在本决策窗口内的平均占有率。
            arr = np.array(self.elements)
            # SUMO occupancy 原始单位是百分比 [0, 100]，这里除以 100 转成 [0, 1]。
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
                # tls_data shape: (窗口内仿真步数, 当前路口 movement 数)
                tls_data = [step_data[tid] for step_data in self.elements]
                arr = np.array(tls_data)
                # 每个路口独立求平均，因为不同路口 movement 数可能不同。
                avg = np.mean(arr, axis=0, dtype=np.float32)/100
                result[tid] = avg.tolist()
            
            self.clear_elements()
            return result
        
        return None


class TSCEnvWrapper(gym.Wrapper):
    """TSC Env Wrapper for single junction or multiple junctions.

    这个 wrapper 把底层 TSHub/SUMO 返回的完整状态，压缩成交通信号控制
    常用的 observation：

        最近 max_states 个“决策窗口”的 movement occupancy 历史。

    注意：
      - max_states 不是 SUMO 仿真步数，而是历史决策窗口数量。
      - 一个决策窗口通常覆盖多个 SUMO step，例如 27s 绿灯 + 3s 黄灯。
      - 单路口 obs shape: (max_states, 当前路口 movement 数)。
      - 多路口 obs: List[Dict[tls_id, List[float]]]，保留原始结构。
    """
    def __init__(self, env: Env, tls_id: Union[str, List[str]], number_phases:int, max_states:int=5) -> None:
        super().__init__(env)
        # 单路口时 tls_id 是 str，例如 "J1"；多路口时是 List[str]。
        self.tls_id = tls_id
        self.is_multi_agent = isinstance(tls_id, list)

        # 当前场景 prompt/action space 中允许选择的绿灯相位数量。
        # 例如 Hongkong_YMT 是 3，France_Massy 是 2，Jinan/Hangzhou/NewYork 是 4。
        self.number_phases = number_phases

        # 历史状态窗口长度：保留最近 max_states 个决策窗口的平均 occupancy。
        # 默认 5 表示模型/控制器能看到最近 5 次决策周期的交通趋势。
        self.max_states = max_states

        # 单路口静态结构信息：
        # movement_ids 的顺序决定 obs 每一列对应哪个 movement。
        # phase2movements 用于把 movement occupancy 汇总成每个 phase 的 occupancy。
        self.movement_ids = None
        self.phase2movements = None

        # 多路口静态结构信息：
        # 每个路口独立保存 movement_ids/phase2movements，因为异构路口 movement 数不同。
        self.multi_movement_ids = {}
        self.multi_phase2movements = {}
        
        # self.states 是最终 observation 的历史缓存。
        # reset() 前还不知道真实 movement 数，因此这里先用 12 作为兼容 fallback；
        # reset() 拿到底层 state 后，会通过 _reset_state_history() 按真实 movement 数重建。
        #
        # 单路口：deque[List[float]]，最终 get_state() 转成 np.ndarray。
        # 多路口：deque[Dict[str, List[float]]]，get_state() 返回 list(self.states)。
        self.states = deque(
            [copy.deepcopy(self._get_initial_state()) for _ in range(max_states)],
            maxlen=max_states,
        )

        # self.occupancy 不是最终 obs，而是“当前决策窗口内”的临时缓存。
        # step() 内每推进一个 SUMO step，就把当步 occupancy 放进去；
        # 窗口结束时求平均，再 append 到 self.states。
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
        """生成 reset 前的占位历史状态。

        初始化 wrapper 时尚未调用底层 env.reset()，因此还不知道真实 movement 数。
        为兼容原始 12 movement 场景，这里 fallback 到 12；真正运行时会在 reset()
        中根据底层返回的真实 occupancy 长度重建 self.states。
        """
        if self.is_multi_agent:
            return {
                tid: [0] * len(self.multi_movement_ids.get(tid, []) or range(12))
                for tid in self.tls_id
            }
        else:
            return [0] * len(self.movement_ids or range(12))

    def _reset_state_history(self, occupancy) -> None:
        """按当前场景真实 movement 数重建历史状态窗口。"""
        self.states = deque(maxlen=self.max_states)

        if self.is_multi_agent:
            # 多路口时 occupancy 是 {tls_id: [occ_m0, occ_m1, ...]}。
            # 每个 tls 的列表长度来自该路口真实 movement 数，允许路口之间长度不同。
            zero_state = {
                tid: [0.0] * len(occupancy.get(tid, []))
                for tid in self.tls_id
            }
            for _ in range(self.max_states - 1):
                self.states.append(copy.deepcopy(zero_state))
            self.states.append(copy.deepcopy(occupancy))
        else:
            # 单路口时 occupancy 是 List[float]。
            # 用 max_states-1 帧全零 + 当前 occupancy 作为 episode 初始历史窗口。
            zero_state = [0.0] * len(occupancy)
            for _ in range(self.max_states - 1):
                self.states.append(zero_state.copy())
            self.states.append(list(occupancy))
    
    def get_state(self):
        """返回对外 observation。

        单路口返回 np.ndarray，shape=(max_states, movement_count)。
        多路口因为不同路口 movement_count 可能不同，不能安全拼成规则 ndarray，
        因此保留 list of dicts 的结构。
        """
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
        """声明 observation 的取值范围和 shape。

        reset 前 movement 数未知时 fallback 到 12；reset 后如果外部再次访问
        observation_space，会使用真实 movement 数。
        """
        if self.is_multi_agent:
             return gym.spaces.Dict({
                tid: gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_states, len(self.multi_movement_ids.get(tid, []) or range(12))),
                    dtype=np.float32,
                )
                for tid in self.tls_id
            })

        state_dim = len(self.movement_ids or range(12))
        
        obs_space = gym.spaces.Box(
            low=np.zeros((self.max_states, state_dim), dtype=np.float32),
            high=np.ones((self.max_states, state_dim), dtype=np.float32),
            shape=(self.max_states, state_dim),
            dtype=np.float32,
        ) 
        return obs_space
    
    # Wrapper
    def state_wrapper(self, state):
        """返回当前每个 movement 的 occupancy。

        Args:
            state: 底层环境返回的完整仿真状态，不是最终对外 observation。
                   这里会从 state['tls'][tls_id] 中抽取 last_step_occupancy。

        Returns:
            occupancy:
              - 单路口：List[float]，长度为该路口真实 movement 数。
              - 多路口：Dict[tls_id, List[float]]。
            can_perform_action:
              当前是否到达信号灯可决策时刻。step() 会一直推进底层仿真，
              直到该值为 True。

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
        """截面 reward：在 can_perform_action=True 时刻读取 TLS detector 快照。

        reward 只使用 TLS 中的两个 detector 变量：
          - last_step_occupancy：占有率惩罚项，越高越差；
          - last_step_mean_speed：速度奖励项，越接近限速越好。

        计算方式：
          reward = -w_occ * norm_occ + w_spd * norm_spd
          norm_occ：occupancy > 1% 的 movement 平均占有率 / 100，范围 [0, 1]
          norm_spd：上述有效 movement 的 (mean_speed / movement maxSpeed) 均值，范围 [0, 1]

        空路网或当前 TLS 无有效占有率时，norm_occ=0 且 norm_spd=0，因此 reward=0。

        多路口时返回 {tls_id: reward}，单路口时返回 float。
        """
        def _compute_single(tls_id: str, tls_info: dict) -> float:
            occupancies = tls_info.get('last_step_occupancy', [])
            mean_speeds  = tls_info.get('last_step_mean_speed', [])
            max_speeds   = self._lane_max_speed.get(tls_id, {})

            # 有效 movement 的 occupancy（过滤空车道或探测器噪声 occ <= 1%）。
            # speed 项也只在这些有效 movement 上计算，避免空 movement 的 0 speed 稀释奖励。
            valid_indices = [i for i, occ in enumerate(occupancies) if occ > 1.0]
            valid_occ = [occupancies[i] for i in valid_indices]
            norm_occ  = (sum(valid_occ) / len(valid_occ) / 100.0) if valid_occ else 0.0

            # 有效 movement 的 speed（过滤 speed < 0 的无效探测器值）
            movement_ids_list = self.multi_movement_ids.get(tls_id) or (
                self.movement_ids if not self.is_multi_agent else []
            )
            norm_spd_list = []
            for i in valid_indices:
                if i >= len(mean_speeds):
                    continue
                spd = mean_speeds[i]
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
        """在 info 中加入每个 phase 的占有率。

        occupancy 是当前决策窗口的平均 movement occupancy，不是瞬时 occupancy。
        phase_occ 的计算方式是把同一个 phase 控制的所有 movement occupancy 相加。
        """
        if self.is_multi_agent:
            phase_occ_all = {}
            for tid in self.tls_id:
                tid_occ = occupancy[tid]
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
        # 清理上一个 episode/场景遗留的窗口内 occupancy 缓存。
        self.occupancy.clear_elements()

        # 底层完整仿真状态，包含 tls、vehicle、detector 等信息。
        # 后续会从中抽取 movement_ids、phase2movements 和 last_step_occupancy。
        state = self.env.reset()

        if self.is_multi_agent:
            for tid in self.tls_id:
                # 静态结构信息只需 reset 时读取一次，用于解释 obs 维度和 phase 聚合。
                self.multi_movement_ids[tid]   = state['tls'][tid]['movement_ids']
                self.multi_phase2movements[tid] = state['tls'][tid]['phase2movements']
                self._lane_max_speed[tid]       = self._build_lane_max_speed(tid, state)
        else:
            # movement_ids 的顺序和 last_step_occupancy 的顺序一一对应。
            self.movement_ids   = state['tls'][self.tls_id]['movement_ids']
            self.phase2movements = state['tls'][self.tls_id]['phase2movements']
            self._lane_max_speed[self.tls_id] = self._build_lane_max_speed(self.tls_id, state)

        # 处理路口动态信息
        # occupancy 是 reset 时刻的瞬时 movement occupancy。
        occupancy, _ = self.state_wrapper(state=state)

        # 按真实 movement 数重建 5 帧历史：前 max_states-1 帧为 0，最后一帧为 reset occupancy。
        self._reset_state_history(occupancy)

        # 覆盖变量名：这里的 state 已经不再是底层完整仿真状态，
        # 而是 wrapper 对外返回的 observation。
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
        """执行一次控制决策，并返回下一个可决策时刻的 observation。

        一次 wrapper.step(action) 内部可能调用多次底层 env.step()：
        只有当 can_perform_action=True 时，才认为本次决策窗口结束。
        """
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

            # states 是底层完整仿真状态；occupancy 是从 states 中抽取出的 movement 占有率。
            occupancy, can_perform_action = self.state_wrapper(state=states)

            # 暂存当前仿真 step 的 occupancy，等待决策窗口结束后求平均。
            self.occupancy.add_element(occupancy)

        # 循环结束 → can_perform_action=True 截面时刻，读取一次 e2 快照计算 reward
        # render_json 保留完整仿真状态，用于保存 data.json 或给渲染/数据生成模块使用；
        # 它不同于下面返回的 state/obs。
        render_json = states.copy()

        # reward 基于窗口结束时刻的完整 tls 状态计算。
        rewards = self.reward_wrapper(states)

        # avg_occupancy 是当前决策窗口内 occupancy 的时间平均值，
        # 这是最终 append 到历史状态 self.states 的一帧。
        avg_occupancy = self.occupancy.calculate_average()

        # phase_occ 也基于 avg_occupancy 计算，表示该决策窗口内每个 phase 的平均压力/占有率。
        infos = self.info_wrapper(infos, occupancy=avg_occupancy)
        infos['3d_data'] = sensor_data

        # 历史窗口向前滚动一帧：丢掉最老窗口，加入当前窗口平均 occupancy。
        self.states.append(avg_occupancy)

        # 对外 observation：单路口为 ndarray(max_states, movement_count)。
        state = self.get_state()

        return state, rewards, truncated, dones, infos, render_json
    def close(self) -> None:
        return super().close()
