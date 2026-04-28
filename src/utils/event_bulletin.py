'''
Author: yufei Ji
Date: 2026-04-21
Description: 路口间异步事件广播板（EventBulletin）。

设计原则：
  - 拓扑静态配置：从 SCENARIO_CONFIGS["TOPOLOGY"] 读取，格式 {jid: [neighbor_jid, ...]}
  - TTL 基于 SUMO 仿真时间（秒）：expires_at_sumo = current_sumo_step + green_duration
  - 写入（broadcast）：路口 A 决策完成、检测到 Special 事件后调用
  - 读取（get_context）：路口 B 构建 Prompt 前调用，获取未过期通知文本
  - 过期清理（tick）：每次 env.step() 返回后，以当前 SUMO 时间为基准清理

定向广播规则（仅 Jinan/Hangzhou/NewYork 场景，车道固定 L1(L)/L2(S)/L3(R)）：
  - Emergency / Transit（影响下游）：
      从 Event Recognition 解析进口道方向 + 受影响 Phase 的转向 → 推断出口方向邻居
      Phase 转向映射（4 固定相位）：
        Phase 0 (ETWT/East-West Straight)  → E进直行→W出, W进直行→E出
        Phase 1 (NTST/North-South Straight) → N进直行→S出, S进直行→N出
        Phase 2 (ELWL/East-West Left-Turn)  → E进左转→S出, W进左转→N出
        Phase 3 (NLSL/North-South Left-Turn)→ N进左转→E出, S进左转→W出
      最终找拓扑中位于出口方向的邻居路口。
  - Crash / Obstruction（影响上游）：
      进口道方向即来车方向 → 该方向的上游路口即受影响邻居。
  - 解析失败（方向/Phase 缺失）：有事件前提下降级全播。
'''
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger


# ── Phase 转向映射表 ──────────────────────────────────────────────────────────
# 键：(进口道方向缩写, phase_id)  值：出口方向缩写（供 Emergency/Transit 定向广播）
# 覆盖 Jinan/Hangzhou/NewYork 固定 4 相位
_PHASE_EXIT_DIR: Dict[Tuple[str, int], str] = {
    # Phase 0: ETWT / East-West Straight
    ("E", 0): "W",
    ("W", 0): "E",
    # Phase 1: NTST / North-South Straight
    ("N", 1): "S",
    ("S", 1): "N",
    # Phase 2: ELWL / East-West Left-Turn
    # 东进左转 → 向南出；西进左转 → 向北出
    ("E", 2): "S",
    ("W", 2): "N",
    # Phase 3: NLSL / North-South Left-Turn
    # 北进左转 → 向东出；南进左转 → 向西出
    ("N", 3): "E",
    ("S", 3): "W",
}

# 进口道方向 → 上游邻居方向（Crash/Obstruction：来车方向的上游）
_UPSTREAM_DIR: Dict[str, str] = {
    "N": "N",  # 北进口来车来自北侧 → 上游在北
    "S": "S",
    "E": "E",
    "W": "W",
}

# 方向缩写 → intersection 坐标偏移 (di, dj)，用于在规则路网中找对应邻居
# intersection_i_j：i 为东西轴（东增大），j 为南北轴（北增大）
_DIR_TO_DELTA: Dict[str, Tuple[int, int]] = {
    "N": (0,  1),   # 北邻居：j+1
    "S": (0, -1),   # 南邻居：j-1
    "E": (1,  0),   # 东邻居：i+1
    "W": (-1, 0),   # 西邻居：i-1
}

# 方向全称/变体 → 单字母缩写（解析 Event Recognition 文本用）
_APPROACH_ALIASES: Dict[str, str] = {
    "north": "N", "south": "S", "east": "E", "west": "W",
    "n": "N", "s": "S", "e": "E", "w": "W",
}

# Phase ID 正则候选关键词 → phase_id（解析 "affects Phase X" 用）
_PHASE_KW: Dict[str, int] = {
    "etwt": 0, "east-west straight": 0, "east west straight": 0,
    "ntst": 1, "north-south straight": 1, "north south straight": 1,
    "elwl": 2, "east-west left": 2, "east west left": 2,
    "nlsl": 3, "north-south left": 3, "north south left": 3,
}


@dataclass
class EventNotice:
    """上游路口向下游路口广播的交通事件通知。"""
    from_jid: str           # 来源路口 ID
    from_direction: str     # 下游路口视角下，事件来自哪个方向（N/E/S/W/Unknown）
    event_type: str         # 事件类型（emergency/transit/crash/obstruction/pedestrian/special_event）
    description: str        # VLM 原始事件描述文本（直接注入 prompt）
    expires_at_sumo: float  # 通知过期的 SUMO 仿真时刻（秒），inclusive
    impact_type: str        # "downstream"（Emergency/Transit）或 "upstream"（Crash/Obstruction）


class EventBulletin:
    """
    路口间异步事件广播板。

    拓扑从 scenairo_config.py 的 TOPOLOGY 字段静态配置，
    TTL 以 SUMO 仿真秒数为单位，与绿灯时长动态绑定。
    定向广播仅适用于 Jinan/Hangzhou/NewYork 场景。
    """

    def __init__(self, topology: Dict[str, List[str]] = None) -> None:
        # {目标路口 jid: [EventNotice, ...]}
        self._board: Dict[str, List[EventNotice]] = defaultdict(list)
        # 静态拓扑：{源路口 jid: [邻居路口 jid, ...]}
        self._topology: Dict[str, List[str]] = topology or {}
        if self._topology:
            logger.info(
                f"[Bulletin][拓扑] 加载静态拓扑，共 {len(self._topology)} 个路口有邻居配置"
            )

    # ── 拓扑查询 ──────────────────────────────────────────────────────────────
    def get_neighbors(self, jid: str) -> List[str]:
        """返回路口 jid 的所有已配置邻居路口 ID 列表。"""
        return self._topology.get(jid, [])

    def log_topology(self) -> None:
        """将静态拓扑关系打印到日志（INFO 级别）。"""
        if not self._topology:
            logger.info("[Bulletin][拓扑] 未配置任何路口拓扑")
            return
        logger.info("[Bulletin][拓扑] 静态拓扑（路口 → 邻居列表）：")
        for jid, neighbors in self._topology.items():
            if neighbors:
                logger.info(f"  {jid} → {neighbors}")

    # ── 事件广播 ──────────────────────────────────────────────────────────────
    def broadcast(self, from_jid: str, vlm_response: str,
                  green_duration: int, current_sumo_step: float) -> None:
        """解析 VLM 响应，若存在 Special 事件则向受影响邻居路口定向广播。

        - Emergency/Transit：广播给下游出口方向的邻居。
        - Crash/Obstruction：广播给上游进口方向的邻居。
        - 解析失败（有事件但方向/Phase 不明）：降级全播所有邻居。
        过期时刻 = current_sumo_step + green_duration（SUMO 仿真秒）。
        """
        event_type, description, approach_dir, phase_id = self._extract_event(vlm_response)
        if event_type is None:
            return  # Normal 条件，无需广播

        all_neighbors = self.get_neighbors(from_jid)
        if not all_neighbors:
            logger.debug(f"[Bulletin] {from_jid} 检测到事件但无已知邻居路口，跳过广播。")
            return

        target_neighbors, impact_type, fallback = self._select_target_neighbors(
            from_jid, all_neighbors, event_type, approach_dir, phase_id
        )
        if fallback:
            logger.warning(
                f"[Bulletin][降级全播] {from_jid} | 事件: {event_type} | "
                f"进口: {approach_dir} | Phase: {phase_id} | 原因: 方向/Phase 解析不足，全播所有邻居"
            )

        expires_at = current_sumo_step + green_duration

        for neighbor_jid in target_neighbors:
            from_direction = self._infer_direction(from_jid, neighbor_jid)
            notice = EventNotice(
                from_jid=from_jid,
                from_direction=from_direction,
                event_type=event_type,
                description=description,
                expires_at_sumo=expires_at,
                impact_type=impact_type,
            )
            self._board[neighbor_jid].append(notice)
            logger.info(
                f"[Bulletin][广播] {from_jid} → {neighbor_jid} | "
                f"事件类型: {event_type} | 影响方向: {impact_type} | "
                f"进口: {approach_dir} | Phase: {phase_id} | "
                f"来源方向(邻居视角): {from_direction} | "
                f"TTL: {green_duration}s (expires @ sumo_t={expires_at:.0f}s) | "
                f"描述: {description[:80]}{'...' if len(description) > 80 else ''}"
            )

    def get_context(self, jid: str, current_sumo_step: float) -> str:
        """获取路口 jid 当前所有有效通知，拼接为 prompt 注入文本。返回空串表示无通知。"""
        active = [
            n for n in self._board.get(jid, [])
            if n.expires_at_sumo >= current_sumo_step
        ]
        if not active:
            return ""
        lines = []
        for n in active:
            lines.append(
                f"  - Source: Intersection {n.from_jid} "
                f"(approaching from {n.from_direction} direction, impact: {n.impact_type})\n"
                f"    Event: {n.description}"
            )
        return "\n".join(lines)

    def tick(self, current_sumo_step: float) -> None:
        """清理所有已过期通知，每次 env.step() 返回后调用。"""
        expired_total = 0
        for jid in list(self._board.keys()):
            before = len(self._board[jid])
            self._board[jid] = [
                n for n in self._board[jid]
                if n.expires_at_sumo >= current_sumo_step
            ]
            expired = before - len(self._board[jid])
            if expired > 0:
                expired_total += expired
                logger.debug(f"[Bulletin][过期清理] {jid} 清除 {expired} 条过期通知")
        if expired_total > 0:
            logger.info(
                f"[Bulletin][过期清理] 本步共清除 {expired_total} 条过期通知 "
                f"(sumo_t={current_sumo_step:.0f}s)"
            )

    # ── 定向邻居筛选 ──────────────────────────────────────────────────────────
    def _select_target_neighbors(
        self,
        from_jid: str,
        all_neighbors: List[str],
        event_type: str,
        approach_dir: Optional[str],
        phase_id: Optional[int],
    ) -> Tuple[List[str], str, bool]:
        """根据事件类型和进口道/Phase 信息，从邻居列表中筛选实际受影响的目标路口。

        返回：(目标邻居列表, impact_type, is_fallback)
          - impact_type: "downstream" 或 "upstream"
          - is_fallback: True 表示定向失败，已降级为全播
        """
        if event_type in ("emergency", "transit"):
            return self._select_downstream(from_jid, all_neighbors, approach_dir, phase_id)
        if event_type in ("crash", "obstruction"):
            return self._select_upstream(from_jid, all_neighbors, approach_dir)
        # pedestrian / special_event 等：全播
        return all_neighbors, "unclassified", True

    def _select_downstream(
        self,
        from_jid: str,
        all_neighbors: List[str],
        approach_dir: Optional[str],
        phase_id: Optional[int],
    ) -> Tuple[List[str], str, bool]:
        """Emergency/Transit：找事件车辆驶出方向对应的下游邻居路口。"""
        if approach_dir is None or phase_id is None:
            return all_neighbors, "downstream", True

        exit_dir = _PHASE_EXIT_DIR.get((approach_dir, phase_id))
        if exit_dir is None:
            # 进口道方向与该 Phase 无关（如北进口 + Phase 0 东西直行）→ 降级全播
            logger.debug(
                f"[Bulletin] 定向下游：({approach_dir}, Phase {phase_id}) 无对应出口方向，降级全播"
            )
            return all_neighbors, "downstream", True

        target = self._neighbor_in_direction(from_jid, all_neighbors, exit_dir)
        if target is None:
            logger.debug(
                f"[Bulletin] 定向下游：出口方向 {exit_dir} 无对应邻居，降级全播"
            )
            return all_neighbors, "downstream", True

        return [target], "downstream", False

    def _select_upstream(
        self,
        from_jid: str,
        all_neighbors: List[str],
        approach_dir: Optional[str],
    ) -> Tuple[List[str], str, bool]:
        """Crash/Obstruction：找进口道来车方向的上游邻居路口。"""
        if approach_dir is None:
            return all_neighbors, "upstream", True

        upstream_dir = _UPSTREAM_DIR.get(approach_dir)
        if upstream_dir is None:
            return all_neighbors, "upstream", True

        target = self._neighbor_in_direction(from_jid, all_neighbors, upstream_dir)
        if target is None:
            logger.debug(
                f"[Bulletin] 定向上游：进口方向 {approach_dir} 无对应邻居，降级全播"
            )
            return all_neighbors, "upstream", True

        return [target], "upstream", False

    def _neighbor_in_direction(
        self, from_jid: str, neighbors: List[str], direction: str
    ) -> Optional[str]:
        """从邻居列表中找出位于 from_jid 指定方向上的路口（基于坐标差）。"""
        delta = _DIR_TO_DELTA.get(direction)
        if delta is None:
            return None
        try:
            parts = from_jid.split('_')
            if len(parts) != 3:
                return None
            i_f, j_f = int(parts[1]), int(parts[2])
            di, dj = delta
            target_i, target_j = i_f + di, j_f + dj
            target_jid = f"intersection_{target_i}_{target_j}"
            return target_jid if target_jid in neighbors else None
        except (ValueError, IndexError):
            return None

    # ── 内部工具 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_event(vlm_response: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
        """从 VLM CoT 输出中提取事件信息，返回四元组 (event_type, description, approach_dir, phase_id)。

        解析逻辑：
          1. 必须含 ``Condition Assessment: Special``，否则返回全 None。
          2. description 优先从 ``Broadcast Notice`` 取，回退到 ``Event Recognition``。
          3. approach_dir 和 phase_id 从 ``Event Recognition`` 原始行中解析：
             - approach_dir: "North/South/East/West Approach" → 单字母缩写
             - phase_id: "affects Phase N" → int；或从 Phase 关键词匹配 phase_id
        """
        if not vlm_response or "ERROR" in vlm_response:
            return None, None, None, None
        if not re.search(r"Condition Assessment\s*:\s*Special", vlm_response, re.IGNORECASE):
            return None, None, None, None

        # ── 提取 description ──────────────────────────────────────────────────
        description = None
        notice_match = re.search(
            r"Broadcast Notice\s*:\s*(.+?)(?:\n|$)", vlm_response, re.IGNORECASE
        )
        if notice_match:
            text = notice_match.group(1).strip()
            if text and text.lower() != "none":
                description = text

        event_recog_line = ""
        event_match = re.search(
            r"Event Recognition\s*:\s*(.+?)(?:\n|$)", vlm_response, re.IGNORECASE
        )
        if event_match:
            event_recog_line = event_match.group(1).strip()
            if not description:
                if event_recog_line and event_recog_line.lower() != "none":
                    description = event_recog_line

        if not description:
            return None, None, None, None

        event_type = EventBulletin._map_event_type(description)

        # ── 解析进口道方向（从 Event Recognition 原始行）────────────────────
        approach_dir: Optional[str] = None
        if event_recog_line:
            approach_match = re.search(
                r"\b(north|south|east|west)\s+approach\b",
                event_recog_line, re.IGNORECASE
            )
            if approach_match:
                approach_dir = _APPROACH_ALIASES.get(approach_match.group(1).lower())

        # ── 解析受影响 Phase ID（从 Event Recognition 原始行）───────────────
        phase_id: Optional[int] = None
        if event_recog_line:
            # 优先匹配 "affects Phase N"
            phase_num_match = re.search(
                r"affects\s+Phase\s+(\d+)", event_recog_line, re.IGNORECASE
            )
            if phase_num_match:
                phase_id = int(phase_num_match.group(1))
            else:
                # 回退：从括号内的 Phase 名称关键词推断
                kw_match = re.search(
                    r"\(([^)]+)\)", event_recog_line
                )
                if kw_match:
                    kw = kw_match.group(1).lower()
                    for keyword, pid in _PHASE_KW.items():
                        if keyword in kw:
                            phase_id = pid
                            break

        return event_type, description, approach_dir, phase_id

    @staticmethod
    def _map_event_type(description: str) -> str:
        """将 VLM 文本描述归一化为广播板内部事件类型。"""
        desc_lower = description.lower()
        if any(k in desc_lower for k in ["ambulance", "police", "fire truck", "fire_engine", "emergency"]):
            return "emergency"
        if any(k in desc_lower for k in ["bus", "school bus", "transit"]):
            return "transit"
        if any(k in desc_lower for k in ["pedestrian", "crosswalk", "crossing"]):
            return "pedestrian"
        if any(k in desc_lower for k in ["accident", "collision", "crash"]):
            return "crash"
        if any(k in desc_lower for k in ["construction", "debris", "obstacle", "barrier", "blocked", "obstruction"]):
            return "obstruction"
        return "special_event"

    @staticmethod
    def _infer_direction(from_jid: str, to_jid: str) -> str:
        """从 to_jid 的视角看，from_jid 在哪个方向（基于 intersection_i_j 坐标差）。"""
        try:
            pf = from_jid.split('_')
            pt = to_jid.split('_')
            if len(pf) == 3 and len(pt) == 3:
                i_f, j_f = int(pf[1]), int(pf[2])
                i_t, j_t = int(pt[1]), int(pt[2])
                if i_f < i_t:
                    return "West"
                if i_f > i_t:
                    return "East"
                if j_f < j_t:
                    return "South"
                if j_f > j_t:
                    return "North"
        except (ValueError, IndexError):
            pass
        return from_jid


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="DEBUG",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 测试数据
    # ──────────────────────────────────────────────────────────────────────────

    # 救护车：East Approach，affects Phase 0（ETWT）→ 出口方向 West → 广播西侧邻居
    VLM_RESPONSE_EMERGENCY = """Thought: [
B. Scene Analysis:
- Event Recognition: Ambulance (Emergency) detected at East Approach L2, affects Phase 0
- Neighboring Messages: Inactive
- Condition Assessment: Special
- Broadcast Notice: Ambulance - High Priority Passage required for East-West Straight.
]
Action: {"phase": 0, "duration": 40}"""

    # 施工路障：North Approach，affects Phase 1（NTST）→ Obstruction → 上游方向 North → 广播北侧邻居
    VLM_RESPONSE_CONSTRUCTION = """Thought: [
2.Scene Analysis:
- Event Recognition: Construction Barrier detected at North Approach Lane 3, affects Phase 1
- Neighboring Messages: Inactive
- Condition Assessment: Special
- Broadcast Notice: Construction Barrier - Reduces capacity on North approach right-turn lane.
]
Action: {"phase": 0, "duration": 25}"""

    # 公交车：South Approach，affects Phase 3（NLSL/North-South Left-Turn）→ Transit → 出口方向 West → 广播西侧邻居
    VLM_RESPONSE_BUS = """Thought: [
B. Scene Analysis:
- Event Recognition: City Bus (Transit) detected at South Approach L1, affects Phase 3
- Neighboring Messages: Inactive
- Condition Assessment: Special
- Broadcast Notice: City Bus priority - South approach left-turn requires clearance.
]
Action: {"phase": 3, "duration": 30}"""

    # 交通事故：West Approach → Crash → 上游方向 West → 广播西侧邻居
    VLM_RESPONSE_CRASH = """Thought: [
B. Scene Analysis:
- Event Recognition: Vehicle Collision (Crash) detected at West Approach L2, affects Phase 0
- Neighboring Messages: Inactive
- Condition Assessment: Special
- Broadcast Notice: Crash blocks West approach, expect severe upstream spillback.
]
Action: {"phase": 1, "duration": 35}"""

    # Normal 场景（不应广播）
    VLM_RESPONSE_NORMAL = """Thought: [
Scene Analysis:
- Event Recognition: None
- Condition Assessment: Normal
]
Action: {"phase": 1, "duration": 20}"""

    # Broadcast Notice 为 None，回退到 Event Recognition 解析（East Approach, Phase 2）
    VLM_RESPONSE_NO_NOTICE = """Thought: [
B. Scene Analysis:
- Event Recognition: Police Car (Emergency) detected at East Approach L1, affects Phase 2
- Neighboring Messages: Inactive
- Condition Assessment: Special
- Broadcast Notice: None
]
Action: {"phase": 2, "duration": 25}"""

    # ──────────────────────────────────────────────────────────────────────────
    # 构造 3×1 线形拓扑：intersection_0_0 ↔ intersection_1_0 ↔ intersection_2_0
    # intersection_i_j: i=列(东西), j=行(南北)
    # ──────────────────────────────────────────────────────────────────────────
    topology = {
        "intersection_0_0": ["intersection_1_0"],
        "intersection_1_0": ["intersection_0_0", "intersection_2_0"],
        "intersection_2_0": ["intersection_1_0"],
    }
    bulletin = EventBulletin(topology=topology)
    bulletin.log_topology()

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 1：_extract_event 四元组解析")
    print("=" * 70)
    cases = [
        ("救护车(Emergency, East, Phase0)", VLM_RESPONSE_EMERGENCY),
        ("施工路障(Obstruction, North, Phase1)", VLM_RESPONSE_CONSTRUCTION),
        ("公交车(Transit, South, Phase3)", VLM_RESPONSE_BUS),
        ("碰撞事故(Crash, West, Phase0)", VLM_RESPONSE_CRASH),
        ("Normal场景", VLM_RESPONSE_NORMAL),
        ("无Broadcast Notice回退", VLM_RESPONSE_NO_NOTICE),
    ]
    for label, resp in cases:
        evt, desc, approach, phase = EventBulletin._extract_event(resp)
        print(f"  [{label}]")
        print(f"    event_type={evt!r}  approach_dir={approach!r}  phase_id={phase!r}")
        print(f"    description={desc!r}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 2：定向广播 - 救护车")
    print("  intersection_1_0 → East Approach, Phase 0(ETWT) → 出口 West → 广播 intersection_0_0")
    print("=" * 70)
    bulletin.broadcast("intersection_1_0", VLM_RESPONSE_EMERGENCY, 40, 100)

    print("\n" + "=" * 70)
    print("测试 3：定向广播 - 施工路障（Obstruction）")
    print("  intersection_1_0 → North Approach → 上游 North → intersection_1_0 北侧无邻居 → 降级全播")
    print("=" * 70)
    bulletin.broadcast("intersection_1_0", VLM_RESPONSE_CONSTRUCTION, 25, 110)

    print("\n" + "=" * 70)
    print("测试 4：定向广播 - 公交车（Transit）")
    print("  intersection_1_0 → South Approach, Phase 3(NLSL) → 出口 West → 广播 intersection_0_0")
    print("=" * 70)
    bulletin.broadcast("intersection_1_0", VLM_RESPONSE_BUS, 30, 120)

    print("\n" + "=" * 70)
    print("测试 5：定向广播 - 碰撞事故（Crash）")
    print("  intersection_1_0 → West Approach → 上游 West → 广播 intersection_0_0")
    print("=" * 70)
    bulletin.broadcast("intersection_1_0", VLM_RESPONSE_CRASH, 35, 130)

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 6：get_context - sumo_t=135s 各路口读取通知")
    print("=" * 70)
    for jid in topology:
        ctx = bulletin.get_context(jid, current_sumo_step=135)
        print(f"  [{jid}]\n{ctx if ctx else '  （无有效通知）'}\n")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 7：Normal 场景不触发广播")
    print("=" * 70)
    evt, *_ = EventBulletin._extract_event(VLM_RESPONSE_NORMAL)
    print(f"  _extract_event event_type={evt!r}  （预期 None）")
    bulletin.broadcast("intersection_1_0", VLM_RESPONSE_NORMAL, 20, 200)
    print("  （预期：无任何广播日志）")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 8：Broadcast Notice=None 回退 Event Recognition 解析")
    print("  intersection_1_0 → East Approach, Phase 2(ELWL) → 出口 South")
    print("  3×1 水平拓扑无南侧邻居 → 降级全播")
    print("=" * 70)
    bulletin.broadcast("intersection_1_0", VLM_RESPONSE_NO_NOTICE, 25, 210)

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 9：无邻居路口（不在拓扑中）→ debug 日志跳过广播")
    print("=" * 70)
    bulletin.broadcast("intersection_99_99", VLM_RESPONSE_EMERGENCY, 30, 300)
    print("  （预期：DEBUG 提示无已知邻居路口）")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 10：tick 过期清理 - sumo_t=166s（部分通知过期）")
    print("=" * 70)
    bulletin.tick(current_sumo_step=166)
    for jid in topology:
        ctx = bulletin.get_context(jid, current_sumo_step=166)
        print(f"  [{jid}]: {ctx.strip() if ctx else '（无有效通知）'}")
