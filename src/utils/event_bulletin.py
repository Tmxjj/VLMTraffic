'''
Author: yufei Ji
Date: 2026-04-21
Description: 路口间异步事件广播板（EventBulletin）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一、调用时机（run_eval.py / golden_generation.py 主循环）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每轮 env.step() 前后，按以下顺序调用三个接口：

  ① tick(sumo_t)          —— 每轮循环最先执行，清理 _board 中已过期通知
  ② get_context(jid, t)   —— VLM 推理前，为当前路口拼接邻居通知文本注入 Prompt
  ③ broadcast(jid, resp)  —— VLM 推理后，解析响应并向受影响邻居写入新通知

  仅多路口模式（is_multi_agent=True）才调用 broadcast；
  单路口评测跳过广播（邻居为空时 broadcast 静默返回）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二、广播触发条件（broadcast 内部逻辑）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
broadcast 从 VLM 响应中调用 _extract_events 解析，满足以下条件才触发：
  1. VLM CoT 含 "Condition: Special" 或 "Condition Assessment: Special"（正常场景直接返回，不广播）
  2. Event Recognition 行非空且非 "None"
  3. 当前路口有已配置的邻居（拓扑中存在）

Event Recognition 支持多个交通事件并存，_split_event_segments
用正则逐一匹配 "[Type] ([Category]) detected at [Location], affects Phase [N/None]" 格式，
每个事件独立解析、独立路由、独立写入通知板。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三、定向路由规则（_select_target_neighbors）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
根据事件类型决定广播方向（仅适用于 Jinan/Hangzhou/NewYork 固定 4 相位场景）：

  Emergency / Transit（车辆将驶入下游路口）：
    进口道方向 + Phase ID → 查 _PHASE_EXIT_DIR → 出口方向 → 找该方向邻居
      Phase 0 (ETWT): E进→W出, W进→E出
      Phase 1 (NTST): N进→S出, S进→N出
      Phase 2 (ELWL): E进左转→S出, W进左转→N出
      Phase 3 (NLSL): N进左转→E出, S进左转→W出

  Crash / Obstruction（堵塞将向上游溢出）：
    进口道方向 = 来车来源方向 → 该方向的邻居即受影响的上游路口

  解析失败（方向或 Phase 缺失、进口与 Phase 组合无出口映射、
  出口方向在拓扑中无对应邻居，或事件类型无方向性）：
    静默跳过，不广播任何邻居，记录 DEBUG 日志

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
四、TTL 与通知格式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  可见时刻：Emergency / Transit 延迟 15s（DURATION_OPTIONS 最小值）后生效；
    其他事件立即生效。
  过期时刻：expires_at_sumo = available_at_sumo + green_duration
    即通知从实际广播生效时刻开始，持续覆盖一个绿灯阶段。

  注入文本格式（get_context 拼接后传入 Prompt B2 节）：
    - Source: The intersection to the {North/South/East/West} (impact: {downstream/upstream})
      Event: {原始事件描述，如 "Ambulance (Emergency) detected at East L1, affects Phase 0"}
'''
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger
from configs.prompt_builder import PromptBuilder


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

# Emergency / Transit 车辆到达下游路口前延迟一个最小绿灯时长再广播。
_DELAYED_BROADCAST_EVENT_TYPES = {"emergency", "transit"}
_DELAYED_BROADCAST_AFTER_S = min(PromptBuilder.DURATION_OPTIONS)


@dataclass
class EventNotice:
    """上游路口向下游路口广播的交通事件通知。"""
    from_jid: str           # 来源路口 ID
    from_direction: str     # 下游路口视角下，事件来自哪个方向（N/E/S/W/Unknown）
    event_type: str         # 事件类型（emergency/transit/crash/obstruction/pedestrian/special_event）
    description: str        # VLM 原始事件描述文本（直接注入 prompt）
    available_at_sumo: float  # 通知开始可见的 SUMO 仿真时刻（秒），inclusive
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

        支持 Event Recognition 中包含多个交通事件，每个事件独立广播。
        - Emergency/Transit：广播给下游出口方向的邻居。
        - Crash/Obstruction：广播给上游进口方向的邻居。
        - 解析失败（有事件但方向/Phase 不明）：跳过广播所有邻居。
        Emergency/Transit 延迟 15s 后可见；其他事件立即可见。
        过期时刻 = available_at_sumo + green_duration（SUMO 仿真秒）。
        """
        events = self._extract_events(vlm_response)
        if not events:
            return  # Normal 条件或无可解析事件，无需广播

        all_neighbors = self.get_neighbors(from_jid)
        if not all_neighbors:
            logger.debug(f"[Bulletin] {from_jid} 检测到事件但无已知邻居路口，跳过广播。")
            return

        for event_type, description, approach_dir, phase_id in events:
            target_neighbors, impact_type, fallback = self._select_target_neighbors(
                from_jid, all_neighbors, event_type, approach_dir, phase_id
            )
            if fallback:
                logger.debug(
                    f"[Bulletin][跳过] {from_jid} | 事件: {event_type} | "
                    f"进口: {approach_dir} | Phase: {phase_id} | 原因: 方向/Phase 解析不足或拓扑无对应邻居"
                )

            for neighbor_jid in target_neighbors:
                delay_s = (
                    _DELAYED_BROADCAST_AFTER_S
                    if event_type in _DELAYED_BROADCAST_EVENT_TYPES
                    else 0
                )
                available_at = current_sumo_step + delay_s
                expires_at = available_at + green_duration
                from_direction = self._infer_direction(from_jid, neighbor_jid)
                notice = EventNotice(
                    from_jid=from_jid,
                    from_direction=from_direction,
                    event_type=event_type,
                    description=description,
                    available_at_sumo=available_at,
                    expires_at_sumo=expires_at,
                    impact_type=impact_type,
                )
                self._board[neighbor_jid].append(notice)
                log_prefix = "[Bulletin][延迟广播]" if delay_s else "[Bulletin][广播]"
                logger.info(
                    f"{log_prefix} {from_jid} → {neighbor_jid} | "
                    f"事件类型: {event_type} | 影响方向: {impact_type} | "
                    f"进口: {approach_dir} | Phase: {phase_id} | "
                    f"来源方向(邻居视角): {from_direction} | "
                    f"delay: {delay_s}s | "
                    f"visible @ sumo_t={available_at:.0f}s | "
                    f"TTL: {green_duration}s (expires @ sumo_t={expires_at:.0f}s) | "
                    f"描述: {description[:80]}{'...' if len(description) > 80 else ''}"
                )

    def get_context(self, jid: str, current_sumo_step: float) -> str:
        """获取路口 jid 当前所有有效通知，拼接为 prompt 注入文本。返回空串表示无通知。"""
        active = [
            n for n in self._board.get(jid, [])
            if n.available_at_sumo <= current_sumo_step <= n.expires_at_sumo
        ]
        if not active:
            return ""
        _CARDINAL = {"North", "South", "East", "West"}
        lines = []
        for n in active:
            # 用方向描述替代路口 ID，使单路口场景下的 VLM 也能理解事件来源位置
            if n.from_direction in _CARDINAL:
                source_label = f"The intersection to the {n.from_direction}"
            else:
                source_label = "A neighboring intersection"
            lines.append(
                f"  - Source: {source_label} (impact: {n.impact_type})\n"
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
          - is_fallback: True 表示定向失败，跳过广播（target 列表为空）
        """
        if event_type in ("emergency", "transit"):
            return self._select_downstream(from_jid, all_neighbors, approach_dir, phase_id)
        if event_type in ("crash", "obstruction"):
            return self._select_upstream(from_jid, all_neighbors, approach_dir)
        # pedestrian / special_event 等无明确方向性：跳过广播
        return [], "unclassified", True

    def _select_downstream(
        self,
        from_jid: str,
        all_neighbors: List[str],
        approach_dir: Optional[str],
        phase_id: Optional[int],
    ) -> Tuple[List[str], str, bool]:
        """Emergency/Transit：找事件车辆驶出方向对应的下游邻居路口。解析失败则跳过广播。"""
        if approach_dir is None or phase_id is None:
            logger.debug(
                f"[Bulletin] 定向下游：进口方向或 Phase 缺失，跳过广播"
            )
            return [], "downstream", True

        exit_dir = _PHASE_EXIT_DIR.get((approach_dir, phase_id))
        if exit_dir is None:
            # 进口道方向与该 Phase 无关（如北进口 + Phase 0 东西直行）
            logger.debug(
                f"[Bulletin] 定向下游：({approach_dir}, Phase {phase_id}) 无对应出口方向，跳过广播"
            )
            return [], "downstream", True

        target = self._neighbor_in_direction(from_jid, all_neighbors, exit_dir)
        if target is None:
            logger.debug(
                f"[Bulletin] 定向下游：出口方向 {exit_dir} 在拓扑中无对应邻居，跳过广播"
            )
            return [], "downstream", True

        return [target], "downstream", False

    def _select_upstream(
        self,
        from_jid: str,
        all_neighbors: List[str],
        approach_dir: Optional[str],
    ) -> Tuple[List[str], str, bool]:
        """Crash/Obstruction：找进口道来车方向的上游邻居路口。解析失败则跳过广播。"""
        if approach_dir is None:
            logger.debug(
                f"[Bulletin] 定向上游：进口方向缺失，跳过广播"
            )
            return [], "upstream", True

        upstream_dir = _UPSTREAM_DIR.get(approach_dir)
        if upstream_dir is None:
            logger.debug(
                f"[Bulletin] 定向上游：进口方向 {approach_dir} 无法映射上游方向，跳过广播"
            )
            return [], "upstream", True

        target = self._neighbor_in_direction(from_jid, all_neighbors, upstream_dir)
        if target is None:
            logger.debug(
                f"[Bulletin] 定向上游：进口方向 {approach_dir} 在拓扑中无对应邻居，跳过广播"
            )
            return [], "upstream", True

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
    def _split_event_segments(text: str) -> List[str]:
        """将 Event Recognition 内容分割为单个事件描述字符串列表。

        优先用正则匹配标准格式 "XXX (XXX) detected at XXX, affects Phase N"，
        回退到分号/换行分割，最终回退为整体作为单个事件。
        """
        if not text or text.lower() == "none":
            return []
        # 匹配每个独立事件描述：类型 (类别) detected at 位置, affects Phase ID/None
        # affects Phase None 是合法输出（表示事件在非管控车道），用 (?:\d+|None) 兼容
        _SINGLE_RE = re.compile(
            r'[\w][\w\s]*\([^)]+\)\s+detected\s+at\s+[^,]+,\s*affects\s+Phase\s+(?:\d+|None)',
            re.IGNORECASE
        )
        segments = _SINGLE_RE.findall(text)
        if segments:
            return [s.strip() for s in segments]
        # 回退：按分号或换行分割
        parts = re.split(r'[;\n]+', text)
        filtered = [p.strip() for p in parts if p.strip() and 'detected' in p.lower()]
        if filtered:
            return filtered
        return [text.strip()]

    @staticmethod
    def _extract_events(vlm_response: str) -> List[Tuple[Optional[str], str, Optional[str], Optional[int]]]:
        """从 VLM CoT 输出中提取所有事件，返回列表，每项为 (event_type, description, approach_dir, phase_id)。

        解析逻辑：
          1. 必须含 ``Condition: Special`` 或 ``Condition Assessment: Special``，否则返回空列表。
          2. 从 ``Event Recognition`` 行分割出各独立事件描述。
          3. 对每个事件段：
             - approach_dir: 优先 "detected at {dir}"，回退 "{dir} approach"
             - phase_id: "affects Phase N" → int；"affects Phase None" → None（非管控车道，跳过回退推断）
        """
        if not vlm_response or "ERROR" in vlm_response:
            return []
        if not re.search(r"Condition(?:\s+Assessment)?\s*:\s*Special", vlm_response, re.IGNORECASE):
            return []

        event_match = re.search(
            r"Event Recognition\s*:\s*(.+?)(?:\n|$)", vlm_response, re.IGNORECASE
        )
        if not event_match:
            return []
        event_recog_text = event_match.group(1).strip()

        segments = EventBulletin._split_event_segments(event_recog_text)
        if not segments:
            return []

        results = []
        for seg in segments:
            event_type = EventBulletin._map_event_type(seg)

            # 进口道方向：优先 "detected at {dir}"，回退 "{dir} approach"
            approach_dir: Optional[str] = None
            det_match = re.search(
                r"detected\s+at\s+(north|south|east|west)", seg, re.IGNORECASE
            )
            if det_match:
                approach_dir = _APPROACH_ALIASES.get(det_match.group(1).lower())
            else:
                appr_match = re.search(
                    r"\b(north|south|east|west)\s+approach\b", seg, re.IGNORECASE
                )
                if appr_match:
                    approach_dir = _APPROACH_ALIASES.get(appr_match.group(1).lower())

            # Phase ID
            # "affects Phase None" 是 prompt 合法输出（非管控车道），此时 phase_id 保持 None
            phase_id: Optional[int] = None
            if re.search(r"affects\s+Phase\s+None", seg, re.IGNORECASE):
                phase_id = None  # 明确标记为无 Phase，跳过回退推断
            else:
                phase_match = re.search(r"affects\s+Phase\s+(\d+)", seg, re.IGNORECASE)
                if phase_match:
                    phase_id = int(phase_match.group(1))
                else:
                    # 回退：从括号内 Phase 名称关键词推断
                    kw_match = re.search(r"\(([^)]+)\)", seg)
                    if kw_match:
                        kw = kw_match.group(1).lower()
                        for keyword, pid in _PHASE_KW.items():
                            if keyword in kw:
                                phase_id = pid
                                break

            results.append((event_type, seg, approach_dir, phase_id))

        return results

    @staticmethod
    def _extract_event(vlm_response: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
        """兼容接口（单事件）：调用 _extract_events 并返回第一个事件，无事件时返回全 None 四元组。"""
        events = EventBulletin._extract_events(vlm_response)
        return events[0] if events else (None, None, None, None)

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
- Event Recognition: Ambulance (Emergency) detected at East Approach L2(S), affects Phase 0
- Neighboring Messages: Inactive
- Condition Assessment: Special
- Broadcast Notice: Ambulance - High Priority Passage required for East-West Straight.
]
Action: {"phase": 0, "duration": 40}"""

    # 施工路障：North Approach，affects Phase 1（NTST）→ Obstruction → 上游方向 North → 广播北侧邻居
    VLM_RESPONSE_CONSTRUCTION = """Thought: [
2.Scene Analysis:
- Event Recognition: Construction Barrier detected at North Approach & Lane 3, affects Phase 1
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

    # Police Car：无 Broadcast Notice（East Approach, Phase 2）
    VLM_RESPONSE_NO_NOTICE = """Thought: [
B. Scene Analysis:
- Event Recognition: Police Car (Emergency) detected at East Approach L1, affects Phase 2
- Neighboring Messages: Inactive
- Condition Assessment: Special
- Broadcast Notice: None
]
Action: {"phase": 2, "duration": 25}"""

    # 多事件：Fire Truck (East, Phase 2) + Ambulance (South, Phase 1)
    VLM_RESPONSE_MULTI = """Thought: [
B. Scene Analysis:
- Event Recognition: Fire Truck (Emergency) detected at East L1, affects Phase 2. Ambulance (Emergency) detected at South L4, affects Phase 1
- Neighboring Messages: Inactive
- Condition Assessment: Special
- Broadcast Notice: Multiple emergency vehicles detected - Fire Truck at East, Ambulance at South.
]
Action: {"phase": 1, "duration": 40}"""

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
    print("测试 1：_extract_events 多事件解析")
    print("=" * 70)
    cases = [
        ("救护车(Emergency, East, Phase0)", VLM_RESPONSE_EMERGENCY),
        ("施工路障(Obstruction, North, Phase1)", VLM_RESPONSE_CONSTRUCTION),
        ("公交车(Transit, South, Phase3)", VLM_RESPONSE_BUS),
        ("碰撞事故(Crash, West, Phase0)", VLM_RESPONSE_CRASH),
        ("Normal场景", VLM_RESPONSE_NORMAL),
        ("无Broadcast Notice回退", VLM_RESPONSE_NO_NOTICE),
        ("多事件(FireTruck+Ambulance)", VLM_RESPONSE_MULTI),
    ]
    for label, resp in cases:
        events = EventBulletin._extract_events(resp)
        print(f"  [{label}] → {len(events)} 个事件")
        for evt, desc, approach, phase in events:
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
    print("  intersection_1_0 → North Approach → 上游 North → intersection_1_0 北侧无邻居 → 跳过广播")
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
    events_normal = EventBulletin._extract_events(VLM_RESPONSE_NORMAL)
    print(f"  _extract_events 返回事件数={len(events_normal)}  （预期 0）")
    bulletin.broadcast("intersection_1_0", VLM_RESPONSE_NORMAL, 20, 200)
    print("  （预期：无任何广播日志）")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 8：Broadcast Notice=None，仅依赖 Event Recognition 解析")
    print("  intersection_1_0 → East Approach, Phase 2(ELWL) → 出口 South")
    print("  3×1 水平拓扑无南侧邻居 → 跳过广播")
    print("=" * 70)
    bulletin.broadcast("intersection_1_0", VLM_RESPONSE_NO_NOTICE, 25, 210)

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 9：多事件广播 - Fire Truck(East,Phase2) + Ambulance(South,Phase1)")
    print("  Phase2(ELWL) East进左转→South出，3×1水平拓扑无南侧邻居→跳过广播")
    print("  Phase1(NTST) South进直行→North出，3×1水平拓扑无北侧邻居→跳过广播")
    print("=" * 70)
    bulletin.broadcast("intersection_1_0", VLM_RESPONSE_MULTI, 40, 220)

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 10：无邻居路口（不在拓扑中）→ debug 日志跳过广播")
    print("=" * 70)
    bulletin.broadcast("intersection_99_99", VLM_RESPONSE_EMERGENCY, 30, 300)
    print("  （预期：DEBUG 提示无已知邻居路口）")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 11：get_context - sumo_t=225s 各路口读取通知（含多事件广播结果）")
    print("=" * 70)
    for jid in topology:
        ctx = bulletin.get_context(jid, current_sumo_step=225)
        print(f"  [{jid}]\n{ctx if ctx else '  （无有效通知）'}\n")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试 12：tick 过期清理 - sumo_t=266s（部分通知过期）")
    print("=" * 70)
    bulletin.tick(current_sumo_step=266)
    for jid in topology:
        ctx = bulletin.get_context(jid, current_sumo_step=266)
        print(f"  [{jid}]: {ctx.strip() if ctx else '（无有效通知）'}")
