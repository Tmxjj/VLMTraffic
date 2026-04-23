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
'''
import re
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger


@dataclass
class EventNotice:
    """上游路口向下游路口广播的交通事件通知。"""
    from_jid: str         # 来源路口 ID
    from_direction: str   # 下游路口视角下，事件来自哪个方向（N/E/S/W/Unknown）
    event_type: str       # 事件类型（emergency / transit / crash / obstruction / pedestrian / special_event）
    description: str      # VLM 原始事件描述文本（直接注入 prompt）
    expires_at_sumo: float  # 通知过期的 SUMO 仿真时刻（秒），inclusive


class EventBulletin:
    """
    路口间异步事件广播板。

    拓扑从 scenairo_config.py 的 TOPOLOGY 字段静态配置，
    TTL 以 SUMO 仿真秒数为单位，与绿灯时长动态绑定。
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
        """解析 VLM 响应，若存在 Special 事件则向所有邻居路口广播。

        过期时刻 = current_sumo_step + green_duration（SUMO 仿真秒）。
        """
        event_type, description = self._extract_event(vlm_response)
        if event_type is None:
            return  # Normal 条件，无需广播

        neighbors = self.get_neighbors(from_jid)
        if not neighbors:
            logger.debug(f"[Bulletin] {from_jid} 检测到事件但无已知邻居路口，跳过广播。")
            return

        expires_at = current_sumo_step + green_duration

        for neighbor_jid in neighbors:
            from_direction = self._infer_direction(from_jid, neighbor_jid)
            notice = EventNotice(
                from_jid=from_jid,
                from_direction=from_direction,
                event_type=event_type,
                description=description,
                expires_at_sumo=expires_at,
            )
            self._board[neighbor_jid].append(notice)
            logger.info(
                f"[Bulletin][广播] {from_jid} → {neighbor_jid} | "
                f"事件类型: {event_type} | "
                f"来源方向: {from_direction} | "
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
                f"(approaching from {n.from_direction} direction)\n"
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

    # ── 内部工具 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_event(vlm_response: str) -> tuple:
        """从 VLM CoT 输出中提取本地事件类型和描述。

        仅当本地事件存在时返回结果：
          1. ``Condition Assessment: Special`` 只是进入慢思考的必要条件；
             若 Special 仅由邻居通知触发，不应再次广播。
          2. 优先读取 ``Broadcast Notice``，因为它是专门为邻居广播设计的精简描述。
          3. 若未输出 ``Broadcast Notice``，则回退到 ``Event Recognition``。
        """
        if not vlm_response or "ERROR" in vlm_response:
            return None, None
        if not re.search(r"Condition Assessment\s*:\s*Special", vlm_response, re.IGNORECASE):
            return None, None

        notice_match = re.search(
            r"Broadcast Notice\s*:\s*(.+?)(?:\n|$)",
            vlm_response,
            re.IGNORECASE,
        )
        if notice_match:
            description = notice_match.group(1).strip()
            if description and description.lower() != "none":
                return EventBulletin._map_event_type(description), description

        event_match = re.search(
            r"Event Recognition\s*:\s*(.+?)(?:\n|$)",
            vlm_response,
            re.IGNORECASE,
        )
        if not event_match:
            return None, None

        description = event_match.group(1).strip()
        if not description or description.lower() == "none":
            return None, None

        return EventBulletin._map_event_type(description), description

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
        """
        在规则 m×n 路网中推断相对方向。
        命名格式假设为 intersection_{i}_{j}，i 为东西轴，j 为南北轴。
        从 to_jid 的视角看，from_jid 在哪个方向。
        """
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
