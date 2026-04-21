import math
import re
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger

_STEP_DURATION_SEC = 30

@dataclass
class EventNotice:
    """上游路口向下游路口（邻居）广播的交通事件通知。"""
    from_jid: str           # 来源路口 ID
    from_direction: str     # 下游路口视角下，事件来自哪个方向（N/E/S/W）
    event_type: str         # 事件类型关键词（emergency_vehicle / incident / congestion）
    description: str        # VLM 原始事件描述文本（直接注入 prompt）
    expires_at_step: int    # 在第几个决策步后过期（含）


class EventBulletin:
    """
    路口间异步事件广播板（移除原先基于 vehicle_next_tls 的动态推断，改为静态配置拓扑）。

    设计原则：
    - 写入（broadcast）：路口 A 决策完毕，检测到 Special 事件后调用
    - 读取（get_context）：路口 B 构建 Prompt 前调用，获取所有未过期通知的文本摘要
    - 过期清理（tick）：每个决策步开始时调用，删除已过期的通知
    """

    def __init__(self, topology: Dict[str, List[str]] = None) -> None:
        # {目标路口 jid: [EventNotice, ...]}
        self._board: Dict[str, List[EventNotice]] = defaultdict(list)
        # 静态网络拓扑（从 scenairo_config 传入）
        self._topology: Dict[str, List[str]] = topology or {}

    def get_neighbors(self, jid: str) -> List[str]:
        """返回路口 jid 的有效邻居路口列表（去除虚拟交叉口）。"""
        return self._topology.get(jid, [])

    # ── 事件广播 ──────────────────────────────────────────────────────────────
    def broadcast(self, from_jid: str, vlm_response: str,
                  green_duration: int, current_step: int) -> None:
        """解析 VLM 响应，若存在 Special 事件则向邻近路口广播。

        过期步数 = ceil(green_duration / _STEP_DURATION_SEC)，最少 1 步。
        """
        event_type, description = self._extract_event(vlm_response)
        if event_type is None:
            return  # Normal 条件，无需广播

        neighbors = self.get_neighbors(from_jid)
        if not neighbors:
            logger.debug(
                f"[Bulletin] {from_jid} 检测到事件但尚无已知邻居路口，跳过广播。"
            )
            return

        ttl_steps = max(1, math.ceil(green_duration / _STEP_DURATION_SEC))
        expires_at = current_step + ttl_steps

        for down_jid in neighbors:
            # 从目标路口的视角看，事件从哪个方向传来
            from_direction = self._infer_direction(from_jid, down_jid)
            
            notice = EventNotice(
                from_jid=from_jid,
                from_direction=from_direction,
                event_type=event_type,
                description=description,
                expires_at_step=expires_at,
            )
            self._board[down_jid].append(notice)

            logger.info(
                f"[Bulletin][广播] {from_jid} → {down_jid} | "
                f"事件类型: {event_type} | "
                f"来源方向: {from_direction} | "
                f"TTL: {ttl_steps} 步 (expires @ step {expires_at}) | "
                f"描述: {description[:80]}{'...' if len(description) > 80 else ''}"
            )

    def get_context(self, jid: str, current_step: int) -> str:
        """获取路口 jid 当前所有有效通知，拼接为 prompt 注入文本。返回空串表示无通知。"""
        active = [n for n in self._board.get(jid, []) if n.expires_at_step >= current_step]
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

    def tick(self, current_step: int) -> None:
        """清理所有已过期通知，每个决策步开始时调用。"""
        expired_total = 0
        for jid in list(self._board.keys()):
            before = len(self._board[jid])
            self._board[jid] = [n for n in self._board[jid] if n.expires_at_step >= current_step]
            expired = before - len(self._board[jid])
            if expired > 0:
                expired_total += expired
                logger.debug(f"[Bulletin][过期清理] {jid} 清除 {expired} 条过期通知")
        if expired_total > 0:
            logger.info(f"[Bulletin][过期清理] 本步共清除 {expired_total} 条过期通知")

    # ── 内部工具 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_event(vlm_response: str) -> tuple:
        """从 VLM CoT 输出中提取事件类型和描述。"""
        if not vlm_response or "ERROR" in vlm_response:
            return None, None

        if not re.search(r"Final Condition\s*:\s*Special", vlm_response, re.IGNORECASE):
            return None, None

        m = re.search(r"Emergency Check\s*:\s*(.+?)(?:\n|$)", vlm_response, re.IGNORECASE)
        description = m.group(1).strip() if m else "Special condition detected"

        desc_lower = description.lower()
        if any(k in desc_lower for k in ["ambulance", "police", "fire", "emergency"]):
            event_type = "emergency_vehicle"
        elif any(k in desc_lower for k in ["accident", "collision", "crash"]):
            event_type = "incident"
        elif any(k in desc_lower for k in ["construction", "debris", "obstacle", "barrier"]):
            event_type = "road_obstruction"
        else:
            event_type = "special_event"

        return event_type, description

    @staticmethod
    def _infer_direction(from_jid: str, to_jid: str) -> str:
        """
        在规则 m*n 路网中推断相对方向。
        假设命名格式为 intersection_{i}_{j}，其中 i 代表横向（东西），j 代表纵向（南北）。
        例如：A(2,2) 广播到 B(3,2)，A 在 B 的西侧。从 B 的视角，事件来自西侧 (West)。
        """
        try:
            parts_f = from_jid.split('_')
            parts_t = to_jid.split('_')
            if len(parts_f) == 3 and len(parts_t) == 3:
                i_f, j_f = int(parts_f[1]), int(parts_f[2])
                i_t, j_t = int(parts_t[1]), int(parts_t[2])
                
                if i_f < i_t: return "West"
                if i_f > i_t: return "East"
                if j_f < j_t: return "South"
                if j_f > j_t: return "North"
        except (ValueError, IndexError):
            pass
            
        return from_jid
