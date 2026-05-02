'''
Author: yufei Ji
Date: 2026-05-02
Description: 混合事件路由文件生成脚本。
    为每个场景的每条路由文件生成两种混合事件路由文件：
      - {BASE}_emergy_bus.rou.xml  ：紧急车辆 + 公交/校车（替换车辆 type）
      - {BASE}_accident_debris.rou.xml：交通事故 + 路面碎片（添加 trip+stop 障碍物）

    核心设计：
      - 空间均匀：按路口 ID 排序后使用 round-robin 轮转分配，每个路口接受大致等量的事件
      - 时间均匀：将 [0, max_depart] 等分为 N 个时间桶，每个桶内随机取一个时间点

    参数：全部沿用 batch_generate_all_scenes.sh 中的默认值，不做更改。
FilePath: /VLMTraffic/scripts/event_scene_generation/generate_mixed_events.py
'''

import os
import sys
import math
import random
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import sumolib
from loguru import logger

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 资产路径 ────────────────────────────────────────────────────────────────
_ASSET_BASE = "TransSimHub/tshub/tshub_env3d/_assets_3d/vehicles_high_poly"

ACCIDENT_MODELS = {
    'crash_vehicle_a': f"{_ASSET_BASE}/event/crash_vehicle_a.glb",
    'crash_vehicle_b': f"{_ASSET_BASE}/event/crash_vehicle_b.glb",
}

DEBRIS_MODELS = {
    'barrier_A':         f"{_ASSET_BASE}/event/barrier_A.glb",
    'barrier_B':         f"{_ASSET_BASE}/event/barrier_B.glb",
    'barrier_C':         f"{_ASSET_BASE}/event/barrier_C.glb",
    'barrier_D':         f"{_ASSET_BASE}/event/barrier_D.glb",
    'barrier_E':         f"{_ASSET_BASE}/event/barrier_E.glb",
    'tree_branch_1lane': f"{_ASSET_BASE}/event/tree_branch_1lane.glb",
}

BARRIER_TYPES    = ['barrier_A', 'barrier_B', 'barrier_C', 'barrier_D', 'barrier_E']
TREEBRANCH_TYPES = ['tree_branch_1lane']


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _get_max_depart(rou_xml: str) -> float:
    """解析路由文件，获取最大出发时间（秒）"""
    max_depart = 0.0
    try:
        for _, elem in ET.iterparse(rou_xml, events=('start',)):
            if elem.tag in ('vehicle', 'trip', 'flow'):
                val = elem.get('depart') or elem.get('begin')
                if val:
                    try:
                        max_depart = max(max_depart, float(val))
                    except ValueError:
                        pass
                elem.clear()
    except Exception:
        pass
    return max_depart if max_depart > 0 else 3600.0


def _uniform_time_points(n: int, max_depart: float, rng: random.Random) -> list:
    """
    在 [0, max_depart] 上生成 n 个时间均匀分布的采样点。
    将时间轴等分为 n 个桶，每个桶内随机取一个值，保证全局时间覆盖均匀。
    """
    if n <= 0:
        return []
    bin_size = max_depart / n
    return [rng.uniform(i * bin_size, (i + 1) * bin_size) for i in range(n)]


def _get_heading_at_offset(shape: list, offset: float) -> float:
    """计算路段在指定 offset 处的行驶方向角（顺时针，正北=0°，正东=90°）"""
    accumulated = 0.0
    for i in range(len(shape) - 1):
        p1, p2 = shape[i], shape[i + 1]
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if accumulated + seg_len >= offset or i == len(shape) - 2:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            return math.degrees(math.atan2(dx, dy)) % 360
        accumulated += seg_len
    return 0.0


def _get_target_nodes(net, target_junction_ids: list):
    """从路网中筛选目标路口节点，按 ID 字典序排序保证空间分配确定性"""
    all_nodes = net.getNodes()
    if target_junction_ids:
        nodes = [n for n in all_nodes if n.getID() in target_junction_ids]
    else:
        nodes = [n for n in all_nodes if n.getType() in
                 ('traffic_light', 'priority', 'traffic_light_right_on_red')]
    return sorted(nodes, key=lambda n: n.getID())


def _pick_incoming_lane(node, rng: random.Random):
    """从路口的进口道中随机选取一条 lane，返回 (edge, lane) 或 None"""
    edges = node.getIncoming()
    if not edges:
        return None, None
    edge = rng.choice(edges)
    lanes = edge.getLanes()
    if not lanes:
        return None, None
    return edge, rng.choice(lanes)


def save_summary(output_path: str, title: str, params: dict, stats: dict) -> None:
    """
    将生成参数和统计信息以文本形式写入 summary 文件。

    Args:
        output_path: 输出路径（.txt）
        title:       标题行
        params:      生成参数字典
        stats:       统计结果字典（事件计数、路口分布等）
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"  {title}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")

        f.write("[Parameters]\n")
        for k, v in params.items():
            f.write(f"  {k:<30} = {v}\n")
        f.write("\n")

        f.write("[Statistics]\n")
        for k, v in stats.items():
            f.write(f"  {k:<30} = {v}\n")
        f.write("\n")
    logger.info(f"[Summary] 已保存: {output_path}")


def save_distribution_plot(
    output_path: str,
    title: str,
    events: list,
    max_depart: float,
    net_xml: str = None,
) -> None:
    """
    生成事件时间和空间分布图并保存为 PNG。

    左图：时间分布直方图（事件出发时间分布）
    右图：空间分布散点图（事件位置，叠加路网骨架，若提供 net_xml）

    Args:
        output_path: 输出 PNG 路径
        title:       图标题
        events:      事件信息列表，每项为 dict：
                       - 'time'   : float  出发时间
                       - 'x'      : float  SUMO x 坐标（可选）
                       - 'y'      : float  SUMO y 坐标（可选）
                       - 'etype'  : str    事件类型标签
        max_depart:  仿真最大出发时间（用于设置直方图 x 轴上限）
        net_xml:     SUMO 路网文件（用于在右图绘制道路骨架，可选）
    """
    if not events:
        logger.warning(f"[Plot] 事件列表为空，跳过绘图: {output_path}")
        return

    # 颜色映射
    _COLOR_MAP = {
        'emergency':         '#FF4444',
        'police':            '#0055FF',
        'fire_engine':       '#FF6600',
        'bus':               '#FFD700',
        'school_bus':        '#FFA500',
        'crash_vehicle_a':   '#CC0033',
        'crash_vehicle_b':   '#FF0077',
        'barrier':           '#9933CC',
        'tree_branch':       '#228B22',
        'unknown':           '#888888',
    }

    def _event_color(etype: str) -> str:
        for k, c in _COLOR_MAP.items():
            if etype.startswith(k):
                return c
        return _COLOR_MAP['unknown']

    times  = [e['time']  for e in events]
    xs     = [e.get('x') for e in events]
    ys     = [e.get('y') for e in events]
    etypes = [e.get('etype', 'unknown') for e in events]
    colors = [_event_color(et) for et in etypes]

    has_coords = all(x is not None and y is not None for x, y in zip(xs, ys))

    fig, axes = plt.subplots(1, 2 if has_coords else 1,
                             figsize=(14 if has_coords else 7, 5))
    if not has_coords:
        axes = [axes]

    # ── 左图：时间分布直方图 ────────────────────────────────────────────────
    ax_t = axes[0]
    ax_t.hist(times, bins=min(30, max(5, len(times) // 3)),
              range=(0, max_depart), color='#4488CC', edgecolor='white', linewidth=0.5)
    ax_t.set_xlabel("Simulation Time (s)")
    ax_t.set_ylabel("Event Count")
    ax_t.set_title("Temporal Distribution")
    ax_t.set_xlim(0, max_depart)
    ax_t.grid(axis='y', alpha=0.3)

    # ── 右图：空间分布散点图（叠加路网骨架）──────────────────────────────────
    if has_coords:
        ax_s = axes[1]

        # 绘制路网骨架（若提供）
        if net_xml and os.path.exists(net_xml):
            try:
                net = sumolib.net.readNet(net_xml)
                for edge in net.getEdges():
                    if edge.getFunction() in ('crossing', 'walkingarea', 'internal'):
                        continue
                    for lane in edge.getLanes():
                        shape = lane.getShape()
                        if len(shape) < 2:
                            continue
                        ex = [p[0] for p in shape]
                        ey = [p[1] for p in shape]
                        ax_s.plot(ex, ey, color='#CCCCCC', linewidth=0.5, zorder=1)
            except Exception as e:
                logger.warning(f"[Plot] 路网绘制失败: {e}")

        ax_s.scatter(xs, ys, c=colors, s=25, alpha=0.8, zorder=2, linewidths=0)
        ax_s.set_xlabel("X (m)")
        ax_s.set_ylabel("Y (m)")
        ax_s.set_title("Spatial Distribution")
        ax_s.set_aspect('equal', adjustable='datalim')

        # 图例
        legend_entries = {}
        for et, c in zip(etypes, colors):
            base_key = next((k for k in _COLOR_MAP if et.startswith(k)), 'unknown')
            if base_key not in legend_entries:
                legend_entries[base_key] = mpatches.Patch(color=c, label=base_key)
        ax_s.legend(handles=list(legend_entries.values()), fontsize=7,
                    loc='upper right', framealpha=0.7)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"[Plot] 已保存: {output_path}")


def _merge_and_save(base_rou_xml: str, output_rou_xml: str,
                    new_vtypes: list, new_elements: list) -> None:
    """将新事件元素按 depart 时间排序后合并到原路由文件并保存"""
    tree = ET.parse(base_rou_xml)
    root = tree.getroot()

    static_elems  = []
    dynamic_elems = []
    for child in list(root):
        if 'depart' in child.attrib or 'begin' in child.attrib:
            dynamic_elems.append(child)
        else:
            static_elems.append(child)
        root.remove(child)

    # vType 去重
    existing_vtype_ids = {e.get('id') for e in static_elems if e.tag == 'vType'}
    for vtype in new_vtypes:
        if vtype.get('id') not in existing_vtype_ids:
            static_elems.append(vtype)
            existing_vtype_ids.add(vtype.get('id'))

    dynamic_elems.extend(new_elements)

    def _get_time(elem):
        try:
            return float(elem.get('depart', elem.get('begin', '0')))
        except ValueError:
            return 0.0

    dynamic_elems.sort(key=_get_time)

    for elem in static_elems:
        root.append(elem)
    for elem in dynamic_elems:
        root.append(elem)

    if hasattr(ET, 'indent'):
        ET.indent(root, space="    ", level=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_rou_xml)), exist_ok=True)
    tree.write(output_rou_xml, encoding='utf-8', xml_declaration=True)
    logger.info(f"[Merge] 已保存: {output_rou_xml}")


# ── 均匀注入：紧急车辆 + 公交/校车 ─────────────────────────────────────────────

# 紧急车辆 vType 定义
_EMERGENCY_VTYPES = [
    {"id": "emergency",  "length": "6.50", "color": "255,165,0", "tau": "1.0", "vClass": "emergency", "guiShape": "emergency"},
    {"id": "police",     "length": "5.00", "color": "blue",      "tau": "1.0", "vClass": "authority", "guiShape": "police"},
    {"id": "fire_engine","length": "8.00", "color": "red",       "vClass": "emergency", "guiShape": "firebrigade"},
]
_EMERGENCY_TYPE_IDS = [v['id'] for v in _EMERGENCY_VTYPES]

# 公交车 vType 定义
_BUS_VTYPES = [
    {"id": "bus",        "length": "12.00", "color": "255,255,0", "tau": "1.5", "maxSpeed": "16.67"},
    {"id": "school_bus", "length": "11.00", "color": "255,165,0", "tau": "1.5", "maxSpeed": "13.89"},
]
_BUS_TYPE_IDS = [v['id'] for v in _BUS_VTYPES]


def generate_emergy_bus(
    base_rou_xml: str,
    output_rou_xml: str,
    emergency_ratio: float = 0.02,
    bus_ratio: float = 0.03,
    seed: int = 42,
) -> dict:
    """
    在基础路由文件中按比例均匀替换车辆类型为紧急车辆和公交/校车，
    生成 *_emergy_bus.rou.xml 混合事件文件。

    均匀性设计（时间 + 空间）：
      - 对所有 <vehicle> 按 depart 时间排序后等分为若干段，
        在每段内固定数量地抽取替换目标，保证替换车辆在时间轴上均匀分布。
      - 每种特种车型（3种紧急+2种公交）在替换集合中均匀出现（round-robin循环分配）。

    Args:
        base_rou_xml:    原始路由文件路径
        output_rou_xml:  输出路由文件路径
        emergency_ratio: 紧急车辆占总车辆比例（默认 0.02）
        bus_ratio:       公交/校车占总车辆比例（默认 0.03）
        seed:            随机种子

    Returns:
        包含事件记录和统计数据的字典，供 save_summary / save_distribution_plot 使用。
    """
    rng = random.Random(seed)
    logger.info(f"[EmgBus] 加载: {base_rou_xml}")

    ET.register_namespace('', "http://sumo.dlr.de/xsd/routes_file.xsd")
    ET.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")

    tree = ET.parse(base_rou_xml)
    root = tree.getroot()

    # 收集所有 vehicle 节点，按出发时间排序（保证时间均匀性）
    vehicles = root.findall('vehicle')
    vehicles_sorted = sorted(vehicles, key=lambda v: float(v.get('depart', '0')))
    total = len(vehicles_sorted)
    if total == 0:
        logger.warning("[EmgBus] 未找到 <vehicle> 节点，跳过。")
        return {"events": [], "max_depart": 0, "stats": {}}

    max_depart = float(vehicles_sorted[-1].get('depart', '0'))
    n_emergency = max(1, int(total * emergency_ratio))
    n_bus       = max(1, int(total * bus_ratio))
    n_special   = n_emergency + n_bus

    logger.info(f"[EmgBus] 共 {total} 辆 → 注入紧急 {n_emergency} + 公交 {n_bus}")

    # 均匀采样索引：将总车辆等分为 n_special 段，每段取中间位置 + 小扰动
    bin_size = total / n_special
    selected_indices = []
    for k in range(n_special):
        lo = int(k * bin_size)
        hi = int((k + 1) * bin_size)
        hi = max(hi, lo + 1)
        selected_indices.append(rng.randint(lo, min(hi - 1, total - 1)))

    # 确保下标不重复
    selected_indices = sorted(set(selected_indices))
    # 若因去重导致数量不足，从剩余中补充
    if len(selected_indices) < n_special:
        taken = set(selected_indices)
        candidates = [i for i in range(total) if i not in taken]
        rng.shuffle(candidates)
        selected_indices = sorted(selected_indices + candidates[:n_special - len(selected_indices)])

    # round-robin 分配车型：先满足 n_emergency 紧急车辆，后分配 n_bus 公交车
    assigned_types = []
    for k in range(n_emergency):
        assigned_types.append(_EMERGENCY_TYPE_IDS[k % len(_EMERGENCY_TYPE_IDS)])
    for k in range(n_bus):
        assigned_types.append(_BUS_TYPE_IDS[k % len(_BUS_TYPE_IDS)])
    rng.shuffle(assigned_types)  # 打乱避免紧急/公交聚集在时间前段

    # 写入 vType 定义（去重）
    existing_vtype_ids = {e.get('id') for e in root.findall('vType')}
    insert_pos = 0
    for vtype_def in _EMERGENCY_VTYPES + _BUS_VTYPES:
        if vtype_def['id'] not in existing_vtype_ids:
            root.insert(insert_pos, ET.Element('vType', attrib=vtype_def))
            existing_vtype_ids.add(vtype_def['id'])
            insert_pos += 1

    # 替换车辆类型，同步收集事件记录（时间可得，坐标无法直接获取）
    idx_map    = {selected_indices[i]: assigned_types[i]
                  for i in range(min(len(selected_indices), len(assigned_types)))}
    event_records = []
    type_counter  = {}
    for i, veh in enumerate(vehicles_sorted):
        if i in idx_map:
            vtype = idx_map[i]
            veh.set('type', vtype)
            depart = float(veh.get('depart', '0'))
            event_records.append({'time': depart, 'x': None, 'y': None, 'etype': vtype})
            type_counter[vtype] = type_counter.get(vtype, 0) + 1

    os.makedirs(os.path.dirname(os.path.abspath(output_rou_xml)), exist_ok=True)
    tree.write(output_rou_xml, encoding='utf-8', xml_declaration=True)
    logger.info(f"[EmgBus] 保存: {output_rou_xml}")

    stats = {
        "total_vehicles":    total,
        "n_emergency":       n_emergency,
        "n_bus":             n_bus,
        "max_depart_s":      f"{max_depart:.1f}",
        **{f"type_{k}": v for k, v in type_counter.items()},
    }
    return {"events": event_records, "max_depart": max_depart, "stats": stats}


# ── 均匀生成：交通事故 + 路面碎片 ──────────────────────────────────────────────

def generate_accident_debris(
    net_xml: str,
    base_rou_xml: str,
    output_rou_xml: str,
    accident_rate: float = 0.8,
    debris_rate: float = 0.8,
    max_range: float = 80.0,
    event_duration: float = 600.0,
    debris_min_gap: float = 5.0,
    debris_max_gap: float = 50.0,
    debris_min_dur: float = 200.0,
    debris_max_dur: float = 600.0,
    barrier_ratio: float = 0.8,
    target_junction_ids: list = None,
    seed: int = 42,
) -> dict:
    """
    在路网进口道上均匀生成交通事故和路面碎片障碍物，
    生成 *_accident_debris.rou.xml 混合事件文件。

    均匀性设计：
      - 空间均匀：将目标路口列表按 ID 排序，用 round-robin 逐一分配事件，
                  每个路口接受大致相同数量的事故/碎片。
      - 时间均匀：用 _uniform_time_points() 将 [0, max_depart] 等分为 N 个桶，
                  每桶内随机取一个时间点作为事件出发时间。

    Args:
        net_xml:              SUMO 路网文件路径
        base_rou_xml:         基础路由文件路径
        output_rou_xml:       输出路由文件路径
        accident_rate:        每小时每路口生成事故数（默认 0.8）
        debris_rate:          每小时每路口生成碎片组数（默认 0.8）
        max_range:            事件距路口最大距离 m（默认 80m）
        event_duration:       事故持续时间 s（默认 600s）
        debris_min_gap:       路障前后间距下限 m（默认 5.0m）
        debris_max_gap:       路障前后间距上限 m（默认 50.0m）
        debris_min_dur:       碎片最小持续时间 s（默认 200s）
        debris_max_dur:       碎片最大持续时间 s（默认 600s）
        barrier_ratio:        碎片中路障（barrier）占比（默认 0.8）
        target_junction_ids:  限定路口 ID 列表；None 则自动从路网中选取信控路口
        seed:                 随机种子

    Returns:
        包含事件记录和统计数据的字典，供 save_summary / save_distribution_plot 使用。
    """
    rng = random.Random(seed)
    logger.info(f"[AccDebris] 读取路网: {net_xml}")
    net = sumolib.net.readNet(net_xml)

    nodes = _get_target_nodes(net, target_junction_ids)
    if not nodes:
        logger.error("[AccDebris] 未找到符合条件的路口，终止。")
        return {"events": [], "max_depart": 0, "stats": {}}

    max_depart = _get_max_depart(base_rou_xml)
    sim_hours  = max_depart / 3600.0

    # 计算总事件数
    n_accidents = int(len(nodes) * accident_rate * sim_hours)
    n_debris    = int(len(nodes) * debris_rate   * sim_hours)
    if n_accidents <= 0 and accident_rate > 0:
        n_accidents = 1
    if n_debris <= 0 and debris_rate > 0:
        n_debris = 1

    logger.info(f"[AccDebris] 路口数={len(nodes)}, 仿真时长={sim_hours:.2f}h → 事故={n_accidents}, 碎片={n_debris}")

    # 均匀时间点
    accident_times = _uniform_time_points(n_accidents, max_depart, rng)
    debris_times   = _uniform_time_points(n_debris,   max_depart, rng)

    new_vtypes    = {}  # vtype_id -> Element
    new_trips     = []
    event_records = []  # 供绘图用

    # ── 生成事故（空间 round-robin） ───────────────────────────────────────────
    accident_models = ['crash_vehicle_a', 'crash_vehicle_b']

    for idx, depart_time in enumerate(accident_times):
        # round-robin 路口选择：idx % len(nodes) 保证均匀覆盖
        node = nodes[idx % len(nodes)]
        edge, lane = _pick_incoming_lane(node, rng)
        if lane is None:
            continue

        lane_len    = lane.getLength()
        min_dist    = 5.0
        max_dist    = min(max_range, max(lane_len - 5.0, min_dist + 1.0))
        dist_from_j = rng.uniform(min_dist, max_dist)
        stop_pos    = max(0.0, lane_len - dist_from_j)

        x, y    = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), stop_pos)
        heading = _get_heading_at_offset(lane.getShape(), stop_pos)

        # round-robin 模型分配：交替 crash_vehicle_a / crash_vehicle_b
        crash_model = accident_models[idx % len(accident_models)]
        event_id    = f"accident_{node.getID()}_{idx}"

        vtype_id = crash_model
        if vtype_id not in new_vtypes:
            new_vtypes[vtype_id] = ET.Element("vType", id=vtype_id, length="10", color="255,0,0")

        trip = ET.Element("trip", id=event_id, type=vtype_id,
                          depart=f"{depart_time:.1f}",
                          departPos=f"{stop_pos:.2f}",
                          **{"from": edge.getID(), "to": edge.getID()})
        ET.SubElement(trip, "stop", lane=lane.getID(),
                      endPos=f"{stop_pos:.2f}",
                      duration=f"{int(event_duration)}")
        ET.SubElement(trip, "param", key="event_type", value=crash_model)
        ET.SubElement(trip, "param", key="model_path",  value=ACCIDENT_MODELS[crash_model])
        ET.SubElement(trip, "param", key="pos_x",       value=f"{x:.4f}")
        ET.SubElement(trip, "param", key="pos_y",       value=f"{y:.4f}")
        ET.SubElement(trip, "param", key="heading",     value=f"{heading:.4f}")
        new_trips.append(trip)
        event_records.append({'time': depart_time, 'x': x, 'y': y, 'etype': crash_model})

    # ── 生成路面碎片（空间 round-robin） ──────────────────────────────────────
    for idx, depart_time in enumerate(debris_times):
        node = nodes[idx % len(nodes)]
        edge, lane = _pick_incoming_lane(node, rng)
        if lane is None:
            continue

        lane_len    = lane.getLength()
        min_dist    = 5.0
        max_dist    = min(max_range, max(lane_len - 5.0, min_dist + 1.0))
        dist_from_j = rng.uniform(min_dist, max_dist)
        center_pos  = max(0.0, lane_len - dist_from_j)

        dur       = rng.uniform(debris_min_dur, debris_max_dur)
        pair_gap  = rng.uniform(debris_min_gap, debris_max_gap)
        elem_id   = f"debris_{node.getID()}_{idx}"

        if rng.random() < barrier_ratio:
            # 路障：round-robin 选类型（均匀覆盖 barrier_A~E）
            debris_type  = BARRIER_TYPES[idx % len(BARRIER_TYPES)]
            event_length = pair_gap
            stop_pos     = center_pos + pair_gap / 2.0
        else:
            # 树枝
            debris_type  = rng.choice(TREEBRANCH_TYPES)
            event_length = 1.0
            stop_pos     = center_pos

        stop_pos = max(0.1, min(lane_len - 0.1, stop_pos))
        phys_pos = max(0.0, stop_pos - event_length / 2.0)
        x, y    = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), phys_pos)
        heading = _get_heading_at_offset(lane.getShape(), phys_pos)

        vtype_id = f"{debris_type}_{event_length:.2f}"
        if vtype_id not in new_vtypes:
            new_vtypes[vtype_id] = ET.Element(
                "vType", id=vtype_id, length=f"{event_length:.2f}", color="128,128,128")

        model_key = debris_type
        trip = ET.Element("trip", id=elem_id, type=vtype_id,
                          depart=f"{depart_time:.1f}",
                          departPos=f"{stop_pos:.2f}",
                          **{"from": edge.getID(), "to": edge.getID()})
        ET.SubElement(trip, "stop", lane=lane.getID(),
                      endPos=f"{stop_pos:.2f}",
                      duration=f"{int(dur)}")
        actual_event_type = f"{debris_type}_{event_length:.2f}" if event_length > 1.0 else debris_type
        ET.SubElement(trip, "param", key="event_type", value=actual_event_type)
        ET.SubElement(trip, "param", key="model_path",  value=DEBRIS_MODELS[model_key])
        ET.SubElement(trip, "param", key="pos_x",       value=f"{x:.4f}")
        ET.SubElement(trip, "param", key="pos_y",       value=f"{y:.4f}")
        ET.SubElement(trip, "param", key="heading",     value=f"{heading:.4f}")
        new_trips.append(trip)
        # 碎片标签统一为基础类型，不带长度后缀，方便绘图颜色归类
        event_records.append({'time': depart_time, 'x': x, 'y': y, 'etype': debris_type})

    n_acc_actual = sum(1 for t in new_trips if t.get('id', '').startswith('accident_'))
    n_deb_actual = sum(1 for t in new_trips if t.get('id', '').startswith('debris_'))
    logger.info(f"[AccDebris] 共生成事故 {n_acc_actual} 个，碎片 {n_deb_actual} 个")
    _merge_and_save(base_rou_xml, output_rou_xml, list(new_vtypes.values()), new_trips)

    stats = {
        "n_junctions":       len(nodes),
        "sim_hours":         f"{sim_hours:.2f}",
        "n_accidents":       n_acc_actual,
        "n_debris":          n_deb_actual,
        "max_depart_s":      f"{max_depart:.1f}",
        "accident_rate":     accident_rate,
        "debris_rate":       debris_rate,
        "max_range_m":       max_range,
        "event_duration_s":  event_duration,
    }
    return {"events": event_records, "max_depart": max_depart, "stats": stats, "net_xml": net_xml}


# ── CLI 入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成 *_emergy_bus 和 *_accident_debris 混合事件路由文件（时空均匀分布）"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── 子命令 1：emergy_bus ────────────────────────────────────────────────
    p_eb = subparsers.add_parser("emergy_bus", help="生成紧急车辆+公交/校车混合路由")
    p_eb.add_argument("--input",             "-i",  type=str,   required=True)
    p_eb.add_argument("--output",            "-o",  type=str,   required=True)
    p_eb.add_argument("--emergency_ratio",   "-er", type=float, default=0.02)
    p_eb.add_argument("--bus_ratio",         "-br", type=float, default=0.03)
    p_eb.add_argument("--seed",              "-s",  type=int,   default=42)

    # ── 子命令 2：accident_debris ───────────────────────────────────────────
    p_ad = subparsers.add_parser("accident_debris", help="生成事故+碎片混合路由")
    p_ad.add_argument("--net",              "-n",   type=str,   required=True)
    p_ad.add_argument("--base_rou",         "-b",   type=str,   required=True)
    p_ad.add_argument("--output",           "-o",   type=str,   required=True)
    p_ad.add_argument("--scenario",         "-sc",  type=str,   default=None,
                      help="场景名称（用于从 SCENARIO_CONFIGS 读取路口列表，可选）")
    p_ad.add_argument("--accident_rate",    "-ar",  type=float, default=0.8)
    p_ad.add_argument("--debris_rate",      "-dr",  type=float, default=0.8)
    p_ad.add_argument("--max_range",        "-r",   type=float, default=80.0)
    p_ad.add_argument("--event_duration",   "-ed",  type=float, default=600.0)
    p_ad.add_argument("--debris_min_gap",   "-ming",type=float, default=5.0)
    p_ad.add_argument("--debris_max_gap",   "-maxg",type=float, default=50.0)
    p_ad.add_argument("--debris_min_dur",   "-mind",type=float, default=200.0)
    p_ad.add_argument("--debris_max_dur",   "-maxd",type=float, default=600.0)
    p_ad.add_argument("--barrier_ratio",    "-bar", type=float, default=0.8)
    p_ad.add_argument("--seed",             "-s",   type=int,   default=42)

    args = parser.parse_args()

    if args.mode == "emergy_bus":
        result = generate_emergy_bus(
            base_rou_xml=args.input,
            output_rou_xml=args.output,
            emergency_ratio=args.emergency_ratio,
            bus_ratio=args.bus_ratio,
            seed=args.seed,
        )

        # 生成 summary 和分布图，保存到与输出路由文件同目录下
        env_dir   = os.path.dirname(os.path.abspath(args.output))
        base_stem = os.path.splitext(os.path.basename(args.output))[0]
        params = {
            "input":          args.input,
            "output":         args.output,
            "emergency_ratio": args.emergency_ratio,
            "bus_ratio":      args.bus_ratio,
            "seed":           args.seed,
        }
        save_summary(
            output_path=os.path.join(env_dir, f"{base_stem}_summary.txt"),
            title=f"EmgBus Generation Summary — {base_stem}",
            params=params,
            stats=result.get("stats", {}),
        )
        save_distribution_plot(
            output_path=os.path.join(env_dir, f"{base_stem}_distribution.png"),
            title=f"EmgBus Event Distribution — {base_stem}",
            events=result.get("events", []),
            max_depart=result.get("max_depart", 3600.0),
            net_xml=None,  # emergy_bus 无坐标，仅绘时间分布图
        )

    else:  # accident_debris
        junction_ids = None
        if args.scenario:
            try:
                from configs.scenairo_config import SCENARIO_CONFIGS
                config = SCENARIO_CONFIGS.get(args.scenario)
                if config:
                    jnames = config.get("JUNCTION_NAME")
                    junction_ids = [jnames] if isinstance(jnames, str) else list(jnames)
            except ImportError:
                pass

        result = generate_accident_debris(
            net_xml=args.net,
            base_rou_xml=args.base_rou,
            output_rou_xml=args.output,
            accident_rate=args.accident_rate,
            debris_rate=args.debris_rate,
            max_range=args.max_range,
            event_duration=args.event_duration,
            debris_min_gap=args.debris_min_gap,
            debris_max_gap=args.debris_max_gap,
            debris_min_dur=args.debris_min_dur,
            debris_max_dur=args.debris_max_dur,
            barrier_ratio=args.barrier_ratio,
            target_junction_ids=junction_ids,
            seed=args.seed,
        )

        # 生成 summary 和分布图
        env_dir   = os.path.dirname(os.path.abspath(args.output))
        base_stem = os.path.splitext(os.path.basename(args.output))[0]
        params = {
            "net":            args.net,
            "base_rou":       args.base_rou,
            "output":         args.output,
            "scenario":       args.scenario or "(auto)",
            "accident_rate":  args.accident_rate,
            "debris_rate":    args.debris_rate,
            "max_range_m":    args.max_range,
            "event_duration_s": args.event_duration,
            "debris_min_gap": args.debris_min_gap,
            "debris_max_gap": args.debris_max_gap,
            "debris_min_dur": args.debris_min_dur,
            "debris_max_dur": args.debris_max_dur,
            "barrier_ratio":  args.barrier_ratio,
            "seed":           args.seed,
        }
        save_summary(
            output_path=os.path.join(env_dir, f"{base_stem}_summary.txt"),
            title=f"AccDebris Generation Summary — {base_stem}",
            params=params,
            stats=result.get("stats", {}),
        )
        save_distribution_plot(
            output_path=os.path.join(env_dir, f"{base_stem}_distribution.png"),
            title=f"AccDebris Event Distribution — {base_stem}",
            events=result.get("events", []),
            max_depart=result.get("max_depart", 3600.0),
            net_xml=result.get("net_xml"),
        )
