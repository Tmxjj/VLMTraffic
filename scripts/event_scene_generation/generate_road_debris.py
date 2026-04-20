'''
Author: yufei Ji
Date: 2026-04-19
LastEditTime: 2026-04-20 11:55:23
Description: Road Debris 事件场景生成脚本。
             在各路口进口道 200m 范围内随机放置路障（barrier_A~E）和树枝（tree_branch）。
             规则：路障占道时，在被占道位置前后各放置一个路障（即每处障碍物对应 2 个 trip+stop）。
FilePath: /VLMTraffic/scripts/event_scene_generation/generate_road_debris.py
'''
import os
import sys
import math
import random
import argparse
import xml.etree.ElementTree as ET

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import sumolib
from loguru import logger
from configs.scenairo_config import SCENARIO_CONFIGS

_ASSET_BASE = "TransSimHub/tshub/tshub_env3d/_assets_3d/vehicles_high_poly"

# 路障与树枝的模型路径映射
DEBRIS_MODELS = {
    'barrier_A':          f"{_ASSET_BASE}/event/barrier_A.glb",
    'barrier_B':          f"{_ASSET_BASE}/event/barrier_B.glb",
    'barrier_C':          f"{_ASSET_BASE}/event/barrier_C.glb",
    'barrier_D':          f"{_ASSET_BASE}/event/barrier_D.glb",
    'barrier_E':          f"{_ASSET_BASE}/event/barrier_E.glb",
    'tree_branch_1lane':  f"{_ASSET_BASE}/event/tree_branch_1lane.glb",
    # 'tree_branch_3lanes': f"{_ASSET_BASE}/event/tree_branch_3lanes.glb",
}

# 路障与树枝的相对权重：路障 70%，树枝 30%
BARRIER_TYPES  = ['barrier_A', 'barrier_B', 'barrier_C', 'barrier_D', 'barrier_E']
TREEBRANCH_TYPES = ['tree_branch_1lane', 
                    # 'tree_branch_3lanes'
                    ]


def _get_max_depart(rou_xml: str) -> float:
    """快速解析路由文件，获取最大出发时间以支撑事件频率计算与分布"""
    if not os.path.exists(rou_xml):
        return 3600.0
    max_depart = 0.0
    try:
        context = ET.iterparse(rou_xml, events=('start',))
        for event, elem in context:
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


def _add_debris_element(
    trip_list: list,
    vtype_dict: dict,
    elem_id: str,
    edge,
    lane,
    stop_pos: float,
    depart_time: float,
    event_duration: float,
    debris_type: str,
    event_length: float = 5.0,
) -> None:
    """
    在指定位置生成一个路障/树枝的 vType + trip 元素对。
    对于物理上占据较大长度的路障，使用 event_length 统一生成单个 vType，并记录在参数中供渲染解析。
    """
    lane_len = lane.getLength()
    # 确保 stop_pos 在合法范围内
    stop_pos = max(0.1, min(lane_len - 0.1, stop_pos))

    # 计算 3D 渲染所需物理中心坐标（SUMO 中 stop 以车头端 endPos 计，所以实体中心位于端点后方 event_length/2 处）
    physical_center_pos = max(0.0, stop_pos - event_length / 2.0)
    x, y = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), physical_center_pos)
    heading = _get_heading_at_offset(lane.getShape(), physical_center_pos)

    # 为了能设置各自独立的 length，vType_id 需要加上长度特征（或使用独立 elem_id）
    vtype_id = f"{debris_type}_{event_length:.2f}"
    if vtype_id not in vtype_dict:
        vtype_dict[vtype_id] = ET.Element("vType", id=vtype_id, length=f"{event_length:.2f}", color="128,128,128")
    
    trip = ET.Element("trip", id=elem_id, type=vtype_id,
                      depart=f"{depart_time:.1f}",
                      departPos=f"{stop_pos:.2f}",
                      **{"from": edge.getID(), "to": edge.getID()})
    ET.SubElement(trip, "stop", lane=lane.getID(),
                  endPos=f"{stop_pos:.2f}",
                  duration=f"{int(event_duration)}")
    
    # 传递给 3D：将长度通过类型后缀传递（如果 event_length > 1，则 EmergencyManager 会解析成双路障两端摆放）
    actual_event_type = f"{debris_type}_{event_length:.2f}" if event_length > 1.0 else debris_type
    ET.SubElement(trip, "param", key="event_type",  value=actual_event_type)
    ET.SubElement(trip, "param", key="model_path",  value=DEBRIS_MODELS[debris_type])
    ET.SubElement(trip, "param", key="pos_x",       value=f"{x:.4f}")
    ET.SubElement(trip, "param", key="pos_y",       value=f"{y:.4f}")
    ET.SubElement(trip, "param", key="heading",     value=f"{heading:.4f}")

    trip_list.append(trip)


def generate_road_debris(
    net_xml: str,
    base_rou_xml: str,
    output_rou_xml: str,
    rate: float = 1.5,
    barrier_ratio: float = 0.7,
    min_gap: float = 2.0,
    max_gap: float = 5.0,
    max_range: float = 200.0,
    min_duration: float = 300.0,
    max_duration: float = 900.0,
    target_junctions: list = None,
    seed: int = 42,
) -> None:
    """
    在路口进口道 max_range 米范围内随机生成路障/树枝路面碎片场景，
    路障类型每处生成前后两个（间隔 pair_gap 米），合并到原路由文件输出。

    Args:
        net_xml:          SUMO 网络文件路径 (.net.xml)
        base_rou_xml:     基础路由文件路径 (.rou.xml)
        output_rou_xml:   输出路由文件路径 (.rou.xml)
        rate:             每小时每交叉口生成路障的组数（默认 1.5）
        barrier_ratio:    路障（barrier）占全部碎片的比例（默认 0.7）
        min_gap:          前后两个路障之间的最小间距（米，默认 2.0m）
        max_gap:          前后两个路障之间的最大间距（米，默认 5.0m）
        max_range:        障碍物距路口的最大距离（米，默认 200m）
        min_duration:     障碍物最小持续时间（秒，默认 300s）
        max_duration:     障碍物最大持续时间（秒，默认 900s）
        target_junctions: 限定路口 ID 列表；None 则使用所有信控路口
        seed:             随机种子
    """
    random.seed(seed)
    logger.info(f"[Debris] 读取路网: {net_xml}")
    net = sumolib.net.readNet(net_xml)

    all_nodes = net.getNodes()
    if target_junctions:
        if isinstance(target_junctions, str):
            target_junctions = [target_junctions]
        nodes = [n for n in all_nodes if n.getID() in target_junctions]
    else:
        nodes = [n for n in all_nodes if n.getType() in
                 ('traffic_light', 'priority', 'traffic_light_right_on_red')]

    if not nodes:
        logger.error("[Debris] 未找到符合条件的路口，终止。")
        return
    
    max_depart = _get_max_depart(base_rou_xml)
    num_debris = int(len(nodes) * rate * (max_depart / 3600.0))
    if num_debris <= 0 and rate > 0:
        num_debris = 1
    
    new_vtypes_by_id = {}  # debris_type -> Element，按 MODEL_MAPPING key 去重
    new_trips        = []
    group_count      = 0   # 成功生成的障碍物组数

    for idx in range(num_debris):
        junction = random.choice(nodes)
        edges = junction.getIncoming()
        if not edges:
            continue

        edge  = random.choice(edges)
        lanes = edge.getLanes()
        if not lanes:
            continue
        lane = random.choice(lanes)

        lane_len = lane.getLength()
        min_dist = 5.0
        max_dist = min(max_range, max(lane_len - 5.0, min_dist + 1.0))
        dist_from_junction = random.uniform(min_dist, max_dist)
        # 进口道：终点近路口，stop_pos = lane_len - dist_from_junction
        center_pos = max(0.0, lane_len - dist_from_junction)

        depart_time = random.uniform(0, max_depart)

        event_duration = random.uniform(min_duration, max_duration)
        pair_gap = random.uniform(min_gap, max_gap)

        # 按比例随机选择障碍类型
        if random.random() < barrier_ratio:
            # ── 路障：单个 trip，占据长路段，在 3D 里渲染前后两个模型 ─────────────
            debris_type = random.choice(BARRIER_TYPES)
            _add_debris_element(
                new_trips, new_vtypes_by_id,
                elem_id=f"debris_{junction.getID()}_{idx}",
                edge=edge, lane=lane,
                stop_pos=center_pos + pair_gap / 2.0, # SUMO 停止点在整个事件空间的前端
                depart_time=depart_time,
                event_duration=event_duration,
                debris_type=debris_type,
                event_length=pair_gap
            )
        else:
            # ── 树枝：单个放置 ───────────────────────────────────
            debris_type = random.choice(TREEBRANCH_TYPES)
            _add_debris_element(
                new_trips, new_vtypes_by_id,
                elem_id=f"debris_{junction.getID()}_{idx}",
                edge=edge, lane=lane,
                stop_pos=center_pos,
                depart_time=depart_time,
                event_duration=event_duration,
                debris_type=debris_type,
            )

        group_count += 1

    logger.info(f"[Debris] 生成障碍物组: {group_count}，共 {len(new_trips)} 个元素")
    _merge_and_save(base_rou_xml, output_rou_xml, list(new_vtypes_by_id.values()), new_trips)


def _merge_and_save(base_rou_xml: str, output_rou_xml: str,
                    new_vtypes: list, new_trips: list) -> None:
    """将事件元素按 depart 时间排序后合并到原路由文件并保存"""
    if not os.path.exists(base_rou_xml):
        logger.error(f"[Merge] 基础路由文件不存在: {base_rou_xml}")
        return

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

    # vType 按 id 去重，避免与 base rou.xml 中已有定义冲突
    existing_vtype_ids = {e.get('id') for e in static_elems if e.tag == 'vType'}
    for vtype in new_vtypes:
        if vtype.get('id') not in existing_vtype_ids:
            static_elems.append(vtype)
            existing_vtype_ids.add(vtype.get('id'))
    dynamic_elems.extend(new_trips)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="在路口进口道生成 Road Debris 事件场景的 .rou.xml 文件"
    )
    parser.add_argument("--scenario",     "-sc", type=str, required=True,
                        help="场景名称（对应 configs/scenairo_config.py 中的键）")
    parser.add_argument("--net",          "-n",  type=str, required=True,  help=".net.xml 路径")
    parser.add_argument("--base_rou",     "-b",  type=str, required=True,  help="基础 .rou.xml 路径")
    parser.add_argument("--output",       "-o",  type=str, required=True,  help="输出 .rou.xml 路径")
    parser.add_argument("--rate",         "-rt", type=float, default=1.5,  help="每小时每交叉口路障组数（默认 1.5）")
    parser.add_argument("--barrier_ratio","-br", type=float, default=0.8,  help="路障比例（默认 0.7）")
    parser.add_argument("--min_gap",      "-ming",type=float, default=2.0, help="最小前后路障间距 m（默认 2.0）")
    parser.add_argument("--max_gap",      "-maxg",type=float, default=5.0, help="最大前后路障间距 m（默认 5.0）")
    parser.add_argument("--range",        "-r",  type=float, default=200.0,help="最大距路口距离 m（默认 200）")
    parser.add_argument("--min_duration", "-mind",type=float, default=300.0,help="最小持续时间 s（默认 300）")
    parser.add_argument("--max_duration", "-maxd",type=float, default=900.0,help="最大持续时间 s（默认 900）")
    parser.add_argument("--seed",         "-s",  type=int,   default=42,   help="随机种子（默认 42）")
    args = parser.parse_args()

    config = SCENARIO_CONFIGS.get(args.scenario)
    if config is None:
        logger.error(f"未找到场景配置: {args.scenario}")
        sys.exit(1)

    junction_names = config.get("JUNCTION_NAME")
    if isinstance(junction_names, str):
        junction_names = [junction_names]

    generate_road_debris(
        net_xml=args.net,
        base_rou_xml=args.base_rou,
        output_rou_xml=args.output,
        rate=args.rate,
        barrier_ratio=args.barrier_ratio,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        max_range=args.range,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        target_junctions=junction_names,
        seed=args.seed,
    )
