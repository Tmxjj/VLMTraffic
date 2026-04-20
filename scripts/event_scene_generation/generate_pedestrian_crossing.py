'''
Author: yufei Ji
Date: 2026-04-19
LastEditTime: 2026-04-20 19:42:41
Description: Pedestrian Crossing 事件场景生成脚本。
             在各路口进口道停止线后方（交叉口内）放置行人过街模型，模拟大规模行人过街场景（如学校放学时段）。
             行人集中分布在横穿马路的方向（离停止线 1~3m 的交叉口区域内）。
FilePath: /VLMTraffic/scripts/event_scene_generation/generate_pedestrian_crossing.py
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

# 行人过街的 3D 模型路径（注意实际文件夹为 pedestrain，存在拼写）
PEDESTRIAN_CROSSING_MODEL = f"{_ASSET_BASE}/pedestrain/pedestrain_a.glb"


def _get_max_depart(rou_xml: str) -> float:
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


def generate_pedestrian_crossing(
    net_xml: str,
    base_rou_xml: str,
    output_rou_xml: str,
    rate: float = 1.0,
    pedestrians_per_crossing: int = 4,
    stop_line_range: tuple = (1.0, 3.0),
    spacing: float = 1.5,
    event_duration: float = 20.0,
    target_junctions: list = None,
    seed: int = 42,
) -> None:
    """
    在路口交叉口区域放置行人过街模型。
    每个过街事件在横穿马路方向上生成 pedestrians_per_crossing 个行人，
    行人之间沿马路横切面均匀间隔 spacing 米。

    Args:
        net_xml:                  SUMO 网络文件路径 (.net.xml)
        base_rou_xml:             基础路由文件路径 (.rou.xml)
        output_rou_xml:           输出路由文件路径 (.rou.xml)
        rate:                     每小时每交叉口生成行人过街事件的组数（默认 1.0）
        pedestrians_per_crossing: 每组事件放置的行人数（默认 4）
        stop_line_range:          行人距停止线的距离范围（米），默认 (1.0, 3.0)
        spacing:                  同组行人之间的间距（米，默认 1.5m）
        event_duration:           行人停留时间（秒，默认 10s）
        target_junctions:         限定路口 ID 列表；None 则使用所有信控路口
        seed:                     随机种子
    """
    random.seed(seed)
    logger.info(f"[Pedestrian] 读取路网: {net_xml}")
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
        logger.error("[Pedestrian] 未找到符合条件的路口，终止。")
        return

    max_depart = _get_max_depart(base_rou_xml)
    num_crossings = int(len(nodes) * rate * (max_depart / 3600.0))
    if num_crossings <= 0 and rate > 0:
        num_crossings = 1

    # 所有行人共享同一 vType 定义，id 与 Vehicle3DElement.MODEL_MAPPING 的 key 一致
    new_vtypes_by_id = {}
    new_trips        = []
    total_peds       = 0

    for grp_idx in range(num_crossings):
        junction = random.choice(nodes)
        # 行人站在进口道停止线前，选进口道
        edges = junction.getIncoming()
        if not edges:
            continue

        edge  = random.choice(edges)
        lanes = edge.getLanes()
        if not lanes:
            continue

        # 优先选 lanes 中最左侧或最右侧车道，确保行人出现在视觉显眼位置
        # 这里随机选择一条车道作为基准
        base_lane = random.choice(lanes)
        lane_len  = base_lane.getLength()

        dist_from_junction = random.uniform(*stop_line_range)
        # 将基础位置设在进口道末端（停止线附近）
        base_pos = max(0.1, lane_len - 0.1)

        depart_time = random.uniform(0, max_depart)

        base_x, base_y = sumolib.geomhelper.positionAtShapeOffset(base_lane.getShape(), base_pos)
        heading = _get_heading_at_offset(base_lane.getShape(), base_pos)
        heading_rad = math.radians(heading)

        # 沿车道方向向前延伸 dist_from_junction 进入交叉口区域
        cross_x = base_x + dist_from_junction * math.sin(heading_rad)
        cross_y = base_y + dist_from_junction * math.cos(heading_rad)

        # 生成随机的横向累积间距，使行人间距随机分布
        random_offsets = [0.0]
        for _ in range(1, pedestrians_per_crossing):
            random_offsets.append(random_offsets[-1] + spacing * random.uniform(0.5, 2))
        
        # 整体居中
        center_offset = random_offsets[-1] / 2.0
        random_offsets = [o - center_offset for o in random_offsets]

        for ped_idx in range(pedestrians_per_crossing):
            # 沿垂直于车道方向（横穿马路）排列，使用随机生成的间距
            lateral_offset = random_offsets[ped_idx]
            
            # 向右侧偏移，由于 heading = 0 为正北，增加 x(向东)
            # 正确的右侧法向量是 dx = cos(heading_rad), dy = -sin(heading_rad)
            ped_x = cross_x + lateral_offset * math.cos(heading_rad)
            ped_y = cross_y - lateral_offset * math.sin(heading_rad)

            ped_pos  = base_pos
            ped_id   = f"ped_{junction.getID()}_{grp_idx}_{ped_idx}"

            # 行人朝向垂直于行车方向（横穿方向）
            crossing_heading = (heading + 90.0) % 360.0

            # vType id 必须与 Vehicle3DElement.MODEL_MAPPING 的 key 完全一致，按 id 去重
            new_vtypes_by_id.setdefault(
                'pedestrian_crossing',
                ET.Element("vType", id="pedestrian_crossing", length="0.5", color="0,200,255")
            )
            trip = ET.Element("trip", id=ped_id,
                              type="pedestrian_crossing",
                              depart=f"{depart_time:.1f}",
                              departPos=f"{ped_pos:.2f}",
                              **{"from": edge.getID(), "to": edge.getID()})
            ET.SubElement(trip, "stop", lane=base_lane.getID(),
                          endPos=f"{ped_pos:.2f}",
                          duration=f"{int(event_duration)}")
            ET.SubElement(trip, "param", key="event_type",  value="pedestrian_crossing")
            ET.SubElement(trip, "param", key="model_path",  value=PEDESTRIAN_CROSSING_MODEL)
            ET.SubElement(trip, "param", key="pos_x",       value=f"{ped_x:.4f}")
            ET.SubElement(trip, "param", key="pos_y",       value=f"{ped_y:.4f}")
            ET.SubElement(trip, "param", key="heading",     value=f"{crossing_heading:.4f}")

            new_trips.append(trip)
            total_peds += 1

    logger.info(f"[Pedestrian] 生成过街事件组: {num_crossings}，共 {total_peds} 个行人")
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
        description="在路口进口道停止线前生成 Pedestrian Crossing 事件场景的 .rou.xml 文件"
    )
    parser.add_argument("--scenario",   "-sc", type=str, required=True,
                        help="场景名称（对应 configs/scenairo_config.py 中的键）")
    parser.add_argument("--net",        "-n",  type=str, required=True,  help=".net.xml 路径")
    parser.add_argument("--base_rou",   "-b",  type=str, required=True,  help="基础 .rou.xml 路径")
    parser.add_argument("--output",     "-o",  type=str, required=True,  help="输出 .rou.xml 路径")
    parser.add_argument("--rate",       "-rt", type=float, default=1.0,  help="每交叉口每小时事件组数（默认 1.0）")
    parser.add_argument("--per_group",  "-pg", type=int,   default=4,    help="每组行人数（默认 4）")
    parser.add_argument("--min_dist",   "-mn", type=float, default=1.0,  help="离停止线后方的最小距离 m（默认 1.0）")
    parser.add_argument("--max_dist",   "-mx", type=float, default=3.0,  help="离停止线后方的最大距离 m（默认 3.0）")
    parser.add_argument("--spacing",    "-sp", type=float, default=1.5,  help="行人间距 m（默认 1.5）")
    parser.add_argument("--duration",   "-d",  type=float, default=20.0, help="持续时间 s（默认 20），考虑老年人过街较慢")
    parser.add_argument("--seed",       "-s",  type=int,   default=42,   help="随机种子（默认 42）")
    args = parser.parse_args()

    config = SCENARIO_CONFIGS.get(args.scenario)
    if config is None:
        logger.error(f"未找到场景配置: {args.scenario}")
        sys.exit(1)

    junction_names = config.get("JUNCTION_NAME")
    if isinstance(junction_names, str):
        junction_names = [junction_names]

    generate_pedestrian_crossing(
        net_xml=args.net,
        base_rou_xml=args.base_rou,
        output_rou_xml=args.output,
        rate=args.rate,
        pedestrians_per_crossing=args.per_group,
        stop_line_range=(args.min_dist, args.max_dist),
        spacing=args.spacing,
        event_duration=args.duration,
        target_junctions=junction_names,
        seed=args.seed,
    )
