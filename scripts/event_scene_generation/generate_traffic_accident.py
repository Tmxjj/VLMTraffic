'''
Author: yufei Ji
Date: 2026-04-19
LastEditTime: 2026-04-20 17:49:55
Description: Traffic Accident 事件场景生成脚本。
             在各路口进口道 200m 范围内随机放置"碰撞残骸车辆"和"倒地行人"障碍物。
             采用与 src/utils/generate_events.py 相同的 trip+stop 物理阻塞 + param 渲染参数设计。
FilePath: /VLMTraffic/scripts/event_scene_generation/generate_traffic_accident.py
'''
import os
import sys
import math
import random
import argparse
import xml.etree.ElementTree as ET

# 将项目根目录加入搜索路径，以便 import configs
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import sumolib
from loguru import logger
from configs.scenairo_config import SCENARIO_CONFIGS

# 事故场景使用的 3D 模型路径（相对项目根目录）
_ASSET_BASE = "TransSimHub/tshub/tshub_env3d/_assets_3d/vehicles_high_poly"
ACCIDENT_MODELS = {
    'crash_vehicle_a':    f"{_ASSET_BASE}/event/crash_vehicle_a.glb",
    'crash_vehicle_b':    f"{_ASSET_BASE}/event/crash_vehicle_b.glb",
    # 'crash_vehicle_lane2': f"{_ASSET_BASE}/event/crash_vehicle_lane2.glb",
    'pedestrian_lying':   f"{_ASSET_BASE}/event/pedestrian.glb",
}


def _get_max_depart(rou_xml: str) -> float:
    """快速解析路由文件，获取最大出发时间以获取可生成事件的时长"""
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


def generate_traffic_accident(
    net_xml: str,
    base_rou_xml: str,
    output_rou_xml: str,
    rate: float = 1.5,
    pedestrian_ratio: float = 0.5,
    max_range: float = 200.0,
    event_duration: float = 600.0,
    target_junctions: list = None,
    seed: int = 42,
) -> None:
    """
    在路口进口道 max_range 米范围内随机生成车辆碰撞事故（及倒地行人）场景，
    并合并到原有路由文件中输出。

    Args:
        net_xml:          SUMO 网络文件路径 (.net.xml)
        base_rou_xml:     基础路由文件路径 (.rou.xml)
        output_rou_xml:   输出路由文件路径 (.rou.xml)
        rate:             事故发生频率（每小时每交叉口生成数），默认 1.5
        pedestrian_ratio: 同时生成倒地行人的概率 (0-1)
        max_range:        事故距路口的最大距离（米），默认 200m
        event_duration:   事故持续时间（秒），默认 600s
        target_junctions: 限定事故发生的路口 ID 列表；None 则使用所有信控路口
        seed:             随机种子
    """
    random.seed(seed)
    logger.info(f"[Accident] 读取路网: {net_xml}")
    net = sumolib.net.readNet(net_xml)

    # 筛选目标路口
    all_nodes = net.getNodes()
    if target_junctions:
        if isinstance(target_junctions, str):
            target_junctions = [target_junctions]
        nodes = [n for n in all_nodes if n.getID() in target_junctions]
    else:
        nodes = [n for n in all_nodes if n.getType() in
                 ('traffic_light', 'priority', 'traffic_light_right_on_red')]

    if not nodes:
        logger.error("[Accident] 未找到符合条件的路口，终止。")
        return

    max_depart = _get_max_depart(base_rou_xml)
    num_accidents = int(len(nodes) * rate * (max_depart / 3600.0))
    if num_accidents <= 0 and rate > 0:
        num_accidents = 1

    # 预分配事故模型，保证如果生成2次或以上，模型A和B都会被包含
    available_models = ['crash_vehicle_a', 'crash_vehicle_b']
    if num_accidents >= len(available_models):
        assigned_models = available_models.copy()
        assigned_models.extend([random.choice(available_models) for _ in range(num_accidents - len(available_models))])
    else:
        assigned_models = [random.choice(available_models) for _ in range(num_accidents)]
    random.shuffle(assigned_models)

    # vType 按 MODEL_MAPPING 的 key 命名，多个事故共享同一 vType 定义
    event_vtypes_by_id = {}  # id -> Element，去重
    event_trips        = []
    event_count        = 0

    for idx in range(num_accidents):
        junction = random.choice(nodes)
        # 仅在进口道（incoming edges）上生成事故
        edges = junction.getIncoming()
        if not edges:
            continue

        edge  = random.choice(edges)
        lanes = edge.getLanes()
        if not lanes:
            continue

        depart_time = random.uniform(0, max_depart)
        event_id = f"accident_{junction.getID()}_{idx}"

        # 始终生成单车道事故，使用 crash_vehicle 或 crash_vehicle_b (各占50%)
        is_lane2_crash = False
        is_lane3_crash = False

        if is_lane3_crash:
            # 随机选择相邻的三个车道
            l_idx = random.randint(0, len(lanes) - 3)
            lane1 = lanes[l_idx]
            lane2 = lanes[l_idx + 1] # 中间车道
            lane3 = lanes[l_idx + 2]

            lane_len = lane2.getLength()
            min_dist = 5.0
            max_dist = min(max_range, max(lane_len - 5.0, min_dist + 1.0))
            dist_from_junction = random.uniform(min_dist, max_dist)
            stop_pos = max(0.0, lane_len - dist_from_junction)

            # 三车道事故将 3D 模型渲染中心放置在中间车道的坐标
            x, y = sumolib.geomhelper.positionAtShapeOffset(lane2.getShape(), stop_pos)
            heading = _get_heading_at_offset(lane2.getShape(), stop_pos)

            # ── 碰撞车辆（跨三车道的 3D 模型配置，绑定于车道 2） ────────────────
            event_vtypes_by_id.setdefault(
                'crash_vehicle_b',
                ET.Element("vType", id="crash_vehicle_b", length="10", color="255,0,0")
            )
            trip1 = ET.Element("trip", id=event_id,
                               type="crash_vehicle_b",
                               depart=f"{depart_time:.1f}",
                               **{"from": edge.getID(), "to": edge.getID()})
            ET.SubElement(trip1, "stop", lane=lane2.getID(),
                          endPos=f"{stop_pos:.2f}",
                          duration=f"{int(event_duration)}")
            ET.SubElement(trip1, "param", key="event_type",  value="crash3")
            ET.SubElement(trip1, "param", key="model_path",  value=ACCIDENT_MODELS['crash_vehicle_b'])
            ET.SubElement(trip1, "param", key="pos_x",       value=f"{x:.4f}")
            ET.SubElement(trip1, "param", key="pos_y",       value=f"{y:.4f}")
            ET.SubElement(trip1, "param", key="heading",     value=f"{heading:.4f}")

            # ── 物理阻塞伪装车（隐藏 3D param 渲染），分别阻塞车道 1 和车道 3 ──────
            event_vtypes_by_id.setdefault(
                'crash_vehicle_a',
                ET.Element("vType", id="crash_vehicle_a", length="10", color="255,0,0")
            )
            trip2 = ET.Element("trip", id=event_id + "_blocker1",
                               type="crash_vehicle_a",
                               depart=f"{depart_time:.1f}",
                               **{"from": edge.getID(), "to": edge.getID()})
            ET.SubElement(trip2, "stop", lane=lane1.getID(),
                          endPos=f"{stop_pos:.2f}",
                          duration=f"{int(event_duration)}")

            trip3 = ET.Element("trip", id=event_id + "_blocker2",
                               type="crash_vehicle_a",
                               depart=f"{depart_time:.1f}",
                               **{"from": edge.getID(), "to": edge.getID()})
            ET.SubElement(trip3, "stop", lane=lane3.getID(),
                          endPos=f"{stop_pos:.2f}",
                          duration=f"{int(event_duration)}")

            event_trips.extend([trip1, trip2, trip3])
            event_count += 1
            lane = lane2  # 保留 lane 引用供下方的行人生成逻辑使用

        # elif is_lane2_crash:
        #     # 随机选择相邻的两个车道
        #     l_idx = random.randint(0, len(lanes) - 2)
        #     lane1 = lanes[l_idx]
        #     lane2 = lanes[l_idx + 1]

        #     lane_len = lane1.getLength()
        #     min_dist = 5.0
        #     max_dist = min(max_range, max(lane_len - 5.0, min_dist + 1.0))
        #     dist_from_junction = random.uniform(min_dist, max_dist)
        #     stop_pos = max(0.0, lane_len - dist_from_junction)

        #     # 获取相邻两个车道的坐标
        #     x1, y1 = sumolib.geomhelper.positionAtShapeOffset(lane1.getShape(), stop_pos)
        #     x2, y2 = sumolib.geomhelper.positionAtShapeOffset(lane2.getShape(), stop_pos)
            
        #     # 双车道事故将 3D 模型渲染中心放置在两个车道坐标空间的中点
        #     x = (x1 + x2) / 2.0
        #     y = (y1 + y2) / 2.0
        #     heading = _get_heading_at_offset(lane1.getShape(), stop_pos)

        #     # ── 碰撞车辆（跨两车道的 3D 模型配置，绑定于车道 1） ────────────────
        #     event_vtypes_by_id.setdefault(
        #         'crash_vehicle_lane2',
        #         ET.Element("vType", id="crash_vehicle_lane2", length="10", color="255,0,0")
        #     )
        #     trip1 = ET.Element("trip", id=event_id,
        #                        type="crash_vehicle_lane2",
        #                        depart=f"{depart_time:.1f}",
        #                        **{"from": edge.getID(), "to": edge.getID()})
        #     ET.SubElement(trip1, "stop", lane=lane1.getID(),
        #                   endPos=f"{stop_pos:.2f}",
        #                   duration=f"{int(event_duration)}")
        #     ET.SubElement(trip1, "param", key="event_type",  value="crash2")
        #     ET.SubElement(trip1, "param", key="model_path",  value=ACCIDENT_MODELS['crash_vehicle_lane2'])
        #     ET.SubElement(trip1, "param", key="pos_x",       value=f"{x:.4f}")
        #     ET.SubElement(trip1, "param", key="pos_y",       value=f"{y:.4f}")
        #     ET.SubElement(trip1, "param", key="heading",     value=f"{heading:.4f}")

        #     # ── 物理阻塞伪装车（隐藏 3D param 渲染，仅实现 SUMO 物理拥堵） ──────
        #     event_vtypes_by_id.setdefault(
        #         'crash_vehicle_a',
        #         ET.Element("vType", id="crash_vehicle_a", length="10", color="255,0,0")
        #     )
        #     trip2 = ET.Element("trip", id=event_id + "_blocker",
        #                        type="crash_vehicle_a",
        #                        depart=f"{depart_time:.1f}",
        #                        **{"from": edge.getID(), "to": edge.getID()})
        #     ET.SubElement(trip2, "stop", lane=lane2.getID(),
        #                   endPos=f"{stop_pos:.2f}",
        #                   duration=f"{int(event_duration)}")

        #     event_trips.extend([trip1, trip2])
        #     event_count += 1
        #     lane = lane1  # 保留 lane 引用供下方的行人生成逻辑使用
        
        else:
            lane = random.choice(lanes)

            lane_len = lane.getLength()
            # 事故位置：距路口 5m ~ min(max_range, lane_len - 5m)
            min_dist = 5.0
            max_dist = min(max_range, max(lane_len - 5.0, min_dist + 1.0))
            dist_from_junction = random.uniform(min_dist, max_dist)
            # 进口道终点靠近路口，stop_pos = 从起点算起 lane_len - dist
            stop_pos = max(0.0, lane_len - dist_from_junction)

            x, y = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), stop_pos)
            heading = _get_heading_at_offset(lane.getShape(), stop_pos)
            
            # 按预分配列表获取模型，保证生成两次以上时模型A和B都会出现
            crash_model = assigned_models[idx]

            # ── 单车道碰撞车辆 ──────────────────────────────────────────────
            event_vtypes_by_id.setdefault(
                crash_model,
                ET.Element("vType", id=crash_model, length="10", color="255,0,0")
            )
            trip_crash = ET.Element("trip", id=event_id,
                                    type=crash_model,
                                    depart=f"{depart_time:.1f}",
                                    **{"from": edge.getID(), "to": edge.getID()})
            ET.SubElement(trip_crash, "stop", lane=lane.getID(),
                          endPos=f"{stop_pos:.2f}",
                          duration=f"{int(event_duration)}")
            ET.SubElement(trip_crash, "param", key="event_type",  value=crash_model)
            ET.SubElement(trip_crash, "param", key="model_path",  value=ACCIDENT_MODELS[crash_model])
            ET.SubElement(trip_crash, "param", key="pos_x",       value=f"{x:.4f}")
            ET.SubElement(trip_crash, "param", key="pos_y",       value=f"{y:.4f}")
            ET.SubElement(trip_crash, "param", key="heading",     value=f"{heading:.4f}")

            event_trips.append(trip_crash)
            event_count += 1

        # ── 倒地行人（按概率生成）────────────────────────────────
        # if random.random() < pedestrian_ratio:
        #     # 行人略微偏离车辆停放位置（±2m）
        #     ped_offset = stop_pos + random.uniform(-2.0, 2.0)
        #     ped_offset = max(0.0, min(lane_len, ped_offset))
        #     px, py = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), ped_offset)
        #     ped_id = f"pedestrian_{junction.getID()}_{idx}"

        #     event_vtypes_by_id.setdefault(
        #         'pedestrian_lying',
        #         ET.Element("vType", id="pedestrian_lying", length="0.5", color="255,200,0")
        #     )
        #     trip_ped = ET.Element("trip", id=ped_id,
        #                           type="pedestrian_lying",
        #                           depart=f"{depart_time:.1f}",
        #                           **{"from": edge.getID(), "to": edge.getID()})
        #     ET.SubElement(trip_ped, "stop", lane=lane.getID(),
        #                   endPos=f"{ped_offset:.2f}",
        #                   duration=f"{int(event_duration)}")
        #     ET.SubElement(trip_ped, "param", key="event_type",  value="pedestrian_lying")
        #     ET.SubElement(trip_ped, "param", key="model_path",  value=ACCIDENT_MODELS['pedestrian_lying'])
        #     ET.SubElement(trip_ped, "param", key="pos_x",       value=f"{px:.4f}")
        #     ET.SubElement(trip_ped, "param", key="pos_y",       value=f"{py:.4f}")
        #     ET.SubElement(trip_ped, "param", key="heading",     value=f"{heading:.4f}")

        #     event_trips.append(trip_ped)

    logger.info(f"[Accident] 生成事故数: {event_count}（含行人 {len(event_trips) - event_count} 个）")
    _merge_and_save(base_rou_xml, output_rou_xml, list(event_vtypes_by_id.values()), event_trips)


def _merge_and_save(base_rou_xml: str, output_rou_xml: str,
                    new_vtypes: list, new_trips: list) -> None:
    """将新生成的事件元素按 depart 时间排序后合并到原路由文件并保存"""
    if not os.path.exists(base_rou_xml):
        logger.error(f"[Merge] 基础路由文件不存在: {base_rou_xml}")
        return

    tree = ET.parse(base_rou_xml)
    root = tree.getroot()

    # 分离静态元素（无 depart/begin 属性，如 vType）和动态元素（vehicle/trip/flow）
    static_elems   = []
    dynamic_elems  = []
    for child in list(root):
        if 'depart' in child.attrib or 'begin' in child.attrib:
            dynamic_elems.append(child)
        else:
            static_elems.append(child)
        root.remove(child)

    # 融合新事件：vType 按 id 去重，避免与 base rou.xml 中已有定义冲突
    existing_vtype_ids = {e.get('id') for e in static_elems if e.tag == 'vType'}
    for vtype in new_vtypes:
        if vtype.get('id') not in existing_vtype_ids:
            static_elems.append(vtype)
            existing_vtype_ids.add(vtype.get('id'))
    dynamic_elems.extend(new_trips)

    # 按 depart/begin 时间升序排列
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

    # 修复缩进（Python ≥ 3.9 支持 ET.indent）
    if hasattr(ET, 'indent'):
        ET.indent(root, space="    ", level=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_rou_xml)), exist_ok=True)
    tree.write(output_rou_xml, encoding='utf-8', xml_declaration=True)
    logger.info(f"[Merge] 已保存: {output_rou_xml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="在路口进口道生成 Traffic Accident 事件场景的 .rou.xml 文件"
    )
    parser.add_argument("--scenario",  "-sc", type=str, required=True,
                        help="场景名称，对应 configs/scenairo_config.py 中的键（如 JiNan）")
    parser.add_argument("--net",       "-n",  type=str, required=True,  help=".net.xml 路径")
    parser.add_argument("--base_rou",  "-b",  type=str, required=True,  help="基础 .rou.xml 路径")
    parser.add_argument("--output",    "-o",  type=str, required=True,  help="输出 .rou.xml 路径")
    parser.add_argument("--rate",      "-rt", type=float, default=1.5,  help="每小时每交叉口生成数（默认 1.5）")
    parser.add_argument("--ped_ratio", "-p",  type=float, default=0.5,  help="倒地行人生成概率（默认 0.5）")
    parser.add_argument("--range",     "-r",  type=float, default=200.0,help="距路口最大距离 m（默认 200）")
    parser.add_argument("--duration",  "-d",  type=float, default=600.0,help="事故持续时间 s（默认 600）")
    parser.add_argument("--seed",      "-s",  type=int,   default=42,   help="随机种子（默认 42）")
    args = parser.parse_args()

    config = SCENARIO_CONFIGS.get(args.scenario)
    if config is None:
        logger.error(f"未找到场景配置: {args.scenario}")
        sys.exit(1)

    junction_names = config.get("JUNCTION_NAME")
    if isinstance(junction_names, str):
        junction_names = [junction_names]

    generate_traffic_accident(
        net_xml=args.net,
        base_rou_xml=args.base_rou,
        output_rou_xml=args.output,
        rate=args.rate,
        pedestrian_ratio=args.ped_ratio,
        max_range=args.range,
        event_duration=args.duration,
        target_junctions=junction_names,
        seed=args.seed,
    )
