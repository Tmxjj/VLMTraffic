'''
Author: yufei Ji
Date: 2026-03-22 19:34:05
LastEditTime: 2026-03-24 20:30:06
Description: 这是一个用于离线生成 SUMO 紧急事件（如车辆碰撞、树枝掉落）的脚本。
             采用“物理和视觉解耦但数据同源”的设计：通过向原有的车辆路由文件中
             插入携带 <stop> 标签的虚拟车辆来实现 SUMO 底层物理阻塞；同时写入 
             三维模型所需的坐标和时间参数。最后将所有事件与原有真实车流严格按 
             时间（depart）进行合并排序，生成新的 rou.xml 供仿真和 3D 环境渲染使用。
FilePath: /VLMTraffic/src/utils/generate_events.py
'''
import os
import sys
import random
import math
import sumolib
import xml.etree.ElementTree as ET
from xml.dom import minidom
from loguru import logger

from configs.scenairo_config import SCENARIO_CONFIGS

EVENT_MODELS = {
    'crash': "TransSimHub/tshub/tshub_env3d/_assets_3d/vehicles_high_poly/event/crash_vehicle.glb",
    'tree_branch': "TransSimHub/tshub/tshub_env3d/_assets_3d/vehicles_high_poly/event/tree_branch_1lane.glb"
}

class EventManager:
    def __init__(self, net_xml_path=None):
        self.net_xml_path = net_xml_path
        if net_xml_path and os.path.exists(net_xml_path):
            self.net = sumolib.net.readNet(net_xml_path)
        else:
            self.net = None
        self.active_events = []

    def _get_heading_at_offset(self, shape, offset):
        """计算路段在特定 offset 处的偏航角（顺时针，正北 0，正东 90）"""
        length = 0.0
        for i in range(len(shape) - 1):
            p1 = shape[i]
            p2 = shape[i+1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if length + dist >= offset or i == len(shape) - 2:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # 在 SUMO 中，North=0, East=90. 通过 math.atan2(dx, dy) 可以得到正确的 heading弧度
                angle = math.degrees(math.atan2(dx, dy))
                return angle % 360
            length += dist
        return 0.0

    def load_events_from_xml(self, add_xml_path):
        """在仿真阶段调用：解析预先生成好的 .add.xml 文件，直接获取事件和渲染用参数"""
        self.active_events = []
        if not os.path.exists(add_xml_path):
            logger.warning(f"Event file not found: {add_xml_path}")
            return self.active_events
            
        tree = ET.parse(add_xml_path)
        root = tree.getroot()
        
        for trip in root.findall("trip"):
            event_id = trip.get("id")
            start_time = float(trip.get("depart", 0))
            
            # 解析 stop 标签，获取持续时间
            stop = trip.find("stop")
            duration = 0.0
            if stop is not None:
                duration = float(stop.get("duration", 0))
                
            # 获取专门保存给 3D 渲染器的参数
            params = {p.get("key"): p.get("value") for p in trip.findall("param")}
            
            if "pos_x" in params and "pos_y" in params:
                event = {
                    'id': event_id,
                    'type': params.get("event_type", "unknown"),
                    'x': float(params["pos_x"]),
                    'y': float(params["pos_y"]),
                    'heading': float(params.get("heading", 0.0)),
                    'start_time': start_time,
                    'end_time': start_time + duration,
                    'model_path': params.get("model_path", "")
                }
                self.active_events.append(event)
                
        logger.info(f"Loaded {len(self.active_events)} offline events from {add_xml_path}")
        return self.active_events

    def generate_events_offline(self, num_events, start_time, max_duration, base_rou_xml, output_rou_xml, target_junction_names=None):
        """离线阶段调用：随机在特定路口 60m 范围内生成事件，并合并到原路网文件中进行时间排序"""
        self.active_events = []
        if not self.net:
            logger.error("Network file not loaded. Cannot generate events.")
            return []
            
        # 获取所有节点
        all_nodes = self.net.getNodes()
        nodes = []
        
        # 筛选符合条件的路口：必须在 target_junction_names 内
        if target_junction_names:
            if not isinstance(target_junction_names, list):
                target_junction_names = [target_junction_names]
            nodes = [n for n in all_nodes if n.getID() in target_junction_names]
        else:
            # Fallback 策略
            nodes = [n for n in all_nodes if n.getType() in ['traffic_light', 'priority', 'traffic_light_right_on_red']]
        
        if not nodes:
            logger.warning("No suitable target junctions found for events!")
            return self.active_events
            
        event_vtypes = []
        event_trips = []
        
        for i in range(num_events):
            junction = random.choice(nodes)
            
            # 随机选择是在交叉口的入口道(上游)还是出口道(下游)
            direction = random.choice(['upstream', 'downstream'])
            edges = junction.getIncoming() if direction == 'upstream' else junction.getOutgoing()
            
            if not edges:
                continue
                
            edge = random.choice(edges)
            lanes = edge.getLanes()
            if not lanes: continue
            lane = random.choice(lanes)
            
            lane_len = lane.getLength()
            
            # 距离交叉口的距离：控制在5到55米范围内
            event_distance_from_intersection = random.uniform(5.0, min(55.0, lane_len))
            
            if direction == 'upstream':
                # 对于入口道(上游)，车道终点靠近交叉口，坐标为 车道总长 - 距离
                stop_pos = max(0.0, lane_len - event_distance_from_intersection)
            else:
                # 对于出口道(下游)，车道起点靠近交叉口，坐标直接为 距离
                stop_pos = event_distance_from_intersection
            
            x, y = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), stop_pos)
            heading = self._get_heading_at_offset(lane.getShape(), stop_pos)
            
            # 使用 weight 控制生成事件的概率：碰撞车辆 70%，树木 30%
            events_types = ['crash', 'tree_branch']
            events_weights = [0.7, 0.3]
            event_type = random.choices(events_types, weights=events_weights, k=1)[0]
            
            event_start = start_time + random.uniform(0, 3600) 
            event_duration = random.uniform(20, max_duration)
            
            event_id = f"event_{junction.getID()}_{i}"
            logger.info(f"Generating Event Logic: {event_id} at {lane.getID()} pos:{stop_pos}")
            
            # 构建独立的 Element
            vtype = ET.Element("vType", id=f"vtype_{event_id}", length="5", color="255,0,0")
            trip = ET.Element("trip", id=event_id, type=f"vtype_{event_id}", 
                              depart=str(float(event_start)), attrib={"from": edge.getID(), "to": edge.getID()})
            stop = ET.SubElement(trip, "stop", lane=lane.getID(), endPos=str(stop_pos), duration=str(int(event_duration)))
            
            ET.SubElement(trip, "param", key="event_type", value=event_type)
            ET.SubElement(trip, "param", key="model_path", value=EVENT_MODELS[event_type])
            ET.SubElement(trip, "param", key="pos_x", value=str(x))
            ET.SubElement(trip, "param", key="pos_y", value=str(y))
            ET.SubElement(trip, "param", key="heading", value=str(heading))
            
            event_vtypes.append(vtype)
            event_trips.append(trip)

        # 解析旧的 rou.xml
        if base_rou_xml and os.path.exists(base_rou_xml):
            tree = ET.parse(base_rou_xml)
            root = tree.getroot()
            
            static_elements = []
            dynamic_elements = []
            
            for child in list(root):
                if 'depart' in child.attrib or 'begin' in child.attrib:
                    dynamic_elements.append(child)
                else:
                    static_elements.append(child)
                # clear original
                root.remove(child)
                    
            # 融合新事件
            static_elements.extend(event_vtypes)
            dynamic_elements.extend(event_trips)
            
            # 按时间排序
            def get_time(elem):
                t_str = elem.get('depart', elem.get('begin', '0'))
                try:
                    return float(t_str)
                except ValueError:
                    return 0.0
                    
            dynamic_elements.sort(key=get_time)
            
            # 重新添加排序后的子元素
            for elem in static_elements:
                root.append(elem)
            for elem in dynamic_elements:
                root.append(elem)
                
            # 修复 XML 格式缩进问题
            if hasattr(ET, 'indent'):
                ET.indent(root, space="    ", level=0)
            else:
                def indent(elem, level=0):
                    i = "\n" + level*"    "
                    if len(elem):
                        if not elem.text or not elem.text.strip():
                            elem.text = i + "    "
                        if not elem.tail or not elem.tail.strip():
                            elem.tail = i
                        for e in elem:
                            indent(e, level+1)
                        if not elem.tail or not elem.tail.strip():
                            elem.tail = i
                    else:
                        if level and (not elem.tail or not elem.tail.strip()):
                            elem.tail = i
                indent(root)
                
            # 写入新的按时间排序完成的文件
            if output_rou_xml:
                os.makedirs(os.path.dirname(os.path.abspath(output_rou_xml)), exist_ok=True)
                tree.write(output_rou_xml, encoding="utf-8", xml_declaration=True)
                logger.info(f"💾 Saved offline SUMO merged route configuration to: {output_rou_xml}")
        else:
            logger.error(f"Base route file not found: {base_rou_xml}")
            
        return self.active_events

    def get_active_events(self, current_time):
        """获取当前时间处于激活状态的事件"""
        return [e for e in self.active_events if e['start_time'] <= current_time <= e['end_time']]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline Event Generator (Merged Sort)")

    parser.add_argument("--scenario", type=str, default="JiNan", help="Scenario key in scenairo_config.py (e.g., JiNan)")
    parser.add_argument("--net", default='data/raw/JiNan/env/jinan.net.xml', help="Path to sumo net.xml")
    parser.add_argument("--base_rou", default='data/raw/JiNan/env/anon_3_4_jinan_real.rou.xml', help="Path to base sumo rou.xml")
    parser.add_argument("--output_rou", default='data/raw/JiNan/env/anon_3_4_jinan_real_incidents.rou.xml', help="Path to output rou.xml")
    parser.add_argument("--num", type=int, default=150, help="Number of events to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # 增加对随机种子的固定
    random.seed(args.seed)
    logger.info(f"Using random seed: {args.seed}")
    
    # 动态获取配置
    config = SCENARIO_CONFIGS.get(args.scenario)
    if not config:
        logger.error(f"Error: 找不到场景配置 {args.scenario}")
        sys.exit(1)
        
    junction_names = config.get("JUNCTION_NAME")

    manager = EventManager(args.net)
    manager.generate_events_offline(
        num_events=args.num, 
        start_time=0, 
        max_duration=600, 
        base_rou_xml=args.base_rou,
        output_rou_xml=args.output_rou,
        target_junction_names=junction_names
    )

