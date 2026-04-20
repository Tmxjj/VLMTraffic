'''
Author: yufei Ji
Date: 2026-04-19
LastEditTime: 2026-04-19
Description: School/City Bus 事件场景生成脚本。
             向 SUMO .rou.xml 文件中随机注入公交车和校车，
             逻辑与 scripts/add_emergency_vehicles.py 保持一致。
FilePath: /VLMTraffic/scripts/event_scene_generation/add_bus_vehicles.py
'''
import os
import random
import argparse
import xml.etree.ElementTree as ET


def add_bus_vehicles(input_file: str, output_file: str, ratio: float = 0.05) -> None:
    """
    修改 SUMO 的 .rou.xml 文件，随机将指定比例的普通车辆改为公交车或校车。
    生成的文件供 School/City Bus 事件场景评测使用。

    Args:
        input_file:  原始路由文件路径（.rou.xml）
        output_file: 输出路由文件路径（.rou.xml）
        ratio:       注入比例，默认 5%
    """
    print(f"[Bus] Loading '{input_file}' ...")

    # 注册命名空间，防止写出时丢失 xmlns 声明
    ET.register_namespace('', "http://sumo.dlr.de/xsd/routes_file.xsd")
    ET.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")

    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except Exception as e:
        print(f"[Bus] 解析 XML 失败: {e}")
        return

    # 公交车和校车的 vType 定义（与 3D 渲染侧的 MODEL_MAPPING 保持一致）
    BUS_VTYPES = [
        {"id": "bus",        "length": "12.00", "color": "255,255,0",  "tau": "1.5", "maxSpeed": "16.67"},
        {"id": "school_bus", "length": "11.00", "color": "255,165,0",  "tau": "1.5", "maxSpeed": "13.89"},
    ]

    # 仅在文件中尚未包含对应 vType 时才插入
    existing_vtype_ids = {elem.get('id') for elem in root.findall('vType')}
    for vtype_dict in BUS_VTYPES:
        if vtype_dict['id'] not in existing_vtype_ids:
            vtype_elem = ET.Element('vType', attrib=vtype_dict)
            root.insert(0, vtype_elem)

    # 找到所有普通车辆节点（<vehicle> 元素）
    vehicles = root.findall('vehicle')
    total = len(vehicles)
    print(f"[Bus] 共发现 {total} 辆车辆")

    if total == 0:
        print("[Bus] 未找到 <vehicle> 节点，退出。")
        return

    num_to_modify = max(1, int(total * ratio))
    print(f"[Bus] 注入比例 {ratio*100:.1f}% → 将修改 {num_to_modify} 辆为公交/校车")

    # 随机抽样需要修改的车辆索引
    indices = set(random.sample(range(total), min(num_to_modify, total)))
    bus_ids = [vt['id'] for vt in BUS_VTYPES]

    modified = 0
    for i, veh in enumerate(vehicles):
        if i in indices:
            veh.set('type', random.choice(bus_ids))
            modified += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"[Bus] 成功注入 {modified} 辆公交/校车 → '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="向 SUMO 路由文件注入公交车/校车（School/City Bus 事件场景）"
    )
    parser.add_argument("--input",  "-i", type=str, required=True,  help="输入 .rou.xml 路径")
    parser.add_argument("--output", "-o", type=str, required=True,  help="输出 .rou.xml 路径")
    parser.add_argument("--ratio",  "-r", type=float, default=0.05, help="注入比例（默认 0.05 = 5%%）")
    parser.add_argument("--seed",   "-s", type=int,   default=42,   help="随机种子（默认 42）")
    args = parser.parse_args()

    random.seed(args.seed)
    add_bus_vehicles(args.input, args.output, args.ratio)
