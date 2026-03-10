'''
Author: yufei Ji
Date: 2026-03-08 16:40:58
LastEditTime: 2026-03-08 16:47:27
Description: this script is used to 
FilePath: /VLMTraffic/scripts/add_emergency_vehicles.py
'''
import os
import random
import xml.etree.ElementTree as ET
import argparse

def add_emergency_vehicles(input_file, output_file, ratio=0.02):
    """
    修改 SUMO 的 .rou.xml 文件，随机将指定比例的普通车辆（默认）修改为紧急车辆。
    保留原文件的其他内容不变。
    """
    print(f"Loading '{input_file}'...")
    
    # 注册命名空间以防丢失
    ET.register_namespace('', "http://sumo.dlr.de/xsd/routes_file.xsd")
    ET.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")
    
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return

    # 定义紧急车辆的 vType
    emergency_types = [
        {"id": "emergency", "length": "6.50", "color": "255,165,0", "tau": "1.0"},
        {"id": "police", "length": "5.00", "color": "blue", "tau": "1.0"},
        {"id": "fire_engine", "length": "8.00", "color": "red", "vClass": "emergency"}
    ]
    
    # 检查是否已经有了对应的 vType，如果没有则添加到根节点开头
    existing_vtypes = [elem.get('id') for elem in root.findall('vType')]
    for vtype in emergency_types:
        if vtype['id'] not in existing_vtypes:
            vtype_elem = ET.Element('vType', attrib=vtype)
            root.insert(0, vtype_elem)  # 插入到最前面

    # 找到所有车辆节点
    vehicles = root.findall('vehicle')
    total_vehicles = len(vehicles)
    print(f"Total vehicles found: {total_vehicles}")
    
    if total_vehicles == 0:
        print("No <vehicle> nodes found. Exiting.")
        return
        
    num_to_modify = int(total_vehicles * ratio)
    print(f"Targeting ~{ratio*100}% of vehicles -> {num_to_modify} vehicles will be turned into emergencies.")
    
    # 随机抽取车辆进行修改
    indices_to_modify = random.sample(range(total_vehicles), num_to_modify)
    indices_to_modify_set = set(indices_to_modify)
    
    modified_count = 0
    emergency_type_ids = [vt['id'] for vt in emergency_types]
    
    for i, vehicle in enumerate(vehicles):
        if i in indices_to_modify_set:
            # 随机选择一种特殊的紧急车型
            chosen_type = random.choice(emergency_type_ids)
            vehicle.set('type', chosen_type)
            modified_count += 1
            
    # 保存输出文件
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Successfully modified {modified_count} vehicles.")
    print(f"Saved to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject random emergency vehicles into SUMO standard route files.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .rou.xml file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output .rou.xml file")
    parser.add_argument("--ratio", "-r", type=float, default=0.04, help="Ratio of emergency vehicles (default: 0.02 = 2%%)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    add_emergency_vehicles(args.input, args.output, args.ratio)

