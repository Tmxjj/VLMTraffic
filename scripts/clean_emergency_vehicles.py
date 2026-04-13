'''
Author: yufei Ji
Date: 2026-04-13
Description: 将 SUMO .rou.xml 文件中的紧急车辆（emergency/police/fire_engine/ambulance）
             替换为普通背景车辆（background），并删除对应的 vType 定义。
             用于净化泛化性验证场景的路由文件，确保不含紧急车辆干扰。

使用方式：
    python scripts/clean_emergency_vehicles.py        # 处理下方 FILES 列表中的所有文件
    python scripts/clean_emergency_vehicles.py --dry-run  # 仅打印统计，不修改文件
'''
import os
import argparse
import xml.etree.ElementTree as ET

# 需要被替换为普通车辆的车型 id
EMERGENCY_TYPES = {"emergency", "police", "fire_engine", "ambulance"}
# 替换目标车型（必须在 rou 文件中已存在的 vType）
NORMAL_TYPE = "background"

# 待处理文件列表（原地修改）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    os.path.join(PROJECT_ROOT, "data/raw/France_Massy/env/massy.rou.xml"),
    os.path.join(PROJECT_ROOT, "data/raw/Hongkong_YMT/env/YMT.rou.xml"),
    os.path.join(PROJECT_ROOT, "data/raw/SouthKorea_Songdo/env/songdo.rou.xml"),
]


def clean_file(path: str, dry_run: bool = False) -> None:
    """
    将文件中所有 type 属于 EMERGENCY_TYPES 的 <vehicle> 元素改为 NORMAL_TYPE，
    并移除对应的 <vType> 定义节点。
    """
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return

    # 保留注释和命名空间声明：用 iterparse 遇到问题时降级到直接 parse
    ET.register_namespace('', "http://sumo.dlr.de/xsd/routes_file.xsd")
    ET.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(path)
    root = tree.getroot()

    # 1. 统计并替换 <vehicle type="xxx"> 中的紧急车型
    replaced = 0
    for vehicle in root.findall('vehicle'):
        vtype = vehicle.get('type', '')
        if vtype in EMERGENCY_TYPES:
            vehicle.set('type', NORMAL_TYPE)
            replaced += 1

    # 2. 删除紧急车辆的 <vType> 定义节点
    removed_vtypes = []
    for vtype_elem in root.findall('vType'):
        if vtype_elem.get('id', '') in EMERGENCY_TYPES:
            removed_vtypes.append(vtype_elem.get('id'))
            if not dry_run:
                root.remove(vtype_elem)

    print(f"{'[DRY-RUN] ' if dry_run else ''}  {os.path.basename(path)}: "
          f"替换车辆 {replaced} 辆，移除 vType {removed_vtypes}")

    if not dry_run:
        tree.write(path, encoding='utf-8', xml_declaration=True)
        print(f"  ✅ 已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="Remove emergency vehicles from SUMO rou.xml files.")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不修改文件")
    args = parser.parse_args()

    print("=" * 60)
    print("清理泛化场景路由文件中的紧急车辆")
    print("=" * 60)
    for f in FILES:
        clean_file(f, dry_run=args.dry_run)
    print("=" * 60)
    print("完成。")


if __name__ == "__main__":
    main()
