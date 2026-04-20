'''
Author: yufei Ji
Date: 2026-04-19
LastEditTime: 2026-04-19 17:42:08
Description: 生成俯视路网图，标注各类事件的发生位置。
             支持同时叠加多种事件类型的标注（用不同颜色和图标区分），
             输出为 PNG 图像文件，每个数据集生成一张。
FilePath: /VLMTraffic/scripts/event_scene_generation/visualize_event_network.py
'''
import os
import sys
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import sumolib
import matplotlib
matplotlib.use('Agg')  # 服务器端无显示器时必须使用 Agg 后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 各事件类型的颜色与标记配置
EVENT_STYLE: Dict[str, dict] = {
    # 事故 / 障碍场景
    'crash_vehicle_a':    {'color': '#FF4444', 'marker': 'X',  'size': 80,  'label': 'Accident (Crash Vehicle A)'},
    'crash_vehicle_b':    {'color': '#FF0044', 'marker': 'X',  'size': 80,  'label': 'Accident (Crash Vehicle B)'},
    'pedestrian_lying':   {'color': '#FF8800', 'marker': 'v',  'size': 80,  'label': 'Accident (Pedestrian Lying)'},
    'barrier_A':          {'color': '#9933CC', 'marker': 's',  'size': 70,  'label': 'Road Debris (Barrier)'},
    'barrier_B':          {'color': '#9933CC', 'marker': 's',  'size': 70,  'label': 'Road Debris (Barrier)'},
    'barrier_C':          {'color': '#9933CC', 'marker': 's',  'size': 70,  'label': 'Road Debris (Barrier)'},
    'barrier_D':          {'color': '#9933CC', 'marker': 's',  'size': 70,  'label': 'Road Debris (Barrier)'},
    'barrier_E':          {'color': '#9933CC', 'marker': 's',  'size': 70,  'label': 'Road Debris (Barrier)'},
    'tree_branch_1lane':  {'color': '#228B22', 'marker': 'P',  'size': 80,  'label': 'Road Debris (Tree Branch)'},
    'tree_branch_3lanes': {'color': '#228B22', 'marker': 'P',  'size': 80,  'label': 'Road Debris (Tree Branch)'},
    # 行人过街
    'pedestrian_crossing':{'color': '#00AAFF', 'marker': 'o',  'size': 60,  'label': 'Pedestrian Crossing'},
    # 紧急车辆（来自 _emergy.rou.xml，无 param 标注，用车辆起点路径可视化）
    'emergency':          {'color': '#FF0000', 'marker': '*',  'size': 120, 'label': 'Emergency Vehicle'},
    'police':             {'color': '#0055FF', 'marker': '*',  'size': 120, 'label': 'Police Vehicle'},
    'fire_engine':        {'color': '#FF6600', 'marker': '*',  'size': 120, 'label': 'Fire Engine'},
    # 公交/校车
    'bus':                {'color': '#FFD700', 'marker': 'D',  'size': 80,  'label': 'City Bus'},
    'school_bus':         {'color': '#FFA500', 'marker': 'D',  'size': 80,  'label': 'School Bus'},
    # 默认
    'unknown':            {'color': '#888888', 'marker': '.',  'size': 40,  'label': 'Unknown Event'},
}


def _extract_events_from_rou(rou_xml: str) -> List[Tuple[float, float, str]]:
    """
    从路由文件中提取事件坐标（通过 <param key="pos_x"> / <param key="pos_y"> 标签），
    或从车辆 type 属性提取公交/紧急车辆类型（此类无 param 坐标，需通过网络定位）。

    返回: [(x, y, event_type), ...]
    """
    if not os.path.exists(rou_xml):
        return []

    events: List[Tuple[float, float, str]] = []
    try:
        tree = ET.parse(rou_xml)
        root = tree.getroot()
    except Exception as e:
        print(f"[Vis] 解析路由文件失败 {rou_xml}: {e}")
        return events

    for elem in root:
        # ── 静态障碍物 trip（含 param 坐标）─────────────────────
        if elem.tag == 'trip':
            params = {p.get('key'): p.get('value') for p in elem.findall('param')}
            if 'pos_x' in params and 'pos_y' in params:
                try:
                    x = float(params['pos_x'])
                    y = float(params['pos_y'])
                    etype = params.get('event_type', 'unknown')
                    events.append((x, y, etype))
                except ValueError:
                    pass

    return events


def _extract_special_vehicles(rou_xml: str, net: sumolib.net.Net,
                               special_types: list) -> List[Tuple[float, float, str]]:
    """
    提取公交/校车/紧急车辆的近似位置（取车辆路径第一条边的中点）。
    用于 bus/school_bus/emergency 等无 param 坐标的车辆类型。
    """
    if not os.path.exists(rou_xml):
        return []

    events = []
    try:
        tree = ET.parse(rou_xml)
        root = tree.getroot()
    except Exception:
        return events

    for veh in root.findall('vehicle'):
        vtype = veh.get('type', '')
        if vtype not in special_types:
            continue

        route_elem = veh.find('route')
        if route_elem is None:
            continue

        edge_ids = route_elem.get('edges', '').split()
        if not edge_ids:
            continue

        # 取第一条边的中点作为车辆近似显示位置
        try:
            edge = net.getEdge(edge_ids[0])
            shape = edge.getLanes()[0].getShape()
            mid = len(shape) // 2
            x, y = shape[mid][0], shape[mid][1]
            events.append((x, y, vtype))
        except Exception:
            pass

    return events


def draw_sumo_network(ax: plt.Axes, net: sumolib.net.Net) -> Tuple[float, float, float, float]:
    """
    在 matplotlib Axes 上绘制 SUMO 路网（仅绘制普通道路，忽略行人道）。
    返回网络的坐标范围 (xmin, xmax, ymin, ymax)。
    """
    xmin, ymin = float('inf'), float('inf')
    xmax, ymax = float('-inf'), float('-inf')

    for edge in net.getEdges():
        # 跳过行人道和自行车道
        if edge.getFunction() in ('crossing', 'walkingarea', 'internal'):
            continue

        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            ax.plot(xs, ys, color='#AAAAAA', linewidth=0.6, zorder=1)
            xmin = min(xmin, min(xs))
            xmax = max(xmax, max(xs))
            ymin = min(ymin, min(ys))
            ymax = max(ymax, max(ys))

    return xmin, xmax, ymin, ymax


def draw_junctions(ax: plt.Axes, net: sumolib.net.Net, target_junctions: Optional[List[str]] = None, marker_scale: float = 1.0) -> None:
    """在路网图上标注信控路口的位置（蓝色空心圆）"""
    for node in net.getNodes():
        if target_junctions and node.getID() not in target_junctions:
            continue
        # 只绘制信控路口
        if node.getType() in ('traffic_light', 'priority', 'traffic_light_right_on_red'):
            x, y = node.getCoord()
            ax.scatter(x, y, s=30 * marker_scale, c='#2266CC', marker='o',
                       edgecolors='white', linewidths=0.5, zorder=3, alpha=0.7)


def visualize_event_network(
    net_xml: str,
    rou_xml_dict: Dict[str, str],
    output_path: str,
    scenario_name: str,
    target_junctions: Optional[List[str]] = None,
    dpi: int = 150,
) -> None:
    """
    生成俯视路网图，叠加各类事件标注并保存为 PNG。

    Args:
        net_xml:          SUMO 网络文件路径 (.net.xml)
        rou_xml_dict:     事件类型名称 -> 路由文件路径的映射字典
                          例如: {'accident': '...accident.rou.xml', 'debris': '...debris.rou.xml'}
        output_path:      输出图像路径 (.png)
        scenario_name:    场景名称（显示在图标题中）
        target_junctions: 需要标注的目标路口 ID；None 则标注全部信控路口
        dpi:              输出分辨率（默认 150 DPI）
    """
    print(f"[Vis] 读取路网: {net_xml}")
    net = sumolib.net.readNet(net_xml)

    # 动态适应不同规模路网尺度
    bbox = net.getBBoxXY()
    n_xmin, n_ymin = bbox[0]
    n_xmax, n_ymax = bbox[1]
    net_width = max(1.0, n_xmax - n_xmin)
    net_height = max(1.0, n_ymax - n_ymin)

    # 按路网最大规模设置初始画布，约束在 [14, 60] 之内
    max_dim = max(net_width, net_height)
    fig_max = min(60.0, max(14.0, max_dim / 80.0))
    if net_width > net_height:
        figsize = (fig_max, max(8.0, fig_max * (net_height / net_width)))
    else:
        figsize = (max(8.0, fig_max * (net_width / net_height)), fig_max)

    # 对于大型路网，自动提高 DPI
    adjusted_dpi = max(dpi, 300)
    if max_dim > 3000.0:
        adjusted_dpi = min(800, int(300 * (max_dim / 3000.0)))
        
    print(f"[Vis] 场景尺度: {net_width:.0f}x{net_height:.0f}m, 自动优化 figsize={figsize}, dpi={adjusted_dpi}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#1A1A2E')
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_aspect('equal')

    marker_scale = (fig_max / 14.0) ** 1.5 
    font_scale = max(1.0, fig_max / 14.0)

    # ── 绘制路网底图 ───────────────────────────────────────────
    xmin, xmax, ymin, ymax = draw_sumo_network(ax, net)
    padding = max((xmax - xmin), (ymax - ymin)) * 0.04
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)

    # ── 标注目标路口 ───────────────────────────────────────────
    draw_junctions(ax, net, target_junctions, marker_scale=marker_scale)

    # ── 叠加各类事件坐标 ──────────────────────────────────────
    legend_handles: Dict[str, mpatches.Patch] = {}
    special_veh_types = ['bus', 'school_bus', 'emergency', 'police', 'fire_engine']

    for scene_key, rou_path in rou_xml_dict.items():
        if not rou_path or not os.path.exists(rou_path):
            print(f"[Vis] 跳过不存在的路由文件: {rou_path}")
            continue

        # 提取静态障碍物坐标（trip + param 方式）
        static_events = _extract_events_from_rou(rou_path)

        # 提取公交/校车/紧急车辆坐标
        vehicle_events = _extract_special_vehicles(rou_path, net, special_veh_types)

        all_events = static_events + vehicle_events
        if not all_events:
            print(f"[Vis] {scene_key}: 未提取到事件坐标")
            continue

        print(f"[Vis] {scene_key}: 提取到 {len(all_events)} 个事件点")

        for x, y, etype in all_events:
            style = EVENT_STYLE.get(etype)
            if style is None:
                for base_type in EVENT_STYLE.keys():
                    if etype.startswith(base_type):
                        style = EVENT_STYLE[base_type]
                        break
                if style is None:
                    style = EVENT_STYLE['unknown']

            ax.scatter(x, y, s=style['size'] * marker_scale, c=style['color'],
                       marker=style['marker'], zorder=5, alpha=0.85,
                       edgecolors='white', linewidths=0.3)

            # 图例去重
            label = style['label']
            if label not in legend_handles:
                legend_handles[label] = Line2D(
                    [0], [0], marker=style['marker'], color='w',
                    markerfacecolor=style['color'],
                    markersize=8 * font_scale, label=label, linestyle='None'
                )

    # ── 路口图例 ───────────────────────────────────────────────
    junction_handle = Line2D(
        [0], [0], marker='o', color='w', markerfacecolor='#2266CC',
        markeredgecolor='white', markersize=7 * font_scale,
        label='Traffic Signal Junction', linestyle='None'
    )
    all_handles = list(legend_handles.values()) + [junction_handle]

    ax.legend(handles=all_handles, loc='lower right', fontsize=7 * font_scale,
              facecolor='#2A2A3E', edgecolor='gray', labelcolor='white', framealpha=0.9)

    # ── 标题与坐标轴 ──────────────────────────────────────────
    ax.set_title(f"{scenario_name} — Event Scene Distribution (Top-Down View)",
                 color='white', fontsize=13 * font_scale, pad=12 * font_scale)
    ax.tick_params(colors='#777777', labelsize=7 * font_scale)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444455')

    ax.set_xlabel("X (m)", color='#888888', fontsize=8 * font_scale)
    ax.set_ylabel("Y (m)", color='#888888', fontsize=8 * font_scale)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=adjusted_dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Vis] 已保存: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成包含各类事件标注的俯视路网图 (PNG)"
    )
    parser.add_argument("--net",        "-n",  type=str, required=True, help=".net.xml 路径")
    parser.add_argument("--output",     "-o",  type=str, required=True, help="输出 PNG 路径")
    parser.add_argument("--scenario",   "-sc", type=str, required=True,
                        help="场景名称（对应 configs/scenairo_config.py 中的键，同时用作图标题）")
    # 各类事件路由文件（可选，按需传入）
    parser.add_argument("--emergency",  type=str, default=None, help="紧急车辆路由文件 (_emergy.rou.xml)")
    parser.add_argument("--bus",        type=str, default=None, help="公交/校车路由文件 (_bus.rou.xml)")
    parser.add_argument("--accident",   type=str, default=None, help="交通事故路由文件 (_accident.rou.xml)")
    parser.add_argument("--debris",     type=str, default=None, help="路面碎片路由文件 (_debris.rou.xml)")
    parser.add_argument("--pedestrian", type=str, default=None, help="行人过街路由文件 (_pedestrian.rou.xml)")
    parser.add_argument("--dpi",        type=int, default=150, help="输出分辨率（默认 150）")
    args = parser.parse_args()

    rou_dict = {}
    if args.emergency:
        rou_dict['emergency']  = args.emergency
    if args.bus:
        rou_dict['bus']        = args.bus
    if args.accident:
        rou_dict['accident']   = args.accident
    if args.debris:
        rou_dict['debris']     = args.debris
    if args.pedestrian:
        rou_dict['pedestrian'] = args.pedestrian

    if not rou_dict:
        print("[Vis] 警告：未传入任何事件路由文件，将仅绘制路网底图。")

    # 自动从 SCENARIO_CONFIGS 加载目标路口 ID 列表
    target_junctions = None
    try:
        config = SCENARIO_CONFIGS.get(args.scenario)
        if config:
            jn = config.get("JUNCTION_NAME")
            if isinstance(jn, str):
                target_junctions = [jn]
            elif isinstance(jn, list):
                target_junctions = jn
    except Exception:
        pass

    visualize_event_network(
        net_xml=args.net,
        rou_xml_dict=rou_dict,
        output_path=args.output,
        scenario_name=args.scenario,
        target_junctions=target_junctions,
        dpi=args.dpi,
    )
