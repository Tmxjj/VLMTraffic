'''
Author: yufei Ji
Date: 2026-01-13
Description: Configuration for TSHub Environment （class TshubEnvironment in TransSimHub/tshub/tshub_env3d/tshub_env3d.py）
'''

TSHUB_ENV_CONFIG = {
    # Builder initialization
    # 地图构建器初始化开关：如果为 False，则不处理地图相关逻辑（如获取车道信息、路网结构）
    "is_map_builder_initialized": False,
    # 车辆构建器初始化开关：如果为 True，处理车辆的生成、位置更新和状态获取
    "is_vehicle_builder_initialized": True,
    # 飞行器构建器初始化开关：如果为 True，处理无人机或空中视角的生成逻辑
    "is_aircraft_builder_initialized": True,
    # 信号灯构建器初始化开关：如果为 True，处理交通信号灯的状态获取和控制
    "is_traffic_light_builder_initialized": True,
    # 行人构建器初始化开关：如果为 True，处理行人的生成和移动逻辑
    "is_person_builder_initialized": True,

    # File paths (defaults)
    # 多边形文件路径（通常用于定义区域或建筑物轮廓），默认为 None
    "poly_file": None,
    # OpenStreetMap 文件路径，用于导入真实地图数据，默认为 None
    "osm_file": None,
    # 无线电地图文件字典（用于通信仿真等），默认为 None
    "radio_map_files": None,
    
    # Aircraft and Vehicles
    # 飞行器初始配置字典（位置、类型等），默认为 None
    "aircraft_inits": {},
    # 车辆的动作类型：'lane' 表示基于车道的控制（如更车模型），'micro' 可能表示微观控制
    "vehicle_action_type": 'lane',
    
    # Visualization
    # 是否高亮显示某些元素（如被控制的车辆或拥堵区域），默认为 False
    "hightlight": False,
    
    # TLS (Traffic Light System)
    # 信号灯动作类型：'next_or_not' (切换/保持), 'choose_next_phase' (指定相位) 等
    "tls_action_type": 'next_or_not', 
    # 动作决策的时间间隔（秒），即每隔多少秒执行一次控制动作
    "delta_time": 10,
    
    # Files
    # 额外的 NET 文件路径（在 sumo.cfg 之外指定），默认为 None
    "net_file": None, 
    # 额外的 Route 文件路径（在 sumo.cfg 之外指定），默认为 None
    "route_file": None, 

    # Simulation settings
    # sumo_gui or sumo
    "use_gui": False,
    # libsumo or Traci，默认为 （not use_gui），当你通过 use_gui=True 初始化环境时，通常会强制关闭 libsumo（使用 traci）；而当你进行后台大规模训练时，开启 libsumo 以获得最高效率。
    # "is_libsumo": False, 
    # 仿真开始时间（秒）
    "begin_time": 0,
    # 仿真总时长（秒），到达该时间后仿真强制结束
    "num_seconds": 20, 
    # 最大出发延迟（秒）：如果车辆积压太久无法进入路网，超过此时间可能会被丢弃
    "max_depart_delay": 100000,
    # 车辆瞬移等待时间（秒）：如果车辆拥堵不动超过此时间，会被瞬移或移除（-1 表示禁用）
    "time_to_teleport": -1,
    # SUMO 随机种子：'random' 表示随机，或指定固定整数以复现结果
    "sumo_seed": 'random',
    # 是否在仿真结束时输出未完成行程的车辆 tripinfo
    "tripinfo_output_unfinished": True,
    # 车辆碰撞后的动作：'warn', 'teleport', 'remove' 或 None (默认为 SUMO 配置)
    "collision_action": None,
    
    # Connection
    # 远程连接端口（用于多客户端 TraCI 连接），None 表示自动选择或单机运行
    "remote_port": None,
    # 客户端数量：如果有多个智能体同时连接 SUMO，需设置此值
    "num_clients": 1,
    
}
