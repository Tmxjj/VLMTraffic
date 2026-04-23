'''
Author: yufei Ji
Date: 2026-01-12 17:09:21
LastEditTime: 2026-04-23 21:37:58
Description: 各场景配置。
    SENSOR_CFG 格式说明：
      - tls: {tls_id: {sensor_types, tls_camera_height}}   进口道摄像头（停止线处俯拍）
      - upstream: {tls_id: {sensor_types, tls_camera_height}}  上游摄像头（进口道 lane 起点处俯拍）
      - aircraft: {sensor_types, height, [camera_heading]}  全局 BEV（评测对比保留）
    多路口场景（JiNan / Hangzhou / NewYork）的 tls/upstream 键用列表推导式批量生成。
FilePath: /VLMTraffic/configs/scenairo_config.py
'''

# JiNan 路口 ID 列表（4×3=12 个路口）
_JINAN_JUNCTIONS  = [f"intersection_{i}_{j}" for i in range(1, 5) for j in range(1, 4)]
# Hangzhou 路口 ID 列表（4×4=16 个路口）
_HANGZHOU_JUNCTIONS = [f"intersection_{i}_{j}" for i in range(1, 5) for j in range(1, 5)]
# NewYork 路口 ID 列表（7×28=196 个路口）
_NEWYORK_JUNCTIONS = [f"intersection_{i}_{j}" for i in range(1, 8) for j in range(1, 29)]

# 通用进口道/上游摄像头参数
_APPROACH_CAM_CFG = {"sensor_types": ["junction_front_all"], "tls_camera_height": 15}

def _generate_grid_topology(m: int, n: int) -> dict:
    """
    针对 m*n 规则路网（横向东西方向 m 个，纵向南北方向 n 个），构建合法拓扑字典。
    只在这个内部相互传播，排除掉外围的虚拟交叉口。
    """
    topology = {}
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            jid = f"intersection_{i}_{j}"
            neighbors = []
            if i < m: neighbors.append(f"intersection_{i+1}_{j}")
            if i > 1: neighbors.append(f"intersection_{i-1}_{j}")
            if j < n: neighbors.append(f"intersection_{i}_{j+1}")
            if j > 1: neighbors.append(f"intersection_{i}_{j-1}")
            topology[jid] = neighbors
    return topology

def _tls_sensor_cfg(junction_ids):
    """为路口 ID 列表批量生成 tls/upstream sensor_cfg 字典。"""
    if isinstance(junction_ids, str):
        junction_ids = [junction_ids]
    return {jid: _APPROACH_CAM_CFG for jid in junction_ids}


SCENARIO_CONFIGS = {
    "Hongkong_YMT": {
        "SCENARIO_NAME": "Hongkong_YMT",
        "NETFILE": "ymt_eval",
        "JUNCTION_NAME": "J1",
        "PHASE_NUMBER": 3,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (172, 201, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.8,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg("J1"),         # 进口道摄像头（停止线处）
            # "upstream": _tls_sensor_cfg("J1"),     # 上游摄像头（lane 起点处）
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 65.0}
        },
        "TOPOLOGY": {}  # 单交叉口场景不考虑事件广播
    },

    "SouthKorea_Songdo": {
        "SCENARIO_NAME": "SouthKorea_Songdo",
        "NETFILE": "songdo_eval",
        "JUNCTION_NAME": "J2",
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (900, 1641, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:1, 3:0},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.85,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg("J2"),
            # "upstream": _tls_sensor_cfg("J2"),
            # "aircraft": {
            #     "sensor_types": ["aircraft_all"],
            #     "height": 85.0,
            #     # Songdo 路口道路走向为 NE/NW/SE/SW（约 45° 角），
            #     # [-1, 1, 0] → 相机顺时针旋转 45°，使 NE 进口道显示在图像顶部（北方）
            #     "camera_heading": [-1, 1, 0]
            # }
        },
        "TOPOLOGY": {}  # 单交叉口场景不考虑事件广播
    },

    "France_Massy": {
        "SCENARIO_NAME": "France_Massy",
        "NETFILE": "massy_eval",
        "JUNCTION_NAME": "INT1",
        "PHASE_NUMBER": 2,
        # T字路口：仅 North / South / West 三个进口道，无 East 进口道
        "APPROACH_DIRS": ["N", "S", "W"],
        "CENTER_COORDINATES": (173, 244, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:1, 2:0},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.75,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False,
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg("INT1"),
            # "upstream": _tls_sensor_cfg("INT1"),
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 55.0}
        },
        "TOPOLOGY": {}  # 单交叉口场景不考虑事件广播
    },

    "JiNan": {
        "SCENARIO_NAME": "JiNan",
        "NETFILE": "jinan",
        "JUNCTION_NAME": _JINAN_JUNCTIONS,
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (173, 244, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:1, 3:0},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.85,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg(_JINAN_JUNCTIONS),
            # "upstream": _tls_sensor_cfg(_JINAN_JUNCTIONS),
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 60.0}
        },
        "TOPOLOGY": _generate_grid_topology(m=4, n=3)  # JiNan 是 4x3 路网，排除虚拟路口
    },

    "NewYork": {
        "SCENARIO_NAME": "NewYork",
        "NETFILE": "NewYork",
        "JUNCTION_NAME": _NEWYORK_JUNCTIONS,
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (172, 201, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.85,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg(_NEWYORK_JUNCTIONS),
            # "upstream": _tls_sensor_cfg(_NEWYORK_JUNCTIONS),
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 60.0}
        },
        "TOPOLOGY": _generate_grid_topology(m=7, n=28)  # NewYork 是 7*28 路网网络
    },

    "Hangzhou": {
        "SCENARIO_NAME": "Hangzhou",
        "NETFILE": "Hangzhou",
        "JUNCTION_NAME": _HANGZHOU_JUNCTIONS,
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (172, 201, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.85,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg(_HANGZHOU_JUNCTIONS),
            # "upstream": _tls_sensor_cfg(_HANGZHOU_JUNCTIONS),
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 60.0}
        },
        "TOPOLOGY": _generate_grid_topology(m=4, n=4)   # Hangzhou 是 4x4 路网网络
    },
}
