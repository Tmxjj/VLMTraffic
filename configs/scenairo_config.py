'''
Author: yufei Ji
Date: 2026-01-12 17:09:21
LastEditTime: 2026-01-22 21:48:38
Description: this script is used to store the configuration of different TSC scenarios
FilePath: /VLMTraffic/configs/scenairo_config.py
'''
SCENARIO_CONFIGS = {
    "Hongkong_YMT": {
        "SCENARIO_NAME": "Hongkong_YMT",
        "NETFILE": "ymt_eval",
        "JUNCTION_NAME": "J1",
        "PHASE_NUMBER": 4,
        "CENTER_COORDINATES": (172, 201, 100), # 这个参数不影响
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1}, # 路口传感器索引到信号灯相位的映射，对于BEV视角下不需要处理
        "RENDERER_CFG": {
            "preset": "SQUARE_1024",
            #BUG: 变焦作用，胶片（传感器）越小：不仅拍摄到的范围变小了，为了填满同样的屏幕（1920x1080），画面会被放大（Zoom In）。 但实测下来没什么作用（暂不影响）
            "resolution": 2,
            "vehicle_model": "high",
            "render_mode": "offscreen", # onscreen or offscreen 服务器端必须是 offscreen
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False # 是否每一帧都渲染

        },
        "SENSOR_CFG": {
            "aircraft": {
                "sensor_types": ["aircraft_all"],
                "height": 55.0
            }
        } # 传感器支持三类：vehicle, tls, aircraft
    },
    "SouthKorea_Songdo": {
        "SCENARIO_NAME": "SouthKorea_Songdo",
        "NETFILE": "songdo_eval",
        "JUNCTION_NAME": "J2",
        "PHASE_NUMBER": 4,
        "CENTER_COORDINATES": (900, 1641, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:1, 3:0},
        "RENDERER_CFG": {
            "preset": "720P",
            "resolution": 1.0,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": False,
            "is_every_frame": False # 是否每一帧都渲染
        },
        "SENSOR_CFG": {
            "tls": {
                "sensor_types": ["junction_front_all"],
                "tls_camera_height": 15
            },
            "aircraft": {
                "sensor_types": ["aircraft_all"],
                "height": 55.0
            }       
        }
    },
    "France_Massy": {
        "SCENARIO_NAME": "France_Massy",
        "NETFILE": "massy_eval",
        "JUNCTION_NAME": "INT1",
        "PHASE_NUMBER": 3,
        "CENTER_COORDINATES": (173, 244, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:1, 2:0},
        "RENDERER_CFG": {
            "preset": "720P",
            "resolution": 1.0,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": False,
            "is_every_frame": False, # 是否每一帧都渲染
        },
        "SENSOR_CFG": {
            "tls": {
                "sensor_types": ["junction_front_all"],
                "tls_camera_height": 15
            },
            "aircraft": {
                "sensor_types": ["aircraft_all"],
                "height": 55.0
            }       
        }
    },
    "JiNan": {
        "SCENARIO_NAME": "JiNan",
        "NETFILE": "jinan",
        "JUNCTION_NAME": "intersection_1_1",
        "PHASE_NUMBER": 4,
        "CENTER_COORDINATES": (173, 244, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:1, 3:0}, 
        "RENDERER_CFG": {
            "preset": "SQUARE_1024",
            #BUG: 变焦作用，胶片（传感器）越小：不仅拍摄到的范围变小了，为了填满同样的屏幕（1920x1080），画面会被放大（Zoom In）。 但实测下来没什么作用（暂不影响）
            "resolution": 2,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False # 是否每一帧都渲染
        },
        "SENSOR_CFG": {
            "aircraft": {
                "sensor_types": ["aircraft_all"],
                "height": 55.0
            }       
        }
    },
    "NewYork": {
        "SCENARIO_NAME": "NewYork",
        "NETFILE": "NewYork",
        "JUNCTION_NAME": "intersection_2_2",
        "PHASE_NUMBER": 4,
        "CENTER_COORDINATES": (172, 201, 100), # 这个参数不影响
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1}, # 路口传感器索引到信号灯相位的映射，对于BEV视角下不需要处理
        "RENDERER_CFG": {
            "preset": "SQUARE_1024",
            #BUG: 变焦作用，胶片（传感器）越小：不仅拍摄到的范围变小了，为了填满同样的屏幕（1920x1080），画面会被放大（Zoom In）。 但实测下来没什么作用（暂不影响）
            "resolution": 2,
            "vehicle_model": "high",
            "render_mode": "offscreen", # onscreen or offscreen 服务器端必须是 offscreen
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False # 是否每一帧都渲染

        },
        "SENSOR_CFG": {
            "aircraft": {
                "sensor_types": ["aircraft_all"],
                "height": 55.0
            }
        }
    },
    "Hangzhou": {
        "SCENARIO_NAME": "Hangzhou",
        "NETFILE": "Hangzhou",
        "JUNCTION_NAME": [
            "intersection_1_1", "intersection_1_2", "intersection_1_3", "intersection_1_4", 
            "intersection_2_1", "intersection_2_2", "intersection_2_3", "intersection_2_4", 
            "intersection_3_1", "intersection_3_2", "intersection_3_3", "intersection_3_4", 
            "intersection_4_1", "intersection_4_2", "intersection_4_3", "intersection_4_4"
        ],
        "PHASE_NUMBER": 4,
        "CENTER_COORDINATES": (172, 201, 100), # 这个参数不影响
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1}, # 路口传感器索引到信号灯相位的映射，对于BEV视角下不需要处理
        "RENDERER_CFG": {
            "preset": "SQUARE_1024",
            #BUG: 变焦作用，胶片（传感器）越小：不仅拍摄到的范围变小了，为了填满同样的屏幕（1920x1080），画面会被放大（Zoom In）。 但实测下来没什么作用（暂不影响）
            "resolution": 2,
            "vehicle_model": "high",
            "render_mode": "offscreen", # onscreen or offscreen 服务器端必须是 offscreen
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False # 是否每一帧都渲染

        },
        "SENSOR_CFG": {
            "aircraft": {
                "sensor_types": ["aircraft_all"],
                "height": 55.0
            }   
        }
    },
}