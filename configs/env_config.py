'''
Author: yufei Ji
Date: 2026-01-12 17:09:21
LastEditTime: 2026-01-13 12:01:03
Description: this script is used to store the env config
FilePath: /VLMTraffic/configs/env_config.py
'''
SCENARIO_CONFIGS = {
    "Hongkong_YMT": {
        "SCENARIO_NAME": "Hongkong_YMT",
        "NETFILE": "ymt_eval",
        "JUNCTION_NAME": "J1",
        "PHASE_NUMBER": 4,
        "CENTER_COORDINATES": (172, 201, 100), # 这个参数不影响
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1},
        "RENDERER_CFG": {
            "preset": "SQUARE_1024",
            #BUG: 变焦作用，胶片（传感器）越小：不仅拍摄到的范围变小了，为了填满同样的屏幕（1920x1080），画面会被放大（Zoom In）。 但实测下来没什么作用（暂不影响）
            "resolution": 2,
            "vehicle_model": "high",
            "render_mode": "offscreen", # onscreen or offscreen 服务器端必须是 offscreen
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False
        },
        "SENSOR_CFG": {
            # "tls": {
            #     "sensor_types": ["junction_front_all"], # 从路口红绿灯角度拍摄
            #     "tls_camera_height": 15
            # },
            "aircraft": {
                "junction_cam_1": {
                    "sensor_types": ["aircraft_all"],
                    "height": 50.0
                }
            }
        }
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
            "debuger_spin_camera": False
        },
        "SENSOR_CFG": {
            "tls": {
                "sensor_types": ["junction_front_all"],
                "tls_camera_height": 15
            },
            "aircraft": {
                "junction_cam_1": {
                    "sensor_types": ["aircraft_all"],
                    "height": 50.0
                }
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
            "debuger_spin_camera": False
        },
        "SENSOR_CFG": {
            "tls": {
                "sensor_types": ["junction_front_all"],
                "tls_camera_height": 15
            },
            "aircraft": {
                "junction_cam_1": {
                    "sensor_types": ["aircraft_all"],
                    "height": 50.0
                }
            }
        }
    },
    "JiNan": {
        "SCENARIO_NAME": "JiNan",
        "NETFILE": "jinan",
        "JUNCTION_NAME": "intersection_1_1",
        "PHASE_NUMBER": 4,
        "CENTER_COORDINATES": (173, 244, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:1, 3:0}, # 传感器索引到信号灯相位的映射，未理清其逻辑，赞不管
        "RENDERER_CFG": {
            "preset": "SQUARE_1024",
            #BUG: 变焦作用，胶片（传感器）越小：不仅拍摄到的范围变小了，为了填满同样的屏幕（1920x1080），画面会被放大（Zoom In）。 但实测下来没什么作用（暂不影响）
            "resolution": 2,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False
        },
        "SENSOR_CFG": {
            "tls": {
                "sensor_types": ["junction_front_all"],
                "tls_camera_height": 15
            },
            "aircraft": {
                "junction_cam_1": {
                    "sensor_types": ["aircraft_all"],
                    "height": 50.0
                }
            }
        }
    }
}