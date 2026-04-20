'''
Author: yufei Ji
Date: 2026-01-12
LastEditTime: 2026-04-20 21:15:06
Description: 核心指标计算引擎。
             支持在线（仿真步流式更新）和离线（SUMO 输出文件批量解析）两种模式。
             离线模式新增参数：
               event_id_prefixes  - 事件车辆 ID 前缀列表；提供后从 tripinfo 重算 ATT/AWT/TPT
               special_vtypes     - 专项车辆 vType 集合（默认紧急车辆；可覆盖为 bus/school_bus）
            新增返回字段：
               MaxQL  - 全程队列长度峰值（来自 queue_output.xml）
               TPT    - 到达普通车辆数（event_id_prefixes 模式下有值）
            数据流  
                SUMO 仿真运行                                                  
                    ↓                                                          
                三个输出文件：                                                 
                    statistic_output.xml  → 全局聚合均值（ATT/AWT）              
                    queue_output.xml      → 每步各车道排队长度                   
                    tripinfo.out.xml      →                                      
                每辆车的完整行程记录（duration/waitingTime/vType/id）          
                    ↓                                                          
                MetricsCalculator.calculate_from_files()                       
                    ↓                                                     
                results/*.csv

            ---                                                            
            两条计算路径
                                                                                
                路径一（普通场景 / 泛化场景）：ATT/AWT 直接读             
                statistic_output.xml 的聚合均值，AQL 从 queue_output.xml       
                逐步求均值再除以路口数。
                                                                                
                路径二（事件场景）：ATT/AWT 改从 tripinfo.out.xml              
                逐车计算——先按 event_id_prefixes 过滤掉事件占位假车（accident_*
                、debris_*、ped_*），只对剩余普通车辆求均值，TPT =             
                剩余车辆计数。                                            

            ---
            专项指标
                        
                - EATT/EAWT（紧急车辆）、BATT/BAWT（公交）：在 tripinfo 中按
                vType ∈ special_vtypes 过滤后单独求均值                        
                - MaxQL：queue_output.xml 所有步的全路网排队总长峰值
FilePath: /VLMTraffic/src/evaluation/metrics.py
'''
import csv
import numpy as np
import xml.etree.ElementTree as ET
from loguru import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.scenairo_config import SCENARIO_CONFIGS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMPARISON_CSV_PATH = os.path.join(PROJECT_ROOT, "results", "comparsion_result.csv")

# 路由文件名（带 .rou 后缀）→ 对比 CSV 中 (ATT列, AWT列, AQL列) 的 0-indexed 列号
ROUTE_TO_CSV_COLS = {
    "anon_3_4_jinan_real.rou":                     (2,  3,  4),
    "anon_3_4_jinan_real_2000.rou":                (5,  6,  7),
    "anon_3_4_jinan_real_2500.rou":                (8,  9,  10),
    "anon_3_4_jinan_synthetic_24000_60min.rou":    (11, 12, 13),
    "anon_4_4_hangzhou_real.rou":                  (14, 15, 16),
    "anon_4_4_hangzhou_real_5816.rou":             (17, 18, 19),
    "anon_4_4_hangzhou_synthetic_24000_60min.rou": (20, 21, 22),
}

METHOD_TO_CSV_ROW_KEY = {
    "max_pressure": "MaxPressure",
    "fixed_time":   "FixedTime",
}

# 紧急车辆 vType 集合（与 scripts/event_scene_generation/add_emergency_vehicles.py 中的 vType id 严格对齐）
_DEFAULT_SPECIAL_VTYPES = {"emergency", "police", "fire_engine"}

# 默认事件车辆 ID 前缀（不同场景可能只传其中的子集）
_ALL_EVENT_PREFIXES = ("accident_", "pedestrian_", "debris_", "ped_")


def update_comparison_csv(metrics: dict, route_key: str, row_key: str):
    """将 ATT/AWT/AQL 写入 results/comparsion_result.csv 对应的行列位置。"""
    col_indices = ROUTE_TO_CSV_COLS.get(route_key)
    if col_indices is None:
        print(f"⚠️  [CSV] 路由 '{route_key}' 未在 ROUTE_TO_CSV_COLS 中配置，跳过 CSV 更新。")
        return

    if not os.path.exists(COMPARISON_CSV_PATH):
        print(f"⚠️  [CSV] 对比 CSV 文件不存在: {COMPARISON_CSV_PATH}")
        return

    att_col, awt_col, aql_col = col_indices

    try:
        with open(COMPARISON_CSV_PATH, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))

        target_row_idx = None
        for idx, row in enumerate(rows):
            if len(row) > 1 and row[1].strip() == row_key:
                target_row_idx = idx
                break

        if target_row_idx is None:
            print(f"⚠️  [CSV] 未找到 Method='{row_key}' 的行，跳过 CSV 更新。")
            return

        target_row = rows[target_row_idx]
        max_col = max(att_col, awt_col, aql_col)
        while len(target_row) <= max_col:
            target_row.append('')

        target_row[att_col] = f"{metrics['ATT']:.6f}"
        target_row[awt_col] = f"{metrics['AWT']:.6f}"
        target_row[aql_col] = f"{metrics['AQL']:.6f}"
        rows[target_row_idx] = target_row

        with open(COMPARISON_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"✅ [CSV] 已更新 '{row_key}' | 路由='{route_key}' | "
              f"ATT={metrics['ATT']:.2f}, AWT={metrics['AWT']:.2f}, AQL={metrics['AQL']:.6f}")
    except Exception as e:
        print(f"❌ [CSV] 写入对比 CSV 失败: {e}")


class MetricsCalculator:
    """交通指标计算器，支持在线（仿真步）和离线（SUMO 文件）两种模式。"""

    def __init__(self):
        self.reset()

    def reset(self):
        # {veh_id: {start_time, end_time, trip_time, waiting_time, type}}
        self.vehicle_data = {}
        self.queue_lengths = []

    def update(self, step_data):
        """在线模式：逐步更新仿真数据。"""
        if 'current_queue_lengths' in step_data:
            self.queue_lengths.append(np.mean(step_data['current_queue_lengths']))

        if 'arrived_vehicles' in step_data:
            for veh in step_data['arrived_vehicles']:
                self.vehicle_data[veh['id']] = veh

    def compute_metrics(self, scenario_name=None):
        """在线模式：汇总计算最终指标（ATT / AWT / AQL / Special_ATT / Special_AWT）。"""
        num_junctions = 1
        if scenario_name and scenario_name in SCENARIO_CONFIGS:
            junction_name = SCENARIO_CONFIGS[scenario_name].get("JUNCTION_NAME", [])
            num_junctions = len(junction_name) if isinstance(junction_name, list) else 1

        all_travel_times  = [v['travel_time'] for v in self.vehicle_data.values()]
        all_waiting_times = [v['waiting_time'] for v in self.vehicle_data.values()]
        queue_mean = np.mean(self.queue_lengths) if self.queue_lengths else 0.0

        metrics = {
            "ATT":   np.mean(all_travel_times)  if all_travel_times  else 0.0,
            "AWT":   np.mean(all_waiting_times) if all_waiting_times else 0.0,
            "AQL":   queue_mean / num_junctions,
            "MaxQL": max(self.queue_lengths) if self.queue_lengths else 0.0,
            "TPT":   0,
        }

        special_types = _DEFAULT_SPECIAL_VTYPES
        special_travel_times  = [v['travel_time']  for v in self.vehicle_data.values() if v.get('type') in special_types]
        special_waiting_times = [v['waiting_time'] for v in self.vehicle_data.values() if v.get('type') in special_types]
        metrics["Special_ATT"] = float(np.mean(special_travel_times))  if special_travel_times  else 0.0
        metrics["Special_AWT"] = float(np.mean(special_waiting_times)) if special_waiting_times else 0.0

        return metrics

    # ------------------------------------------------------------------
    def calculate_from_files(
        self,
        statistic_file: str,
        queue_file: str,
        scenario_name: str,
        tripinfo_file: str = None,
        event_id_prefixes: list = None,
        special_vtypes: set = None,
    ) -> dict:
        """
        离线模式：从 SUMO 输出文件计算全套指标。

        Args:
            statistic_file    : statistic_output.xml 路径
            queue_file        : queue_output.xml 路径
            scenario_name     : 场景名称（用于获取路口数量做 AQL 归一化）
            tripinfo_file     : tripinfo.out.xml 路径（可选）
            event_id_prefixes : 事件车辆 ID 前缀列表，如 ["accident_", "pedestrian_"]
                                提供后：
                                  - ATT/AWT 从 tripinfo 重算（排除事件车辆）
                                  - TPT = 到达的普通车辆数
                                不提供时 ATT/AWT 来自 statistic_output.xml（向后兼容）
            special_vtypes    : 专项车辆 vType 集合（默认紧急车辆类型）
                                传入 {"bus","school_bus"} 即可统计公交专项指标

        Returns:
            dict: {ATT, AWT, AQL, MaxQL, TPT, Special_ATT, Special_AWT}
        """
        if special_vtypes is None:
            special_vtypes = _DEFAULT_SPECIAL_VTYPES

        metrics = {
            "ATT": 0.0, "AWT": 0.0, "AQL": 0.0,
            "MaxQL": 0.0, "TPT": 0,
            "Special_ATT": 0.0, "Special_AWT": 0.0,
        }

        num_junctions = 1
        if scenario_name and scenario_name in SCENARIO_CONFIGS:
            junction_name = SCENARIO_CONFIGS[scenario_name].get("JUNCTION_NAME", [])
            num_junctions = len(junction_name) if isinstance(junction_name, list) else 1

        # ── 1. ATT / AWT ──────────────────────────────────────────────
        if event_id_prefixes is None:
            # 向后兼容：从 statistic_output.xml 读取聚合均值
            try:
                tree = ET.parse(statistic_file)
                root = tree.getroot()
                veh_stats = root.find('vehicleTripStatistics')
                if veh_stats is not None:
                    metrics["ATT"] = float(veh_stats.get('duration', 0.0))
                    metrics["AWT"] = float(veh_stats.get('waitingTime', 0.0))
            except Exception as e:
                logger.error(f"[EVAL] 解析 statistic 文件失败: {e} - {statistic_file}")
        else:
            # 事件场景：从 tripinfo 过滤事件车辆后重算 ATT/AWT
            if tripinfo_file and os.path.exists(tripinfo_file):
                try:
                    tree = ET.parse(tripinfo_file)
                    root = tree.getroot()
                    reg_durations, reg_waits = [], []
                    for ti in root.findall('tripinfo'):
                        veh_id = ti.get('id', '')
                        # 排除所有前缀匹配的事件车辆
                        if any(veh_id.startswith(p) for p in event_id_prefixes):
                            continue
                        dur = ti.get('duration')
                        wt  = ti.get('waitingTime')
                        if dur is not None:
                            reg_durations.append(float(dur))
                        if wt is not None:
                            reg_waits.append(float(wt))
                    metrics["ATT"] = float(np.mean(reg_durations)) if reg_durations else 0.0
                    metrics["AWT"] = float(np.mean(reg_waits))     if reg_waits     else 0.0
                    # TPT = 成功到达的普通车辆数（可在方法间横向比较通行能力）
                    metrics["TPT"] = len(reg_durations)
                except Exception as e:
                    logger.error(f"[EVAL] 解析 tripinfo（事件过滤）失败: {e} - {tripinfo_file}")
            else:
                # tripinfo 不存在时降级到 statistic_output.xml
                logger.warning(f"[EVAL] event_id_prefixes 已指定但 tripinfo 不存在，降级读取 statistic_output。")
                try:
                    tree = ET.parse(statistic_file)
                    root = tree.getroot()
                    veh_stats = root.find('vehicleTripStatistics')
                    if veh_stats is not None:
                        metrics["ATT"] = float(veh_stats.get('duration', 0.0))
                        metrics["AWT"] = float(veh_stats.get('waitingTime', 0.0))
                except Exception as e:
                    logger.error(f"[EVAL] 解析 statistic 文件失败: {e} - {statistic_file}")

        # ── 2. AQL / MaxQL ─────────────────────────────────────────────
        try:
            tree = ET.parse(queue_file)
            root = tree.getroot()
            step_totals = []
            for data in root.findall('data'):
                lanes = data.find('lanes')
                step_queue = 0.0
                if lanes is not None:
                    for lane in lanes.findall('lane'):
                        step_queue += float(lane.get('queueing_length', 0.0))
                step_totals.append(step_queue)

            if step_totals:
                metrics["AQL"]   = (float(np.mean(step_totals))) / num_junctions
                metrics["MaxQL"] = float(np.max(step_totals))
        except Exception as e:
            logger.error(f"[EVAL] 解析 queue 文件失败: {e} - {queue_file}")

        # ── 3. Special_ATT / Special_AWT ───────────────────────────────
        if tripinfo_file and os.path.exists(tripinfo_file):
            try:
                tree = ET.parse(tripinfo_file)
                root = tree.getroot()
                sp_durations, sp_waits = [], []
                for ti in root.findall('tripinfo'):
                    if ti.get('vType', '') in special_vtypes:
                        dur = ti.get('duration')
                        wt  = ti.get('waitingTime')
                        if dur is not None:
                            sp_durations.append(float(dur))
                        if wt is not None:
                            sp_waits.append(float(wt))
                if sp_durations:
                    metrics["Special_ATT"] = float(np.mean(sp_durations))
                if sp_waits:
                    metrics["Special_AWT"] = float(np.mean(sp_waits))
            except Exception as e:
                logger.error(f"[EVAL] 解析 tripinfo（专项车辆）失败: {e} - {tripinfo_file}")

        return metrics


if __name__ == "__main__":
    secnario_dict = {
        "JiNan": [
            "anon_3_4_jinan_real.rou",
            "anon_3_4_jinan_real_2000.rou",
            "anon_3_4_jinan_real_2500.rou",
            "anon_3_4_jinan_synthetic_24000_60min.rou",
        ],
        "Hangzhou": [
            "anon_4_4_hangzhou_real.rou",
            "anon_4_4_hangzhou_real_5816.rou",
            "anon_4_4_hangzhou_synthetic_24000_60min.rou",
        ]
    }
    methods = ['max_pressure']

    for secnario_name, net_files in secnario_dict.items():
        for net_file in net_files:
            for method in methods:
                base_path  = f"data/eval/{secnario_name}/{net_file}/{method}"
                stat_path  = f"{base_path}/statistic_output.xml"
                queue_path = f"{base_path}/queue_output.xml"

                if not os.path.exists(stat_path) or not os.path.exists(queue_path):
                    print(f"⚠️  [Skip] 文件缺失，跳过目录: {base_path}")
                    continue

                try:
                    calculator = MetricsCalculator()
                    metrics = calculator.calculate_from_files(stat_path, queue_path, secnario_name)
                    print(f"✅ {secnario_name}/{net_file}/{method}", metrics)
                except Exception as e:
                    print(f"❌ {secnario_name}/{net_file}/{method} 处理出错: {e}")
                    continue

                row_key = METHOD_TO_CSV_ROW_KEY.get(method)
                if row_key:
                    update_comparison_csv(metrics, net_file, row_key)
