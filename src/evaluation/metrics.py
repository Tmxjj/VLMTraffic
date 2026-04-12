import csv
import numpy as np
import xml.etree.ElementTree as ET
from loguru import logger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.scenairo_config import SCENARIO_CONFIGS

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 对比结果 CSV 路径
COMPARISON_CSV_PATH = os.path.join(PROJECT_ROOT, "results", "comparsion_result.csv")

# 路由文件名（带 .rou 后缀，去掉 .xml）→ 对比 CSV 中 (ATT列, AWT列, AQL列) 的 0-indexed 列号
# CSV 列结构: col0=Category, col1=Method,
#   col2-4=JiNan_real, col5-7=JiNan_real_2000, col8-10=JiNan_real_2500,
#   col11-13=Jinan_synthetic, col14-16=Hangzhou, col17-19=Hangzhou_5816, col20-22=Hangzhou_synthetic
ROUTE_TO_CSV_COLS = {
    "anon_3_4_jinan_real.rou":                     (2,  3,  4),
    "anon_3_4_jinan_real_2000.rou":                (5,  6,  7),
    "anon_3_4_jinan_real_2500.rou":                (8,  9,  10),
    "anon_3_4_jinan_synthetic_24000_60min.rou":    (11, 12, 13),
    "anon_4_4_hangzhou_real.rou":                  (14, 15, 16),
    "anon_4_4_hangzhou_real_5816.rou":             (17, 18, 19),
    "anon_4_4_hangzhou_synthetic_24000_60min.rou": (20, 21, 22),
}

# 方法目录名 → CSV 中 Method 列的字符串
METHOD_TO_CSV_ROW_KEY = {
    "max_pressure": "MaxPressure",
    "fixed_time":   "FixedTime",
}


def update_comparison_csv(metrics: dict, route_key: str, row_key: str):
    """
    将 ATT/AWT/AQL 写入 results/comparsion_result.csv 对应的行列位置。

    Args:
        metrics (dict): 包含 ATT, AWT, AQL 的指标字典
        route_key (str): 路由文件名，带 .rou 后缀（如 "anon_3_4_jinan_real_2000.rou"）
        row_key (str): CSV 中 Method 列的值（如 "MaxPressure", "FixedTime"）
    """
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

        # 找到目标行（Method 列 == row_key）
        target_row_idx = None
        for idx, row in enumerate(rows):
            if len(row) > 1 and row[1].strip() == row_key:
                target_row_idx = idx
                break

        if target_row_idx is None:
            print(f"⚠️  [CSV] 未找到 Method='{row_key}' 的行，跳过 CSV 更新。")
            return

        target_row = rows[target_row_idx]

        # 确保行长度足够
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
    """
    Calculates traffic metrics: ATT, AQL, AWT.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.vehicle_data = {} # {veh_id: {start_time, end_time, trip_time, waiting_time, type}}
        self.queue_lengths = []
        
    def update(self, step_data):
        """
        Updates metrics with data from a single simulation step.
        step_data should contain:
        - current_queue_lengths: list of queue lengths for all lanes
        - arrived_vehicles: list of dicts {id, type, travel_time, waiting_time}
        """
        # AQL
        if 'current_queue_lengths' in step_data:
            self.queue_lengths.append(np.mean(step_data['current_queue_lengths']))
            
        # Vehicle Stats (ATT, AWT)
        if 'arrived_vehicles' in step_data:
            for veh in step_data['arrived_vehicles']:
                self.vehicle_data[veh['id']] = veh
                
    def compute_metrics(self, scenario_name=None):
        """
        Computes the final metrics based on online updated data.
        """
        num_junctions = 1
        if scenario_name and scenario_name in SCENARIO_CONFIGS:
            junction_name = SCENARIO_CONFIGS[scenario_name].get("JUNCTION_NAME", [])
            num_junctions = len(junction_name) if isinstance(junction_name, list) else 1

        all_travel_times = [v['travel_time'] for v in self.vehicle_data.values()]
        all_waiting_times = [v['waiting_time'] for v in self.vehicle_data.values()]
        
        queue_mean = np.mean(self.queue_lengths) if self.queue_lengths else 0.0
        metrics = {
            "ATT": np.mean(all_travel_times) if all_travel_times else 0.0,
            "AWT": np.mean(all_waiting_times) if all_waiting_times else 0.0,
            "AQL": queue_mean / num_junctions
        }
        
        # Special Vehicles
        special_types = ['ambulance', 'fire', 'police']
        special_travel_times = [v['travel_time'] for v in self.vehicle_data.values() if v.get('type') in special_types]
        special_waiting_times = [v['waiting_time'] for v in self.vehicle_data.values() if v.get('type') in special_types]
        
        metrics["Special_ATT"] = np.mean(special_travel_times) if special_travel_times else 0.0
        metrics["Special_AWT"] = np.mean(special_waiting_times) if special_waiting_times else 0.0
        
        return metrics

    def calculate_from_files(self, statistic_file, queue_file, scenario_name):
        """
        Calculates metrics from SUMO output files.
        
        Args:
            statistic_file (str): Path to statistic_output.xml
            queue_file (str): Path to queue_output.xml
            scenario_name (str): Name of the scenario to get junction number
            
        Returns:
            dict: {ATT, AWT, AQL}
        """
        metrics = {"ATT": 0.0, "AWT": 0.0, "AQL": 0.0}
        
        num_junctions = 1
        if scenario_name and scenario_name in SCENARIO_CONFIGS:
            junction_name = SCENARIO_CONFIGS[scenario_name].get("JUNCTION_NAME", [])
            num_junctions = len(junction_name) if isinstance(junction_name, list) else 1

        # 1. Parse Statistic Output for ATT and AWT
        try:
            tree = ET.parse(statistic_file)
            root = tree.getroot()
            # <vehicleTripStatistics count="31" duration="20.61" waitingTime="7.90" ... />
            # duration is the average trip duration (ATT)
            # waitingTime is the average waiting time (AWT)
            veh_stats = root.find('vehicleTripStatistics')
            if veh_stats is not None:
                metrics["ATT"] = float(veh_stats.get('duration', 0.0))
                metrics["AWT"] = float(veh_stats.get('waitingTime', 0.0))
        except Exception as e:
            logger.error(f"[EVAL] Error parsing statistic file: {e} - {statistic_file}")

        # 2. Parse Queue Output for AQL
        try:
            tree = ET.parse(queue_file)
            root = tree.getroot()
            
            total_queue_len = 0.0
            step_count = 0
            
            for data in root.findall('data'):
                # timestep = data.get('timestep')
                lanes = data.find('lanes')
                current_step_queue = 0.0
                if lanes is not None:
                    # Sum up queue length of all lanes in this step
                    for lane in lanes.findall('lane'):
                        # Use 'queueing_length' (meters)
                        current_step_queue += float(lane.get('queueing_length', 0.0))
                
                total_queue_len += current_step_queue
                step_count += 1
            
            # AQL = Average Total Queue Length over time
            if step_count > 0:
                metrics["AQL"] = (total_queue_len / step_count) / num_junctions
                
        except Exception as e:
            logger.error(f"[EVAL] Error parsing queue file: {e} - {queue_file}")
            
        return metrics

if __name__ == "__main__":
    # 扫描 data/eval/ 下的所有场景/路由/方法目录，计算指标并写入对比 CSV
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

    methods = [
        # 'fixed_time',
        'max_pressure',
        # 'qwen3-vl-8b'
        # 'qwen3-vl-4b'
    ]

    for secnario_name, net_files in secnario_dict.items():
        for net_file in net_files:
            for method in methods:
                # 1. 提取路径变量
                base_path = f"data/eval/{secnario_name}/{net_file}/{method}"
                stat_path = f"{base_path}/statistic_output.xml"
                queue_path = f"{base_path}/queue_output.xml"

                # 2. 检查必要文件是否存在
                if not os.path.exists(stat_path) or not os.path.exists(queue_path):
                    print(f"⚠️  [Skip] 文件缺失，跳过目录: {base_path}")
                    continue

                # 3. 计算指标
                try:
                    calculator = MetricsCalculator()
                    metrics = calculator.calculate_from_files(stat_path, queue_path, secnario_name)
                    print(f"✅ {secnario_name}/{net_file}/{method}", metrics)
                except Exception as e:
                    print(f"❌ {secnario_name}/{net_file}/{method} 处理出错: {e}")
                    continue

                # 4. 将指标写入对比 CSV（仅支持已配置的基线方法）
                row_key = METHOD_TO_CSV_ROW_KEY.get(method)
                if row_key:
                    update_comparison_csv(metrics, net_file, row_key)