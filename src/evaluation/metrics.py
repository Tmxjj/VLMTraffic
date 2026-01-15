import numpy as np
import xml.etree.ElementTree as ET
from loguru import logger

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
                
    def compute_metrics(self):
        """
        Computes the final metrics based on online updated data.
        """
        all_travel_times = [v['travel_time'] for v in self.vehicle_data.values()]
        all_waiting_times = [v['waiting_time'] for v in self.vehicle_data.values()]
        
        metrics = {
            "ATT": np.mean(all_travel_times) if all_travel_times else 0.0,
            "AWT": np.mean(all_waiting_times) if all_waiting_times else 0.0,
            "AQL": np.mean(self.queue_lengths) if self.queue_lengths else 0.0
        }
        
        # Special Vehicles
        special_types = ['ambulance', 'fire', 'police']
        special_travel_times = [v['travel_time'] for v in self.vehicle_data.values() if v.get('type') in special_types]
        special_waiting_times = [v['waiting_time'] for v in self.vehicle_data.values() if v.get('type') in special_types]
        
        metrics["Special_ATT"] = np.mean(special_travel_times) if special_travel_times else 0.0
        metrics["Special_AWT"] = np.mean(special_waiting_times) if special_waiting_times else 0.0
        
        return metrics

    def calculate_from_files(self, statistic_file, queue_file):
        """
        Calculates metrics from SUMO output files.
        
        Args:
            statistic_file (str): Path to statistic_output.xml
            queue_file (str): Path to queue_output.xml
            
        Returns:
            dict: {ATT, AWT, AQL}
        """
        metrics = {"ATT": 0.0, "AWT": 0.0, "AQL": 0.0}
        
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
            logger.error(f"[EVAL] Error parsing statistic file: {e}")

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
                metrics["AQL"] = total_queue_len / step_count
                
        except Exception as e:
            logger.error(f"[EVAL] Error parsing queue file: {e}")
            
        return metrics
