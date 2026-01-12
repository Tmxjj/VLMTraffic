import numpy as np

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
        Computes the final metrics.
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
        # Note: AQL is usually lane-based, not vehicle-based, so Special AQL might need specific lane filtering logic if required
        
        return metrics
