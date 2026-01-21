'''
Author: TrafficVLM Assistant
Date: 2026-01-20
Description: Analyze SUMO route file (.rou.xml) to calculate vehicle count, simulation duration, and arrival rates.
Usage: python scripts/analyze_route_info.py --file <path_to_rou_xml>
'''

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
import math

def analyze_route_file(file_path):
    print(f"Analyzing file: {file_path} ...")
    
    vehicle_count = 0
    min_depart = float('inf')
    max_depart = float('-inf')
    
    # Bucket size in seconds (5 minutes = 300 seconds)
    bucket_size = 300
    # Dictionary to store counts: key is bucket index, value is count
    arrival_buckets = defaultdict(int)
    
    try:
        # Use iterparse for memory efficiency on large XML files
        context = ET.iterparse(file_path, events=("start", "end"))
        context = iter(context)
        event, root = next(context)
        
        for event, elem in context:
            if event == "end" and elem.tag == "vehicle":
                depart_str = elem.get("depart")
                if depart_str:
                    try:
                        depart_time = float(depart_str)
                        
                        # 1. Update Counts
                        vehicle_count += 1
                        
                        # 2. Update Time Range
                        if depart_time < min_depart:
                            min_depart = depart_time
                        if depart_time > max_depart:
                            max_depart = depart_time
                            
                        # 3. Update 5-min Arrival Buckets
                        bucket_index = int(depart_time // bucket_size)
                        arrival_buckets[bucket_index] += 1
                        
                    except ValueError:
                        pass # Ignore vehicles with invalid depart times 'triggered', etc.
                
                # Clear element to save memory
                root.clear()
                
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return

    if vehicle_count == 0:
        print("No vehicles found in the file.")
        return

    # Calculations
    if not arrival_buckets:
        print("No arrival data collected.")
        return

    import statistics

    simulation_duration = max_depart - min_depart
    last_bucket_idx = int(max_depart // bucket_size)
    # Fill in all buckets including empty ones from 0 to last_bucket_idx
    rates_list = [arrival_buckets[i] for i in range(last_bucket_idx + 1)]
    
    mean_rate = statistics.mean(rates_list)
    try:
        variance_rate = statistics.variance(rates_list)
    except statistics.StatisticsError:
        variance_rate = 0.0
    
    max_rate = max(rates_list)
    min_rate = min(rates_list)

    print("-" * 50)
    print(f"STATISTICS REPORT")
    print("-" * 50)
    print(f"Total Vehicles     : {vehicle_count}")
    print(f"First Departure    : {min_depart:.2f} s")
    print(f"Last Departure     : {max_depart:.2f} s")
    print(f"Effective Duration : {simulation_duration:.2f} s ({simulation_duration/60:.2f} min)")
    print("-" * 50)
    print(f"ARRIVAL RATE ANALYSIS (Interval: 5 mins)")
    print("-" * 50)
    
    # Print non-zero buckets for detail, or all? Usually non-zero is enough for detail list, 
    # but stats cover all. Let's keep printing non-zero explicitly or the filled list.
    # To keep output concise, stick to printing valid entries in dict, 
    # but clarify stats calculate over full duration.
    sorted_buckets = sorted(arrival_buckets.items())
    
    for bucket_idx, count in sorted_buckets:
        start_time = bucket_idx * bucket_size
        end_time = (bucket_idx + 1) * bucket_size
        print(f"Time {start_time/60:04.1f}m - {end_time/60:04.1f}m : {count} vehicles")
            
    print("-" * 50)
    print(f"Max Arrival Rate   : {max_rate} vehicles/5min")
    print(f"Min Arrival Rate   : {min_rate} vehicles/5min")
    print(f"Avg Arrival Rate   : {mean_rate:.2f} vehicles/5min")
    print(f"Variance           : {variance_rate:.2f}")
    print(f"Stdev              : {variance_rate**0.5:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SUMO route file statistics.")
    parser.add_argument("--file", type=str, required=True, help="Path to the .rou.xml file")
    
    args = parser.parse_args()
    
    analyze_route_file(args.file)
