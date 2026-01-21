'''
Author: TrafficVLM Assistant
Date: 2026-01-21
Description: Generate e2.add.xml and tls_programs.add.xml from a SUMO net.xml file.
Usage: python scripts/generate_detectors.py --net <path_to_net_xml> --output <output_directory>
'''

import argparse
import os
import sys

# Ensure sumolib is available
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import sumolib

def generate_detectors(net_file, output_dir):
    print(f"Reading net file: {net_file}")
    net = sumolib.net.readNet(net_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    e2_file_path = os.path.join(output_dir, "e2.add.xml")
    tls_file_path = os.path.join(output_dir, "tls_programs.add.xml")
    
    print(f"Generating {e2_file_path}...")
    print(f"Generating {tls_file_path}...")

    with open(e2_file_path, "w") as f_e2, open(tls_file_path, "w") as f_tls:
        header = '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n'
        f_e2.write(header)
        f_tls.write(header)

        # Track processed TLS IDs to avoid duplicates if multiple nodes share a TLS
        processed_tls = set()

        # Iterate over all nodes to find traffic light controlled intersections
        for node in net.getNodes():
            if node.getType() == "traffic_light_right_on_red":
                junction_id = node.getID()
                
                # ---- Generate TLS Program Recorder ----
                # In CityFlow converted nets, Junction ID usually equals TLS ID.
                # Use the junction ID as the source.
                if junction_id not in processed_tls:
                    f_tls.write(f'    <timedEvent type="SaveTLSProgram" source="{junction_id}" dest="tls_programs.out.xml"/>\n')
                    processed_tls.add(junction_id)

                # ---- Generate E2 Detectors for Incoming Lanes ----
                for edge in node.getIncoming():
                    edge_id = edge.getID()
                    # Skip internal connections (edges starting with :)
                    if edge_id.startswith(":"):
                        continue
                    
                    for lane in edge.getLanes():
                        lane_id = lane.getID()
                        length = lane.getLength()
                        
                        # Detector configuration
                        # Assuming detector length 45m, placed at the stop line
                        # pos is the start position of the detector relative to lane start
                        det_length = 45.0
                        pos = max(0, length - det_length) 
                        
                        # Determine direction suffix (r, s, l)
                        directions = []
                        for conn in lane.getOutgoing():
                            d = conn.getDirection()
                            if d in ['r', 'R']: directions.append('r')
                            elif d in ['s']: directions.append('s')
                            elif d in ['l', 'L']: directions.append('l')
                            elif d in ['t', 'T']: directions.append('t')
                        
                        # Heuristic to pick one suffix if multiple movements exist
                        suffix = 's' # default
                        if directions:
                            if 's' in directions: suffix = 's'
                            elif 'l' in directions: suffix = 'l'
                            elif 'r' in directions: suffix = 'r'
                            elif 't' in directions: suffix = 't'
                            else: suffix = directions[0]
                        
                        # ID Format: e2det--<JunctionID>--<EdgeID>--<LaneID>--<Turn>
                        det_id = f"e2det--{junction_id}--{edge_id}--{lane_id}--{suffix}"
                        
                        f_e2.write(
                            f'    <laneAreaDetector file="e2.output.xml" freq="60" friendlyPos="x" '
                            f'id="{det_id}" lane="{lane_id}" pos="{pos:.2f}" length="{det_length}"/>\n'
                        )

        f_e2.write('</additional>\n')
        f_tls.write('</additional>\n')

    print("Generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SUMO detector definitions.")
    parser.add_argument("--net", required=True, help="Path to SUMO .net.xml file")
    parser.add_argument("--output", required=True, help="Output directory for .add.xml files")
    
    args = parser.parse_args()
    
    generate_detectors(args.net, args.output)
    # python scripts/generate_detectors.py --net data/raw/Hangzhou/env/Hangzhou.net.xml --output data/raw/Hangzhou/add/
    # python scripts/generate_detectors.py --net data/raw/JiNan/env/JiNan.net.xml --output data/raw/JiNan/add/
    # python scripts/generate_detectors.py --net data/raw/NewYork/env/NewYork.net.xml --output data/raw/NewYork/add/