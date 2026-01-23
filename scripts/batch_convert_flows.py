
import os
import sys
import glob
import argparse

# Add current directory to path so we can import cityflow2sumo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cityflow2sumo import cityflow2sumo_flow
except ImportError:
    # If running from root, scripts might be needed prefix
    sys.path.append(os.path.join(os.getcwd(), 'scripts'))
    from cityflow2sumo import cityflow2sumo_flow

class MockArgs:
    def __init__(self, or_cityflowtraffic, sumotraffic):
        self.or_cityflowtraffic = or_cityflowtraffic
        self.sumotraffic = sumotraffic

def convert_scenario(scenario_name, roadnet_pattern):
    base_dir = f"data/raw/{scenario_name}"
    cityflow_dir = os.path.join(base_dir, "cityflow")
    env_dir = os.path.join(base_dir, "env")
    
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
        
    # Get all json files
    json_files = glob.glob(os.path.join(cityflow_dir, "*.json"))
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        if "roadnet" in filename:
            continue
            
        # Target output file
        # Convert .json to .rou.xml
        output_filename = filename.replace(".json", ".rou.xml")
        output_path = os.path.join(env_dir, output_filename)
        
        print(f"Converting {filename} to {output_filename}...")
        
        args = MockArgs(
            or_cityflowtraffic=json_file,
            sumotraffic=output_path
        )
        
        try:
            cityflow2sumo_flow(args)
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

def main():
    # JiNan
    convert_scenario("JiNan", "roadnet_3_4.json")
    
    # NewYork
    convert_scenario("NewYork", "roadnet_28_7.json")
    
    # Hangzhou
    convert_scenario("Hangzhou", "roadnet_4_4.json")

if __name__ == "__main__":
    main()
