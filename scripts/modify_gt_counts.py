'''
Author: yufei Ji
Date: 2026-03-11 15:02:05
LastEditTime: 2026-03-11 16:23:53
Description: this script is used to 
FilePath: /VLMTraffic/scripts/modify_gt_counts.py
'''
import os
import json
import re

def modify_gt_counts(file_path):
    print(f"Reading {file_path} ...")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as fin:
        content = fin.read()

    # Split the chunks by the separator "\n-----\n" which was used in the dataset pipeline
    chunks = [c.strip() for c in re.split(r'\n-----\n+', content) if c.strip()]
    
    modified_count = 0
    record_count = 0
    
    # We will modify the file in-place
    with open(file_path, 'w', encoding='utf-8') as fout:
        for chunk in chunks:
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError as e:
                print(f"Skipping a chunk due to JSON parse error: {e}")
                fout.write(chunk + "\n-----\n\n\n\n\n")
                continue
            
            # Modify the gt_vehicle_counts
            if "gt_vehicle_counts" in data:
                for key, val in data["gt_vehicle_counts"].items():
                    if val == 7:
                        data["gt_vehicle_counts"][key] = 6
                        modified_count += 1
                        
            # Write back maintaining the existing structure
            fout.write(json.dumps(data, indent=4) + "\n-----\n\n\n\n\n")
            record_count += 1
            
    print(f"Successfully processed {record_count} records.")
    print(f"Modified {modified_count} values from 7 to 6 in `gt_vehicle_counts`.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modify gt_vehicle_counts replacing 7 with 6")
    parser.add_argument("--jsonl", type=str, default="data/sft_dataset/JiNan/anon_3_4_jinan_real_2000.rou/dataset.jsonl", help="Path to input jsonl file")
    args = parser.parse_args()
    
    modify_gt_counts(args.jsonl)
