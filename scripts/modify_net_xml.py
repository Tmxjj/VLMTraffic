import xml.etree.ElementTree as ET
import os

def modify_net_file(file_path):
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"File not found: {abs_path}")
        return

    print(f"Processing {abs_path}...")
    try:
        tree = ET.parse(abs_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {abs_path}: {e}")
        return

    modified_count = 0

    # Iterate over all tlLogic elements
    for tl_logic in root.findall('tlLogic'):
        phases = list(tl_logic.findall('phase'))
        
        # Requirement: Process intersections with 16 phases
        if len(phases) == 16:
            # 1. Clear existing phases
            for p in phases:
                tl_logic.remove(p)
            
            # 2. Keep only the first 8 phases
            phases_to_keep = phases[:8]
            
            # 3. Modify transition phases (duration 5s -> 3s) and append back
            for phase in phases_to_keep:
                duration = phase.get('duration')
                if duration == '5':
                    phase.set('duration', '3')
                tl_logic.append(phase)
            
            modified_count += 1

    if modified_count > 0:
        tree.write(abs_path, encoding="UTF-8", xml_declaration=True)
        print(f"Successfully modified {modified_count} tlLogic elements in {abs_path}")
    else:
        print(f"No matching tlLogic elements (16 phases) found in {abs_path}")

if __name__ == "__main__":
    files_to_process = [
        "data/raw/JiNan/env/jinan.net.xml",
        "data/raw/NewYork/env/NewYork.net.xml",
        "data/raw/Hangzhou/env/Hangzhou.net.xml"
    ]

    for f in files_to_process:
        modify_net_file(f)
