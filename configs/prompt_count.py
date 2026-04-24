import inspect

class PromptBuilder:
    """
    Constructs text prompts for the VLM based on traffic states and scenarios.
    """

    # Define Phase Descriptions and scenario descriptions for different scenario types
    PHASE_DESCRIPTIONS = {
        "4_PHASE_STANDARD": '''
Phase 0: ETWT, East-West Straight
Phase 1: NTST, North-South Straight
Phase 2: ELWL, East-West Left-Turn
Phase 3: NLSL, North-South Left-Turn
Note: Right-turning vehicles are unrestricted and permitted to turn at any time
        ''',

        "2_PHASE_T_JUNCTION": '''
Phase 0: Major Road Straight/ Left-Turn / Merging
Phase 1: Minor Road Entry
Note: Right-turning vehicles are unrestricted and permitted to turn at any time
        ''',

        "Hongkong_SPECIAL_PHASE": '''
Phase 0: ETWT, East-West Straight
Phase 1: NTST, North-South Straight
Phase 2: SR, Southern Right-Turn
Note: Left-turning vehicles are unrestricted and permitted to turn at any time
        '''
    }

    PHASE_MAP = {
        "JiNan": "4_PHASE_STANDARD",
        "NewYork": "4_PHASE_STANDARD",
        "Hangzhou": "4_PHASE_STANDARD",
        "SouthKorea_Songdo": "4_PHASE_STANDARD",  # 与 Jinan/Hangzhou 相同的四相位标准方案
        "Hongkong_YMT": "Hongkong_SPECIAL_PHASE",
        "France_Massy": "2_PHASE_T_JUNCTION",
        "JiNan_test": "4_PHASE_STANDARD"
    }

    SCENARIO_DESCRIPTIONS = {
    "4_JUNCTION": '''
A standard four-way intersection under **Right-Hand Traffic (RHT)** rules.
- **Visual Input Mapping (4 Images)**:
Identify each approach strictly by reading its on-image text in this sequence:
  - Image 1: North
  - Image 2: East
  - Image 3: South
  - Image 4: West

- **Lane Layout & Numbering**:
Each approach consists of **3 lanes**. 
As indicated by the on-road numbering (1, 2, 3) in the images, lanes MUST be identified from the yellow median (center line) outward to the white curb (road edge):
  - **Lane 1 (Innermost)**: Dedicated Left-Turn
  - **Lane 2 (Middle)**: Straight
  - **Lane 3 (Outermost)**: Dedicated Right-Turn
''',
        
    "Hongkong_SPECIAL_JUNCTION": '''
A high-capacity four-way urban junction operating under **Left-Hand Traffic (LHT)** rules.
- **Visual Input Mapping (4 Images)**:
Identify each approach strictly by reading its on-image text in this sequence:
  - Image 1: North
  - Image 2: East
  - Image 3: South
  - Image 4: West

- **Lane Layout & Numbering**:
Each approach consists of **3 lanes**. 
As indicated by the on-road numbering (1, 2, 3) in the images, lanes MUST be identified from the yellow median (center line) outward to the white curb (road edge):
  - **North & West Approaches**: All three lanes are straight-only (Lanes 1-3).
  - **South Approach**: Lane 1 (Innermost) is a dedicated right-turn lane, while Lanes 2 and 3 are straight-only.
  - **East Approach**: Lane 3 (Outermost) is a dedicated left-turn lane, while Lanes 1 and 2 are straight-only.
''',

    "T_JUNCTION": '''
A T-shaped three-way junction under **Right-Hand Traffic (RHT)** rules.
- **Intersection Topology**:
The East approach is missing. The North and South approaches form the main continuous road, while the West approach is the terminating branch.
- **Visual Input Mapping (3 Images)**:
Identify each approach strictly by reading its on-image text in this sequence:
  - Image 1: North
  - Image 2: South
  - Image 3: West

- **Lane Layout & Numbering**:
Each approach consists of **2 lanes**. 
As indicated by the on-road numbering (1, 2) in the images, lanes MUST be identified from the yellow median (center line) outward to the white curb (road edge):
  - **North Approach** (Major Road, Heading South): Lane 1 (Straight-Through), Lane 2 (Right-Turn toward West)
  - **South Approach** (Major Road, Heading North): Lane 1 (Left-Turn toward West), Lane 2 (Straight-Through)
  - **West Approach** (Minor Road, Heading East): Lane 1 (Left-Turn toward North), Lane 2 (Right-Turn toward South)
''',

    "SONGDO_5LANE_JUNCTION": '''
A high-capacity four-way intersection under **Right-Hand Traffic (RHT)** rules, designed for heavy urban traffic.
- **Visual Input Mapping (4 Images)**:
Identify each approach strictly by reading its on-image text in this sequence:
  - Image 1: North
  - Image 2: East
  - Image 3: South
  - Image 4: West

- **Lane Layout & Numbering**:
As indicated by the on-road numbering in the images, lanes MUST be identified from the **yellow median (center line) outward to the white curb (road edge)**. 
Approach lane counts are **asymmetric**:

  **North Approach (6 lanes)**:
  - Lane 1 (Innermost): Dedicated Left-Turn
  - Lane 2: Straight
  - Lane 3: Straight
  - Lane 4: Straight
  - Lane 5: Straight
  - Lane 6 (Outermost): Dedicated Right-Turn

  **East Approach (6 lanes)**:
  - Lane 1 (Innermost): Dedicated Left-Turn
  - Lane 2: Straight
  - Lane 3: Straight
  - Lane 4: Straight
  - Lane 5: Straight
  - Lane 6 (Outermost): Dedicated Right-Turn

  **South Approach (5 lanes)**:
  - Lane 1 (Innermost): Dedicated Left-Turn
  - Lane 2: Straight
  - Lane 3: Straight
  - Lane 4: Straight
  - Lane 5 (Outermost): Dedicated Right-Turn

  **West Approach (5 lanes)**:
  - Lane 1 (Innermost): Dedicated Left-Turn
  - Lane 2: Straight
  - Lane 3: Straight
  - Lane 4: Straight
  - Lane 5 (Outermost): Dedicated Right-Turn
'''
}

    SCENARIO_MAP = {
        "JiNan": "4_JUNCTION",
        "NewYork": "4_JUNCTION",
        "Hangzhou": "4_JUNCTION",
        "SouthKorea_Songdo": "SONGDO_5LANE_JUNCTION",  # 5 车道/进口道的高容量路口
        "Hongkong_YMT": "Hongkong_SPECIAL_JUNCTION",
        "France_Massy": "T_JUNCTION",
        "JiNan_test": "4_JUNCTION"
    }
    
    COT_LANE_TEMPLATES = {
    "4_JUNCTION": '''
North: Lane 1(Left):<int>, Lane 2(Straight):<int>
East: Lane 1(Left):<int>, Lane 2(Straight):<int>
South: Lane 1(Left):<int>, Lane 2(Straight):<int>
West: Lane 1(Left):<int>, Lane 2(Straight):<int>
    ''',
    
    "Hongkong_SPECIAL_JUNCTION": '''
North: Lane 1(Straight):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>
East: Lane 1(Straight):<int>, Lane 2(Straight):<int>
South: Lane 1(Right):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>
West: Lane 1(Straight):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>
    ''',
    
    "T_JUNCTION": '''
North: Lane 1(Straight):<int>
South: Lane 1(Left):<int>, Lane 2(Straight):<int>
West: Lane 1(Left):<int>
    ''',

    "SONGDO_5LANE_JUNCTION": '''
North: Lane 1(Left):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>, Lane 4(Straight):<int>, Lane 5(Straight):<int>
East: Lane 1(Left):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>, Lane 4(Straight):<int>, Lane 5(Straight):<int>
South: Lane 1(Left):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>, Lane 4(Straight):<int>
West: Lane 1(Left):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>, Lane 4(Straight):<int>
    '''
}
    
    @staticmethod
    def get_cot_lane_template(scenario_name: str) -> str:
        """获取对应场景的 CoT 车道分析模板"""
        config_key = PromptBuilder.SCENARIO_MAP.get(scenario_name, "4_JUNCTION")
        template = PromptBuilder.COT_LANE_TEMPLATES.get(config_key, PromptBuilder.COT_LANE_TEMPLATES["4_JUNCTION"])
        # 使用 inspect.cleandoc 去除多行字符串内部的多余缩进，保证注入时对齐
        return inspect.cleandoc(template)   
    
    @staticmethod
    def get_phase_description(scenario_name: str) -> str:
        config_key = PromptBuilder.PHASE_MAP.get(scenario_name, "4_PHASE_STANDARD")
        return PromptBuilder.PHASE_DESCRIPTIONS.get(config_key, PromptBuilder.PHASE_DESCRIPTIONS["4_PHASE_STANDARD"]).strip()
    
    @staticmethod
    def get_scenario_description(scenario_name: str) -> str:
        config_key = PromptBuilder.SCENARIO_MAP.get(scenario_name, "4_JUNCTION")
        return PromptBuilder.SCENARIO_DESCRIPTIONS.get(config_key, PromptBuilder.SCENARIO_DESCRIPTIONS["4_JUNCTION"]).strip()
    
    @staticmethod
    def build_decision_prompt(current_phase_id: int, scenario_name: str = "JiNan", neighbor_messages: str = "") -> str:
        phase_explanation = PromptBuilder.get_phase_description(scenario_name)
        scenario_description = PromptBuilder.get_scenario_description(scenario_name)
        cot_lane_template = PromptBuilder.get_cot_lane_template(scenario_name)
        neighbor_info = neighbor_messages.strip() if neighbor_messages and neighbor_messages.strip() else "None"
        
        prompt = f"""
1. Role Description
You are an expert in traffic signal control and computer vision. Your goal is to ensure safety, minimize delays, and handle special traffic events by analyzing visual inputs and asynchronous messages from neighboring intersections.

2. Scenario Information
{scenario_description}
- **Note:** Unrestricted turning lanes are excluded from the analytical output.

3. Action Space
You must select EXACTLY ONE Phase ID and EXACTLY ONE Duration.
**A. Phase Selection:**
{phase_explanation}
**B. Green Duration Selection:**
Options = [15, 20, 25, 30] (unit: seconds)

4. Current State
Currently Active Phase: **[ Phase {current_phase_id} ]**

5. Task Definition
Base on the **Bird's-Eye Oblique Images** of each approach, current **Scenario Information**, and **Action Space**, execute:

A. Scene Understanding
**Step 1: Lane-Level Vehicle Detection**
- Identify each approach lane and count the exact number of queuing vehicles.

**Step 2: Phase Mapping**
- Map the lane counts to corresponding Phase IDs to calculate:
  1. **P_Sum**: Sum of queued vehicles across all lanes in the phase.
  2. **P_Max**: Max queued vehicles in any single lane of the phase.
  
**Step 3: Congestion Assessment**
- Synthesize the visual queuing state with the calculated **P_Sum** to assign a congestion level to each phase: Gridlock > High > Medium > Low.
- **Note:** Strictly ignore visual queues in unrestricted lanes.
  
B. Scene Analysis
**Step 1: Event Recognition**
Scan all images for specific traffic events.
   - Type: Emergency (Ambulance, Police, Fire), Transit (Public Bus, School Bus), Crash (Accidents, Collisions), Obstruction (Debris, Barriers).
   - Impact:
     * Priority Passage: Applies ONLY to Emergency and Transit (Hierarchy: Emergency > Transit). 
     * Capacity Reduction: Applies ONLY to Crash and Obstruction.
   - Rule: If detected, specify [Type], [Location: Approach & Lane ID], and [Directly Affected Phase ID]. If not, output "None".

**Step 2: Neighboring Messages**
   - Status: "ACTIVE" if {neighbor_info} contains data, otherwise "INACTIVE".
   - Content: {neighbor_info}

**Step 3: Condition Assessment**
   - Set to "SPECIAL" if an Event is detected OR Neighboring Messages status is ACTIVE.
   - Otherwise, set to "NORMAL".

C. Adaptive Reasoning :
Based on your `Condition Assessment`, you MUST choose ONLY ONE of the following reasoning paths:

[Path 1] IF Condition == "NORMAL":
Keep reasoning extremely concise (exactly ONE clear sentence per field).
- Phase Reasoning: Select the phase based on the highest **P_Sum**.
- Duration Reasoning: Select the duration based on the highest **P_Max**.

[Path 2] IF Condition == "SPECIAL":
Think step-by-step to handle the specific event impact.
- Impact Analysis: Evaluate how the local event or neighboring message physically impacts the intersection.
- Phase Reasoning: Select the phase by prioritizing the event's impact, using the visual queuing state as secondary context.
- Duration Reasoning: Dynamically adjust the duration to alleviate the event's severity while preventing gridlock in other directions.
- Broadcast Notice: Strictly format as "[Type] - [Brief warning on upstream/downstream impact]" if event detected, else "None".

6. Output Format
Thought: [
1.Scene Understanding: 
- Lane Analysis:
{cot_lane_template}
- Phase Mapping: 
Phase <ID> (<Direction>): P_Sum: <int> | P_Max: <int> | Congestion: <Level>
2.Scene Analysis: 
- Event Recognition: <"None" OR "[Type] detected at [Location], affects Phase [ID]">
- Neighboring Messages: <"Inactive" OR "Active">
- Final Condition: <"Normal" OR "Special">
3.Adaptive Reasoning: 
<Strictly follow [Path 1] OR [Path 2] formatting based on your Final Condition above.>
]
Action: {{"phase": <ID>, "duration": <Duration>}}
"""
        return inspect.cleandoc(prompt).strip()

if __name__ == "__main__":
    print("--- JiNan Example ---")
    print(PromptBuilder.build_decision_prompt(0, "JiNan_test"))