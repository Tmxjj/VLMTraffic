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

    # OPTIMIZATION: Explicitly number the lanes from a fixed physical reference point
    SCENARIO_DESCRIPTIONS = {
        "4_JUNCTION": '''
A standard four-way intersection under **Right-Hand Traffic (RHT)** rules.
- **Visual Input Mapping (8 Images in Sequence)**:
You will receive 8 sequential images. **Identify each view strictly by reading its on-image text**. The expected sequence is:

  [Downstream Views] (Looking outward from the intersection)
  - Image 1: "Approach: North", "Type: Downstream"
  - Image 2: "Approach: East", "Type: Downstream"
  - Image 3: "Approach: South", "Type: Downstream"
  - Image 4: "Approach: West", "Type: Downstream"

  [Upstream Views] (Looking towards the intersection)
  - Image 5: "Approach: North", "Type: Upstream"
  - Image 6: "Approach: East", "Type: Upstream"
  - Image 7: "Approach: South", "Type: Upstream"
  - Image 8: "Approach: West", "Type: Upstream"

- **Lane Layout & Numbering**:
Each approach consists of **3 lanes**. 
As indicated by the on-road numbering (1, 2, 3) in the images, lanes MUST be identified from the yellow median (center line) outward to the white curb (road edge):
- **Lane 1 (Innermost)**: Dedicated Left-Turn
- **Lane 2 (Middle)**: Straight
- **Lane 3 (Outermost)**: Dedicated Right-Turn
''',
        
        "Hongkong_SPECIAL_JUNCTION": '''
The intersection is a high-capacity 4-way urban junction operating under **Left-Hand Traffic (LHT)** rules.
- **LHT Geometry & Lane Numbering**:
    - **Keep Left**: Vehicles drive on the left.
    - **Lane Numbers**: Counted from the **Yellow Median (Innermost)** outward to the **White Road Edge/Curb (Outermost)**.
- **Lane Layout**:
    - **East Inlet**: Has a **Left-Turn Lane** located at the **OUTERMOST Lane** (Lane 3, next to white curb).
    - **South Inlet**: Has a **Right-Turn Lane** located at the **INNERMOST Lane** (Lane 1, next to yellow median).
    - **North & West Inlets**: No dedicated turn lanes; all lanes are straight-only.
        ''',

        "T_JUNCTION": '''
A T-shaped three-way junction under **Right-Hand Traffic (RHT)** rules.
- **Intersection Layout**:
    - **Major Road (North–South)**: Runs vertically through the junction. Vehicles approach from both the **North** (top) and **South** (bottom).
    - **Minor Road (West)**: Enters from the **left side**, terminating at the junction (no through-route).
- **Lane Layout & Numbering**:
  Each approach has **2 lanes**. From the yellow median (center line) outward to the white curb:
    - **North Approach** (Major Road): Lane 1 (Straight-Through), Lane 2 (Right-Turn toward West)
    - **South Approach** (Major Road): Lane 1 (Left-Turn toward West), Lane 2 (Straight-Through)
    - **West Approach** (Minor Road): Lane 1 (Left-Turn toward South), Lane 2 (Right-Turn toward North)
        ''',

        "SONGDO_5LANE_JUNCTION": '''
A high-capacity four-way intersection under **Right-Hand Traffic (RHT)** rules, designed for heavy urban traffic.
- **Lane Layout & Numbering**:
  Lanes are numbered from the **Median (Innermost)** outward to the **Curb (Outermost)**.
  Approach lane counts are **asymmetric**:
    - **North Approach** (6 lanes):
        - Lane 1 (Innermost): Left-Turn
        - Lane 2: Left-Turn
        - Lane 3: Straight
        - Lane 4: Straight
        - Lane 5: Straight
        - Lane 6 (Outermost): Right-Turn
    - **West Approach** (6 lanes):
        - Lane 1 (Innermost): Left-Turn
        - Lane 2: Straight
        - Lane 3: Straight
        - Lane 4: Straight
        - Lane 5: Straight
        - Lane 6 (Outermost): Right-Turn
    - **East Approach** (5 lanes):
        - Lane 1 (Innermost): Left-Turn
        - Lane 2: Straight
        - Lane 3: Straight
        - Lane 4: Straight
        - Lane 5 (Outermost): Right-Turn
    - **South Approach** (5 lanes):
        - Lane 1 (Innermost): Left-Turn
        - Lane 2: Straight
        - Lane 3: Straight
        - Lane 4: Straight
        - Lane 5 (Outermost): Right-Turn
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
North Approach: 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Right): [Queue:<int>, Upstream:<int>]
South Approach: 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Right): [Queue:<int>, Upstream:<int>]
East Approach: 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Right): [Queue:<int>, Upstream:<int>]
West Approach: 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Right): [Queue:<int>, Upstream:<int>]
    ''',
    
    "Hongkong_SPECIAL_JUNCTION": '''
North Approach: 
  Lane 1(Straight): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Straight): [Queue:<int>, Upstream:<int>]
South Approach: 
  Lane 1(Right): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Straight): [Queue:<int>, Upstream:<int>]
East Approach: 
  Lane 1(Straight): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Left): [Queue:<int>, Upstream:<int>]
West Approach: 
  Lane 1(Straight): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Straight): [Queue:<int>, Upstream:<int>]
    ''',
    
    "T_JUNCTION": '''
North Approach (Major): 
  Lane 1(Straight): [Queue:<int>, Upstream:<int>], Lane 2(Right): [Queue:<int>, Upstream:<int>]
South Approach (Major): 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>]
West Approach (Minor): 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Right): [Queue:<int>, Upstream:<int>]
    ''',

    "SONGDO_5LANE_JUNCTION": '''
North Approach (6 lanes): 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Left): [Queue:<int>, Upstream:<int>], Lane 3(Straight): [Queue:<int>, Upstream:<int>], Lane 4(Straight): [Queue:<int>, Upstream:<int>], Lane 5(Straight): [Queue:<int>, Upstream:<int>], Lane 6(Right): [Queue:<int>, Upstream:<int>]
West Approach (6 lanes): 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Straight): [Queue:<int>, Upstream:<int>], Lane 4(Straight): [Queue:<int>, Upstream:<int>], Lane 5(Straight): [Queue:<int>, Upstream:<int>], Lane 6(Right): [Queue:<int>, Upstream:<int>]
East Approach (5 lanes): 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Straight): [Queue:<int>, Upstream:<int>], Lane 4(Straight): [Queue:<int>, Upstream:<int>], Lane 5(Right): [Queue:<int>, Upstream:<int>]
South Approach (5 lanes): 
  Lane 1(Left): [Queue:<int>, Upstream:<int>], Lane 2(Straight): [Queue:<int>, Upstream:<int>], Lane 3(Straight): [Queue:<int>, Upstream:<int>], Lane 4(Straight): [Queue:<int>, Upstream:<int>], Lane 5(Right): [Queue:<int>, Upstream:<int>]
    '''
}
    

    DURATION_OPTIONS = [15, 20, 25, 30]

    EVENT_DESCRIPTIONS = '''
        - Event Types:
            - Emergency: Ambulances, police cars, fire trucks.
            - Transit: Public buses, school buses.
            - Crash: Traffic accidents, vehicle collisions.
            - Obstruction: Road debris, construction barriers.
        - Event Impacts:
            - Priority Passage: Emergency > Transit.
            - Capacity Reduction: Crash, Obstruction.
    '''
    
    @staticmethod
    def get_duration_description() -> str:
        """生成时长选择的描述文本"""
        options_str = ", ".join([f"{d}s" for d in PromptBuilder.DURATION_OPTIONS])
        return f"Options: [{options_str}]"
    @staticmethod
    def get_event_description() -> str:
        """获取交通事件类别的纯净文本描述"""
        return inspect.cleandoc(PromptBuilder.EVENT_DESCRIPTIONS).strip()

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
        duration_explanation = PromptBuilder.get_duration_description()
        event_description = PromptBuilder.get_event_description()

        neighbor_info = neighbor_messages.strip() if neighbor_messages and neighbor_messages.strip() else "None"
        
        prompt = f"""
1. Role Description
You are an expert in traffic signal control and computer vision. Your goal is to ensure safety, minimize delays, and handle special traffic events by analyzing visual inputs and asynchronous messages from neighboring intersections.

2. Scenario Information
{scenario_description}

3. Action Space
You must select EXACTLY ONE Phase ID and EXACTLY ONE Duration.
**A. Phase Selection:**
{phase_explanation}

**B. Green Duration Selection:**
{duration_explanation}


4. Current State
Currently Active Phase: **[ Phase {current_phase_id} ]**

5. Task Definition
Base on the **8 multi-view images**, current **Scenario Information**, and **Action Space**, execute:

A. Scene Understanding
Analyze Images 1-4 (downstream) and 5-8 (upstream) to assess lane-level and phase-level traffic states. Strictly follow these steps:

**Step 1: Lane-Level Vehicle Detection**
- **Downstream (Images 1-4):** Identify each approach lane and count the exact number of queuing vehicles.
- **Upstream (Images 5-8):** Identify the corresponding incoming lanes and count the vehicles approaching the intersection.

**Step 2: Phase Mapping**
- Map the identified lanes counts to the specific Phase IDs listed in the Action Space. 
- *Note:* Ensure that both the stop-line lane and its corresponding upstream lane are unified under the exact same Phase ID.

**Step 3: Phase-Level State Synthesis**
For each mapped Phase ID, output the final assessment:
- **Queue:** The integer vehicle count waiting at the stop line.
- **Congestion Assessment**: Categorize each phase based on density:
    1. `Low`: Free-flowing traffic
    2. `Medium`: Steady movement with spacing
    3. `High`: Slow-moving with minimal gaps
    4. `Gridlock`: Stationary vehicles
- **Expected_Arrivals:** Estimate integer count of vehicles that will reach the stop line during the upcoming green duration.

B. Scene Analysis
**Step 1: Event Recognition**
- Detection Task: Scan ALL 8 images to determine if any traffic events are present based on the defined categories below.{event_description}
- Localization: 
  - IF an event is detected: Specify the `[Type]`, `[Location: Approach & Lane ID]`, and the `[Directly Affected Phase ID]`.
  - IF NO event is present: Strictly output `None`.

**Step 2: Neighboring Messages**
- Status: {"ACTIVE" if neighbor_info != "None" else "INACTIVE"}
- Content: {neighbor_info}

**Step 3: Condition Assessment**
- Output strictly `Special` IF a local event is detected OR Neighboring Messages Status is ACTIVE.
- Otherwise, output strictly `Normal`.

C. Adaptive Reasoning :
Based on your `Condition Assessment`, you MUST choose ONLY ONE of the following reasoning paths:

**[Path 1] IF Condition == Normal:**
Keep reasoning extremely concise. Limit each part to EXACTLY ONE clear sentence.
- Phase Reasoning: Briefly assess the 'Total Demand' (Current Queue + Expected Upstream Arrivals), prioritizing the Phase ID that faces the highest combined traffic pressure.
- Duration Reasoning: Assign a longer green duration for heavy queues and expected upstream arrivals, or a shorter duration for light traffic.

**[Path 2] IF Condition == Special:**
You must think step-by-step to handle the impact of the event:
- Impact Analysis: Evaluate how the detected local event AND/OR the received neighboring messages physically impact the current intersection.
- Phase Reasoning: Synthesizing the Impact Analysis and Scene Understanding, prioritizing event mitigation over standard traffic pressure.
- Duration Reasoning: Assign duration based on the severity and clearance time of the event.
- Broadcast Notice: Output exactly "None" if no event. Otherwise, strictly format as: "[Event Type] - [Brief warning on upstream/downstream impact]".

6. Output Format
Thought: [
1.Scene Understanding: 
- Lane Analysis:
{cot_lane_template}
- Phase Mapping: 
Phase ID (<Direction, e.g., NTST>): Queue: <int> | Congestion: <Level> | Upstream: <int> | Expected_Arrivals: <int>

2.Scene Analysis: 
- Event Recognition: <"None" OR "[Type] detected at [Location], affects Phase [ID]">
- Neighboring Messages: <"Inactive" OR "Active">
- Final Condition: <"Normal" OR "Special">

3.Adaptive Reasoning: 
<Strictly follow [Path 1] OR [Path 2] formatting based on your Final Condition above.>
]

Action: {{ "phase": <ID>, "duration": <Duration> }}
"""
        return inspect.cleandoc(prompt).strip()

if __name__ == "__main__":
    print("--- JiNan Example ---")
    print(PromptBuilder.build_decision_prompt(0, "JiNan_test"))