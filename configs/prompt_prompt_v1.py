import inspect

class PromptBuilder:
    """
    Constructs text prompts for the VLM based on traffic states and scenarios.
    """

    PHASE_DESCRIPTIONS = {
        "4_PHASE_STANDARD": '''
Phase 0: ETWT, East-West Straight
Phase 1: NTST, North-South Straight
Phase 2: ELWL, East-West Left-Turn
Phase 3: NLSL, North-South Left-Turn
Note: Right-turning vehicles are unrestricted and permitted to turn at any time.
        ''',

        "2_PHASE_T_JUNCTION": '''
Phase 0: Major Road Straight/ Left-Turn / Merging
Phase 1: Minor Road Entry
Note: Right-turning vehicles are unrestricted and permitted to turn at any time.
        ''',

        "Hongkong_SPECIAL_PHASE": '''
Phase 0: ETWT, East-West Straight
Phase 1: NTST, North-South Straight
Phase 2: SR, Southern Right-Turn
Note: Left-turning vehicles are unrestricted and permitted to turn at any time.
        '''
    }

    PHASE_MAP = {
        "JiNan": "4_PHASE_STANDARD",
        "NewYork": "4_PHASE_STANDARD",
        "Hangzhou": "4_PHASE_STANDARD",
        "SouthKorea_Songdo": "4_PHASE_STANDARD",  
        "Hongkong_YMT": "Hongkong_SPECIAL_PHASE",
        "France_Massy": "2_PHASE_T_JUNCTION",
        "JiNan_test": "4_PHASE_STANDARD"
    }

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
  - **East Inlet**: Has a **Left-Turn Lane** located at the **OUTERMOST Lane** (Lane 3).
  - **South Inlet**: Has a **Right-Turn Lane** located at the **INNERMOST Lane** (Lane 1).
  - **North & West Inlets**: No dedicated turn lanes; all lanes are straight-only.
        ''',

        "T_JUNCTION": '''
A T-shaped three-way junction under **Right-Hand Traffic (RHT)** rules.
- **Visual Input Mapping (8 Images in Sequence)**:
You will receive 8 sequential images. **Identify each view strictly by reading its on-image text**. The expected sequence is:

  [Downstream Views] (Looking outward from the intersection)
  - Image 1: "Approach: North", "Type: Downstream"
  - Image 2: "Approach: South", "Type: Downstream"
  - Image 3: "Approach: West", "Type: Downstream"

  [Upstream Views] (Looking towards the intersection)
  - Image 4: "Approach: North", "Type: Upstream"
  - Image 5: "Approach: South", "Type: Upstream"
  - Image 6: "Approach: West", "Type: Upstream"

- **Lane Layout & Numbering**:
  Each approach has **2 lanes**. From the yellow median (center line) outward to the white curb:
    - **North Approach** (Major Road): Lane 1 (Straight-Through), Lane 2 (Right-Turn toward West)
    - **South Approach** (Major Road): Lane 1 (Left-Turn toward West), Lane 2 (Straight-Through)
    - **West Approach** (Minor Road): Lane 1 (Left-Turn toward South), Lane 2 (Right-Turn toward North)
        ''',

        "SONGDO_5LANE_JUNCTION": '''
A high-capacity four-way intersection under **Right-Hand Traffic (RHT)** rules.
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
  Lanes are numbered from the **Median (Innermost)** outward to the **Curb (Outermost)**.
    - **North/West Approach** (6 lanes): Lane 1-2(Left), Lane 3-5(Straight), Lane 6(Right)
    - **East/South Approach** (5 lanes): Lane 1(Left), Lane 2-4(Straight), Lane 5(Right)
        '''
    }

    SCENARIO_MAP = {
        "JiNan": "4_JUNCTION",
        "NewYork": "4_JUNCTION",
        "Hangzhou": "4_JUNCTION",
        "SouthKorea_Songdo": "SONGDO_5LANE_JUNCTION", 
        "Hongkong_YMT": "Hongkong_SPECIAL_JUNCTION",
        "France_Massy": "T_JUNCTION",
        "JiNan_test": "4_JUNCTION"
    }
    
    # OPTIMIZATION 1 & 2: Change <int> to <Level> and mask Right-Turn/Unrestricted lanes as Ignored
    COT_LANE_TEMPLATES = {
    "4_JUNCTION": '''
North Approach: 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 3(Right): Ignored
South Approach: 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 3(Right): Ignored
East Approach: 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 3(Right): Ignored
West Approach: 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 3(Right): Ignored
    ''',
    
    "Hongkong_SPECIAL_JUNCTION": '''
North Approach: 
  Lane 1(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 2(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 3(Straight): [Queue:<Level>, Arrivals:<Level>]
South Approach: 
  Lane 1(Right): [Queue:<Level>, Arrivals:<Level>], Lane 2(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 3(Straight): [Queue:<Level>, Arrivals:<Level>]
East Approach: 
  Lane 1(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 2(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 3(Left): Ignored
West Approach: 
  Lane 1(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 2(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 3(Straight): [Queue:<Level>, Arrivals:<Level>]
    ''',
    
    "T_JUNCTION": '''
North Approach (Major): 
  Lane 1(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 2(Right): Ignored
South Approach (Major): 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2(Straight): [Queue:<Level>, Arrivals:<Level>]
West Approach (Minor): 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2(Right): Ignored
    ''',

    "SONGDO_5LANE_JUNCTION": '''
North Approach (6 lanes): 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2(Left): [Queue:<Level>, Arrivals:<Level>], Lane 3-5(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 6(Right): Ignored
West Approach (6 lanes): 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2-5(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 6(Right): Ignored
East Approach (5 lanes): 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2-4(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 5(Right): Ignored
South Approach (5 lanes): 
  Lane 1(Left): [Queue:<Level>, Arrivals:<Level>], Lane 2-4(Straight): [Queue:<Level>, Arrivals:<Level>], Lane 5(Right): Ignored
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
        options_str = ", ".join([f"{d}s" for d in PromptBuilder.DURATION_OPTIONS])
        return f"Options: [{options_str}]"

    @staticmethod
    def get_event_description() -> str:
        return inspect.cleandoc(PromptBuilder.EVENT_DESCRIPTIONS).strip()

    @staticmethod
    def get_cot_lane_template(scenario_name: str) -> str:
        config_key = PromptBuilder.SCENARIO_MAP.get(scenario_name, "4_JUNCTION")
        template = PromptBuilder.COT_LANE_TEMPLATES.get(config_key, PromptBuilder.COT_LANE_TEMPLATES["4_JUNCTION"])
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

**Step 1: Lane-Level Density Assessment **
Categorize the traffic density using the following levels: `Empty`, `Short`, `Medium`, `Long`.
- **Downstream (Images 1-4):** Assess the queuing density waiting at the stop line. **This is your PRIMARY decision basis for Phase Selection.**
- **Upstream (Images 5-8):** Assess the density of incoming/approaching vehicles. Use this as an **AUXILIARY factor** to estimate vehicle arrivals during the green light for Duration Selection.
- *Note:* Strictly **IGNORE** unrestricted turn lanes (e.g., dedicated Right-Turn lanes).

**Step 2: Phase Mapping**
- **Mapping(MAX Rule):** Map the identified lane densities to the specific Phase IDs. For bi-directional phases (e.g., NTST, ELWL), determine the final Phase Density by taking the **MAXIMUM** density level between the two opposing lanes (e.g., if North Lane 2 Queue is 'Long' and South Lane 2 Queue is 'Short', the NTST Phase Queue Density is 'Long'). The heavier queue dictates the phase urgency.
- **Output Assessment:** For each mapped Phase ID, synthesize and output:
  - **Queue Density:** <Level> *(Derived from Downstream images)*
  - **Expected Arrivals:** <Level> *(Derived directly from Upstream images)*
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
- Phase Reasoning: To maximize intersection throughput and minimize overall delay, prioritize the Phase ID with the highest **Upstream** density. **Tie-breaker:** If two phases have the identical density level, visually compare them and select the one with the longer physical queue/cluster.
- Duration Reasoning: To optimize continuous traffic flow efficiency, assign duration based on the dominant density level and the auxiliary Downstream expected arrivals.

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
Phase ID (<Direction, e.g., NTST>): Upstream Density: <Level> | Expected Arrivals: <Level>
]

2.Scene Analysis: 
- Event Recognition: <"None" OR "[Type] detected at [Location], affects Phase [ID]">
- Neighboring Messages: <"Inactive" OR "Active">
- Final Condition: <"Normal" OR "Special">

3.Adaptive Reasoning: 
<Strictly follow [Path 1] OR [Path 2] formatting based on your Final Condition above.>
]

Action:
{
  "phase": <ID>,
  "duration": <Duration>
}
"""
        return inspect.cleandoc(prompt).strip()

if __name__ == "main":
  print("--- JiNan Example ---")
  print(PromptBuilder.build_decision_prompt(0, "JiNan_test"))