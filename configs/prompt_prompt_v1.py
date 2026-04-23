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

    # Queue = downstream stop-line queue density; UpstreamDensity = upstream approaching density
    COT_LANE_TEMPLATES = {
    "4_JUNCTION": '''
North Approach:
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 3(Right): Ignored
South Approach:
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 3(Right): Ignored
East Approach:
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 3(Right): Ignored
West Approach:
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 3(Right): Ignored
    ''',

    "Hongkong_SPECIAL_JUNCTION": '''
North Approach:
  Lane 1(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 3(Straight): [Queue:<Level>, UpstreamDensity:<Level>]
South Approach:
  Lane 1(Right): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 3(Straight): [Queue:<Level>, UpstreamDensity:<Level>]
East Approach:
  Lane 1(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 3(Left): Ignored
West Approach:
  Lane 1(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 3(Straight): [Queue:<Level>, UpstreamDensity:<Level>]
    ''',

    "T_JUNCTION": '''
North Approach (Major):
  Lane 1(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Right): Ignored
South Approach (Major):
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Straight): [Queue:<Level>, UpstreamDensity:<Level>]
West Approach (Minor):
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Right): Ignored
    ''',

    "SONGDO_5LANE_JUNCTION": '''
North Approach (6 lanes):
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 3-5(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 6(Right): Ignored
West Approach (6 lanes):
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2-5(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 6(Right): Ignored
East Approach (5 lanes):
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2-4(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 5(Right): Ignored
South Approach (5 lanes):
  Lane 1(Left): [Queue:<Level>, UpstreamDensity:<Level>], Lane 2-4(Straight): [Queue:<Level>, UpstreamDensity:<Level>], Lane 5(Right): Ignored
    '''
}

    # 与 tsc_wrapper.py 中的 GREEN_DURATION_CANDIDATES 保持一致
    DURATION_OPTIONS = [15, 20, 25, 30, 35, 40]

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

    TRAFFIC_KNOWLEDGE = '''
**Traffic Engineering Principles (Apply as World Knowledge):**
- **Signal Starvation Prevention**: A phase that has been waiting the longest (i.e., the phase that is NOT the currently active phase) should be prioritized when queue levels are equal, to prevent any direction from being indefinitely delayed.
- **Emergency Vehicle Preemption**: When an emergency vehicle (ambulance, fire truck, police car) is detected approaching, immediately grant and hold a green phase on its travel path. Emergency preemption overrides all other considerations.
- **Transit Priority**: Buses and school buses carry high passenger loads. Grant priority passage when detected, but this is secondary to emergency preemption.
- **Incident Capacity Reduction**: A crash or road obstruction effectively removes one or more lanes from service. Avoid selecting a phase whose movement is blocked; if all phases are partially blocked, minimize green time on the most-blocked phase.
- **Queue Discharge Rate**: A fully loaded lane (Long queue) needs more green time to discharge than a short queue. Upstream density predicts future demand and should lengthen the green window when the current queue is already heavy.
- **Minimum Green Time**: Even with an empty queue, a minimum green duration is needed to allow stopped vehicles to react and start moving safely (reaction time + acceleration).
    '''

    DENSITY_LEVELS = """
    - **Density Quantification Standards**:
        - `Empty`: 0 vehicles.
        - `Short`: 1-3 vehicles.
        - `Medium`: 4-7 vehicles.
        - `Long`: 8-10 vehicles.
        - `Critical`: 10+ vehicles (Severe congestion).
        """
    @staticmethod
    def get_duration_description() -> str:
        options_str = ", ".join([f"{d}s" for d in PromptBuilder.DURATION_OPTIONS])
        return f"Options: [{options_str}]"

    @staticmethod
    def get_event_description() -> str:
        return inspect.cleandoc(PromptBuilder.EVENT_DESCRIPTIONS).strip()

    @staticmethod
    def get_traffic_knowledge() -> str:
        return inspect.cleandoc(PromptBuilder.TRAFFIC_KNOWLEDGE).strip()

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
        traffic_knowledge = PromptBuilder.get_traffic_knowledge()
        density_standards = inspect.cleandoc(PromptBuilder.DENSITY_LEVELS)

        neighbor_info = neighbor_messages.strip() if neighbor_messages and neighbor_messages.strip() else "None"

        prompt = f"""
1. Role Description
You are an expert in traffic signal control and computer vision. Your goal is to ensure safety, minimize delays, and handle special traffic events by analyzing visual inputs and asynchronous messages from neighboring intersections.

2. Traffic Engineering Knowledge
{traffic_knowledge}

3. Scenario Information
{scenario_description}

4. Action Space
You must select EXACTLY ONE Phase ID and EXACTLY ONE Duration.
**A. Phase Selection:**
{phase_explanation}

**B. Green Duration Selection:**
{duration_explanation}

5. Current State
Currently Active Phase: **[ Phase {current_phase_id} ]**

6. Task Definition
Based on the **8 multi-view images**, current **Scenario Information**, **Traffic Engineering Knowledge**, and **Action Space**, execute:

A. Scene Understanding
Analyze Images 1-4 (Downstream) and Images 5-8 (Upstream) to assess lane-level and phase-level traffic states. Strictly follow these steps:

**Step 1: Lane-Level Density Assessment**
{density_standards}
- **Queue** (from Downstream Images 1-4): The stop-line queue density of vehicles WAITING at the intersection. This is your **PRIMARY basis for Phase Selection**.
- **UpstreamDensity** (from Upstream Images 5-8): The density of vehicles currently approaching but not yet queued. This is your **AUXILIARY basis for Duration Selection** — it estimates how much demand will arrive during the upcoming green window.
- *Note:* Strictly **IGNORE** unrestricted turn lanes (e.g., dedicated Right-Turn lanes under RHT).

**Step 2: Phase Mapping**
Map lane densities to Phase IDs.For each phase, compute summary statistics combining ALL constituent lanes belonging to that phase:
- **OverallPressure**: Evaluate the overall traffic demand for this phase (Output as: Low, Medium, High, or Severe) by holistically synthesizing the Queue densities of its governed lanes. *(Captures total demand; used for Phase Selection.)*
- **CriticalQueue**: The MAX Queue density level between the two opposing directions. *(Captures the worst-case lane that sets the discharge time; used for Duration Selection.)*
- **CriticalUpstream**: The MAX upstream density among the constituent lanes.

B. Scene Analysis
**Step 1: Event Recognition**
- Detection Task: Scan ALL 8 images for traffic events based on the defined categories below.
{event_description}
- Localization:
  - IF an event is detected: Specify the `[Type]`, `[Location: Approach & Lane ID]`, and the `[Directly Affected Phase ID]`.
  - IF NO event is present: Strictly output `None`.

**Step 2: Neighboring Messages**
- Status: {"ACTIVE" if neighbor_info != "None" else "INACTIVE"}
- Content: {neighbor_info}

**Step 3: Condition Assessment**
- Output strictly `Special` IF a local event is detected OR Neighboring Messages Status is ACTIVE.
- Otherwise, output strictly `Normal`.

C. Adaptive Reasoning
Based on your `Condition Assessment`, you MUST choose ONLY ONE of the following reasoning paths:

**[Path 1] IF Condition == Normal:**
Keep reasoning extremely concise. Limit each part to EXACTLY ONE clear sentence.
- Phase Reasoning: Select Phase ID with the highest **OverallPressure**.
- Duration Reasoning: Scale duration considering CriticalQueue and CriticalUpstream. Use longer durations for `Long`/`Critical`.

**[Path 2] IF Condition == Special:**
Apply your traffic engineering world knowledge to reason step-by-step:
- Impact Analysis: Evaluate how the detected local event AND/OR the received neighboring messages physically impact the current intersection's capacity and safety.
- Phase Reasoning: Synthesize Impact Analysis and Scene Understanding; prioritize event mitigation over standard traffic pressure per the Traffic Engineering Knowledge principles.
- Duration Reasoning: Apply your knowledge of the specific event type, its severity, and standard traffic engineering practice to determine the appropriate green duration.
- Broadcast Notice: Output exactly "None" if no local event is detected. Otherwise, strictly format as: "[Event Type] - [Brief warning on upstream/downstream impact]".

7. Output Format
Thought: [
1.Scene Understanding:
- Lane Analysis:
{cot_lane_template}
- Phase Mapping:
Phase ID (<Direction, e.g., NTST>): OverallPressure: <Level> | CriticalQueue: <Level> | CriticalUpstream: <Level>

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
