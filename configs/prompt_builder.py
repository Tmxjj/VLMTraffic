'''
Author: yufei Ji
Date: 2026-04-23
Description: 交叉口进口道的图片输入（4张）不需要输入上游的图片进行流量预测了，因为上游的流量预测主要是为了辅助绿灯时长的决策，而现在我们改成了直接根据当前的车道级别的排队长度来选择绿灯时长，所以上游的图片对于决策的帮助不大，反而会增加模型的输入复杂度和理解难度。因此在新的版本中，我们将不再提供上游图片作为输入，而是专注于分析进口道的图片来评估当前的交通状态和排队情况，从而做出更精准的信号控制决策。 
'''

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

- **Visual Input Mapping (4 Images in Sequence)**:
You will receive **4 sequential tilted-overhead images** of the approach stop lines. **Strictly identify each approach using its on-image text**. Expected sequence:
  - Image 1: North
  - Image 2: East
  - Image 3: South
  - Image 4: West

- **Lane Layout** (numbered from median outward; L=Left, S=Straight, R=Right):
Each approach: **L1(L), L2(S), L3(R)**
''',

        "Hongkong_SPECIAL_JUNCTION": '''
The intersection is a high-capacity 4-way urban junction operating under **Left-Hand Traffic (LHT)** rules.
- **Visual Input Mapping (4 Images in Sequence)**:
You will receive **4 sequential tilted-overhead images** of the approach stop lines. **Strictly identify each approach using its on-image text**. Expected sequence:
  - Image 1: North
  - Image 2: East
  - Image 3: South
  - Image 4: West

- **Lane Layout** (numbered from median outward; L=Left, S=Straight, R=Right):
  - **N/W**: L1(S), L2(S), L3(S)
  - **E**: L1(S), L2(S), L3(L)
  - **S**: L1(R), L2(S), L3(S)
        ''',

        "T_JUNCTION": '''
A T-shaped three-way junction under **Right-Hand Traffic (RHT)** rules.
- **Visual Input Mapping (3 Images in Sequence)**:
You will receive **3 sequential tilted-overhead images** of the approach stop lines. **Strictly identify each approach using its on-image text**. Expected sequence:
  - Image 1: North
  - Image 2: South
  - Image 3: West

  **Note: The East approach does NOT exist at this T-junction. There is no Image 4.**

- **Lane Layout** (numbered from median outward; L=Left, S=Straight, R=Right):
  - **N** (Major): L1(S), L2(R→W)
  - **S** (Major): L1(L→W), L2(S)
  - **W** (Minor): L1(L→N), L2(R→S)
        ''',

        "SONGDO_5LANE_JUNCTION": '''
A high-capacity four-way intersection under **Right-Hand Traffic (RHT)** rules.
- **Visual Input Mapping (4 Images in Sequence)**:
You will receive **4 sequential tilted-overhead images** of the approach stop lines. **Strictly identify each approach using its on-image text**. Expected sequence:
  - Image 1: North
  - Image 2: East
  - Image 3: South
  - Image 4: West

You must interpret the input images as viewed from inside the intersection, with vehicles facing the camera. This reverses left and right:
- Image left = Vehicle right
- Image right = Vehicle left
**Note:** L1 (left-turn lane) is on the RIGHT side of the image

- **Lane Layout** (Lane Locations:numbered from vehicle's left to vehicle's right; Lane Functions:L=Left, S=Straight, R=Right):
  - **N** (6): L1(L), L2(L), L3-5(S), L6(R)
  - **W** (6): L1(L), L2-5(S), L6(R)
  - **E/S** (5): L1(L), L2-4(S), L5(R)
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

    # Queue = downstream stop-line queue density; L=Left, S=Straight, R=Right
    COT_LANE_TEMPLATES = {
    "4_JUNCTION": '''
N: L1(L):<Level>, L2(S):<Level>
S: L1(L):<Level>, L2(S):<Level>
E: L1(L):<Level>, L2(S):<Level>
W: L1(L):<Level>, L2(S):<Level>''',

    "Hongkong_SPECIAL_JUNCTION": '''
N: L1(S):<Level>, L2(S):<Level>, L3(S):<Level>
S: L1(R):<Level>, L2(S):<Level>, L3(S):<Level>
E: L1(S):<Level>, L2(S):<Level>
W: L1(S):<Level>, L2(S):<Level>, L3(S):<Level>''',

    "T_JUNCTION": '''
N(Major): L1(S):<Level>
S(Major): L1(L):<Level>, L2(S):<Level>
W(Minor): L1(L):<Level>''',

    "SONGDO_5LANE_JUNCTION": '''
N(6): L1(L):<Level>, L2(L):<Level>, L3(S):<Level>, L4(S):<Level>, L5(S):<Level>
S(5): L1(L):<Level>, L2(S):<Level>, L3(S):<Level>, L4(S):<Level>
E(5): L1(L):<Level>, L2(S):<Level>, L3(S):<Level>, L4(S):<Level>
W(6): L1(L):<Level>, L2(S):<Level>, L3(S):<Level>, L4(S):<Level>, L5(S):<Level>
'''
    }

    # 与 tsc_wrapper.py 中的 GREEN_DURATION_CANDIDATES 保持一致
    DURATION_OPTIONS = [15, 20, 25, 30, 35, 40]
    EVENT_DESCRIPTIONS = '''
| Specific Type        | Category    | Traffic Control Impact     |
|----------------------|-------------|----------------------------|
| Ambulance            | Emergency   | High Priority Passage      |
| Police Car           | Emergency   | High Priority Passage      |
| Fire Truck           | Emergency   | High Priority Passage      |
| Public Bus           | Transit     | Low Priority Passage       |
| School Bus           | Transit     | Low Priority Passage       |
| Traffic Accident     | Crash       | Capacity Reduction         |
| Road Debris          | Obstruction | Capacity Reduction         |
| Construction Barrier | Obstruction | Capacity Reduction         |

**Visual Identification Cues:**
- Ambulance: Boxy medical vehicle with distinct roof lightbars.
- Police Car: Law enforcement vehicle with distinct roof lightbars.
- Fire Truck: Large red emergency truck with roof lightbars.
- Public Bus: Long, large, flat-front, white city passenger vehicle with a rectangular body.
- School Bus: Long, large passenger vehicle with a protruding cowl-front hood.
- Traffic Accident: Crashed vehicles with visible structural deformation, positioned abnormally (e.g., stopped diagonally across lanes).
- Road Debris: Non-vehicle scattered objects (e.g., logs, fallen cargo) blocking the lane.
- Construction Barrier: Orange longitudinal roadblock with red/orange striped end-supports.
    '''

    TRAFFIC_KNOWLEDGE = '''
**Traffic Engineering Principles:**
- **Signal Starvation Prevention**: When queue levels are equal, prioritize the longest-waiting inactive phase to prevent any direction from being indefinitely delayed.
- **Emergency Preemption**: Immediately grant and hold a green phase for approaching emergency vehicles(ambulance, fire truck, police car). This overrides ALL other rules.
- **Transit Priority**: Prioritize detected buses and school buses due to their high passenger loads, secondary only to emergency preemption.
- **Capacity Reduction**: A crash or road obstruction effectively removes one or more lanes from service. Avoid selecting a phase whose movement is blocked. If all phases are partially blocked, minimize green time on the most-blocked phase.
- **Queue Discharge Rate**: A fully loaded lane (Long or Critical queue) requires significantly more green time to complete discharge compared to a Short or Medium queue.
- **Minimum Green Time**: Assign a minimum green duration even for empty queues to allow safe reaction and start-up times.
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

        neighbor_status = "Active" if neighbor_info != "None" else "Inactive"

        prompt = f"""
1. Role Description
You are an expert in traffic signal control and computer vision. Your goal is to ensure safety, minimize delays, and handle special traffic events by analyzing visual inputs from the intersection approaches and asynchronous messages from neighboring intersections.

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
Based on the **Approach Bird's-Eye Oblique Images**, current **Scenario Information**, **Traffic Engineering Knowledge**, and **Action Space**, execute the following three-part analysis:

**A. Scene Understanding**

**A1. Lane-Level Density Assessment**
{density_standards}
- **Queue Density**: Count ONLY vehicles that are **stopped or queuing behind the stop line** in each controlled lane. Vehicles that have already passed the stop line or are beyond the intersection must be excluded. This visual assessment is your basis for both Phase Selection and Duration Selection.
- *Note:* Strictly **IGNORE** unrestricted turn lanes (e.g., dedicated Right-Turn lanes under RHT) as they do not require signal control.

**A2. Phase Mapping**
Map the observed lane densities to their respective Phase IDs. For each phase, aggregate the states of its governed lanes into the following metrics:
- **OverallPressure**: A holistic synthesis of traffic demand for the phase. Output: [Low, Medium, High, or Severe]. *(Primary factor for Phase Selection)*
- **CriticalQueue**: The MAX density level among the phase's constituent lanes. *(Primary factor for Duration Selection)*
- **Tie-Breaker** *(Active ONLY if multiple phases share the same OverallPressure)*: Perform a direct visual comparison of the physical congestion and output the conclusion as: "Phase X queue appears longer than Phase Y". Otherwise, output "None".

**B. Scene Analysis**

**B1. Event Recognition**
Scan ALL images for traffic events. Use the table below to identify Specific Type, Category, and Traffic Control Impact:

{event_description}

**Visual Localization:**
- IF an event is detected: Specify [Specific Type], [Category], [Location: Approach & Lane ID], and [Directly Affected Phase ID].
- IF NO event is present: Strictly output `None`.

**B2. Neighboring Messages**
- Status: {neighbor_status}
- Content: {neighbor_info}

**B3. Condition Assessment**
- Set to `SPECIAL` if an event is detected OR Neighboring Messages status is Active.
- Otherwise, set to `NORMAL`.

**C. Adaptive Reasoning**
Based on your Condition Assessment, you MUST follow ONLY ONE of the paths below:

**[Path 1] IF Condition == Normal:**
Keep reasoning extremely concise (exactly ONE clear sentence per field).
- Phase Reasoning: Select the Phase ID with the highest **OverallPressure** based on the observed visual queues.
- Duration Reasoning: Scale duration based solely on the **CriticalQueue** level of the selected phase. Use longer durations for `Long`/`Critical` queues to ensure complete discharge.

**[Path 2] IF Condition == Special:**
Apply traffic engineering knowledge to reason step-by-step:
- Impact Analysis: Evaluate how the detected local event AND/OR the neighboring messages physically impact the current intersection's capacity and safety.
- Phase Reasoning: Synthesize Impact Analysis and visible queue lengths, prioritizing event mitigation over traffic pressure, and select the appropriate phase.
- Duration Reasoning: Synthesize Impact Analysis and visible queue lengths, prioritizing event mitigation over traffic pressure, and select the appropriate duration.
- Broadcast Notice: Format as "[Specific Type] - [Brief impact on upstream/downstream]" if an event is detected, else "None".

7. Output Format
Thought: [
A. Scene Understanding:
- Lane Analysis:
{cot_lane_template}
- Phase Mapping:
Phase <ID> (<Direction>): OverallPressure: <Level> | CriticalQueue: <Level>
Tie-Breaker: <"None" OR "Phase X queue appears longer than Phase Y">

B. Scene Analysis:
- Event Recognition: <"None" OR "[Specific Type] ([Category]) detected at [Approach & Lane ID], affects Phase [ID]">
- Neighboring Messages: <"Inactive" OR "Active">
- Condition Assessment: <"Normal" OR "Special">

C. Adaptive Reasoning:
Strictly follow [Path 1] OR [Path 2] formatting based on your Condition Assessment.
]
Action: {{"phase": <ID>, "duration": <Duration>}}
"""
        return inspect.cleandoc(prompt).strip()

if __name__ == "__main__":
    # Testing the generated prompt for a standard 4-way junction scenario
    print(PromptBuilder.build_decision_prompt(current_phase_id=0, scenario_name="JiNan_test"))