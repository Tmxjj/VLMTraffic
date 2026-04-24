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
You will receive **4 sequential images**. These are **tilted overhead views** of each inlet approach, focusing on the lanes near the stop line. **Identify each approach strictly by reading its on-image text**. The expected sequence is:

  - Image 1: View of the North Approach
  - Image 2: View of the East Approach
  - Image 3: View of the South Approach
  - Image 4: View of the West Approach

- **Lane Layout** (numbered from median outward; L=Left, S=Straight, R=Right):
Each approach: **L1(L), L2(S), L3(R)**
''',

        "Hongkong_SPECIAL_JUNCTION": '''
The intersection is a high-capacity 4-way urban junction operating under **Left-Hand Traffic (LHT)** rules.
- **Visual Input Mapping (4 Images in Sequence)**:
You will receive **4 sequential images**. These are **tilted overhead views** of each inlet approach, focusing on the lanes near the stop line. **Identify each approach strictly by reading its on-image text**. The expected sequence is:

  - Image 1: View of the North Approach
  - Image 2: View of the East Approach
  - Image 3: View of the South Approach
  - Image 4: View of the West Approach

- **Lane Layout** (numbered from median outward; L=Left, S=Straight, R=Right):
  - **N/W**: L1(S), L2(S), L3(S)
  - **E**: L1(S), L2(S), L3(L)
  - **S**: L1(R), L2(S), L3(S)
        ''',

        "T_JUNCTION": '''
A T-shaped three-way junction under **Right-Hand Traffic (RHT)** rules.
- **Visual Input Mapping (3 Images in Sequence)**:
You will receive **3 sequential images**. These are **tilted overhead views** of each inlet approach, focusing on the lanes near the stop line. **Identify each approach strictly by reading its on-image text**. The expected sequence is:

  - Image 1: View of the North Approach
  - Image 2: View of the South Approach
  - Image 3: View of the West Approach

  **Note: The East approach does NOT exist at this T-junction. There is no Image 4.**

- **Lane Layout** (numbered from median outward; L=Left, S=Straight, R=Right):
  - **N** (Major): L1(S), L2(R→W)
  - **S** (Major): L1(L→W), L2(S)
  - **W** (Minor): L1(L→S), L2(R→N)
        ''',

        "SONGDO_5LANE_JUNCTION": '''
A high-capacity four-way intersection under **Right-Hand Traffic (RHT)** rules.
- **Visual Input Mapping (4 Images in Sequence)**:
You will receive **4 sequential images**. These are **tilted overhead views** of each inlet approach, focusing on the lanes near the stop line. **Identify each approach strictly by reading its on-image text**. The expected sequence is:

  - Image 1: View of the North Approach
  - Image 2: View of the East Approach
  - Image 3: View of the South Approach
  - Image 4: View of the West Approach

- **Lane Layout** (numbered from median outward; L=Left, S=Straight, R=Right):
  - **N/W** (6): L1-2(L), L3-5(S), L6(R)
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
W: L1(L):<Level>, L2(S):<Level>
    ''',

    "Hongkong_SPECIAL_JUNCTION": '''
N: L1(S):<Level>, L2(S):<Level>, L3(S):<Level>
S: L1(R):<Level>, L2(S):<Level>, L3(S):<Level>
E: L1(S):<Level>, L2(S):<Level>
W: L1(S):<Level>, L2(S):<Level>, L3(S):<Level>
    ''',

    "T_JUNCTION": '''
N(Major): L1(S):<Level>
S(Major): L1(L):<Level>, L2(S):<Level>
W(Minor): L1(L):<Level>
    ''',

    "SONGDO_5LANE_JUNCTION": '''
N(6): L1(L):<Level>, L2(S):<Level>, L3(S):<Level>, L4(S):<Level>, L5(S):<Level>
W(6): L1(L):<Level>, L2(S):<Level>, L3(S):<Level>, L4(S):<Level>, L5(S):<Level>
E(5): L1(L):<Level>, L2(S):<Level>, L3(S):<Level>, L4(S):<Level>
S(5): L1(L):<Level>, L2(S):<Level>, L3(S):<Level>, L4(S):<Level>
'''
    }

    # 与 tsc_wrapper.py 中的 GREEN_DURATION_CANDIDATES 保持一致
    DURATION_OPTIONS = [15, 20, 25, 30, 35, 40]
    #TODO：type和具体类型分离
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
- **Signal Starvation Prevention**: A phase that has been waiting the longest (i.e., the phase that is NOT the currently active phase) should be prioritized when queue levels are equal across approaches, to prevent any direction from being indefinitely delayed.
- **Emergency Vehicle Preemption**: When an emergency vehicle (ambulance, fire truck, police car) is detected approaching, immediately grant and hold a green phase on its travel path. Emergency preemption overrides all other considerations.
- **Transit Priority**: Buses and school buses carry high passenger loads. Grant priority passage when detected, but this is secondary to emergency preemption.
- **Incident Capacity Reduction**: A crash or road obstruction effectively removes one or more lanes from service. Avoid selecting a phase whose movement is blocked; if all phases are partially blocked, minimize green time on the most-blocked phase.
- **Queue Discharge Rate**: A fully loaded lane (Long or Critical queue) requires significantly more green time to complete discharge compared to a Short or Medium queue.
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
Based on the **approach view images**, current **Scenario Information**, **Traffic Engineering Knowledge**, and **Action Space**, execute:

A. Scene Understanding
Analyze the provided images, which represent the tilted overhead views of each approach near the stop line. Strictly follow these steps:

**Step 1: Lane-Level Density Assessment**
{density_standards}
- **Queue Density**: Count ONLY vehicles that are **stopped or queuing behind the stop line** in each controlled lane. Vehicles that have already passed the stop line or are beyond the intersection must be excluded. This visual assessment is your basis for both Phase Selection and Duration Selection.
- *Note:* Strictly **IGNORE** unrestricted turn lanes (e.g., dedicated Right-Turn lanes under RHT) as they do not require signal control.

**Step 2: Phase Mapping**
Map the observed lane queue densities to Phase IDs. For each phase, compute summary statistics combining ALL constituent lanes belonging to that phase:
- **OverallPressure**: Evaluate the overall traffic demand for this phase (Output as: Low, Medium, High, or Severe) by holistically synthesizing the Queue Densities of its governed lanes. *(Captures total demand; used for Phase Selection.)*
- **CriticalQueue**: The MAX Queue Density level among the constituent lanes of the phase. *(Captures the worst-case lane that sets the required discharge time; used for Duration Selection.)*
- **Tie-Breaker** *(Fill in ONLY when two or more phases share the same OverallPressure level)*: Visually compare the total queuing length across the tied phases and output which phase appears to have more vehicles (e.g., "Phase 0 queue appears longer than Phase 1"). If no tie exists, output `N/A`.
B. Scene Analysis
**Step 1: Event Recognition**
- Detection Task: Scan ALL 4 images for traffic events based on the defined categories below.
{event_description}
- Localization:
  - IF an event is detected: Specify the `[Type]` , `[Location: Approach & Lane ID]`, and the `[Directly Affected Phase ID]`.
  - IF NO event is present: Strictly output `None`.

**Step 2: Neighboring Messages**
- Status: {"Active" if neighbor_info != "None" else "Inactive"}
- Content: {neighbor_info}

**Step 3: Condition Assessment**
- Output strictly `Special` IF a local event is detected OR Neighboring Messages Status is Active.
- Otherwise, output strictly `Normal`.

C. Adaptive Reasoning
Based on your `Condition Assessment`, you MUST choose ONLY ONE of the following reasoning paths:

**[Path 1] IF Condition == Normal:**
Keep reasoning extremely concise. Limit each part to EXACTLY ONE clear sentence.
- Phase Reasoning: Select Phase ID with the highest **OverallPressure** based on the observed queues.
- Duration Reasoning: Scale duration based solely on the **CriticalQueue** level of the selected phase. Use longer durations for `Long`/`Critical` queues to ensure complete discharge.

**[Path 2] IF Condition == Special:**
Apply your traffic engineering world knowledge to reason step-by-step:
- Impact Analysis: Evaluate how the detected local event AND/OR the received neighboring messages physically impact the current intersection's capacity and safety.
- Phase Reasoning: Synthesize Impact Analysis and visible queue lengths; prioritize event mitigation over standard traffic pressure per the Traffic Engineering Knowledge principles.
- Duration Reasoning: Apply your knowledge of the specific event type, its severity, and current queue lengths to determine the appropriate green duration.
- Broadcast Notice: Output exactly "None" if no local event is detected. Otherwise, strictly format as: "[Event Type] - [Brief warning on impact]".

7. Output Format
Thought: [
1.Scene Understanding:
- Lane Analysis (Queue Assessment):
{cot_lane_template}
- Phase Mapping:
[List all valid Phase IDs]: Phase <ID> (<Direction>): OverallPressure: <Level> | CriticalQueue: <Level>
- Tie-Breaker: <"N/A" OR "Phase X queue appears longer than Phase Y">
2.Scene Analysis:
- Event Recognition: <"None" OR "[Type] detected at [Location], affects Phase [ID]">
- Neighboring Messages: <"Inactive" OR "Active">
- Condition Assessment: <"Normal" OR "Special">
3.Adaptive Reasoning:
Strictly follow [Path 1] OR [Path 2] formatting based on your Final Condition above.
]
Action: {{"phase": <ID>, "duration": <Duration>}}
"""
        return inspect.cleandoc(prompt).strip()

if __name__ == "__main__":
    # Testing the generated prompt for a standard 4-way junction scenario
    print(PromptBuilder.build_decision_prompt(current_phase_id=0, scenario_name="JiNan_test"))