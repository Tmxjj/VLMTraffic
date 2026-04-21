'''
Author: yufei Ji
Date: 2026-01-12 16:48:42
LastEditTime: 2026-04-13 21:00:35
Description: Optimized Prompt Builder (Visual-Only Analysis with Lane Numbering)
FilePath: /VLMTraffic/configs/prompt_builder.py
'''
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
- **Lane Layout & Numbering**:
Each approach (North, South, East, West) consists of **3 lanes**. 
From the yellow median (center line ) outward to the white curb (road edge), lanes MUST be identified by number:
- **Lane 1 (Innermost)**: Dedicated Left-Turn
- **Lane 2 (Middle)**: Straight
- **Lane 3 (Outermost)**: Dedicated Right-Turn
        ''',

        "T_JUNCTION": "",
        
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
North Approach: Lane 1(Left-Turn):<int>, Lane 2(Straight):<int>, Lane 3(Right-Turn):<int>
South Approach: Lane 1(Left-Turn):<int>, Lane 2(Straight):<int>, Lane 3(Right-Turn):<int>
East Approach: Lane 1(Left-Turn):<int>, Lane 2(Straight):<int>, Lane 3(Right-Turn):<int>
West Approach: Lane 1(Left-Turn):<int>, Lane 2(Straight):<int>, Lane 3(Right-Turn):<int>
        ''',
        
        "Hongkong_SPECIAL_JUNCTION": '''
North Approach: Lane 1 (Straight):<int>, Lane 2 (Straight):<int>, Lane 3 (Straight):<int>.
South Approach: Lane 1 (Right-Turn):<int>, Lane 2 (Straight):<int>, Lane 3 (Straight):<int>.
East Approach: Lane 1 (Straight):<int>, Lane 2 (Straight):<int>, Lane 3 (Left-Turn):<int>.
West Approach: Lane 1 (Straight):<int>, Lane 2 (Straight):<int>, Lane 3 (Straight):<int>.
        ''',
        
        "T_JUNCTION": '''
North Approach (Major Road): Lane 1(Straight):<int>, Lane 2(Right-Turn):<int>
South Approach (Major Road): Lane 1(Left-Turn):<int>, Lane 2(Straight):<int>
West Approach (Minor Road): Lane 1(Left-Turn):<int>, Lane 2(Right-Turn):<int>
        ''',

        "SONGDO_5LANE_JUNCTION": '''
North Approach (6 lanes): Lane 1(Left-Turn):<int>, Lane 2(Left-Turn):<int>, Lane 3(Straight):<int>, Lane 4(Straight):<int>, Lane 5(Straight):<int>, Lane 6(Right-Turn):<int>
West Approach (6 lanes): Lane 1(Left-Turn):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>, Lane 4(Straight):<int>, Lane 5(Straight):<int>, Lane 6(Right-Turn):<int>
East Approach (5 lanes): Lane 1(Left-Turn):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>, Lane 4(Straight):<int>, Lane 5(Right-Turn):<int>
South Approach (5 lanes): Lane 1(Left-Turn):<int>, Lane 2(Straight):<int>, Lane 3(Straight):<int>, Lane 4(Straight):<int>, Lane 5(Right-Turn):<int>
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
    def build_decision_prompt(current_phase_id: int, scenario_name: str = "JiNan",
                              coordination_context: str = "") -> str:
        """构建 VLM 决策 Prompt。

        Args:
            current_phase_id: 当前激活相位编号
            scenario_name: 场景名称，用于查找相位/场景描述
            coordination_context: 来自上游路口的事件广播文本（非空时插入协同章节）
        """
        phase_explanation = PromptBuilder.get_phase_description(scenario_name)
        scenario_description = PromptBuilder.get_scenario_description(scenario_name)
        cot_lane_template = PromptBuilder.get_cot_lane_template(scenario_name)

        # 协同章节：仅当上游有有效广播时注入
        if coordination_context and coordination_context.strip():
            coordination_section = f"""
6. Upstream Coordination Context [ACTIVE]
A neighboring upstream intersection has detected a traffic event that may affect YOUR intersection:
{coordination_context.strip()}
  ⚠️ You MUST account for this incoming event in your phase selection AND duration choice.
     If an emergency vehicle is en route, prioritize the phase that clears its path.
     If heavy spillback is expected, extend the green duration to absorb the incoming flow.
"""
        else:
            coordination_section = ""

        # 若有协同章节，后续 Task Definition 编号顺延
        task_section_num = 7 if coordination_section else 6

        prompt = f"""
1. Role Description
You are an expert in traffic management and computer vision.
You use your knowledge of traffic engineering to solve traffic signal control tasks.
Your goal is to maximize intersection efficiency and ensure emergency vehicle priority by analyzing visual data.

2. Input Images
You are provided with 8 images of the intersection (in order):
  [Image 1] North approach — stop-line view (vehicles queued behind stop line, driving southward)
  [Image 2] East approach  — stop-line view (vehicles queued behind stop line, driving westward)
  [Image 3] South approach — stop-line view (vehicles queued behind stop line, driving northward)
  [Image 4] West approach  — stop-line view (vehicles queued behind stop line, driving eastward)
  [Image 5] North upstream — road view upstream of the North approach (vehicles approaching from the north)
  [Image 6] East upstream  — road view upstream of the East approach  (vehicles approaching from the east)
  [Image 7] South upstream — road view upstream of the South approach (vehicles approaching from the south)
  [Image 8] West upstream  — road view upstream of the West approach  (vehicles approaching from the west)
Images 1-4 show current queue length at stop lines. Images 5-8 show vehicles en route and expected to arrive during the next green phase.

3. Scenario Information
{scenario_description}
*Reference: Top=North (N), Bottom=South (S), Left=West (W), Right=East (E).*

4. Action Space
The intersection operates on the following discrete signal phases.
You must select ONE phase index AND ONE green duration from the candidates below:
  Phase candidates:
{phase_explanation}
  Green duration candidates (seconds): [10, 15, 20, 25, 30, 35]
  Selection basis: balance current stop-line queue (Images 1-4) with expected upstream arrivals (Images 5-8).
  Longer queues or heavy upstream flow → select longer duration. Light traffic → select shorter duration.

5. Current State
Currently Active Phase: **[ Phase {current_phase_id} ]**
{coordination_section}
{task_section_num}. Task Definition
Based on the **multi-view approach images and upstream road images**, current **Scenario Information**, and **Action Space**, execute:

A. Scene Understanding:
- **Lane Scanning**: For each approach, report the integer queue length for ALL lanes identified in the Scenario Information.
- **Visual Constraints**: 
    Stop Line Constraint: Count ONLY vehicles located behind the stop line. Do NOT identify or count vehicles that have already crossed the stop line and entered the intersection interior (the "box").
    Directional Constraint: Identify ONLY Inward-facing vehicles (Inlet Lanes). Strictly IGNORE all vehicles in Outlet Lanes (those driving away from the intersection center).
- **Phase Mapping**: Map the identified lane counts to the specific Phase IDs listed in the Action Space.
- **Congestion Assessment**: Categorize each phase based on density:
    1. `Low`: Free-flowing traffic
    2. `Medium`: Steady movement with spacing
    3. `High`: Slow-moving with minimal gaps
    4. `Gridlock`: Stationary vehicles (Critical state requiring immediate attention)

B. Scene Analysis :
- **Emergency **: Scan for:
    1. **Emergency Vehicles**: Ambulance, Police, or Fire trucks with active lights.
    2. **Incidents**: Traffic accidents (collisions), road construction/maintenance, or road obstacles
- **Mapping**: If detected, specify [Type], [Location - Approach & Lane ID], and the **Directly Affected Phase ID**.
- **Classification**: State `Special` (Emergency present) or `Normal`.

C. Selection Logic :
**[Global Note on Visual Context]**: Do not rely solely on the extracted numerical lane counts. You MUST holistically evaluate the BEV image. Consider the overall intersection geometry, the spatial distribution of vehicles, and the intuitive visual queuing pressure/density at each approach to make the most contextually optimal decision.
**IF Special Condition**:
    1. [Rule: Emergency_Priority]: Select the Phase ID that directly serves the emergency vehicle's lane. 
    2. [Rule: Incident_Avoidance]: Select the Phase ID that moves traffic AWAY from or BYPASSES the accident/construction site.

**IF Normal Condition**:
    1. [Rule: Fallback_Static]: If ALL lanes have 0 waiting vehicles, ensure phase rotation by selecting the NEXT Phase relative to the Current Phase
    2. [Rule: Bottleneck_Rule]: Select the Phase ID with the **HIGHEST** cumulative queue length and congestion across its permitted movements.
    3. [Rule: Tie_Breaker]: If multiple phases tie for the highest queue, resolve STRICTLY in this order:
        - (a) Straight > Left: Prioritize Straight-moving phases over Left-Turn phases.
        - (b) Max Single Lane: Prioritize the phase with the longest single-lane queue.
        - (c) Index_Order: If still tied, strictly select the Phase with the LOWEST Phase ID among the tied candidates (excluding the Current Phase ID).
    4. [Rule: Contextual_Adaptation]: If the visual context presents atypical dynamics or complex nuances not adequately resolved by Rules 1-3, apply general traffic engineering common sense to holistically evaluate the scene. Select the optimal Phase ID to maximize overall throughput, and base your decision on a logical assessment of the complete visual state.
    
    Note: Always prioritize safety and emergency response over regular traffic flow.
    
{task_section_num + 1}. Chain-of-Thought Reasoning
You must think step-by-step follow Task Definition. The output format must be strictly as follows (without indentation and other extra text):

Thought: [
Scene Understanding: 
- Lane Analysis (Mandatory):
{cot_lane_template}
- Phase Mapping: 
Phase ID (<Direction, e.g., NTST>): <Congestion Level> | <Reasoning>
Scene Analysis: 
- Emergency Check: <"None" OR "[Type] detected at [Location], affects Phase [ID]">
- Final Condition: <Normal / Special>
Selection Logic:
- Rule Identification: <Exact Rule Name from Section 5C>
- Reasoning: <State the direct cause for the selection in a sentence. Focus purely on facts, without conversational filler or self-correction.>
- Conclusion: Phase <ID>
Duration Selection:
- Stop-line queue pressure (Images 1-4): <brief assessment>
- Upstream arrival estimate (Images 5-8): <brief assessment>
- Selected Duration: <X> seconds | Reasoning: <one sentence>
]

Action: phase=<phase_id>, duration=<seconds>, e.g., Action: phase=1, duration=25
        """
        return inspect.cleandoc(prompt).strip()

if __name__ == "__main__":
    print("--- JiNan Example ---")
    print(PromptBuilder.build_decision_prompt(0, "JiNan_test"))