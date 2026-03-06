'''
Author: yufei Ji
Date: 2026-01-12 16:48:42
LastEditTime: 2026-03-06 11:00:06
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
        "SouthKorea_Songdo": "todo",  
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
        '''
    }

    SCENARIO_MAP = {
        "JiNan": "4_JUNCTION",
        "NewYork": "4_JUNCTION",
        "Hangzhou": "4_JUNCTION",
        "SouthKorea_Songdo": "todo",  
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
        
        "T_JUNCTION": "todo"
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
    def build_decision_prompt(current_phase_id: int, scenario_name: str = "JiNan") -> str:
        phase_explanation = PromptBuilder.get_phase_description(scenario_name)
        scenario_description = PromptBuilder.get_scenario_description(scenario_name)
        # 动态获取当前场景的 CoT 车道模板
        cot_lane_template = PromptBuilder.get_cot_lane_template(scenario_name)

        # OPTIMIZATION: Updated Task Definition and CoT format to enforce lane-by-lane inspection
        prompt = f"""
1. Role Description
You are an expert in traffic management and computer vision. You use your knowledge of traffic engineering (commonsense) to solve traffic signal control tasks. Your goal is to maximize intersection efficiency and ensure emergency vehicle priority by analyzing visual data.

2. Scenario Information
{scenario_description}
*Reference: Top=North (N), Bottom=South (S), Left=West (W), Right=East (E).*

3. Action Space 
The intersection operates on the following discrete signal phases. You must choose one index:
{phase_explanation}

4. Current State
Currently Active Phase: **[ Phase {current_phase_id} ]**

5. Task Definition
Base on the **Bird's-Eye-View (BEV) image**, current **Scenario Information**, and **Action Space**, execute:


A. Scene Understanding:
- **Lane Scanning**: For each approach, report the integer queue length for ALL lanes identified in the Scenario Information.
- **Visual Constraints**: 
    * Count ONLY **Inlet Lanes** (vehicles facing INWARD, behind the stop line).
    * IGNORE **Outlet Lanes** (driving away) and vehicles already inside the intersection.
- **Phase Mapping**: Map the identified lane counts to the specific Phase IDs listed in the Action Space.
- **Congestion Assessment**: Categorize each phase based on density:
    1. `Low`: Free-flowing traffic
    2. `Medium`: Steady movement with spacing
    3. `High`: Slow-moving with minimal gaps
    4. `Gridlock`: Stationary vehicles (Critical state requiring immediate attention)

B. Scene Analysis :
- **Emergency **: Scan for:
    1. **Emergency Vehicles**: Ambulance, Police, or Fire trucks with active lights.
    2. **Incidents**: Traffic accidents (collisions), road construction/maintenance, or broken-down vehicles.
- **Mapping**: If detected, specify [Type], [Location - Approach & Lane ID], and the **Directly Affected Phase ID**.
- **Classification**: State `Special` (Emergency present) or `Normal`.

C. Selection Logic :
**IF Special Condition**:
    1. [Rule: Emergency_Priority]: Select the Phase ID that directly serves the emergency vehicle's lane. 
    2. [Rule: Incident_Avoidance]: Select the Phase ID that moves traffic AWAY from or BYPASSES the accident/construction site.


**IF Normal Condition**:
    1. [Rule: Bottleneck_Rule]: Select the Phase ID with the **HIGHEST** cumulative queue length across its permitted movements.
    2. [Rule: Empty_Lane_Constraint]: NEVER select a phase if its corresponding lanes have 0 waiting vehicles. (Note: If ALL phases have 0 vehicles, Rule 4 Fallback applies).
    3. [Rule: Tie_Breaker]: If congestion is equal among multiple candidates, select a Phase ID **DIFFERENT** from the current one.
    4. [Rule: Fallback_Cycle]: If all lanes in all directions are empty, ensure phase rotation by selecting the NEXT Phase relative to the Current Phase.
    5. [Rule: Contextual_Adaptation]: If the scenario involves complex traffic patterns, potential upstream/downstream blockages, or nuances not fully captured by the rules above, autonomously evaluate the overall scene dynamics and determine the most optimal Phase ID to maximize intersection efficiency.


6. Chain-of-Thought Reasoning
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
- Reasoning: <A sentence explaining the choice. No "however", "wait", or self-correction.>
- Conclusion: Phase <ID>
]

Action: The Selected Phase Index, e.g., 0
        """
        return inspect.cleandoc(prompt).strip()

if __name__ == "__main__":
    print("--- JiNan Example ---")
    print(PromptBuilder.build_decision_prompt(0, "JiNan_test"))