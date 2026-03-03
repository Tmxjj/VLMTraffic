'''
Author: yufei Ji
Date: 2026-01-12 16:48:42
LastEditTime: 2026-03-03 21:06:49
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
            - Phase 0: ETWT, East-West Straight
            - Phase 1: NTST, North-South Straight
            - Phase 2: ELWL, East-West Left-Turn
            - Phase 3: NLSL, North-South Left-Turn
            Note: Right-turning vehicles are unrestricted and permitted to turn at any time
        ''',

        "2_PHASE_T_JUNCTION": '''
            - Phase 0: Major Road Straight/ Left-Turn / Merging
            - Phase 1: Minor Road Entry
            Note: Right-turning vehicles are unrestricted and permitted to turn at any time
        ''',

        "Hongkong_SPECIAL_PHASE": '''
            - Phase 0: ETWT, East-West Straight
            - Phase 1: NTST, North-South Straight
            - Phase 2: SR, Southern Right-Turn
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
                    - North Approach: Lane 1 (Left-Turn): <queue length>, Lane 2 (Straight): <queue length>, Lane 3 (Right-Turn): <queue length>>.
                    - South Approach: Lane 1 (Left-Turn): <queue length>, Lane 2 (Straight): <queue length>, Lane 3 (Right-Turn): <queue length>>.
                    - East Approach: Lane 1 (Left-Turn): <queue length>, Lane 2 (Straight): <queue length>, Lane 3 (Right-Turn): <queue length>>.
                    - West Approach: Lane 1 (Left-Turn): <queue length>, Lane 2 (Straight): <queue length>, Lane 3 (Right-Turn): <queue length>>.
        ''',
        
        "Hongkong_SPECIAL_JUNCTION": '''
                    - North Approach: Lane 1 (Straight): <queue length>, Lane 2 (Straight): <queue length>, Lane 3 (Straight): <queue length>>.
                    - South Approach: Lane 1 (Right-Turn): <queue length>, Lane 2 (Straight): <queue length>, Lane 3 (Straight): <queue length>>.
                    - East Approach: Lane 1 (Straight): <queue length>, Lane 2 (Straight): <queue length>, Lane 3 (Left-Turn): <queue length>>.
                    - West Approach: Lane 1 (Straight): <queue length>, Lane 2 (Straight): <queue length>, Lane 3 (Straight): <queue length>>.
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
            Base on the **Bird's-Eye-View (BEV) image** of an intersection and scenario information,
            Your task is to:
            A. Scene Understanding: Visually Analyze the image approach by approach (North, South, East, West). **For each approach, explicitly scan and report the queue length of EACH lane by its number (Lane 1, Lane 2, etc., as defined in the Scenario Information).** Identify vehicle queues, congestion levels, and emergency vehicles corresponding to each lane, then map them to the correct signal phase. 
                - **Congestion levels** (inferred from density and queue lengths) can be categorized as:
                    -  `Low`: Free-flowing traffic
                    -  `Medium`: Steady movement with spacing
                    -  `High`: Slow-moving with minimal gaps
                    -  `Gridlock`: Stationary vehicles (Critical state requiring immediate attention)
                - **Visual Constraints & Anti-Hallucination (CRITICAL)**:
                    - **INLET vs. OUTLET DEDUCTION**: Do NOT confuse approach (inlet) lanes with departure (outlet) lanes. 
                        * **Inlet Lanes (Count these)**: Vehicles face INWARD toward the intersection center. They are stopped BEHIND the thick white STOP LINE. 
                        * **Outlet Lanes (IGNORE these)**: Vehicles face OUTWARD, driving away from the intersection. You will see their rear ends. DO NOT count any vehicles in outlet lanes.
                    - **Waiting Only**: Count ONLY vehicles currently waiting at the stop line. Ignore vehicles that have already crossed into the intersection.
                    - **Emergency ID**: Identify Ambulance, Police, or Fire truck clearly.
            B. Scene Analysis: Determine the traffic condition based solely on the presence of confirmed emergency vehicles.
                - **Special Condition**: ONLY if a confirmed emergency vehicle is identified.
                    **Constraint**: Disregard vehicles that are too far away or moving away from the intersection. If uncertain, classify as `Regular`.
                - **Normal Condition**: If NO confirmed emergency vehicles are present (default to this if uncertain).
            C. Selection Logic: Select the optimal next signal phase index by applying the following hierarchical rules:
                **IF Special Condition:**
                    - **Emergency Priority**: IMMEDIATELY select the phase serving the emergency vehicle to ensure rapid passage.

                **IF Normal Condition (Standard Logic):**
                    1. **Bottleneck Rule**: Prioritize the phase with the **HIGHEST** congestion level or `Gridlock` status to prevent spillover.
                    2. **Empty Lane Constraint**: NEVER select a phase if its corresponding lanes are completely empty (0 waiting vehicles).
                    3. **Tie-Breaker**: If multiple phases have equal highest congestion, select a phase different from the currently active one (promote fairness).
                    4. **No Traffic Fallback**: If ALL directions are empty, maintain the **Current Phase** (do not switch).


            6. Chain-of-Thought Reasoning
            You must think step-by-step follow Task Definition. The output format must be strictly as follows (without indentation and other extra text):

            Thought: [
            1. Scene Understanding: 
                - Lane Analysis (Mandatory):
                   {cot_lane_template}
                - Phase Mapping:
                    - Phase ID (<Direction, e.g., NTST>): <Congestion Level> | <Details summarized STRICTLY from the corresponding numbered lanes above>.
            2. Scene Analysis: 
                - Emergency Check: Reason about the presence of emergency vehicles in specific numbered lanes. 
                - Final Condition: State the "Final Condition" (Normal or Special).
            3. Selection Logic: 
                - Rule Identification: Identify which rule from the "Task Definition" applies.
                - Conclusion: Select the target Phase ID.
                - Reasoning: Explain why this phase was selected over others, referencing the lane-specific data.
            ]
            
            Action: The Selected Phase Index, e.g., 0
        """
        return inspect.cleandoc(prompt).strip()

if __name__ == "__main__":
    print("--- JiNan Example ---")
    print(PromptBuilder.build_decision_prompt(0, "JiNan_test"))