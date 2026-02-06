'''
Author: yufei Ji
Date: 2026-01-12 16:48:42
LastEditTime: 2026-02-06 19:47:04
Description: Optimized Prompt Builder (Visual-Only Analysis)
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
        "SouthKorea_Songdo": "todo",  # Placeholder for future implementation
        "Hongkong_YMT": "Hongkong_SPECIAL_PHASE", 
        "France_Massy": "2_PHASE_T_JUNCTION",
        "JiNan_test": "4_PHASE_STANDARD"
    }

    SCENARIO_DESCRIPTIONS = {
        "4_JUNCTION": '''
        A standard four-way intersection with dedicated left-turn lanes and dedicated right-turn lanes under **Right-Hand Traffic (RHT)** rules.
        - **Lane Layout**:
            Each approach (N/S/E/W) consists of **3 lanes**, arranged from the median (left) to the curb (right) as: **[ Dedicated Left-Turn | Straight | Dedicated Right-Turn ]**.
        ''',

        "T_JUNCTION": "",
        
        "Hongkong_SPECIAL_JUNCTION": '''
        The intersection is a high-capacity 4-way urban junction operating under **Left-Hand Traffic (LHT)** rules.
        - **LHT Geometry**:
            - **Keep Left**: Vehicles drive on the left.
            - **Left Turn (Curb-side)**: Located at the **OUTERMOST Lane** (Road Edge).
            - **Right Turn (Center-side)**: Located at the **INNERMOST Lane** (Road Divider).
        - **Lane Layout**:
            - **East Inlet**: Has a **Left-Turn Lane** at the bottom edge.
            - **South Inlet**: Has a **Right-Turn Lane** at the right edge (next to divider).
            - **North & West Inlets**: No dedicated turn lanes; all lanes are straight-only.
        '''
    }

    # Map specific scenarios to their phase configuration
   

    SCENARIO_MAP = {
        "JiNan": "4_JUNCTION",
        "NewYork": "4_JUNCTION",
        "Hangzhou": "4_JUNCTION",
        "SouthKorea_Songdo": "todo",  # Placeholder for future implementation
        "Hongkong_YMT": "Hongkong_SPECIAL_JUNCTION", 
        "France_Massy": "T_JUNCTION",
        "JiNan_test": "4_JUNCTION"
    }
    
    
    @staticmethod
    def get_phase_description(scenario_name: str) -> str:
        """Retrieve the phase description for a given scenario."""
        config_key = PromptBuilder.PHASE_MAP.get(scenario_name, "4_PHASE_STANDARD")
        return PromptBuilder.PHASE_DESCRIPTIONS.get(config_key, PromptBuilder.PHASE_DESCRIPTIONS["4_PHASE_STANDARD"]).strip()
    
    @staticmethod
    def get_scenario_description(scenario_name: str) -> str:
        """
        Retrieve the textual description of the intersection environment based on the scenario name.
        """
        config_key = PromptBuilder.SCENARIO_MAP.get(scenario_name, "4_JUNCTION")
        return PromptBuilder.SCENARIO_DESCRIPTIONS.get(config_key, PromptBuilder.SCENARIO_DESCRIPTIONS["4_JUNCTION"]).strip()

    @staticmethod
    def build_decision_prompt(current_phase_id: int, scenario_name: str = "JiNan") -> str:
        """
        Build a prompt for the VLM to make traffic signal decisions based solely on Visual Input and Phase Info.
        
        Args:
            current_phase_id (int): The index of the currently active traffic signal phase.
            scenario_name (str): The name of the scenario (e.g., "JiNan", "NewYork").
        """
        
        phase_explanation = PromptBuilder.get_phase_description(scenario_name)
        scenario_description = PromptBuilder.get_scenario_description(scenario_name)


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
            A. Scene Understanding: Visually Analyze the image to understand lane turning directions (Left/Straight/Right) for each approach and identify **vehicle queues, congestion levels, and emergency vehicles (Ambulance, Police, Fire truck) corresponding to each signal phase**. 
                - **Congestion levels** (inferred from density and queue lengths) can be categorized as:
                    -  `Low`: Free-flowing traffic
                    -  `Medium`: Steady movement with spacing
                    -  `High`: Slow-moving with minimal gaps
                    -  `Gridlock`: Stationary vehicles (Critical state requiring immediate attention)
                - **Visual Constraints**:
                    - **Waiting Only**: Count ONLY vehicles currently waiting at the stop line. Ignore vehicles that have already crossed into the intersection.
                    - **Emergency ID**: Identify Ambulance, Police, or Fire truck clearly.
            B. Scene Analysis: Determine the traffic condition based solely on the presence of confirmed emergency vehicles:
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
            1. Scene Understanding: Describe the status for **EACH phase** (Phase 0, Phase X...) and output strictly using this format:
                - [Phase ID] (<Direction, e.g., NTST>): <Congestion Level> | <Details about queue  & special vehicles in corresponding lanes>.
            2. Scene Analysis: 
                - Emergency Check: Reason about the presence of emergency vehicles. 
                - Final Condition: State the "Final Condition" (Normal or Special).
            3. Selection Logic: 
                - Rule Identification: Identify which rule from the "Task Definition" applies (e.g., Emergency Priority, Bottleneck Rule, Tie-Breaker, or No Traffic Fallback).
                - Conclusion: Select the target Phase ID.
                - Reasoning: Explain why this phase was selected over others, citing specific congestion levels or emergency priority.
            ]
            
            Action: The Selected Phase Index, e.g., 0
        """
        # 使用 inspect.cleandoc 去除多余缩进
        return inspect.cleandoc(prompt).strip()

if __name__ == "__main__":
    # Test generated prompts for different scenarios
    print("--- JiNan Example ---")
    print(PromptBuilder.build_decision_prompt(0, "JiNan_test"))
    print("\n--- Hongkong_YMT Example ---")
    print(PromptBuilder.build_decision_prompt(0, "Hongkong_YMT"))