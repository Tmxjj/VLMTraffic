'''
Author: yufei Ji
Date: 2026-01-12 16:48:42
LastEditTime: 2026-01-26 17:15:27
Description: Optimized Prompt Builder (Visual-Only Analysis)
FilePath: /VLMTraffic/configs/prompt_builder.py
'''

class PromptBuilder:
    """
    Constructs text prompts for the VLM based on traffic states and scenarios.
    """

    # Define Phase Descriptions for different scenario types
    PHASE_DESCRIPTIONS = {
        "4_PHASE_STANDARD": '''
            - Phase 0: North-South Straight
            - Phase 1: North-South Left-Turn
            - Phase 2: East-West Straight
            - Phase 3: East-West Left-Turn
        ''',
        "3_PHASE_T_JUNCTION": '''
            - Phase 0: Major Road Straight
            - Phase 1: Major Road Left-Turn / Merging
            - Phase 2: Minor Road Entry
        '''
    }

    # Map specific scenarios to their phase configuration
    SCENARIO_MAP = {
        "JiNan": "4_PHASE_STANDARD",
        "NewYork": "4_PHASE_STANDARD",
        "Hangzhou": "4_PHASE_STANDARD",
        "SouthKorea_Songdo": "4_PHASE_STANDARD",
        "Hongkong_YMT": "3_PHASE_T_JUNCTION", 
        "France_Massy": "3_PHASE_T_JUNCTION" 
    }
    
    @staticmethod
    def get_phase_description(scenario_name: str) -> str:
        """Retrieve the phase description for a given scenario."""
        config_key = PromptBuilder.SCENARIO_MAP.get(scenario_name, "4_PHASE_STANDARD")
        return PromptBuilder.PHASE_DESCRIPTIONS.get(config_key, PromptBuilder.PHASE_DESCRIPTIONS["4_PHASE_STANDARD"])

    @staticmethod
    def build_decision_prompt(current_phase_id: int, scenario_name: str = "JiNan") -> str:
        """
        Build a prompt for the VLM to make traffic signal decisions based solely on Visual Input and Phase Info.
        
        Args:
            current_phase_id (int): The index of the currently active traffic signal phase.
            scenario_name (str): The name of the scenario (e.g., "JiNan", "NewYork").
        """
        
        phase_explanation = PromptBuilder.get_phase_description(scenario_name)

        prompt = f"""
            ### 1. Role Description
            You are an expert in traffic management and computer vision. You use your knowledge of traffic engineering (commonsense) to solve traffic signal control tasks. Your goal is to maximize intersection efficiency by analyzing visual data.

            ### 2. Task Definition
            You are provided with a **Bird's-Eye-View (BEV) image** of an intersection. 
            Your task is to:
            1.  **Visually Analyze** the image to identify vehicle queues, congestion levels, and empty lanes on all approaches.
            2.  **Evaluate** the current signal state.
            3.  **Select** the next optimal signal phase index (Action) to activate.

            ### 3. Action Space (Phase Definitions)
            The intersection operates on the following discrete signal phases. You must choose one index:
            {phase_explanation}

            ### 4. Current State
            **Currently Active Phase**: **[ Phase {current_phase_id} ]**
                *(Instruction: Look at the image. Is this current phase effective? Are the lanes served by Phase {current_phase_id} empty? If so, you should switch.)*

            ### 5. Knowledge Injection (Traffic Commonsense)
            Use these rules to guide your visual reasoning:
            * **Queue Length (Pressure)**: Give priority to the phase serving the longest visible queue of vehicles.
            * **Green Time Utilization**: Do not select a phase if its corresponding lanes are empty in the image.
            * **Anti-Congestion**: If one direction is heavily blocked (gridlock), prioritize clearing it to prevent spillover.
            * **Inertia**: If the current phase still has a continuous flow of cars crossing the stop line, maintain it.

            ### 6. Chain-of-Thought Reasoning
            You must think step-by-step. The output format must be strictly as follows:

            Thought: [
            1. Visual Observation: Describe what you see in the image. Which lanes have the most cars? Which are empty?
            2. Current Phase Analysis: Is Phase {current_phase_id} wasting time?
            3. Selection Logic: Based on "Queue Length" rules, Phase X is the best choice because...
            ]
            Action: The Selected Phase Index, e.g., 0
        """
        return prompt.strip()

if __name__ == "__main__":
    # Test generated prompts for different scenarios
    print("--- JiNan Example ---")
    print(PromptBuilder.build_decision_prompt(0, "JiNan"))
    print("\n--- Hongkong_YMT Example ---")
    print(PromptBuilder.build_decision_prompt(0, "Hongkong_YMT"))