'''
Author: yufei Ji
Date: 2026-01-12 16:48:42
LastEditTime: 2026-01-14 19:36:19
Description: Optimized Prompt Builder (Visual-Only Analysis)
FilePath: /VLMTraffic/src/inference/prompt_builder.py
'''

class PromptBuilder:
    """
    Constructs text prompts for the VLM based on traffic states.
    """

    PHASE_EXPLANATION = '''
        - Phase 0: NS Straight
        - Phase 1: NS Left
        - Phase 2: EW Straight
        - Phase 3: EW Leff

    '''
    
    @staticmethod
    def build_decision_prompt(current_phase_id: int) -> str:
        """
        Build a prompt for the VLM to make traffic signal decisions based solely on Visual Input and Phase Info.
        
        Args:
            current_phase_id (int): The index of the currently active traffic signal phase.
        """
        
        prompt = f"""
            ### 1. Role Description
            You are an expert in traffic management and computer vision. You use your knowledge of traffic engineering (commonsense) to solve traffic signal control tasks. Your goal is to maximize intersection efficiency by analyzing visual data.

            ### 2. Task Definition
            You are provided with a **Bird's-Eye-View (BEV) image** of a 4-way intersection. 
            Your task is to:
            1.  **Visually Analyze** the image to identify vehicle queues, congestion levels, and empty lanes on all approaches (North, South, East, West).
            2.  **Evaluate** the current signal state.
            3.  **Select** the next optimal signal phase index (Action) to activate.

            ### 3. Action Space (Phase Definitions)
            The intersection operates on the following discrete signal phases. You must choose one index:
            {PromptBuilder.PHASE_EXPLANATION}

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
    # 测试生成的 Prompt
    test_phase_id = 1
    
    print(PromptBuilder.build_decision_prompt(test_phase_id))