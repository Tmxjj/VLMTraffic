'''
Author: AI Assistant
Date: 2026-03-07
Description: Step 3 Prompt Builder (Using GT Lane Analysis to force accurate reasoning)
FilePath: /VLMTraffic/configs/prompt_builder_golden_step3.py
'''
import inspect
from configs.prompt_builder import PromptBuilder

class PromptBuilderStep3:
    """
    Constructs text prompts for the VLM leveraging strictly correct Ground Truth (GT) Lane Analysis data.
    Forces the VLM to rely on provided numeric data while using BEV images to verify overall spatial context.
    """
    
    @staticmethod
    def build_step3_prompt(current_phase_id: int, scenario_name: str, gt_lane_analysis: str) -> str:
        phase_explanation = PromptBuilder.get_phase_description(scenario_name)
        scenario_description = PromptBuilder.get_scenario_description(scenario_name)

        prompt = f"""
1. Role Description
You are an expert in traffic management and computer vision. You use your knowledge of traffic engineering (commonsense) to solve traffic signal control tasks. Your goal is to maximize intersection efficiency and ensure emergency vehicle priority by combining accurate numeric lane counts with spatial visual analysis from the provided image.

2. Scenario Information
{scenario_description}
*Reference: Top=North (N), Bottom=South (S), Left=West (W), Right=East (E).*

3. Action Space 
The intersection operates on the following discrete signal phases. You must choose one index:
{phase_explanation}

4. Current State
Currently Active Phase: **[ Phase {current_phase_id} ]**

5. Ground Truth (GT) Lane Analysis
You are provided with the absolute true vehicle counts for the intersection. 
{gt_lane_analysis}

6. Task Definition
Based heavily on the **GT Lane Analysis** provided above and cross-referencing with the **Bird's-Eye-View (BEV) image** for spatial and density confirmation, execute the following steps:

A. Scene Understanding:
- **Phase Mapping**: Map the exact numeric lane counts from Section 5 to the specific Phase IDs listed in the Action Space.
- **Congestion Assessment**: Categorize each phase based on density (verify spatial crowding and gaps visually in the BEV image):
    1. `Low`: Free-flowing traffic or empty
    2. `Medium`: Steady movement with spacing
    3. `High`: Slow-moving with minimal gaps
    4. `Gridlock`: Stationary vehicles densely packed (Critical state requiring immediate attention)

B. Scene Analysis :
- **Emergency **: Visually scan the BEV image for:
    1. **Emergency Vehicles**: Ambulance, Police, or Fire trucks with active lights.
    2. **Incidents**: Traffic accidents (collisions), road construction/maintenance, or broken-down vehicles.
- **Mapping**: If visually detected, specify [Type], [Location - Approach & Lane ID], and the **Directly Affected Phase ID**.
- **Classification**: State `Special` (Emergency present) or `Normal`.

C. Selection Logic :
**[Global Note on Visual Context]**: While you have exact lane counts in Section 5, you MUST holistically evaluate the BEV image. Consider the overall intersection geometry, spatial distribution, queue density, and potential spillback to make the most contextually optimal decision.
**IF Special Condition**:
    1. [Rule: Emergency_Priority]: Select the Phase ID that directly serves the emergency vehicle's lane. 
    2. [Rule: Incident_Avoidance]: Select the Phase ID that moves traffic AWAY from or BYPASSES the accident/construction site.

**IF Normal Condition**:
    1. [Rule: Bottleneck_Rule]: Select the Phase ID with the **HIGHEST** cumulative queue length across its permitted movements (using the GT data).
    2. [Rule: Empty_Lane_Constraint]: NEVER select a phase if its corresponding lanes have 0 waiting vehicles (using the GT data). (Note: If ALL phases have 0 vehicles, Rule 4 Fallback applies).
    3. [Rule: Tie_Breaker]: If congestion is equal among multiple candidates, visually prioritize which queue requires more urgent clearance, or select a Phase ID **DIFFERENT** from the current one.
    4. [Rule: Fallback_Cycle]: If all lanes in all directions are empty, ensure phase rotation by selecting the NEXT Phase relative to the Current Phase.
    5. [Rule: Contextual_Adaptation]: If visual evidence reveals complex traffic patterns (e.g., downstream blockages) not captured by pure counts, autonomously prioritize the optimal Phase ID to maximize intersection efficiency.

    Note: Always prioritize safety and emergency response over regular traffic flow.

7. Chain-of-Thought Reasoning
You must think step-by-step follow Task Definition. The output format must be strictly as follows (without indentation and other extra text):

Thought: [
Scene Understanding: 
- Phase Mapping: 
Phase ID (<Direction, e.g., NTST>): <Congestion Level> | <Reasoning based on GT counts and visual density>
Scene Analysis: 
- Emergency Check: <"None" OR "[Type] detected at [Location], affects Phase [ID]">
- Final Condition: <Normal / Special>
Selection Logic: 
- Rule Identification: <Exact Rule Name from Section 6C>
- Reasoning: <A sentence explaining the choice utilizing both visible spatial characteristics and GT numeric counts . No "however", "wait", or self-correction.>
- Conclusion: Phase <ID>
]

Action: The Selected Phase Index, e.g., 0
"""
        return inspect.cleandoc(prompt).strip()

if __name__ == "__main__":
    example_gt = '''
North Approach: Lane 1(Left-Turn):1, Lane 2(Straight):0, Lane 3(Right-Turn):1
South Approach: Lane 1(Left-Turn):0, Lane 2(Straight):0, Lane 3(Right-Turn):0
East Approach: Lane 1(Left-Turn):0, Lane 2(Straight):0, Lane 3(Right-Turn):0
West Approach: Lane 1(Left-Turn):2, Lane 2(Straight):3, Lane 3(Right-Turn):0
    '''.strip()
    print("--- Step 3 Prompt Example ---")
    print(PromptBuilderStep3.build_step3_prompt(0, "JiNan", example_gt))
