class PromptBuilder:
    """
    Constructs text prompts for the VLM based on traffic states.
    """
    
    @staticmethod
    def build_decision_prompt(traffic_state_description: str = ""):
        """
        Creates the prompt for the signal control decision task.
        """
        base_prompt = (
            "You are a traffic signal control expert. "
            "Analyze the provided Bird's Eye View (BEV) image of the intersection. "
            "Identify the congestion levels and queue lengths on each lane. "
            "Decide the next phase for the traffic light to minimize average travel time and waiting time. "
            "Output your decision as the phase index (e.g., 'Phase 1')."
        )
        if traffic_state_description:
            base_prompt += f"\nAdditional Context: {traffic_state_description}"
            
        return base_prompt

    @staticmethod
    def build_explanation_prompt(decision: str):
        """
        Creates a prompt asking for an explanation of a previous decision.
        """
        return f"Explain why you chose {decision} for this traffic situation."
