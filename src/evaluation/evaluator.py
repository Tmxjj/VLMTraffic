from ..simulation.env_wrapper import TrafficEnvWrapper
from ..inference.vlm_agent import VLMAgent
from ..inference.prompt_builder import PromptBuilder
from ..bev_generation.bev_generator import BEVGenerator
from .metrics import MetricsCalculator
import os

class Evaluator:
    """
    Runs the end-to-end evaluation loop.
    """
    def __init__(self, config, model_path):
        self.config = config
        self.env = TrafficEnvWrapper(config)
        self.agent = VLMAgent(model_path)
        self.metrics_calc = MetricsCalculator()
        self.bev_gen = BEVGenerator()
        
    def run_eval(self, num_episodes=1):
        """
        Runs the evaluation for a specific number of episodes.
        """
        self.agent.load_model()
        
        for episode in range(num_episodes):
            print(f"Starting Episode {episode+1}")
            self.env.start()
            self.metrics_calc.reset()
            
            done = False
            step = 0
            while not done:
                # 1. Get Simulation State
                # obs = self.env.get_observation()
                
                # 2. Generate BEV Image
                # bev_path = os.path.join(self.config.DATA_DIR, f"eval_ep{episode}_step{step}.png")
                # self.bev_gen.generate_bev(obs, bev_path)
                
                # 3. Construct Prompt
                # prompt = PromptBuilder.build_decision_prompt()
                
                # 4. Get Agent Decision
                # decision = self.agent.get_decision(bev_path, prompt)
                
                # 5. Apply Action
                # self.env.apply_action(decision)
                
                # 6. Step Simulation
                # self.env.step()
                
                # 7. Collect Metrics Data
                # step_info = self.env.get_metrics() # Should return arrived vehicles etc
                # self.metrics_calc.update(step_info)
                
                step += 1
                if step > 1000: # Max steps
                    done = True
            
            self.env.close()
            
            # Output Final Metrics
            final_metrics = self.metrics_calc.compute_metrics()
            print(f"Episode {episode+1} Metrics: {final_metrics}")
