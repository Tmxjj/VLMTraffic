import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import MODEL_PATH
from evaluation.evaluator import Evaluator

def main():
    # Configuration stub
    config = {
        # simulation config params
    }
    
    # Initialize evaluator
    # Ensure model is downloaded first!
    evaluator = Evaluator(config, model_path=MODEL_PATH)
    
    # Run
    evaluator.run_eval(num_episodes=1)

if __name__ == "__main__":
    main()
