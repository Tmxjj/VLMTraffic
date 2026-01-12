import os
import sys
from modelscope.hub.snapshot_download import snapshot_download

# Add src to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import MODELS_DIR

def download_base_model(model_id="qwen/Qwen-VL-Chat", revision=None):
    """
    Downloads the specified model from ModelScope to the models directory.
    """
    print(f"Downloading model: {model_id}...")
    
    # Ensure directory exists
    model_dir = os.path.join(MODELS_DIR, "base_models")
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        model_path = snapshot_download(model_id, cache_dir=model_dir, revision=revision)
        print(f"Model downloaded successfully to: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    # Example usage:
    # python scripts/download_model.py
    # You can change the model_id as needed
    download_base_model(model_id="qwen/Qwen-VL-Chat")
