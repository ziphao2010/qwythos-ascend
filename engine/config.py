"""Configuration management."""
import yaml, os

class EngineConfig:
    def __init__(self, path: str = None):
        self.SOC_VERSION = os.environ.get("SOC_VERSION", "Ascend310")
        self.NUM_NPU_CHIPS = 4
        self.MODEL_PATH = os.environ.get("QWYTHOS_WEIGHT_PATH", "/root/models/Qwythos-9B-Claude-Mythos-5-1M")
        self.TENSOR_PARALLEL_SIZE = 4
        self.DTYPE = "float16"
        self.MAX_MODEL_LEN = 8192
        self.GPU_MEMORY_UTILIZATION = 0.90
        self.TEMPERATURE = 0.6
        self.TOP_P = 0.95
        self.TOP_K = 20
        self.REPETITION_PENALTY = 1.05
        self.MAX_TOKENS = 16384
        self.API_KEY = "your-api-key"
        self.API_HOST = "0.0.0.0"
        self.API_PORT = 8000
        self.OPS_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "ops", "build")
        if path and os.path.exists(path):
            with open(path) as f:
                data = yaml.safe_load(f)
                for k, v in data.items():
                    attr = k.upper()
                    if hasattr(self, attr):
                        setattr(self, attr, v)

config = EngineConfig()
