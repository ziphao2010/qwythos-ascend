import json, os, sys
import re
import numpy as np
import torch
from safetensors import safe_open


def normalize_key(key):
    """Strip known prefixes to normalize weight names."""
    # Remove model.language_model. prefix -> model.
    key = re.sub(r'^model\.language_model\.', 'model.', key)
    # Remove model.model. prefix
    key = re.sub(r'^model\.model\.', 'model.', key)
    # Remove transformer prefix
    key = re.sub(r'^transformer\.', 'model.', key)
    return key


class WeightLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.config = self._load_config()
        self.weights = {}

    def _load_config(self):
        with open(os.path.join(self.model_path, "config.json")) as f:
            return json.load(f)

    def _find_safetensors(self):
        files = []
        for f in os.listdir(self.model_path):
            if f.endswith(".safetensors") or f.endswith(".bin"):
                files.append(os.path.join(self.model_path, f))
        return sorted(files)

    def load_all(self):
        files = self._find_safetensors()
        if not files:
            f = os.path.join(self.model_path, "model.safetensors")
            if os.path.exists(f):
                files = [f]
            else:
                raise FileNotFoundError(f"No weights at {self.model_path}")

        for fpath in files:
            print(f"  Loading {os.path.basename(fpath)}")
            if fpath.endswith(".safetensors"):
                with safe_open(fpath, framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.float().numpy().astype(np.float16)
                        else:
                            tensor = tensor.numpy().astype(np.float16)
                        nkey = normalize_key(key)
                        self.weights[nkey] = tensor
            elif fpath.endswith(".bin"):
                state = torch.load(fpath, map_location="cpu")
                for key, val in state.items():
                    val = val.float().numpy().astype(np.float16)
                    self.weights[key] = val
        return self.weights

    def get_layer_weights(self, layer_idx):
        prefix = f"model.layers.{layer_idx}"
        return {k[len(prefix)+1:]: v for k, v in self.weights.items() if k.startswith(prefix)}

    def shard_weights(self, num_chips=4):
        shards = [{} for _ in range(num_chips)]
        for key, val in self.weights.items():
            if "down" in key and len(val.shape) == 2:
                chunk = val.shape[1] // num_chips
                for i in range(num_chips):
                    shards[i][key] = val[:, i*chunk : (i+1)*chunk]
            elif any(x in key for x in ["qkv","gate","up","proj"]) and len(val.shape) == 2:
                chunk = val.shape[0] // num_chips
                for i in range(num_chips):
                    shards[i][key] = val[i*chunk : (i+1)*chunk]
            else:
                for i in range(num_chips):
                    shards[i][key] = val
        return shards
