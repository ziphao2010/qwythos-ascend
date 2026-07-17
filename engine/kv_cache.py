"""KV cache management for hybrid attention model.

Full attention layers: standard KV cache (key, value tensors)
DeltaNet layers: recurrent state (state vector per head)

Memory budget (8192 ctx, 4 chips):
  Full attention KV: 8 layers x 2 x 4KV_heads x 8192 x 256 x 2bytes = ~0.5 GB
  DeltaNet states: 24 layers x 2 x 32V_heads x 128 x 2bytes x batch = ~0.4 GB
"""
import numpy as np


class KVCache:
    def __init__(self, num_full_attn_layers=8, num_delta_layers=24,
                 num_heads=4, head_dim=256, max_seq_len=8192,
                 num_value_heads=32, value_head_dim=128, batch=1):

        self.num_full = num_full_attn_layers
        self.num_delta = num_delta_layers

        # Full attention KV cache: [layers, 2, batch, kv_heads, seq_len, head_dim]
        self.kv_cache = np.zeros(
            (num_full_attn_layers, 2, batch, num_heads, max_seq_len, head_dim),
            dtype=np.float16
        )
        self.seq_len = 0

        # DeltaNet recurrent states: [layers, 2, batch, value_heads]
        # scalar state per head (d_state=1, Mamba2-style)
        self.delta_states = np.zeros(
            (num_delta_layers, 2, batch, num_value_heads),
            dtype=np.float16
        )

    def update_kv(self, layer_idx, key, value):
        """Append key/value to full attention cache at given layer."""
        seq_start = self.seq_len
        seq_end = seq_start + key.shape[-2]
        self.kv_cache[layer_idx, 0, :, :, seq_start:seq_end, :] = key
        self.kv_cache[layer_idx, 1, :, :, seq_start:seq_end, :] = value
        return self.kv_cache[layer_idx, 0, :, :, :seq_end, :], \
               self.kv_cache[layer_idx, 1, :, :, :seq_end, :]

    def update_delta_state(self, layer_idx, state_k, state_v):
        """Update DeltaNet recurrent state (scalar per head)."""
        if state_k.ndim == 2:
            self.delta_states[layer_idx, 0] = state_k
            self.delta_states[layer_idx, 1] = state_v
        else:
            self.delta_states[layer_idx, 0] = state_k.squeeze()
            self.delta_states[layer_idx, 1] = state_v.squeeze()

    def get_delta_state(self, layer_idx):
        return self.delta_states[layer_idx, 0], self.delta_states[layer_idx, 1]

    def increment_seq_len(self, n=1):
        self.seq_len += n

    def reset(self):
        self.seq_len = 0
        self.kv_cache.fill(0)
        self.delta_states.fill(0)
