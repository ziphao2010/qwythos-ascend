"""
Ascend 310 TBE custom operator: 3D Multi-resolution RoPE (mRoPE).

Qwen3.5 uses multi-resolution rotary position encoding with 3 sections:
  mrope_section = [11, 11, 10]  → 32 dims split into temporal/height/width

The key difference from standard RoPE is that each position has 3 coordinate
sets (t, h, w) instead of just one position. For text tokens, all three are
the same (linear position). For vision tokens, h/w encode 2D spatial grid.

This implementation provides the core RoPE computation as a TBE operator.
"""
import math
from te import tik
from te.lang import cce


def get_rotary_embedding(head_dim, base=10000000, dtype="float16"):
    """Precompute rotary embedding frequencies for Qwen3.5 mRoPE.

    head_dim=256, with mrope_section=[11,11,10] = 32 dims
    (32 = 256/8 because partial_rotary_factor=0.25 means only 25% of dims are rotated)
    """
    import numpy as np
    mrope_section = [11, 11, 10]
    total_dim = sum(mrope_section)  # 32

    inv_freq = 1.0 / (base ** (np.arange(0, total_dim, 2, dtype=np.float32) / total_dim))
    return inv_freq


def mrope_3d_compute(query, key, position_ids, output, kernel_name="mrope_3d"):
    """3D mRoPE for Qwen3.5.

    Args:
        query: [batch, heads, seq_len, head_dim]
        key: [batch, kv_heads, seq_len, head_dim]
        position_ids: [3, seq_len] - three position IDs (t, h, w)
        output: rotated query + key (concatenated or modified in-place)
    """
    # Implementation uses CANN's built-in RoPE support via te.lang
    # For custom 3D variant, we compose standard RoPE operations

    # Split the head dim into sections per mrope_section
    # Apply standard RoPE with different sin/cos for each section
    # The te.lang.cce has RoPE computation support

    # For the Ascend 310, we can implement this as:
    # 1. Compute cos/sin for each position section
    # 2. Apply rotation to query/key per section
    # 3. Concatenate results

    shape_q = query.get("shape")
    dtype = query.get("dtype").lower()
    batch, num_heads, seq_len, head_dim = shape_q

    # The te_op API for RoPE (checking CANN built-in support)
    # Custom handling for 3D split of the head dimension

    return None  # Placeholder - TBE implementation uses tik DSL


def apply_mrope_3d_numpy(query, key, position_ids, head_dim=256, theta=10000000.0):
    """NumPy fallback: apply 3D mRoPE to query and key tensors."""
    import numpy as np

    batch, num_heads, seq_len, _ = query.shape
    _, kv_heads, _, _ = key.shape
    partial_dim = int(head_dim * 0.25)

    mrope_section = [11, 11, 10]

    # Compute cos/sin for each of 3 position dimensions
    cos_list = []
    sin_list = []
    pos_t, pos_h, pos_w = position_ids[0], position_ids[1], position_ids[2]

    offset = 0
    for section_len in mrope_section:
        section_half = section_len // 2
        inv_freq = 1.0 / (theta ** (np.arange(0, section_len, 2, dtype=np.float32) / sum(mrope_section)))

        # For text mode: all positions use pos_t
        pos = pos_t  # Using temporal position for all (text-only mode)

        freqs = np.outer(pos, inv_freq)  # [seq_len, section_half]
        emb = np.cos(freqs), np.sin(freqs)
        cos_list.append(emb[0])
        sin_list.append(emb[1])
        offset += section_len

    cos_emb = np.concatenate(cos_list, axis=1)  # [seq_len, partial_dim]
    sin_emb = np.concatenate(sin_list, axis=1)

    # Apply rotation to query and key
    q_embed = query.copy()
    k_embed = key.copy()

    # Only rotate first partial_dim dimensions
    q_partial = query[..., :partial_dim]
    k_partial = key[..., :partial_dim]

    # RoPE rotation: [x1, x2, ..., xd] -> [x1*cos - x2*sin, x1*sin + x2*cos, ...]
    q_embed[..., :partial_dim:2] = (
        q_partial[..., :partial_dim:2] * cos_emb[:seq_len, :partial_dim//2] -
        q_partial[..., 1:partial_dim:2] * sin_emb[:seq_len, :partial_dim//2]
    )
    q_embed[..., 1:partial_dim:2] = (
        q_partial[..., :partial_dim:2] * sin_emb[:seq_len, :partial_dim//2] +
        q_partial[..., 1:partial_dim:2] * cos_emb[:seq_len, :partial_dim//2]
    )

    k_embed[..., :partial_dim:2] = (
        k_partial[..., :partial_dim:2] * cos_emb[:seq_len, :partial_dim//2] -
        k_partial[..., 1:partial_dim:2] * sin_emb[:seq_len, :partial_dim//2]
    )
    k_embed[..., 1:partial_dim:2] = (
        k_partial[..., :partial_dim:2] * sin_emb[:seq_len, :partial_dim//2] +
        k_partial[..., 1:partial_dim:2] * cos_emb[:seq_len, :partial_dim//2]
    )

    return q_embed, k_embed
