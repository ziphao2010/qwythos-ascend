"""Full attention layer implementation for Qwen3.5.

Standard GQA (Grouped Query Attention) with YaRN RoPE.
Config: 16 heads, 4 KV heads, 256 head_dim, 12288 intermediate
"""
import numpy as np


def full_attention_forward(q, k, v, mask=None, softmax_scale=None):
    """Standard scaled dot-product attention.

    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, kv_heads, seq_len, head_dim]
        v: [batch, kv_heads, seq_len, head_dim]
        mask: optional attention mask [batch, 1, seq_len, seq_len]
        softmax_scale: scaling factor (default: 1/sqrt(head_dim))

    Returns:
        output: [batch, heads, seq_len, head_dim]
    """
    head_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / np.sqrt(head_dim)

    # GQA: expand KV heads to match Q heads
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_kv_heads < num_q_heads:
        repeat = num_q_heads // num_kv_heads
        k = np.repeat(k, repeat, axis=1)
        v = np.repeat(v, repeat, axis=1)

    # Score: Q @ K^T
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * softmax_scale

    if mask is not None:
        scores = scores + mask

    # Softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

    # Output: attn_weights @ V
    output = np.matmul(attn_weights, v)
    return output


def rms_norm(x, weight, eps=1e-6):
    """Root Mean Square Layer Normalization."""
    variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    x_norm = x / np.sqrt(variance + eps)
    return (x_norm * weight).astype(x.dtype)


def silu_gate(gate, x):
    """SiLU gated activation: silu(gate) * x."""
    return x * (gate * (1.0 / (1.0 + np.exp(-gate))))


def swiglu_mlp(x, gate_weight, up_weight, down_weight):
    """SwiGLU MLP forward: down @ (silu(gate @ x) * (up @ x))"""
    gate_out = x @ gate_weight.T
    up_out = x @ up_weight.T
    hidden = silu_gate(gate_out, up_out)
    return hidden @ down_weight.T
