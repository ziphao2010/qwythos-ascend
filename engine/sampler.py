"""Token sampling with NaN/Inf protection."""
import numpy as np
import warnings


class Sampler:
    def __init__(self, vocab_size=248320):
        self.vocab_size = vocab_size

    def sample(self, logits, temperature=0.6, top_p=0.95, top_k=20,
               repetition_penalty=1.05, input_ids=None):
        """Sample next token with numerical safety."""
        logits = logits.copy().astype(np.float64)

        # Guard: clip extreme values that cause exp overflow
        logits = np.clip(logits, -100.0, 100.0)

        # Guard: replace NaN/Inf in logits
        if not np.all(np.isfinite(logits)):
            warnings.warn("NaN/Inf in logits, replacing with -100")
            logits[~np.isfinite(logits)] = -100.0

        # Repetition penalty
        if repetition_penalty != 1.0 and input_ids is not None:
            for token_id in set(input_ids):
                if token_id < len(logits):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty

        # Temperature
        if temperature > 0:
            logits = logits / max(temperature, 1e-8)
        else:
            return int(np.argmax(logits))

        # Numerically stable softmax
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        sum_exp = np.sum(exp_logits)

        if sum_exp <= 0 or not np.isfinite(sum_exp):
            # Fallback: uniform distribution
            probs = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size
        else:
            probs = exp_logits / sum_exp

        # Top-K
        if top_k > 0 and top_k < self.vocab_size:
            indices = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros_like(probs)
            mask[indices] = 1.0
            probs = probs * mask
            probs /= np.sum(probs)

        # Top-P
        if top_p < 1.0:
            sorted_i = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_i])
            cutoff = sorted_i[cumsum > top_p]
            if len(cutoff) > 0:
                probs[cutoff[1:]] = 0.0
                probs /= np.sum(probs)

        # Final guard
        if not np.all(np.isfinite(probs)) or np.sum(probs) <= 0:
            probs = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size

        token = np.random.choice(self.vocab_size, p=probs)
        return int(token)
