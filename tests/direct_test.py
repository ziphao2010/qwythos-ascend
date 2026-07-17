"""Direct model test: generate tokens with lm_head."""
import sys, time, numpy as np
sys.path.insert(0, "/root/qwythos_engine")
from engine.qwythos_npu_v5 import QNPU, H

print("Init NPU...")
q = QNPU()

print("Load lm_head...")
from engine.weights import WeightLoader
wl = WeightLoader("/root/models/Qwythos-9B-Claude-Mythos-5-1M")
cw = wl.load_all()
lm = cw["lm_head.weight"].astype(np.float32)

from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("/root/models/Qwythos-9B-Claude-Mythos-5-1M", trust_remote_code=True)

# Generate 3 tokens with timing
prompt = tk.apply_chat_template(
    [{"role": "user", "content": "Say hello in one word."}],
    add_generation_prompt=True, tokenize=False)
ids = tk.encode(prompt)[:512]
print(f"Prompt: {len(ids)} tokens")

for step in range(3):
    t0 = time.time()
    q.forward(32)
    fwd_t = time.time() - t0

    h = np.empty(H, dtype=np.float32)
    q.a.d2h(h, q.h)
    logits = h @ lm.T

    # Sample with temperature
    logits = logits / 0.6
    logits -= np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    token = int(np.random.choice(248320, p=probs))
    ids.append(token)

    text = tk.decode([token])
    print(f"  [{fwd_t:.1f}s] Token {step}: {repr(text)}")
    if token in (248044, 248046):
        print("  (EOS)")
        break

print(f"\nFull output: {repr(tk.decode(ids[len(ids)-3:]))}")
print("Done")
