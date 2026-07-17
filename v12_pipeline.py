"""
Qwythos-9B v12: Pipeline Parallelism (4-chip throughput).
Sequential per-token latency ~ same, throughput 4x in steady state.

Fixes over the original:
  ✅ Per-chip per-slot hidden state buffers (4 slots × 4 chips)
  ✅ D2D copy between stages (push model: stage N → buf[N+1] before signaling)
  ✅ aclrtSetDevice per thread (each thread sets own chip context)
  ✅ Fixed queue routing: Main → Stage 0 → … → Stage 3 → Main (no loop)
  ✅ Per-slot KV cache (no race across concurrent token processing)
  ✅ h_buf defined (4 ping-pong slots, not 2)
"""
import sys, time, json, queue, threading, numpy as np
sys.path.insert(0, "/root/qwythos_engine")
from engine.qwythos_npu_v11 import Chip, load_layer, run_layer, H

Wp = "/root/models/Qwythos-9B-Claude-Mythos-5-1M"
from engine.weights import WeightLoader

NUM_CHIPS = 4
LAYERS_PER_CHIP = 8
NUM_LAYERS = 32
NUM_SLOTS = 4  # 4 tokens in pipeline at once
H_BYTES = H * 2  # 8192 bytes FP16

print("Loading...", end=" ", flush=True)
t0 = time.time()
wl = WeightLoader(Wp)
wl.load_all()
lt = json.load(open(f"{Wp}/config.json")).get("text_config",
                                               {}).get("layer_types", [])
chips = [Chip(i) for i in range(NUM_CHIPS)]

# Pre-load weights (8 layers per chip, with correct device context)
wc = [None] * NUM_LAYERS
for i in range(NUM_LAYERS):
    ci = i // LAYERS_PER_CHIP
    chips[ci].L.aclrtSetDevice(ci)
    w = load_layer(wl, chips[ci], i)
    if w:
        wc[i] = w
    else:
        print(f"  WARN: Layer {i} weights not loaded!")
print(f"done ({time.time() - t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════
# VERIFIED SEQUENTIAL (4 tokens, baseline)
# ═══════════════════════════════════════════════════════════════════
# Allocate master + shadow buffers for sequential path
chips[0].L.aclrtSetDevice(0)
h_master = chips[0].malloc(H_BYTES)
ch_shadow = [None] * NUM_CHIPS
for c in chips:
    c.L.aclrtSetDevice(c.dev)
    ch_shadow[c.dev] = c.malloc(H_BYTES)

def forward_sequential(h_master, token_kv):
    """Forward one token through all 32 layers (v11 verified pattern)."""
    for i in range(NUM_LAYERS):
        ci = i // LAYERS_PER_CHIP
        c = chips[ci]
        c.L.aclrtSetDevice(ci)
        if ci > 0:
            # D2D copy: master (chip 0) → shadow (this chip)
            chips[0].L.aclrtMemcpy(ch_shadow[ci], H_BYTES,
                                   h_master, H_BYTES, 3)
            h = ch_shadow[ci]
        else:
            h = h_master
        if wc[i]:
            run_layer(c, h, wc[i], lt, i, token_kv)
        if ci > 0:
            # D2D copy back: shadow → master
            chips[ci].L.aclrtMemcpy(h_master, H_BYTES,
                                    ch_shadow[ci], H_BYTES, 3)

print("\n=== Sequential (v11-style, 4 tokens) ===")
t_seq = time.time()
seq_kv = [[] for _ in range(NUM_LAYERS)]
for _ in range(4):
    chips[0].L.aclrtSetDevice(0)
    chips[0].memset(h_master, H_BYTES)
    forward_sequential(h_master, seq_kv)
t_seq = time.time() - t_seq
print(f"  4 tokens: {t_seq:.1f}s = {t_seq/4:.2f}s/token")
print(f"  Throughput: {4/t_seq:.2f} tok/s")

# Free sequential buffers
chips[0].free(h_master)
for c in chips:
    c.free(ch_shadow[c.dev])

# ═══════════════════════════════════════════════════════════════════
# PIPELINE PARALLEL (4 threads × 8 layers)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Pipeline (4 threads, 8 layers each) ===")

# 1. Per-chip per-slot hidden state buffers
#    buf[ci][slot] allocated on chip ci's device
buf = [[None] * NUM_SLOTS for _ in range(NUM_CHIPS)]
for ci in range(NUM_CHIPS):
    chips[ci].L.aclrtSetDevice(ci)
    for si in range(NUM_SLOTS):
        buf[ci][si] = chips[ci].malloc(H_BYTES)
        chips[ci].memset(buf[ci][si], H_BYTES)

# 2. Per-slot KV caches (one per concurrently-processed sequence)
kv_slots = [[[] for _ in range(NUM_LAYERS)] for _ in range(NUM_SLOTS)]

# 3. Pipeline queues
#   Main → in_q → Stage0 → q[0] → Stage1 → q[1] → Stage2 → q[2] → Stage3 → out_q → Main
in_q = queue.Queue()
stage_qs = [queue.Queue() for _ in range(NUM_CHIPS - 1)]
out_q = queue.Queue()


def stage_worker(ci, q_in, q_out):
    """Pipeline stage: processes 8 layers on chip ci.

    Push model: after processing, D2D-copy result to next chip's buffer
    before signaling via queue. Next stage finds data already waiting.
    """
    c = chips[ci]
    c.L.aclrtSetDevice(ci)       # ⚠ each thread must set its own device
    base = ci * LAYERS_PER_CHIP

    while True:
        slot = q_in.get()
        if slot is None:          # sentinel: propagate and die
            q_out.put(None)
            break

        # This stage's local working buffer (already on this chip's device)
        h_local = buf[ci][slot]

        # Process this chip's 8 layers
        for i in range(base, base + LAYERS_PER_CHIP):
            if wc[i]:
                run_layer(c, h_local, wc[i], lt, i, kv_slots[slot])

        # Push model: D2D copy to next chip BEFORE signaling
        # (so next stage finds data ready when it wakes)
        if ci < NUM_CHIPS - 1:
            chips[ci].L.aclrtMemcpy(buf[ci + 1][slot], H_BYTES,
                                    h_local, H_BYTES, 3)

        # Signal next stage
        q_out.put(slot)


# Start 4 stage threads
threads = []
stages_config = [
    (0, in_q, stage_qs[0]),
    (1, stage_qs[0], stage_qs[1]),
    (2, stage_qs[1], stage_qs[2]),
    (3, stage_qs[2], out_q),
]
for ci, q_in, q_out in stages_config:
    t = threading.Thread(target=stage_worker,
                         args=(ci, q_in, q_out), daemon=True)
    t.start()
    threads.append(t)

# Give threads time to initialize (load .om models, etc.)
time.sleep(1.0)

# Feed 4 tokens into pipeline
t_pipe = time.time()
for slot in range(NUM_SLOTS):
    # Zero the initial hidden state on chip 0 for this slot
    chips[0].L.aclrtSetDevice(0)
    chips[0].memset(buf[0][slot], H_BYTES)
    in_q.put(slot)

# Collect all results from final stage
results = []
for _ in range(NUM_SLOTS):
    try:
        slot = out_q.get(timeout=120)
        results.append(slot)
    except queue.Empty:
        print(f"  ⚠ TIMEOUT waiting for pipeline result ({_})")
t_pipe = time.time() - t_pipe

# Shutdown
in_q.put(None)
for t in threads:
    t.join(timeout=5)

# Free pipeline buffers
for ci in range(NUM_CHIPS):
    for si in range(NUM_SLOTS):
        chips[ci].free(buf[ci][si])

# Report
print(f"\n  4 tokens (pipeline): {t_pipe:.1f}s")
if results:
    print(f"  Effective per-token: {t_pipe / len(results):.3f}s")
    print(f"  Throughput: {len(results) / t_pipe:.2f} tok/s")

print(f"\n── Comparison ──────────────────────────────")
print(f"  Sequential 4 tok:  {t_seq:.1f}s  [{4/t_seq:.2f} tok/s]")
print(f"  Pipeline   4 tok:  {t_pipe:.1f}s  [{4/t_pipe:.2f} tok/s]")
print(f"  Speedup:           {t_seq/max(t_pipe,0.001):.2f}x")
print(f"  (Theoretical max:  {NUM_CHIPS}x throughput in steady state)")
print(f"  For larger batches the pipeline gap widens — 4 tok barely fills it.")

# Cleanup remaining chips
# (Chip destructor / aclrtFree handled implicitly at exit)
