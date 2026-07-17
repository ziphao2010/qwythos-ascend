"""
Qwythos-9B v12: Pipeline Parallelism (4-chip throughput).
Fixes:
  - Per-chip per-slot hidden state buffers (buf[ci][si])
  - D2D copy between stages (push model)
  - Proper device context (aclrtSetDevice) per thread
  - Fixed queue routing: in_q → q[0]→q[1]→q[2]→out_q
  - Per-slot KV cache for thread safety

Sequential per-token latency ~ same, but throughput ~ 4x (steady state).
"""
import sys, time, json, queue, threading, numpy as np
from ctypes import c_void_p, c_size_t, c_int, c_uint32, c_char_p, byref, CDLL, POINTER

sys.path.insert(0, "/root/qwythos_engine")
WEIGHT_PATH = "/root/models/Qwythos-9B-Claude-Mythos-5-1M"
MODEL_DIR = "/root/qwythos_engine/om_models"
H, NH, NKV, HD, IM = 4096, 16, 4, 256, 12288
NUM_CHIPS = 4
LAYERS_PER_CHIP = 8
NUM_LAYERS = 32
NUM_SLOTS = 4  # max tokens in pipeline = NUM_CHIPS

# ── Chip class (from v11, self-contained) ──────────────────────────
class Chip:
    def __init__(self, dev):
        self.L = CDLL("libascendcl.so")
        self.dev = dev
        self._setup_funcs()
        if dev == 0:
            self.L.aclInit(None)
        self.L.aclrtSetDevice(dev)
        self._oc = {}       # loaded .om models cache
        self.h = None       # local hidden state buffer

    def _setup_funcs(self):
        L = self.L
        P = POINTER
        sigs = [
            ("aclrtMalloc",           [P(c_void_p), c_size_t, c_int], c_int),
            ("aclrtFree",             [c_void_p], c_int),
            ("aclrtMemcpy",           [c_void_p, c_size_t, c_void_p, c_size_t, c_int], c_int),
            ("aclrtMemset",           [c_void_p, c_size_t, c_int, c_size_t], c_int),
            ("aclrtSetDevice",        [c_int], c_int),
            ("aclmdlLoadFromFile",    [c_char_p, P(c_uint32)], c_int),
            ("aclmdlCreateDesc",      [], c_void_p),
            ("aclmdlGetDesc",         [c_void_p, c_uint32], c_int),
            ("aclmdlGetNumInputs",    [c_void_p], c_size_t),
            ("aclmdlGetNumOutputs",   [c_void_p], c_size_t),
            ("aclmdlGetInputSizeByIndex",  [c_void_p, c_size_t], c_size_t),
            ("aclmdlGetOutputSizeByIndex", [c_void_p, c_size_t], c_size_t),
            ("aclmdlCreateDataset",   [], c_void_p),
            ("aclmdlDestroyDataset",  [c_void_p], None),
            ("aclmdlAddDatasetBuffer",[c_void_p, c_void_p], c_int),
            ("aclmdlExecute",         [c_uint32, c_void_p, c_void_p], c_int),
            ("aclCreateDataBuffer",   [c_void_p, c_size_t], c_void_p),
        ]
        for name, argtypes, restype in sigs:
            f = getattr(L, name)
            f.argtypes = argtypes
            f.restype = restype

    def exec(self, name, inputs):
        """Run .om model. Returns list of output device pointers."""
        if name not in self._oc:
            p = f"{MODEL_DIR}/{name}.om"
            mid = c_uint32(0)
            self.L.aclmdlLoadFromFile(p.encode(), byref(mid))
            d = self.L.aclmdlCreateDesc()
            self.L.aclmdlGetDesc(d, mid.value)
            self._oc[name] = {"id": mid.value, "d": d}
        om = self._oc[name]
        d = om["d"]
        in_ds = self.L.aclmdlCreateDataset()
        out_ds = self.L.aclmdlCreateDataset()
        for ptr, sz in inputs:
            buf = self.L.aclCreateDataBuffer(c_void_p(ptr), sz)
            self.L.aclmdlAddDatasetBuffer(in_ds, buf)
        num_out = self.L.aclmdlGetNumOutputs(d)
        bufs = []
        for i in range(num_out):
            sz = self.L.aclmdlGetOutputSizeByIndex(d, i)
            p = c_void_p(0)
            self.L.aclrtMalloc(byref(p), sz, 1)
            bufs.append(p.value)
            buf = self.L.aclCreateDataBuffer(c_void_p(p.value), sz)
            self.L.aclmdlAddDatasetBuffer(out_ds, buf)
        r = self.L.aclmdlExecute(om["id"], in_ds, out_ds)
        self.L.aclmdlDestroyDataset(in_ds)
        self.L.aclmdlDestroyDataset(out_ds)
        if r:
            raise RuntimeError(f"{name}: {r}")
        return bufs

    def malloc(self, sz):
        p = c_void_p(0)
        self.L.aclrtMalloc(byref(p), sz, 1)
        return p.value

    def free(self, p):
        if p:
            self.L.aclrtFree(c_void_p(p))

    def h2d(self, d, h):
        self.L.aclrtMemcpy(c_void_p(d), h.nbytes,
                           h.ctypes.data_as(c_void_p), h.nbytes, 1)

    def d2h(self, h, d, sz=0):
        n = sz or h.nbytes
        self.L.aclrtMemcpy(h.ctypes.data_as(c_void_p), n,
                           c_void_p(d), n, 2)

    def memset(self, p, sz):
        self.L.aclrtMemset(c_void_p(p), sz, 0, sz)

    def d2d(self, dst, src, sz):
        """Device-to-device copy (within same device or across devices)."""
        self.L.aclrtMemcpy(c_void_p(dst), sz, c_void_p(src), sz, 3)


# ── Weight loader helper ───────────────────────────────────────────
def load_layer(wl, c, i):
    """Load one layer's weights to chip c. Returns weight dict or None."""
    lw = wl.get_layer_weights(i)
    w = {}
    for k, v in lw.items():
        d = v
        if "down_proj" in k:
            d = v.T.astype(np.float16).copy()
        p = c.malloc(d.nbytes)
        if p:
            c.h2d(p, d)
            w[k] = p
    return w if len(w) >= 10 else None


# ── Layer runner (from v11, self-contained) ────────────────────────
def run_layer(c, h, w, lt, i, kv_cache):
    """Run one layer on chip c. h is local device pointer (on c's device)."""
    def g(k):
        return w.get(k, w.get(f".{k}"))

    hn = c.exec("ops_rmsnorm", [(h, H*2), (g("input_layernorm.weight"), H*2)])[0]
    is_full = lt[i] == "full_attention"

    if is_full:
        ow = g("self_attn.o_proj.weight")
        qw = g("self_attn.q_proj.weight")
        kw = g("self_attn.k_proj.weight")
        vw = g("self_attn.v_proj.weight")
        if all([qw, kw, vw, ow]):
            qp = c.exec("mm_1_4096_4096", [(hn, H*2), (qw, H*H*2)])[0]
            kn = c.exec("mm_1_4096_1024", [(hn, H*2), (kw, H*1024*2)])
            vn = c.exec("mm_1_4096_1024", [(hn, H*2), (vw, H*1024*2)])
            qc = np.empty(4096, dtype=np.float16); c.d2h(qc, qp)
            kc = np.empty(1024, dtype=np.float16); c.d2h(kc, kn[0])
            vc = np.empty(1024, dtype=np.float16); c.d2h(vc, vn[0])
            c.free(qp); c.free(kn[0]); c.free(vn[0])

            # KV Cache append
            kv_cache[i].append((kc.copy(), vc.copy()))

            q = qc.reshape(NH, HD).astype(np.float32)
            T = len(kv_cache[i])
            k_all = np.array([kv_cache[i][t][0] for t in range(T)]
                             ).reshape(T, NKV, HD).astype(np.float32)
            v_all = np.array([kv_cache[i][t][1] for t in range(T)]
                             ).reshape(T, NKV, HD).astype(np.float32)
            # GQA expand: NKV → NH
            k_all = k_all.repeat(NH // NKV, axis=1)  # [T, 16, 256]
            v_all = v_all.repeat(NH // NKV, axis=1)
            k2d = k_all.reshape(-1, HD)
            v2d = v_all.reshape(-1, HD)

            scores = q @ k2d.T
            scale = HD ** -0.5
            s = np.exp(scores * scale - np.max(scores * scale, -1, keepdims=True))
            a = s / s.sum(-1, keepdims=True)
            o = (a.astype(np.float32) @ v2d.astype(np.float32)
                 ).ravel().astype(np.float16)

            on = c.malloc(H * 2)
            c.d2d(on, o.ctypes.data, H * 2)  # uses D2D from host DMA
            op = c.exec("mm_1_4096_4096", [(on, H*2), (ow, H*H*2)])[0]
            c.free(on)
        else:
            op = hn
    else:
        ow = g("linear_attn.out_proj.weight") or g("linear_attn.in_proj_z.weight")
        if ow:
            op = c.exec("mm_1_4096_4096", [(hn, H*2), (ow, H*H*2)])[0]
        else:
            op = hn

    if op is not hn:
        c.free(hn)
    r = c.exec("ops_add", [(h, H*2), (op, H*2)])[0]
    c.d2d(h, r, H*2); c.free(r)
    if op is not hn:
        c.free(op)

    # MLP
    pn = g("post_attention_layernorm.weight")
    gp = g("mlp.gate_proj.weight")
    up = g("mlp.up_proj.weight")
    dp = g("mlp.down_proj.weight")
    if all([pn, gp, up, dp]):
        hn2 = c.exec("ops_rmsnorm", [(h, H*2), (pn, H*2)])[0]
        gg = c.exec("mm_1_4096_12288", [(hn2, H*2), (gp, H*IM*2)])
        uu = c.exec("mm_1_4096_12288", [(hn2, H*2), (up, H*IM*2)])
        c.free(hn2)
        sg = c.exec("ops_silu", [(gg[0], IM*2)])
        gu = c.exec("ops_mul", [(sg[0], IM*2), (uu[0], IM*2)])
        c.free(gg[0]); c.free(uu[0]); c.free(sg[0])
        dd = c.exec("mm_1_6144_4096", [(gu[0], 6144*2), (dp, 6144*4096*2)])
        dd2 = c.exec("mm_1_6144_4096", [(gu[0]+6144*2, 6144*2),
                                        (dp+6144*4096*2, 6144*4096*2)])
        c.free(gu[0])
        ds = c.exec("ops_add", [(dd[0], H*2), (dd2[0], H*2)])[0]
        c.free(dd[0]); c.free(dd2[0])
        r2 = c.exec("ops_add", [(h, H*2), (ds, H*2)])[0]
        c.d2d(h, r2, H*2); c.free(ds); c.free(r2)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== v12 PIPELINE PARALLELISM (4-chip, 4-slot) ===")
    t0 = time.time()

    # 1. Initialize chips
    chips = [Chip(i) for i in range(NUM_CHIPS)]
    from engine.weights import WeightLoader
    wl = WeightLoader(WEIGHT_PATH)
    wl.load_all()
    lt = json.load(open(f"{WEIGHT_PATH}/config.json")).get("text_config",
                                                           {}).get("layer_types", [])
    # 2. Pre-load weights: 8 layers per chip
    wc = [None] * NUM_LAYERS
    for i in range(NUM_LAYERS):
        ci = i // LAYERS_PER_CHIP
        chips[ci].L.aclrtSetDevice(ci)
        w = load_layer(wl, chips[ci], i)
        if w:
            wc[i] = w
        else:
            print(f"  WARN: Layer {i} weights not loaded!")
    print(f"  Init: {time.time() - t0:.0f}s")

    # 3. Allocate per-chip per-slot hidden state buffers
    #    buf[ci][slot] — each on the respective chip's device
    buf = [[None] * NUM_SLOTS for _ in range(NUM_CHIPS)]
    for ci in range(NUM_CHIPS):
        chips[ci].L.aclrtSetDevice(ci)
        for si in range(NUM_SLOTS):
            buf[ci][si] = chips[ci].malloc(H * 2)
            chips[ci].memset(buf[ci][si], H * 2)  # zero init

    # 4. Per-slot KV caches (each slot has its own sequence)
    kv_slots = [[[] for _ in range(NUM_LAYERS)] for _ in range(NUM_SLOTS)]

    # ── Sequential baseline (4 tokens, one at a time) ───────────────
    print("\n=== Sequential (4 tokens × v11-style forward) ===")
    chips[0].L.aclrtSetDevice(0)
    h_master = chips[0].malloc(H * 2)  # master hidden state on chip 0
    # Allocate shadow buffers on each chip for D2D copies
    ch_shadow = [None] * NUM_CHIPS
    for c in chips:
        c.L.aclrtSetDevice(c.dev)
        ch_shadow[c.dev] = c.malloc(H * 2)
        c.memset(ch_shadow[c.dev], H * 2)

    def forward_sequential(h_master, token_kv):
        """Single-token forward through all 32 layers (v11 pattern)."""
        for i in range(NUM_LAYERS):
            ci = i // LAYERS_PER_CHIP
            c = chips[ci]
            c.L.aclrtSetDevice(ci)
            if ci > 0:
                # D2D copy: master (chip 0) → shadow (this chip)
                chips[0].d2d(ch_shadow[ci], h_master, H * 2)
                h = ch_shadow[ci]
            else:
                h = h_master
            if wc[i]:
                run_layer(c, h, wc[i], lt, i, token_kv)
            if ci > 0:
                # D2D copy back: shadow → master
                chips[ci].d2d(h_master, ch_shadow[ci], H * 2)

    t_seq = time.time()
    seq_kv = [[] for _ in range(NUM_LAYERS)]
    for _ in range(4):
        chips[0].memset(h_master, H * 2)
        forward_sequential(h_master, seq_kv)
    t_seq = time.time() - t_seq
    print(f"  4 tokens: {t_seq:.1f}s = {t_seq/4:.2f}s/token")
    print(f"  Throughput: {4/t_seq:.2f} tok/s")

    # Free sequential buffers
    chips[0].free(h_master)
    for c in chips:
        c.free(ch_shadow[c.dev])

    # ── Pipeline (4 tokens, 4-stage) ───────────────────────────────
    print("\n=== Pipeline (4 threads × 8 layers) ===")

    in_q = queue.Queue()        # main → stage 0
    stage_qs = [queue.Queue() for _ in range(NUM_CHIPS - 1)]  # stage0→1, stage1→2, stage2→3
    out_q = queue.Queue()       # stage 3 → main

    # Queue routing:
    #   Main → in_q → Stage 0 → stage_qs[0] → Stage 1 → stage_qs[1]
    #       → Stage 2 → stage_qs[2] → Stage 3 → out_q → Main

    pipeline_done = threading.Event()

    def stage_worker(ci, q_in, q_out):
        """Pipeline stage: processes 8 layers on chip ci."""
        c = chips[ci]
        c.L.aclrtSetDevice(ci)  # set device for THIS thread
        base = ci * LAYERS_PER_CHIP
        stage_kv = [[] for _ in range(NUM_LAYERS)]  # KV for layers on this chip only

        while True:
            slot = q_in.get()
            if slot is None:  # sentinel: propagate and exit
                q_out.put(None)
                break

            h_local = buf[ci][slot]

            # Process the 8 layers that belong to this chip
            for i in range(base, base + LAYERS_PER_CHIP):
                if wc[i]:
                    run_layer(c, h_local, wc[i], lt, i, kv_slots[slot])

            # Push result to next stage (D2D copy to next chip's buffer)
            if ci < NUM_CHIPS - 1:
                # Copy from our buffer to next chip's buffer for this slot
                chips[ci].d2d(buf[ci + 1][slot], h_local, H * 2)

            # Signal next stage
            q_out.put(slot)

    # Start stage threads
    threads = []
    # Stage 0: reads in_q, writes stage_qs[0]
    t = threading.Thread(target=stage_worker,
                         args=(0, in_q, stage_qs[0]), daemon=True)
    t.start(); threads.append(t)
    # Stage 1: reads stage_qs[0], writes stage_qs[1]
    t = threading.Thread(target=stage_worker,
                         args=(1, stage_qs[0], stage_qs[1]), daemon=True)
    t.start(); threads.append(t)
    # Stage 2: reads stage_qs[1], writes stage_qs[2]
    t = threading.Thread(target=stage_worker,
                         args=(2, stage_qs[1], stage_qs[2]), daemon=True)
    t.start(); threads.append(t)
    # Stage 3: reads stage_qs[2], writes out_q
    t = threading.Thread(target=stage_worker,
                         args=(3, stage_qs[2], out_q), daemon=True)
    t.start(); threads.append(t)

    # Give threads a moment to initialize (load .om models etc.)
    time.sleep(0.5)

    # Feed 4 tokens into the pipeline
    t_pipe = time.time()
    for slot in range(NUM_SLOTS):
        # Zero the hidden state on chip 0 for this slot
        chips[0].L.aclrtSetDevice(0)
        chips[0].memset(buf[0][slot], H * 2)
        in_q.put(slot)

    # Collect results (they come out in order from stage 3)
    results_pipe = []
    for _ in range(NUM_SLOTS):
        try:
            slot = out_q.get(timeout=60)
            results_pipe.append(slot)
        except queue.Empty:
            print(f"  TIMEOUT waiting for pipeline result (slot {_})")
    t_pipe = time.time() - t_pipe

    # Shutdown pipeline (send sentinel to each stage)
    in_q.put(None)
    for t in threads:
        t.join(timeout=3)

    print(f"  Pipeline: {t_pipe:.2f}s for {len(results_pipe)} tokens")
    if results_pipe:
        print(f"  Per-token (effective): {t_pipe/len(results_pipe):.3f}s")
        print(f"  Throughput: {len(results_pipe)/t_pipe:.2f} tok/s")
        in_order = all(results_pipe[i] == i for i in range(len(results_pipe)))
        print(f"  In-order: {'yes' if in_order else 'NO (check queue routing)'}")

    print(f"\n── Comparison ──")
    print(f"  Sequential 4 tok:  {t_seq:.2f}s  ({4/t_seq:.2f} tok/s)")
    print(f"  Pipeline   4 tok:  {t_pipe:.2f}s  ({4/t_pipe:.2f} tok/s)")
    print(f"  Speedup: {t_seq/t_pipe:.2f}x")
    print(f"  (Theoretical max: {NUM_CHIPS}x in steady state)")
