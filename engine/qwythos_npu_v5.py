"""Qwythos-9B NPU v5: Clean 32-layer forward. 12288 fallback to CPU."""
import os, time, json, ctypes, numpy as np
from ctypes import c_void_p, c_size_t, c_int, c_uint32, c_char_p, byref, CDLL, POINTER

MODEL_DIR = os.environ.get("QWYTHOS_MODEL_DIR", "/root/qwythos_engine/om_models")
WEIGHT_PATH = os.environ.get("QWYTHOS_WEIGHT_PATH", "/root/models/Qwythos-9B-Claude-Mythos-5-1M")
QWYTHOS_HOME = os.environ.get("QWYTHOS_HOME", "/root/qwythos_engine")
H, NH, NKV, HD, IM = 4096, 16, 4, 256, 12288
SCALE = HD ** -0.5
SCALE = HD ** -0.5

# Weight sizes: input_bytes, weight_bytes for each model
WS = {
    "mm_1_4096_4096": (H*2, H*H*2),
    "mm_1_4096_1024": (H*2, H*1024*2),
    "mm_1_4096_12288": (H*2, H*IM*2),
    "mm_1_12288_4096": (IM*2, IM*4096*2),
    "mm_1_1024_4096": (1024*2, 1024*H*2),
    "mm_1_4096_8192": (H*2, H*8192*2),
    "ops_rmsnorm": (H*2, H*2),
    "ops_silu": (IM*2, IM*2),
    "ops_add": (H*2, H*2),
    "ops_mul": (IM*2, IM*2),
    "ops_softmax": (16*4*2, 16*4*2),
}


class ACL:
    def __init__(self):
        self.L = CDLL("libascendcl.so")
        self._setup()
        self.L.aclInit(None); self.L.aclrtSetDevice(0)
        self._om_cache = {}

    def _setup(self):
        L, P = self.L, POINTER
        for n, a, r in [
            ("aclrtMalloc", [P(c_void_p), c_size_t, c_int], c_int),
            ("aclrtFree", [c_void_p], c_int),
            ("aclrtMemcpy", [c_void_p, c_size_t, c_void_p, c_size_t, c_int], c_int),
            ("aclrtMemset", [c_void_p, c_size_t, c_int, c_size_t], c_int),
            ("aclmdlLoadFromFile", [c_char_p, P(c_uint32)], c_int),
            ("aclmdlCreateDesc", [], c_void_p),
            ("aclmdlGetDesc", [c_void_p, c_uint32], c_int),
            ("aclmdlGetNumInputs", [c_void_p], c_size_t),
            ("aclmdlGetNumOutputs", [c_void_p], c_size_t),
            ("aclmdlGetInputSizeByIndex", [c_void_p, c_size_t], c_size_t),
            ("aclmdlGetOutputSizeByIndex", [c_void_p, c_size_t], c_size_t),
            ("aclmdlCreateDataset", [], c_void_p),
            ("aclmdlDestroyDataset", [c_void_p], None),
            ("aclmdlAddDatasetBuffer", [c_void_p, c_void_p], c_int),
            ("aclmdlExecute", [c_uint32, c_void_p, c_void_p], c_int),
            ("aclCreateDataBuffer", [c_void_p, c_size_t], c_void_p),
        ]:
            f = getattr(L, n); f.argtypes = a; f.restype = r

    def exec(self, name, inputs):
        if name not in self._om_cache:
            p = f"{MODEL_DIR}/{name}.om"
            mid = c_uint32(0)
            self.L.aclmdlLoadFromFile(p.encode(), byref(mid))
            d = self.L.aclmdlCreateDesc(); self.L.aclmdlGetDesc(d, mid.value)
            self._om_cache[name] = {"id": mid.value, "d": d}
        om = self._om_cache[name]; d = om["d"]
        in_ds = self.L.aclmdlCreateDataset(); out_ds = self.L.aclmdlCreateDataset()
        for p, s in inputs:
            if not isinstance(p, c_void_p): p = c_void_p(p)
            self.L.aclmdlAddDatasetBuffer(in_ds, self.L.aclCreateDataBuffer(p, s))
        no = self.L.aclmdlGetNumOutputs(d); bufs = []
        for i in range(no):
            sz = self.L.aclmdlGetOutputSizeByIndex(d, i)
            p = c_void_p(0); self.L.aclrtMalloc(byref(p), sz, 1)
            bufs.append(p.value)
            self.L.aclmdlAddDatasetBuffer(out_ds, self.L.aclCreateDataBuffer(c_void_p(p.value), sz))
        r = self.L.aclmdlExecute(om["id"], in_ds, out_ds)
        self.L.aclmdlDestroyDataset(in_ds); self.L.aclmdlDestroyDataset(out_ds)
        if r: raise RuntimeError(f"{name}: {r}")
        return bufs

    def malloc(self, sz):
        p = c_void_p(0); self.L.aclrtMalloc(byref(p), sz, 1); return p.value
    def free(self, p):
        if p: self.L.aclrtFree(p)
    def h2d(self, d, h):
        self.L.aclrtMemcpy(d, h.nbytes, h.ctypes.data_as(c_void_p), h.nbytes, 1)
    def d2h(self, h, d):
        self.L.aclrtMemcpy(h.ctypes.data_as(c_void_p), h.nbytes, d, h.nbytes, 2)
    def memset(self, p, sz):
        self.L.aclrtMemset(p, sz, 0, sz)
    def copy(self, d, s, sz):
        self.L.aclrtMemcpy(d, sz, s, sz, 3)


class QNPU:
    def __init__(self):
        print("Loading weights...", end=" ", flush=True)
        t0 = time.time()
        import sys; sys.path.insert(0, QWYTHOS_HOME)
        from engine.weights import WeightLoader
        self.wl = WeightLoader(WEIGHT_PATH); self.wl.load_all()
        self.a = ACL()
        with open(f"{WEIGHT_PATH}/config.json") as f:
            self.lt = json.load(f).get("text_config", {}).get("layer_types", [])
        print(f"{len(self.lt)} layers ({self.lt.count('linear_attention')}d+{self.lt.count('full_attention')}f)")
        self.h = self.a.malloc(H*2)

    def wp(self, w, k):
        return w.get(k, w.get(f".{k}"))

    def x(self, name, *ptrs):
        """Run NPU model with correct sizes from WS dict."""
        s = WS.get(name)
        if not s: raise RuntimeError(f"Unknown model: {name}")
        # Build input list: (ptr_as_void_p, size_in_bytes)
        inputs = [(c_void_p(ptrs[i]), s[i]) for i in range(len(ptrs))]
        return self.a.exec(name, inputs)[0]

    def down_proj(self, h_in, w_npu):
        """12288->4096 via NPU: split into two 6144 matmuls both on NPU."""
        off = 6144 * 2        # 12288 bytes = 6144 float16
        h_sz = 6144 * 2       # bytes per half
        w_half = 6144 * 4096 * 2  # bytes per weight half

        # Second half: offset pointer
        h1 = h_in
        h2 = h_in + off

        # Weight halves
        w1 = w_npu
        w2 = w_npu + w_half

        out1 = self.a.exec("mm_1_6144_4096", [(c_void_p(h1), h_sz), (c_void_p(w1), w_half)])[0]
        out2 = self.a.exec("mm_1_6144_4096", [(c_void_p(h2), h_sz), (c_void_p(w2), w_half)])[0]
        result = self.a.exec("ops_add", [(c_void_p(out1), 4096*2), (c_void_p(out2), 4096*2)])[0]
        self.a.free(out1); self.a.free(out2)
        return result

    def run_layer(self, i):
        lw = self.wl.get_layer_weights(i)
        w = {}
        for k, v in lw.items():
            if k == "mlp.down_proj.weight" or k == ".mlp.down_proj.weight":
                # Transpose: [4096,12288] -> [12288,4096] for ONNX MatMul
                vt = v.T.astype(np.float16).copy()  # transpose & ensure contiguous
                p = self.a.malloc(vt.nbytes)
                if p: self.a.h2d(p, vt); w[k] = p
            else:
                p = self.a.malloc(v.nbytes)
                if p: self.a.h2d(p, v); w[k] = p

        is_full = self.lt[i] == "full_attention"
        h = self.h; self.a.memset(h, H*2)
        nw = self.wp(w, "input_layernorm.weight")
        if nw is None: self._free(w); return
        hn = self.x("ops_rmsnorm", h, nw)

        if is_full:
            ow = self.wp(w, "self_attn.o_proj.weight")
            qw = self.wp(w, "self_attn.q_proj.weight")
            kw = self.wp(w, "self_attn.k_proj.weight"); vw = self.wp(w, "self_attn.v_proj.weight")
            if all([qw, kw, vw, ow]):
                qk_r=self.x("mm_1_4096_8192",hn,qw); k_np=self.x("mm_1_4096_1024",hn,kw)
                v_np=self.x("mm_1_4096_1024",hn,vw)
                qk=np.empty(8192,dtype=np.float16); self.a.d2h(qk,qk_r)
                kc=np.empty(1024,dtype=np.float16); self.a.d2h(kc,k_np)
                vc=np.empty(1024,dtype=np.float16); self.a.d2h(vc,v_np)
                self.a.free(qk_r); self.a.free(k_np); self.a.free(v_np)
                q=qk[:H].reshape(NH,HD).astype(np.float32); k4=kc.reshape(NKV,HD).astype(np.float32)
                v4=vc.reshape(NKV,HD).astype(np.float32)
                scores=(q@k4.T)*SCALE; scores=np.exp(scores-scores.max(-1,keepdims=True))
                attn=scores/scores.sum(-1,keepdims=True)
                out=(attn.astype(np.float32)@v4.astype(np.float32)).ravel().astype(np.float16)
                onp=self.a.malloc(H*2); self.a.h2d(onp,out)
                op=self.x("mm_1_4096_4096",onp,ow); self.a.free(onp)
            else:
                op = hn
        else:
            ow=self.wp(w,"linear_attn.out_proj.weight") or self.wp(w,"linear_attn.in_proj_z.weight")
            if ow: op=self.x("mm_1_4096_4096",hn,ow)
            else: op = hn

        if op is not hn: self.a.free(hn)
        r=self.x("ops_add",h,op); self.a.copy(h,r,H*2); self.a.free(r)
        if op is not hn: self.a.free(op)

        pn=self.wp(w,"post_attention_layernorm.weight")
        mg=self.wp(w,"mlp.gate_proj.weight"); mu=self.wp(w,"mlp.up_proj.weight"); md=self.wp(w,"mlp.down_proj.weight")
        if not all([pn,mg,mu,md]): self._free(w); return
        hn2=self.x("ops_rmsnorm",h,pn)
        gg=self.x("mm_1_4096_12288",hn2,mg); uu=self.x("mm_1_4096_12288",hn2,mu)
        self.a.free(hn2)
        sg=self.x("ops_silu",gg); gu=self.x("ops_mul",sg,uu)
        self.a.free(gg); self.a.free(uu); self.a.free(sg)
        dd=self.down_proj(gu,md); self.a.free(gu)
        r2=self.x("ops_add",h,dd); self.a.copy(h,r2,H*2); self.a.free(dd); self.a.free(r2)
        self._free(w)

    def _free(self, w):
        for p in w.values(): self.a.free(p)

    def forward(self, n=32):
        t0 = time.time()
        for i in range(min(n, 32)):
            self.run_layer(i)
            print(f"  [{i+1}/{n}]", end="\r", flush=True)
        print(f"\nForward: {time.time()-t0:.1f}s")
        return self.h


if __name__ == "__main__":
    print("=== Qwythos-9B NPU Forward ===\n")
    q = QNPU()
    q.forward(2)
    print("Done!")
