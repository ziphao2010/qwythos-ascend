"""Quick fix: cache weights in init, reuse in forward."""
import sys, time, json, ctypes, numpy as np
from ctypes import c_void_p, byref, CDLL, c_uint32, c_size_t, c_int, c_char_p, POINTER

sys.path.insert(0,"/root/qwythos_engine")
Wp="/root/models/Qwythos-9B-Claude-Mythos-5-1M"
Md="/root/qwythos_engine/om_models"
H,NH,NKV,HD,IM=4096,16,4,256,12288

from engine.qwythos_npu_v11 import Chip, load_layer, run_layer
from engine.weights import WeightLoader

print("=== v11b Fixed ===")
t0=time.time()
chips=[Chip(i) for i in range(4)]
wl=WeightLoader(Wp);wl.load_all()
lt=json.load(open(f"{Wp}/config.json")).get("text_config",{}).get("layer_types",[])
kv=[[]for _ in range(32)]

# Init: pre-load weights ONCE, cache in wcache[]
wc=[None]*32
for i in range(32):
    ci=i//8;c=chips[ci];c.L.aclrtSetDevice(ci)
    w=load_layer(wl,c,i)
    if w: wc[i]=w
print(f"  Init: {time.time()-t0:.0f}s")

# Allocate hidden on each chip
for c in chips: c.L.aclrtSetDevice(c.dev);c.h=c.malloc(H*2)

# Forward: use cached weights, switch chips
t0=time.time()
for i in range(32):
    ci=i//8;c=chips[ci];c.L.aclrtSetDevice(ci)
    if ci>0: chips[0].L.aclrtMemcpy(c.h,H*2,chips[0].h,H*2,3)
    w=wc[i]
    if w: run_layer(c,c.h,w,lt,i,kv)
    if ci>0: chips[ci].L.aclrtMemcpy(chips[0].h,H*2,c.h,H*2,3)

print(f"  32 layers (4-chip cached+KV): {time.time()-t0:.1f}s")
print(f"  KV/cache: layers {sum(1 for k in kv if k)} have KV cache")
