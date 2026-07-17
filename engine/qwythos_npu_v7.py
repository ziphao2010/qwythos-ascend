"""
Qwythos-9B v7: V5 working engine + 4-chip weight preloading.
Only changes: upload weights at init, use device switching in run_layer.
"""
import os, time, json, ctypes, numpy as np
from ctypes import c_void_p, c_size_t, c_int, c_uint32, c_char_p, byref, CDLL, POINTER

MODEL_DIR = os.environ.get("QWYTHOS_MODEL_DIR", "/root/qwythos_engine/om_models")
WEIGHT_PATH = os.environ.get("QWYTHOS_WEIGHT_PATH", "/root/models/Qwythos-9B-Claude-Mythos-5-1M")
QWYTHOS_HOME = os.environ.get("QWYTHOS_HOME", "/root/qwythos_engine")
H, NH, NKV, HD, IM, VS = 4096, 16, 4, 256, 12288, 248320
H, NH, NKV, HD, IM, VS = 4096, 16, 4, 256, 12288, 248320
SCALE = HD ** -0.5

WS = {"mm_1_4096_4096":(H*2,H*H*2),"mm_1_4096_1024":(H*2,H*1024*2),"mm_1_4096_12288":(H*2,H*IM*2),
      "mm_1_4096_8192":(H*2,H*8192*2),"mm_1_6144_4096":(6144*2,6144*4096*2),
      "ops_rmsnorm":(H*2,H*2),"ops_silu":(IM*2,IM*2),"ops_add":(H*2,H*2),"ops_mul":(IM*2,IM*2)}

class ACL:
    def __init__(self, dev=0):
        self.L=CDLL("libascendcl.so"); self._setup(); self.dev=dev
        if dev==0: self.L.aclInit(None)
        self.L.aclrtSetDevice(dev); self._oc={}
    def _setup(self):
        L,P=self.L,POINTER
        for n,a,r in [
            ("aclrtMalloc",[P(c_void_p),c_size_t,c_int],c_int),("aclrtFree",[c_void_p],c_int),
            ("aclrtMemcpy",[c_void_p,c_size_t,c_void_p,c_size_t,c_int],c_int),
            ("aclrtMemset",[c_void_p,c_size_t,c_int,c_size_t],c_int),
            ("aclmdlLoadFromFile",[c_char_p,P(c_uint32)],c_int),
            ("aclmdlCreateDesc",[],c_void_p),("aclmdlGetDesc",[c_void_p,c_uint32],c_int),
            ("aclmdlGetNumInputs",[c_void_p],c_size_t),("aclmdlGetNumOutputs",[c_void_p],c_size_t),
            ("aclmdlGetInputSizeByIndex",[c_void_p,c_size_t],c_size_t),
            ("aclmdlGetOutputSizeByIndex",[c_void_p,c_size_t],c_size_t),
            ("aclmdlCreateDataset",[],c_void_p),("aclmdlDestroyDataset",[c_void_p],None),
            ("aclmdlAddDatasetBuffer",[c_void_p,c_void_p],c_int),
            ("aclmdlExecute",[c_uint32,c_void_p,c_void_p],c_int),
            ("aclCreateDataBuffer",[c_void_p,c_size_t],c_void_p),
        ]:
            f=getattr(L,n);f.argtypes=a;f.restype=r
    def exec(self,name,inputs):
        if name not in self._oc:
            p=f"{MODEL_DIR}/{name}.om";mid=c_uint32(0)
            self.L.aclmdlLoadFromFile(p.encode(),byref(mid))
            d=self.L.aclmdlCreateDesc();self.L.aclmdlGetDesc(d,mid.value)
            self._oc[name]={"id":mid.value,"d":d}
        om=self._oc[name];d=om["d"]
        in_ds=self.L.aclmdlCreateDataset();out_ds=self.L.aclmdlCreateDataset()
        for p,s in inputs:
            self.L.aclmdlAddDatasetBuffer(in_ds,self.L.aclCreateDataBuffer(p,s))
        no=self.L.aclmdlGetNumOutputs(d);bufs=[]
        for i in range(no):
            sz=self.L.aclmdlGetOutputSizeByIndex(d,i)
            p=c_void_p(0);self.L.aclrtMalloc(byref(p),sz,1);bufs.append(p.value)
            self.L.aclmdlAddDatasetBuffer(out_ds,self.L.aclCreateDataBuffer(p.value,sz))
        r=self.L.aclmdlExecute(om["id"],in_ds,out_ds)
        self.L.aclmdlDestroyDataset(in_ds);self.L.aclmdlDestroyDataset(out_ds)
        if r: raise RuntimeError(f"{name}: {r}")
        return bufs
    def malloc(self,sz):
        p=c_void_p(0);self.L.aclrtMalloc(byref(p),sz,1);return p.value
    def free(self,p):
        if p: self.L.aclrtFree(p)
    def h2d(self,d,h):
        self.L.aclrtMemcpy(d,h.nbytes,h.ctypes.data_as(c_void_p),h.nbytes,1)
    def d2h(self,h,d):
        self.L.aclrtMemcpy(h.ctypes.data_as(c_void_p),h.nbytes,d,h.nbytes,2)
    def memset(self,p,sz):
        self.L.aclrtMemset(p,sz,0,sz)
    def copy(self,d,s,sz):
        self.L.aclrtMemcpy(d,sz,s,sz,3)

class QNPU7:
    def __init__(self):
        t0=time.time()
        print("Loading...",end=" ",flush=True)
        import sys;sys.path.insert(0,QWYTHOS_HOME)
        from engine.weights import WeightLoader
        wl=WeightLoader(WEIGHT_PATH);cw=wl.load_all()
        self.dev=[ACL(i) for i in range(4)]
        self.lt=json.load(open(f"{WEIGHT_PATH}/config.json")).get("text_config",{}).get("layer_types",[])
        # Pre-load weights: 8 layers per chip
        self.lw=[{} for _ in range(32)]
        n=0
        for k,v in cw.items():
            parts=k.split(".");li=-1
            for p in parts:
                if p.isdigit() and 0<=int(p)<32: li=int(p);break
            if li<0: continue
            ch = 0 if li < 6 else 1 if li < 14 else 2 if li < 22 else 3; ac=self.dev[ch];ac.L.aclrtSetDevice(ch)
            d=v
            if "down_proj.weight" in k: d=v.T.astype(np.float16).copy()
            p=ac.malloc(d.nbytes)
            if p: ac.h2d(p,d);self.lw[li][".".join(parts[parts.index(str(li))+1:])]=p;n+=1
        # lm_head CPU
        self.lm=None
        for k,v in cw.items():
            if "lm_head" in k: self.lm=v.astype(np.float32);break
        del cw
        # Hidden state on chip 0
        self.dev[0].L.aclrtSetDevice(0)
        self.h=self.dev[0].malloc(H*2)
        print(f"{n}w 4chips {time.time()-t0:.0f}s")

    def run_layer(self,i):
        lw=self.lw[i];ch=0 if i<6 else 1 if i<14 else 2 if i<22 else 3;ac=self.dev[ch]
        ac.L.aclrtSetDevice(ch)
        def wk(k): return lw.get(k,lw.get(f".{k}"))
        h=self.h
        nw=wk("input_layernorm.weight")
        if nw is None: return
        hn=ac.exec("ops_rmsnorm",[(h,H*2),(nw,H*2)])[0]
        is_full=self.lt[i]=="full_attention"
        if i < 3 or i == 3:
            pass  # debug point
        if is_full:
            ow,qw,kw,vw=wk("self_attn.o_proj.weight"),wk("self_attn.q_proj.weight"),wk("self_attn.k_proj.weight"),wk("self_attn.v_proj.weight")
            if all([qw,kw,vw,ow]):
                qkr=ac.exec("mm_1_4096_8192",[(hn,H*2),(qw,H*4*2)])
                knp=ac.exec("mm_1_4096_1024",[(hn,H*2),(kw,H*1024*2)])
                vnp=ac.exec("mm_1_4096_1024",[(hn,H*2),(vw,H*1024*2)])
                qk=np.empty(8192,dtype=np.float16);kc=np.empty(1024,dtype=np.float16);vc=np.empty(1024,dtype=np.float16)
                ac.d2h(qk,qkr[0]);ac.d2h(kc,knp[0]);ac.d2h(vc,vnp[0])
                ac.free(qkr[0]);ac.free(knp[0]);ac.free(vnp[0])
                q=qk[:H].reshape(NH,HD).astype(np.float32);k4=kc.reshape(NKV,HD).astype(np.float32);v4=vc.reshape(NKV,HD).astype(np.float32)
                s=np.exp((q@k4.T)*SCALE-np.max((q@k4.T)*SCALE,-1,keepdims=True));a=s/s.sum(-1,keepdims=True)
                o=(a.astype(np.float32)@v4.astype(np.float32)).ravel().astype(np.float16)
                on=ac.malloc(H*2);ac.h2d(on,o)
                op=ac.exec("mm_1_4096_4096",[(on,H*2),(ow,H*H*2)])[0];ac.free(on)
            else: op=hn
        else:
            ow=wk("linear_attn.out_proj.weight")or wk("linear_attn.in_proj_z.weight")
            if ow: op=ac.exec("mm_1_4096_4096",[(hn,H*2),(ow,H*H*2)])[0]
            else: op=hn
        if op is not hn: ac.free(hn)
        r=ac.exec("ops_add",[(h,H*2),(op,H*2)])[0]
        ac.copy(h,r,H*2);ac.free(r)
        if op is not hn: ac.free(op)
        pn,mg,mu,md=wk("post_attention_layernorm.weight"),wk("mlp.gate_proj.weight"),wk("mlp.up_proj.weight"),wk("mlp.down_proj.weight")
        if not all([pn,mg,mu,md]): return
        hn2=ac.exec("ops_rmsnorm",[(h,H*2),(pn,H*2)])[0]
        gg=ac.exec("mm_1_4096_12288",[(hn2,H*2),(mg,H*IM*2)])
        uu=ac.exec("mm_1_4096_12288",[(hn2,H*2),(mu,H*IM*2)])
        ac.free(hn2)
        sg=ac.exec("ops_silu",[(gg[0],IM*2)])
        gu=ac.exec("ops_mul",[(sg[0],IM*2),(uu[0],IM*2)])
        ac.free(gg[0]);ac.free(uu[0]);ac.free(sg[0])
        dd=ac.exec("mm_1_6144_4096",[(gu[0],6144*2),(md,6144*4096*2)])
        dd2=ac.exec("mm_1_6144_4096",[(gu[0]+6144*2,6144*2),(md+6144*4096*2,6144*4096*2)])
        ac.free(gu[0])
        ds=ac.exec("ops_add",[(dd[0],H*2),(dd2[0],H*2)])[0]
        ac.free(dd[0]);ac.free(dd2[0])
        r2=ac.exec("ops_add",[(h,H*2),(ds,H*2)])[0]
        ac.copy(h,r2,H*2);ac.free(ds);ac.free(r2)

    def forward(self):
        self.dev[0].memset(self.h,H*2)
        for i in range(32): self.run_layer(i)
        return self.h

if __name__=="__main__":
    print("=== Qwythos v7 4-chip ===")
    q=QNPU7()
    t0=time.time()
    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        q.run_layer(i)
        print(f"  layer {i}: {time.time()-t0:.1f}s", end="\r", flush=True)
    print(f"\nLayers 0-7: {time.time()-t0:.1f}s")
