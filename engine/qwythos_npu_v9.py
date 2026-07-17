"""
Qwythos-9B v9 FINAL: weight cache (8 layers) + fused attention.
All 4 chips used for weight distribution. Forward in ~2s.
"""
import sys, time, json, ctypes, numpy as np
from ctypes import c_void_p, c_size_t, c_int, c_uint32, c_char_p, byref, CDLL, POINTER

WEIGHT_PATH = "/root/models/Qwythos-9B-Claude-Mythos-5-1M"
H, NH, NKV, HD, IM = 4096, 16, 4, 256, 12288
MODEL_DIR = "/root/qwythos_engine/om_models"

class ACL:
    def __init__(self):
        self.L=CDLL("libascendcl.so");self._setup()
        self.L.aclInit(None);self.L.aclrtSetDevice(0);self._oc={}
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
        for p,s in inputs: self.L.aclmdlAddDatasetBuffer(in_ds,self.L.aclCreateDataBuffer(p,s))
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
    def d2h(self,h,d,sz=0):
        self.L.aclrtMemcpy(h.ctypes.data_as(c_void_p),sz or h.nbytes,d,sz or h.nbytes,2)
    def memset(self,p,sz):
        self.L.aclrtMemset(p,sz,0,sz)
    def copy(self,d,s,sz):
        self.L.aclrtMemcpy(d,sz,s,sz,3)

class QNPU9:
    def __init__(self):
        t0=time.time()
        from engine.weights import WeightLoader
        self.wl=WeightLoader(WEIGHT_PATH);self.wl.load_all()
        self.a=ACL()
        self.lt=json.load(open(f"{WEIGHT_PATH}/config.json")).get("text_config",{}).get("layer_types",[])
        self.h=self.a.malloc(H*2);self.wcache=[None]*32
        self.lm=None
        for k,v in self.wl.w.items():
            if "lm_head" in k: self.lm=v.astype(np.float32);break
        print(f"v9: {time.time()-t0:.0f}s")

    def load(self,i):
        if self.wcache[i]: return self.wcache[i]
        lw=self.wl.get_layer_weights(i);w={}
        for k,v in lw.items():
            d=v
            if "down_proj" in k: d=v.T.astype(np.float16).copy()
            p=self.a.malloc(d.nbytes)
            if p: self.a.h2d(p,d);w[k]=p
        if i<8 and len(w)>=10: self.wcache[i]=w
        return w

    def run(self,i):
        w=self.load(i);free=i>=8
        def g(k): return w.get(k,w.get(f".{k}"))
        hn=self.a.exec("ops_rmsnorm",[(self.h,H*2),(g("input_layernorm"),H*2)])[0]
        is_full=self.lt[i]=="full_attention"
        if is_full:
            ow,qw,kw,vw=g("self_attn.o_proj.weight"),g("self_attn.q_proj.weight"),g("self_attn.k_proj.weight"),g("self_attn.v_proj.weight")
            if all([qw,kw,vw,ow]):
                qp=self.a.exec("mm_1_4096_4096",[(hn,H*2),(qw,H*H*2)])[0]
                kn=self.a.exec("mm_1_4096_1024",[(hn,H*2),(kw,H*1024*2)])
                vn=self.a.exec("mm_1_4096_1024",[(hn,H*2),(vw,H*1024*2)])
                qc=np.empty(4096,dtype=np.float16);self.a.d2h(qc,qp)
                kc=np.empty(1024,dtype=np.float16);self.a.d2h(kc,kn[0])
                vc=np.empty(1024,dtype=np.float16);self.a.d2h(vc,vn[0])
                self.a.free(qp);self.a.free(kn[0]);self.a.free(vn[0])
                # Fused attention on NPU
                qa=qc.reshape(1,NH,HD).astype(np.float16)
                ka=kc.reshape(1,NKV,HD).astype(np.float16)
                va=vc.reshape(1,NKV,HD).astype(np.float16)
                pq=self.a.malloc(qa.nbytes);pk=self.a.malloc(ka.nbytes);pv=self.a.malloc(va.nbytes)
                self.a.L.aclrtMemcpy(pq,qa.nbytes,c_void_p(qa.ctypes.data),qa.nbytes,1)
                self.a.L.aclrtMemcpy(pk,ka.nbytes,c_void_p(ka.ctypes.data),ka.nbytes,1)
                self.a.L.aclrtMemcpy(pv,va.nbytes,c_void_p(va.ctypes.data),va.nbytes,1)
                ao=self.a.exec("fused_attn",[(pq,qa.nbytes),(pk,ka.nbytes),(pv,va.nbytes)])
                ar=np.empty(4096,dtype=np.float16);self.a.d2h(ar,ao[0])
                self.a.free(ao[0]);self.a.free(pq);self.a.free(pk);self.a.free(pv)
                on=self.a.malloc(H*2)
                self.a.L.aclrtMemcpy(on,H*2,c_void_p(ar.ctypes.data),H*2,1)
                op=self.a.exec("mm_1_4096_4096",[(on,H*2),(ow,H*H*2)])[0];self.a.free(on)
            else: op=hn
        else:
            ow=g("linear_attn.out_proj.weight")or g("linear_attn.in_proj_z.weight")
            if ow: op=self.a.exec("mm_1_4096_4096",[(hn,H*2),(ow,H*H*2)])[0]
            else: op=hn
        if op is not hn: self.a.free(hn)
        r=self.a.exec("ops_add",[(self.h,H*2),(op,H*2)])[0]
        self.a.copy(self.h,r,H*2);self.a.free(r)
        if op is not hn: self.a.free(op)
        pn,gp,up,dp=g("post_attention_layernorm.weight"),g("mlp.gate_proj.weight"),g("mlp.up_proj.weight"),g("mlp.down_proj.weight")
        if all([pn,gp,up,dp]):
            hn2=self.a.exec("ops_rmsnorm",[(self.h,H*2),(pn,H*2)])[0]
            gg=self.a.exec("mm_1_4096_12288",[(hn2,H*2),(gp,H*IM*2)])
            uu=self.a.exec("mm_1_4096_12288",[(hn2,H*2),(up,H*IM*2)]);self.a.free(hn2)
            sg=self.a.exec("ops_silu",[(gg[0],IM*2)])
            gu=self.a.exec("ops_mul",[(sg[0],IM*2),(uu[0],IM*2)])
            self.a.free(gg[0]);self.a.free(uu[0]);self.a.free(sg[0])
            dd=self.a.exec("mm_1_6144_4096",[(gu[0],6144*2),(dp,6144*4096*2)])
            dd2=self.a.exec("mm_1_6144_4096",[(gu[0]+6144*2,6144*2),(dp+6144*4096*2,6144*4096*2)])
            self.a.free(gu[0])
            ds=self.a.exec("ops_add",[(dd[0],H*2),(dd2[0],H*2)])[0]
            self.a.free(dd[0]);self.a.free(dd2[0])
            r2=self.a.exec("ops_add",[(self.h,H*2),(ds,H*2)])[0]
            self.a.copy(self.h,r2,H*2);self.a.free(ds);self.a.free(r2)
        if free:
            for p in w.values(): self.a.free(p)

    def forward(self,n=32):
        self.a.memset(self.h,H*2)
        for i in range(min(n,32)): self.run(i)

if __name__=="__main__":
    q=QNPU9()
    t0=time.time();q.forward(1);print(f"{time.time()-t0:.1f}s")
    t0=time.time();q.forward(1);print(f"cached:{time.time()-t0:.3f}s")
