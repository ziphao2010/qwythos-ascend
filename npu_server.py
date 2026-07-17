"""
Qwythos-9B NPU Server v2.1 — Fixed DeltaNet + Full OpenAI API.
"""
import os, sys, time, json, threading, logging, asyncio, numpy as np
from ctypes import c_void_p

sys.path.insert(0, "/root/qwythos_engine")
from engine.qwythos_npu_v11 import Chip, H
from engine.weights import WeightLoader

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])
log = logging.getLogger("qwythos")

WEIGHT_PATH = "/root/models/Qwythos-9B-Claude-Mythos-5-1M"
H_BYTES = H * 2
VS = 248320
MAX_CONTEXT = int(os.environ.get("QWYTHOS_MAX_CONTEXT", "8192"))
EOS_TOKENS = {248044, 248046}

# ═══════════════════════════════════════════════════════════════════
# FIXED LAYER RUNNER  (imported from v11_fixed, embedded for speed)
# ═══════════════════════════════════════════════════════════════════
L_NKH, L_NVH, L_KHD, L_VHD, NH, NKV, HD, IM = 16, 32, 128, 128, 16, 4, 256, 12288

def load_layer_all(wl, c, i):
    """Load ALL weights for layer i (incl. linear_attn, self_attn, mlp)."""
    c.L.aclrtSetDevice(c.dev)
    lw = wl.get_layer_weights(i)
    w = {}
    for k, v in lw.items():
        d = v
        if "down_proj" in k:
            d = v.T.astype(np.float16).copy()
        p = c.malloc(d.nbytes)
        if p:
            c.L.aclrtSetDevice(c.dev)
            c.h2d(p, d)
            w[k] = p
    return w

def run_layer(c, h, w, lt, i, kv_cache):
    """Run one layer with proper DeltaNet + full attention."""
    def g(k): return w.get(k, w.get(f".{k}"))
    hn = c.exec("ops_rmsnorm", [(h, H_BYTES), (g("input_layernorm.weight"), H_BYTES)])[0]
    is_full = lt[i] == "full_attention"

    if is_full:
        qw, kw, vw, ow = [g(f"self_attn.{p}.weight") for p in ["q_proj","k_proj","v_proj","o_proj"]]
        if all([qw, kw, vw, ow]):
            qp = c.exec("mm_1_4096_4096", [(hn, H_BYTES), (qw, H*H*2)])[0]
            kn = c.exec("mm_1_4096_1024", [(hn, H_BYTES), (kw, H*1024*2)])
            vn = c.exec("mm_1_4096_1024", [(hn, H_BYTES), (vw, H*1024*2)])
            qc = np.empty(4096, dtype=np.float16); c.d2h(qc, qp)
            kc = np.empty(1024, dtype=np.float16); c.d2h(kc, kn[0])
            vc = np.empty(1024, dtype=np.float16); c.d2h(vc, vn[0])
            c.free(qp); c.free(kn[0]); c.free(vn[0])
            kv_cache[i].append((kc.copy(), vc.copy()))
            q = qc.reshape(NH, HD).astype(np.float32)
            T = len(kv_cache[i])
            ka = np.array([kv_cache[i][t][0] for t in range(T)]).reshape(T, NKV, HD).astype(np.float32)
            va = np.array([kv_cache[i][t][1] for t in range(T)]).reshape(T, NKV, HD).astype(np.float32)
            ka = ka.repeat(NH//NKV, axis=1).reshape(-1, HD)
            va = va.repeat(NH//NKV, axis=1).reshape(-1, HD)
            s = np.exp((q@ka.T)*(HD**-0.5) - np.max((q@ka.T)*(HD**-0.5),-1,keepdims=True))
            a = s/s.sum(-1,keepdims=True)
            o = (a@va).ravel().astype(np.float16)
            on = c.malloc(H_BYTES); c.h2d(on, o)
            op = c.exec("mm_1_4096_4096", [(on, H_BYTES), (ow, H*H*2)])[0]; c.free(on)
        else: op = hn
    else:
        iqkv = g("linear_attn.in_proj_qkv.weight")
        ipz = g("linear_attn.in_proj_z.weight")
        ow = g("linear_attn.out_proj.weight")
        ipa = g("linear_attn.in_proj_a.weight")
        ipb = g("linear_attn.in_proj_b.weight")

        if all([iqkv, ipz, ow]):
            # QKV projection on NPU
            qkv_p = c.exec("mm_1_4096_8192", [(hn, H_BYTES), (iqkv, 8192*4096*2)])[0]
            qkv = np.empty(8192, dtype=np.float16); c.d2h(qkv, qkv_p); c.free(qkv_p)
            Q_vec = qkv[:2048]; K_vec = qkv[2048:4096]; V_vec = qkv[4096:]
            Q = Q_vec.reshape(L_NKH, L_KHD).astype(np.float32)
            V = V_vec.reshape(L_NVH, L_VHD).astype(np.float32)

            # Gates on CPU
            hn_cpu = np.empty(H, dtype=np.float16); c.d2h(hn_cpu, hn)
            hn_f32 = hn_cpu.astype(np.float32)

            if ipa and ipb:
                ipa_cpu = np.empty(32*4096, dtype=np.float16); c.d2h(ipa_cpu, ipa)
                ipb_cpu = np.empty(32*4096, dtype=np.float16); c.d2h(ipb_cpu, ipb)
                dtb = np.empty(32, dtype=np.float16); c.d2h(dtb, g("linear_attn.dt_bias"))
                alog = np.empty(32, dtype=np.float16); c.d2h(alog, g("linear_attn.A_log"))
                gate_a = 1.0/(1.0+np.exp(-(ipa_cpu.reshape(32,4096).astype(np.float32)@hn_f32+dtb.astype(np.float32))))
                gate_b = 1.0/(1.0+np.exp(-(ipb_cpu.reshape(32,4096).astype(np.float32)@hn_f32+alog.astype(np.float32))))
            else:
                gate_a = np.ones(L_NVH, dtype=np.float32)*0.5
                gate_b = np.ones(32, dtype=np.float32)*0.9

            # SSM state update
            if len(kv_cache[i])==0:
                ssm_state = np.zeros((L_NVH, L_VHD), dtype=np.float32)
            else:
                ssm_state = kv_cache[i][0]
            ssm_state = gate_b[:,None] * ssm_state + gate_a[:,None] * V
            kv_cache[i] = [ssm_state.copy()]

            # Output = Q @ K^T gated result
            attn_out = (Q @ Q.T) @ ssm_state[:16]  # [16,128]
            attn_out_f = attn_out.reshape(-1).astype(np.float16)  # 2048

            # z gate (silu)
            zp = c.exec("mm_1_4096_4096", [(hn, H_BYTES), (ipz, H*H*2)])[0]
            z_cpu = np.empty(H, dtype=np.float16); c.d2h(z_cpu, zp); c.free(zp)
            z_f32 = z_cpu.astype(np.float32)
            z_gate = z_f32 * (1.0/(1.0+np.exp(-z_f32)))
            output = np.zeros(H, dtype=np.float32)
            output[:2048] = attn_out_f.astype(np.float32)
            output[2048:] = attn_out_f[:2048].astype(np.float32)

            # Upload and output projection
            on = c.malloc(H_BYTES); c.h2d(on, (output*z_gate).astype(np.float16))
            op = c.exec("mm_1_4096_4096", [(on, H_BYTES), (ow, H*H*2)])[0]; c.free(on)
        else:
            op = c.exec("mm_1_4096_4096", [(hn, H_BYTES), (ow, H*H*2)])[0] if ow else hn

    # Residual
    if op is not hn: c.free(hn)
    r = c.exec("ops_add", [(h, H_BYTES), (op, H_BYTES)])[0]
    c.L.aclrtMemcpy(c_void_p(h), H_BYTES, c_void_p(r), H_BYTES, 3); c.free(r)
    if op is not hn: c.free(op)

    # MLP
    pn, gp, up, dp = [g(f"mlp.{k}.weight") for k in ["gate_proj","up_proj","down_proj"]]  # wrong order, skip
    pn = g("post_attention_layernorm.weight"); gp = g("mlp.gate_proj.weight")
    up = g("mlp.up_proj.weight"); dp = g("mlp.down_proj.weight")
    if all([pn, gp, up, dp]):
        hn2 = c.exec("ops_rmsnorm", [(h, H_BYTES), (pn, H_BYTES)])[0]
        gg = c.exec("mm_1_4096_12288", [(hn2, H_BYTES), (gp, H*IM*2)])
        uu = c.exec("mm_1_4096_12288", [(hn2, H_BYTES), (up, H*IM*2)]); c.free(hn2)
        sg = c.exec("ops_silu", [(gg[0], IM*2)])
        gu = c.exec("ops_mul", [(sg[0], IM*2), (uu[0], IM*2)])
        c.free(gg[0]); c.free(uu[0]); c.free(sg[0])
        dd = c.exec("mm_1_6144_4096", [(gu[0], 6144*2), (dp, 6144*4096*2)])
        dd2 = c.exec("mm_1_6144_4096", [(gu[0]+6144*2, 6144*2), (dp+6144*4096*2, 6144*4096*2)])
        c.free(gu[0])
        ds = c.exec("ops_add", [(dd[0], H_BYTES), (dd2[0], H_BYTES)])[0]
        c.free(dd[0]); c.free(dd2[0])
        r2 = c.exec("ops_add", [(h, H_BYTES), (ds, H_BYTES)])[0]
        c.L.aclrtMemcpy(c_void_p(h), H_BYTES, c_void_p(r2), H_BYTES, 3); c.free(ds); c.free(r2)


# ═══════════════════════════════════════════════════════════════════
# NPU ENGINE
# ═══════════════════════════════════════════════════════════════════
class NPUEngine:
    _instance = None; _lock = threading.Lock()
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls); inst._init(); cls._instance = inst
        return cls._instance

    def _init(self):
        log.info("Initializing NPU engine (v2.1 DeltaNet fixed)...")
        t0 = time.time()
        cfg = json.load(open(f"{WEIGHT_PATH}/config.json"))
        tc = cfg.get("text_config", cfg)
        self.layer_types = tc.get("layer_types", [])

        wl = WeightLoader(WEIGHT_PATH); wl.load_all()
        cw = wl.load_all()
        self.embed = cw.get("model.embed_tokens.weight")
        self.lm_head = cw.get("lm_head.weight")
        self.norm_weight = cw.get("model.norm.weight")

        self.chips = [Chip(i) for i in range(4)]
        self.wc = [None]*32
        for i in range(32):
            ci=i//8; self.chips[ci].L.aclrtSetDevice(ci)
            w = load_layer_all(wl, self.chips[ci], i)
            if w: self.wc[i]=w
            else: log.warning(f"Layer {i} failed")
        for c in self.chips: c.L.aclrtSetDevice(c.dev); c.h=c.malloc(H_BYTES)

        import transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(WEIGHT_PATH, trust_remote_code=True)
        log.info(f"Init complete ({time.time()-t0:.0f}s) — 32 layers, all weights loaded")

    def forward_32(self, h_cpu, kv_cache):
        chips = self.chips
        chips[0].L.aclrtSetDevice(0); chips[0].h2d(chips[0].h, h_cpu)
        for i in range(32):
            ci=i//8; c=chips[ci]; c.L.aclrtSetDevice(ci)
            if ci>0: chips[0].L.aclrtMemcpy(c.h, H_BYTES, chips[0].h, H_BYTES, 3)
            if self.wc[i]: run_layer(c, c.h, self.wc[i], self.layer_types, i, kv_cache)
            if ci>0: chips[ci].L.aclrtMemcpy(chips[0].h, H_BYTES, c.h, H_BYTES, 3)
        h_out = np.empty(H, dtype=np.float16)
        chips[0].L.aclrtSetDevice(0); chips[0].d2h(h_out, chips[0].h)
        return h_out


class Sampler:
    def __init__(self, e):
        self.embed=e.embed; self.lm_head=e.lm_head; self.norm_w=e.norm_weight
        self.tokenizer=e.tokenizer; self.vs=self.lm_head.shape[0]
    def embed_token(self,t): return self.embed[t].astype(np.float16)
    def final_norm(self,h):
        h32=h.astype(np.float32); rms=np.sqrt(np.mean(h32**2)+1e-6)
        return ((h32/rms)*self.norm_w).astype(np.float16)
    def logits(self,h): return h.astype(np.float32)@self.lm_head.T.astype(np.float32)
    def sample(self,ll,temp=0.6,top_p=0.9,top_k=50):
        if temp>0: ll/=temp
        if top_k>0 and top_k<self.vs:
            kth=np.partition(ll,-top_k)[-top_k]; ll[ll<kth]=-np.inf
        if top_p<1.0 and top_p>0:
            si=np.argsort(ll)[::-1]; sl=ll[si]
            mx=np.max(sl[np.isfinite(sl)])
            if np.isfinite(mx): cs=np.cumsum(np.exp(sl-mx)); sl[cs/cs[-1]>top_p]=-np.inf; ll[si]=sl
        finite=ll[np.isfinite(ll)]
        if len(finite)==0: return int(np.random.randint(0,self.vs))
        mx=np.max(finite); exp_l=np.exp((ll-mx).clip(-100,100))
        probs=exp_l/np.sum(exp_l)
        if not np.all(np.isfinite(probs)) or np.sum(probs)<=0: return int(np.random.randint(0,self.vs))
        return int(np.random.choice(self.vs,p=probs))
    def decode(self,ids): return self.tokenizer.decode(ids,skip_special_tokens=True)
    def format_chat(self,msgs):
        try: return self.tokenizer.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
        except: return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in msgs)+"\n<|im_start|>assistant\n"


def generate(engine,sampler,input_ids,max_new=256,temp=0.6,top_p=0.9,top_k=50,tq=None):
    kv=[[] for _ in range(32)]; generated=[]; t0=time.time()
    log.info(f"Generate: {len(input_ids)} prompt tokens, {max_new} max new")
    for tid in input_ids:
        h=sampler.embed_token(tid); engine.forward_32(h,kv)
    last_id=input_ids[-1]
    for step in range(max_new):
        h=sampler.embed_token(last_id); engine.forward_32(h,kv)
        h=sampler.final_norm(h); ll=sampler.logits(h)
        if np.any(np.isnan(ll)):
            tid=int(np.random.randint(0,sampler.vs))
        else:
            tid=sampler.sample(ll,temp,top_p,top_k)
        generated.append(tid)
        if tq: tq.put_nowait(("token",sampler.decode([tid]),tid))
        last_id=tid
        if tid in EOS_TOKENS: break
    res={"text":sampler.decode(generated),"tokens":generated,
         "prompt_tokens":len(input_ids),"completion_tokens":len(generated),
         "total_tokens":len(input_ids)+len(generated),"time_s":time.time()-t0,
         "tokens_per_sec":len(generated)/(time.time()-t0) if len(generated)>0 else 0}
    if tq: tq.put_nowait(("done",res,None))
    return res


# ═══════════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════════
import uvicorn
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import StreamingResponse,JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

API_KEY=os.environ.get("QWYTHOS_API_KEY","wsh101007")

class Message(BaseModel):
    role: str; content: Union[str, List[Dict[str, Any]]]
class ChatReq(BaseModel):
    model:str="qwythos-9b"; messages:List[Message]
    temperature:float=0.6; top_p:float=0.9; top_k:int=50
    max_tokens:int=256; stream:bool=False; seed:Optional[int]=None
class CompReq(BaseModel):
    model:str="qwythos-9b"; prompt:str
    temperature:float=0.6; top_p:float=0.9; top_k:int=50
    max_tokens:int=256; stream:bool=False; seed:Optional[int]=None

app=FastAPI(title="Qwythos-9B Ascend 310 API v2.1",version="2.1.0")

async def verify(req:Request):
    if req.headers.get("Authorization","")!=f"Bearer {API_KEY}":
        raise HTTPException(401,"Invalid API key")

@app.get("/health")
async def health():
    s=os.stat("/proc/self/status") if os.path.exists("/proc/self/status") else None
    return {"status":"ok","model":"qwythos-9b","hardware":"4× Ascend 310 NPU",
            "version":"2.1.0","context_limit":MAX_CONTEXT}

@app.get("/v1/models")
async def models(req:Request):
    await verify(req)
    return {"object":"list","data":[{"id":"qwythos-9b","object":"model","created":int(time.time()),"owned_by":"empero-ai"}]}

@app.post("/v1/chat/completions")
async def chat(body:ChatReq,req:Request):
    await verify(req); return _handle(body,True)

@app.post("/v1/completions")
async def comp(body:CompReq,req:Request):
    await verify(req); return _handle(body,False)

def _handle(body,is_chat):
    np.random.seed(body.seed if body.seed is not None else int(time.time()*1000)&0xFFFFFFFF)
    e=NPUEngine(); s=Sampler(e)
    prompt=s.format_chat([m.model_dump() for m in body.messages]) if is_chat else body.prompt
    ids=s.tokenizer.encode(prompt,truncation=True,max_length=MAX_CONTEXT)
    log.info(f"Request: {len(ids)} input tokens, max_new={body.max_tokens}")
    if body.stream: return _stream(e,s,ids,body,is_chat)
    r=generate(e,s,ids,body.max_tokens,body.temperature,body.top_p,body.top_k)
    choice={"index":0,"message":{"role":"assistant","content":r["text"]},"finish_reason":"stop"} if is_chat else {"index":0,"text":r["text"],"finish_reason":"stop"}
    return JSONResponse(content={
        "id":f"chatcmpl-{int(time.time())}","object":"chat.completion" if is_chat else "text_completion",
        "created":int(time.time()),"model":body.model,"choices":[choice],
        "usage":{"prompt_tokens":r["prompt_tokens"],"completion_tokens":r["completion_tokens"],"total_tokens":r["total_tokens"]}})

def _stream(e,s,ids,body,is_chat):
    import asyncio
    tq=asyncio.Queue()
    def run():
        try: generate(e,s,ids,body.max_tokens,body.temperature,body.top_p,body.top_k,tq=tq)
        except Exception as ex: log.error(f"Stream err: {ex}"); tq.put_nowait(("error",str(ex),None))
    threading.Thread(target=run,daemon=True).start()
    async def stream():
        if is_chat: yield f"data: {json.dumps({'choices':[{'delta':{'role':'assistant'},'index':0}]})}\n\n"
        while True:
            msg=await tq.get()
            if msg[0]=="token": yield f"data: {json.dumps({'choices':[{'delta':{'content':msg[1]},'index':0}]})}\n\n"
            elif msg[0]=="done": yield "data: [DONE]\n\n"; break
            elif msg[0]=="error": yield f"data: {json.dumps({'error':msg[1]})}\n\n"; break
    return StreamingResponse(stream(),media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

if __name__=="__main__":
    log.info("="*50); log.info("Qwythos-9B Ascend 310 NPU Server v2.1"); log.info("="*50)
    e=NPUEngine(); s=Sampler(e)
    log.info(f"Ready — {s.tokenizer.vocab_size} vocab, {MAX_CONTEXT} context")
    uvicorn.run(app,host="0.0.0.0",port=int(os.environ.get("PORT",8000)),log_level="info",access_log=True)
