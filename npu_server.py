"""
Qwythos-9B NPU Server — Full OpenAI-Compatible API.
Uses v11 4-chip engine with cached weights (2.2s/token).
CPU embedding + final RMSNorm + LM head.
Supports streaming, chat completions, 8K context.
"""
import os, sys, time, json, threading, logging, asyncio, numpy as np

sys.path.insert(0, "/root/qwythos_engine")
from engine.qwythos_npu_v11 import Chip, load_layer, run_layer, H
from engine.weights import WeightLoader

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])
log = logging.getLogger("qwythos")

WEIGHT_PATH = "/root/models/Qwythos-9B-Claude-Mythos-5-1M"
H_BYTES = H * 2
VS = 248320
MAX_CONTEXT = 8192
EOS_TOKENS = {248044, 248046}

# ═══════════════════════════════════════════════════════════════════
# CACHED WEIGHT LOADER (fixes 507011 by setting device before h2d)
# ═══════════════════════════════════════════════════════════════════
def load_layer_cached(wl, c, i):
    """Load one layer's weights to chip c. Sets device before each h2d/malloc."""
    c.L.aclrtSetDevice(c.dev)  # ⚠ MUST set device first!
    lw = wl.get_layer_weights(i)
    w = {}
    for k, v in lw.items():
        d = v
        if "down_proj" in k:
            d = v.T.astype(np.float16).copy()
        p = c.malloc(d.nbytes)
        if p:
            c.L.aclrtSetDevice(c.dev)  # ⚠ device for H2D
            c.h2d(p, d)
            w[k] = p
    return w if len(w) >= 10 else None


# ═══════════════════════════════════════════════════════════════════
# NPU ENGINE
# ═══════════════════════════════════════════════════════════════════
class NPUEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._init()
                    cls._instance = inst
        return cls._instance

    def _init(self):
        log.info("Initializing NPU engine...")
        t0 = time.time()

        cfg = json.load(open(f"{WEIGHT_PATH}/config.json"))
        tc = cfg.get("text_config", cfg)
        self.layer_types = tc.get("layer_types", [])
        log.info(f"  Layers: {len(self.layer_types)} ({tc.get('num_hidden_layers')} hidden)")

        # Load CPU weights
        wl = WeightLoader(WEIGHT_PATH)
        wl.load_all()
        cw = wl.load_all()
        self.embed = cw.get("model.embed_tokens.weight", cw.get("embed_tokens.weight"))
        self.lm_head = cw.get("lm_head.weight")
        self.norm_weight = cw.get("model.norm.weight")
        log.info(f"  embed: {self.embed.shape}  lm_head: {self.lm_head.shape}")

        # Init 4 chips and pre-load weights with CORRECT device context
        self.chips = [Chip(i) for i in range(4)]
        self.wc = [None] * 32
        for i in range(32):
            ci = i // 8
            # load_layer_cached calls setDevice internally
            w = load_layer_cached(wl, self.chips[ci], i)
            if w:
                self.wc[i] = w
            else:
                log.warning(f"  Layer {i} weights not loaded!")

        # Allocate hidden state on each chip
        for c in self.chips:
            c.L.aclrtSetDevice(c.dev)
            c.h = c.malloc(H_BYTES)

        # Load tokenizer
        import transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            WEIGHT_PATH, trust_remote_code=True)
        log.info(f"  Init complete ({time.time()-t0:.0f}s)")

    def forward_32(self, h_cpu, kv_cache):
        """Run 32 NPU layers. Sets device context before every ACL call."""
        chips = self.chips

        # Upload initial state to chip 0
        chips[0].L.aclrtSetDevice(0)
        chips[0].h2d(chips[0].h, h_cpu)

        for i in range(32):
            ci = i // 8; c = chips[ci]
            c.L.aclrtSetDevice(ci)
            # D2D copy from chip 0 to this chip if needed
            if ci > 0:
                chips[0].L.aclrtMemcpy(c.h, H_BYTES, chips[0].h, H_BYTES, 3)
            # Use cached weights
            if self.wc[i]:
                run_layer(c, c.h, self.wc[i], self.layer_types, i, kv_cache)
            # D2D copy back to chip 0
            if ci > 0:
                chips[ci].L.aclrtMemcpy(chips[0].h, H_BYTES, c.h, H_BYTES, 3)

        # Download result
        h_out = np.empty(H, dtype=np.float16)
        chips[0].L.aclrtSetDevice(0)
        chips[0].d2h(h_out, chips[0].h)
        return h_out


# ═══════════════════════════════════════════════════════════════════
# SAMPLER
# ═══════════════════════════════════════════════════════════════════
class Sampler:
    def __init__(self, engine):
        self.embed = engine.embed
        self.lm_head = engine.lm_head
        self.norm_weight = engine.norm_weight
        self.tokenizer = engine.tokenizer
        self.vs = self.lm_head.shape[0]

    def embed_token(self, token_id):
        return self.embed[token_id].astype(np.float16)

    def apply_final_norm(self, h):
        """Final RMSNorm after 32 layers, before LM head."""
        h32 = h.astype(np.float32)
        rms = np.sqrt(np.mean(h32 ** 2) + 1e-6)
        return ((h32 / rms) * self.norm_weight).astype(np.float16)

    def logits(self, h):
        return h.astype(np.float32) @ self.lm_head.T.astype(np.float32)

    def sample(self, logits, temperature=0.6, top_p=0.9, top_k=50):
        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature
        else:
            logits = logits.copy()

        # Top-K filtering
        if top_k > 0 and top_k < self.vs:
            kth = np.partition(logits, -top_k)[-top_k]
            logits[logits < kth] = -np.inf

        # Top-P (nucleus) filtering
        if top_p < 1.0 and top_p > 0:
            sidx = np.argsort(logits)[::-1]
            sl = logits[sidx]
            max_l = np.max(sl[np.isfinite(sl)])
            if np.isfinite(max_l):
                cs = np.cumsum(np.exp(sl - max_l))
                cs = cs / cs[-1]
                sl[cs > top_p] = -np.inf
                logits[sidx] = sl

        # Softmax with numerical guards
        finite = logits[np.isfinite(logits)]
        if len(finite) == 0:
            return int(np.random.randint(0, self.vs))

        max_l = np.max(finite)
        exp_l = np.exp((logits - max_l).clip(-100, 100))
        probs = exp_l / np.sum(exp_l)

        if not np.all(np.isfinite(probs)) or np.sum(probs) <= 0:
            return int(np.random.randint(0, self.vs))

        return int(np.random.choice(self.vs, p=probs))

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def format_chat(self, messages):
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            log.warning(f"Chat template fallback: {e}")
            out = ""
            for m in messages:
                role = m.get("role", "user"); c = m.get("content", "")
                if isinstance(c, list):
                    c = " ".join(x.get("text","") for x in c if x.get("type")=="text")
                out += f"<|im_start|>{role}\n{c}<|im_end|>\n"
            return out + "<|im_start|>assistant\n"


# ═══════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════
def generate(engine, sampler, input_ids, max_new_tokens=256,
             temperature=0.6, top_p=0.9, top_k=50, token_queue=None):
    """Autoregressive generation. Pushes tokens to queue for streaming."""
    kv_cache = [[] for _ in range(32)]
    generated = []
    t0 = time.time()

    # Prefill: process all prompt tokens
    for tid in input_ids:
        h = sampler.embed_token(tid)
        engine.forward_32(h, kv_cache)

    # Decode loop
    last_id = input_ids[-1]
    for step in range(max_new_tokens):
        h = sampler.embed_token(last_id)
        engine.forward_32(h, kv_cache)

        h = sampler.apply_final_norm(h)
        ll = sampler.logits(h)

        if np.any(np.isnan(ll)):
            log.warning(f"NaN logits at step {step}")
            tid = int(np.random.randint(0, sampler.vs))
        else:
            tid = sampler.sample(ll, temperature, top_p, top_k)

        generated.append(tid)
        if token_queue is not None:
            token_queue.put_nowait(("token", sampler.decode([tid]), tid))

        last_id = tid
        if tid in EOS_TOKENS:
            break

    elapsed = time.time() - t0
    output_text = sampler.decode(generated)

    result = {
        "text": output_text, "tokens": generated,
        "prompt_tokens": len(input_ids),
        "completion_tokens": len(generated),
        "total_tokens": len(input_ids) + len(generated),
        "time_s": elapsed,
        "tokens_per_sec": len(generated) / elapsed if elapsed > 0 else 0,
    }
    if token_queue is not None:
        token_queue.put_nowait(("done", result, None))
    return result


# ═══════════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════════
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

API_KEY = os.environ.get("QWYTHOS_API_KEY", "wsh101007")


class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatReq(BaseModel):
    model: str = "qwythos-9b"
    messages: List[Message]
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 256
    stream: bool = False
    seed: Optional[int] = None


class CompReq(BaseModel):
    model: str = "qwythos-9b"
    prompt: str
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 256
    stream: bool = False
    seed: Optional[int] = None


app = FastAPI(title="Qwythos-9B Ascend 310 API", version="2.0.0")


async def verify_auth(req: Request):
    auth = req.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(401, "Invalid API key")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwythos-9b",
            "hardware": "4× Ascend 310 NPU", "version": "2.0.0",
            "context_limit": MAX_CONTEXT}


@app.get("/v1/models")
async def list_models(req: Request):
    await verify_auth(req)
    return {"object": "list", "data": [
        {"id": "qwythos-9b", "object": "model",
         "created": int(time.time()), "owned_by": "empero-ai"}]}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatReq, req: Request):
    await verify_auth(req)
    return _handle(body, is_chat=True)


@app.post("/v1/completions")
async def completions(body: CompReq, req: Request):
    await verify_auth(req)
    return _handle(body, is_chat=False)


def _handle(body, is_chat):
    np.random.seed(body.seed if body.seed is not None
                   else int(time.time() * 1000) & 0xFFFFFFFF)
    engine = NPUEngine()
    sampler = Sampler(engine)

    if is_chat:
        msgs = [m.model_dump() for m in body.messages]
        prompt = sampler.format_chat(msgs)
    else:
        prompt = body.prompt

    input_ids = sampler.tokenizer.encode(prompt, truncation=True, max_length=MAX_CONTEXT)
    log.info(f"Generate: {len(input_ids)} input tokens, {body.max_tokens} max new, "
             f"temp={body.temperature}")

    if body.stream:
        return _stream_generate(engine, sampler, input_ids, body, is_chat)

    result = generate(engine, sampler, input_ids,
                      body.max_tokens, body.temperature, body.top_p, body.top_k)
    log.info(f"Done: {result['completion_tokens']} tokens in {result['time_s']:.1f}s")

    if is_chat:
        choice = {"index": 0,
                  "message": {"role": "assistant", "content": result["text"]},
                  "finish_reason": "stop"}
    else:
        choice = {"index": 0, "text": result["text"], "finish_reason": "stop"}

    return JSONResponse(content={
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion" if is_chat else "text_completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [choice],
        "usage": {"prompt_tokens": result["prompt_tokens"],
                  "completion_tokens": result["completion_tokens"],
                  "total_tokens": result["total_tokens"]},
    })


def _stream_generate(engine, sampler, input_ids, body, is_chat):
    token_queue = asyncio.Queue()

    def run_in_thread():
        try:
            generate(engine, sampler, input_ids,
                     body.max_tokens, body.temperature, body.top_p, body.top_k,
                     token_queue=token_queue)
        except Exception as e:
            log.error(f"Generate error: {e}")
            token_queue.put_nowait(("error", str(e), None))

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    async def event_stream():
        if is_chat:
            yield f"data: {json.dumps({'choices':[{'delta':{'role':'assistant'},'index':0}]})}\n\n"
        while True:
            msg = await token_queue.get()
            kind, data, tid = msg
            if kind == "token":
                yield f"data: {json.dumps({'choices':[{'delta':{'content':data},'index':0}]})}\n\n"
            elif kind == "done":
                yield "data: [DONE]\n\n"
                break
            elif kind == "error":
                yield f"data: {json.dumps({'error':data})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("=" * 50)
    log.info("Qwythos-9B Ascend 310 NPU Server v2.0")
    log.info("=" * 50)

    engine = NPUEngine()
    sampler = Sampler(engine)
    log.info(f"Server ready — {sampler.tokenizer.vocab_size} vocab, {MAX_CONTEXT} context")

    port = int(os.environ.get("PORT", 8000))
    log.info(f"Listening on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)
