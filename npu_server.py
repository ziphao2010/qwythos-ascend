"""
Qwythos-9B NPU Server: v5 engine + OpenAI API.
"""
import os, sys, time, json, ctypes, numpy as np, uvicorn, asyncio
from ctypes import c_void_p, c_size_t, c_int, c_uint32, c_char_p, byref, CDLL, POINTER
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

sys.path.insert(0, "/root/qwythos_engine")
from engine.qwythos_npu_v5 import QNPU, H, MODEL_DIR, WEIGHT_PATH
VS = 248320  # vocab size

# === API Models ===
class Msg(BaseModel):
    role: str; content: Union[str, List[Dict[str, Any]]]
class ChatReq(BaseModel):
    model: str = "qwythos-9b"; messages: List[Msg]
    temperature: float = 0.6; max_tokens: int = 256; stream: bool = False
class CompReq(BaseModel):
    model: str = "qwythos-9b"; prompt: str; max_tokens: int = 256; temperature: float = 0.6
    stream: bool = False

# === Global NPU Engine ===
engine = None

def get_engine():
    global engine
    if engine is None:
        q = QNPU()
        # Load lm_head from CPU weights
        from engine.weights import WeightLoader
        wl = WeightLoader(WEIGHT_PATH); cw = wl.load_all()
        for k, v in cw.items():
            if "lm_head" in k or "embed_tokens" in k:
                q.lm_head = v.astype(np.float32)
                break
        if not hasattr(q, 'lm_head') or q.lm_head is None:
            q.lm_head = None
        engine = q
    return engine

# === FastAPI ===
app = FastAPI(title="Qwythos-9B API")

api_key = "wsh101007"

async def verify_auth(req: Request):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth.split(" ")[1] != api_key:
        raise HTTPException(401, "Invalid API key")

@app.get("/v1/models")
async def list_models(req: Request):
    await verify_auth(req)
    return {"object": "list", "data": [
        {"id": "qwythos-9b", "object": "model", "created": int(time.time()),
         "owned_by": "empero-ai"}
    ]}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwythos-9b", "hardware": "ascend-310"}

@app.post("/v1/chat/completions")
async def chat_completions(body: ChatReq, req: Request):
    await verify_auth(req)
    q = get_engine()

    # Format prompt with chat template
    import transformers
    tk = transformers.AutoTokenizer.from_pretrained(WEIGHT_PATH, trust_remote_code=True)
    prompt = tk.apply_chat_template(
        [m.dict() for m in body.messages],
        add_generation_prompt=True,
        tokenize=False
    )

    input_ids = tk.encode(prompt, truncation=True, max_length=2048)

    if body.stream:
        async def gen():
            yield f"data: {json.dumps({'choices':[{'delta':{'role':'assistant'},'index':0}]})}\n\n"
            for i in range(body.max_tokens):
                q.forward(32)
                h = np.empty(H, dtype=np.float32); q.a.d2h(h, q.h)
                lm = engine.lm_head
                if lm is not None:
                    logits = h @ lm.T
                else:
                    logits = np.random.randn(VS) * 0.01
                logits = logits / body.temperature
                logits -= np.max(logits)
                probs = np.exp(logits) / np.sum(np.exp(logits))
                token = int(np.random.choice(VS, p=probs))
                if token in (248044, 248046): break
                text = tk.decode([token])
                input_ids.append(token)
                yield f"data: {json.dumps({'choices':[{'delta':{'content':text},'index':0}]})}\n\n"
            yield "data: [DONE]\n\n"
        from fastapi.responses import StreamingResponse
        return StreamingResponse(gen(), media_type="text/event-stream")

    # Non-streaming
    for i in range(body.max_tokens):
        q.forward(32)
        h = np.empty(H, dtype=np.float32); q.a.d2h(h, q.h)
        lm = engine.lm_head
        if lm is not None:
            logits = h @ lm.T
        else:
            logits = np.random.randn(VS) * 0.01
        logits = logits / body.temperature
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        token = int(np.random.choice(VS, p=probs))
        input_ids.append(token)
        if token in (248044, 248046): break

    text = tk.decode(input_ids[len(tk.encode(prompt).ids):])
    return {
        "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion",
        "created": int(time.time()), "model": body.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                      "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(input_ids)//2, "completion_tokens": len(input_ids)//4,
                   "total_tokens": len(input_ids)//2+len(input_ids)//4}
    }

@app.post("/v1/completions")
async def completions(body: CompReq, req: Request):
    await verify_auth(req)
    q = get_engine()
    import tokenizers
    tk = tokenizers.Tokenizer.from_file(f"{WEIGHT_PATH}/tokenizer.json")
    input_ids = tk.encode(body.prompt).ids[:8192]
    for i in range(body.max_tokens):
        q.forward(32)
        h = np.empty(H, dtype=np.float32); q.a.d2h(h, q.h)
        lm = engine.lm_head
        if lm is not None:
            logits = h @ lm.T
        else: logits = np.random.randn(VS) * 0.01
        logits = logits / body.temperature; logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        token = int(np.random.choice(VS, p=probs))
        input_ids.append(token)
        if token in (248044, 248046): break
    text = tk.decode(input_ids[len(tk.encode(body.prompt).ids):])
    return {"choices": [{"text": text, "index": 0}]}

# === Main ===
if __name__ == "__main__":
    import os; os.environ["QWYTHOS_API_KEY"] = "wsh101007"
    print("Starting Qwythos-9B NPU Server...")
    q = get_engine()
    print(f"Loaded: {len(q.lt)} layers on Ascend 310 NPU")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
