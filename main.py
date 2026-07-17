"""
Qwythos-9B Inference Server for Ascend 310.

Usage:
  python main.py                          # Start API server
  python main.py --cli "Hello"            # CLI inference
  python main.py --test                   # Smoke test

Environment:
  export SOC_VERSION=Ascend310
  source /etc/profile.d/ascend.sh
"""
import os
import sys
import argparse
import numpy as np

os.environ["QWYTHOS_API_KEY"] = "wsh101007"

from engine.model import QwythosModel
from api.server import create_app
import uvicorn


def load_model():
    """Load Qwythos-9B model weights."""
    model_path = os.environ.get(
        "QWYTHOS_MODEL_PATH",
        "/root/models/Qwythos-9B-Claude-Mythos-5-1M")
    print(f"[Main] Loading model from {model_path}")
    model = QwythosModel(model_path, num_chips=4)
    model.load_weights()
    print(f"[Main] Model loaded: {model.num_layers} layers, "
          f"{model.vocab_size} vocab")
    return model


def start_server(model, host="0.0.0.0", port=8000):
    """Start the OpenAI-compatible API server."""
    app = create_app(model)
    print(f"[Main] Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def run_cli(model, prompt):
    """Run single inference from CLI."""
    import tokenizers
    tk = tokenizers.Tokenizer.from_file(
        f"{model.model_path}/tokenizer.json")
    input_ids = np.array([tk.encode(prompt).ids], dtype=np.int64)
    print(f"[CLI] Input: {len(input_ids[0])} tokens")
    tokens = model.generate(input_ids, max_new_tokens=256)
    response = tk.decode(tokens)
    print(f"[CLI] Response: {response}")


def smoke_test(model):
    """Quick smoke test of the model pipeline."""
    print("[Test] Running smoke test...")
    input_ids = np.array([[1, 100, 200, 300, 400]], dtype=np.int64)
    try:
        tokens = model.generate(input_ids, max_new_tokens=10, stream=False)
        print(f"[Test] Generated {len(tokens)} tokens: {tokens[:5]}...")
        print("[Test] ✅ Smoke test passed!")
    except Exception as e:
        print(f"[Test] ❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwythos-9B Server")
    parser.add_argument("--cli", type=str, help="CLI prompt")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-server", action="store_true",
                       help="Don't start server after test")

    args = parser.parse_args()

    model = load_model()

    if args.test:
        smoke_test(model)

    if args.cli:
        run_cli(model, args.cli)

    if not args.no_server and not args.cli and not args.test:
        start_server(model, args.host, args.port)
    elif args.test and not args.no_server and not args.cli:
        start_server(model, args.host, args.port)
