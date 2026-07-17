"""Step ②: Fused attention ONNX model (all on NPU).
Q[1,16,256] @ K^T[1,256,4] → score[1,16,4]
→ Mul(SCALE=0.0625) → Softmax → @ V[1,4,256] → out[1,16,256]
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT16, [1, 16, 256])
K = helper.make_tensor_value_info("K", TensorProto.FLOAT16, [1, 4, 256])
V = helper.make_tensor_value_info("V", TensorProto.FLOAT16, [1, 4, 256])
Out = helper.make_tensor_value_info("Out", TensorProto.FLOAT16, [1, 16, 256])

# 0.0625 in fp16 = 0x2C00
scale_const = helper.make_tensor("scale", TensorProto.FLOAT16, [], [0x2C00])

nodes = [
    helper.make_node("Transpose", ["K"], ["KT"], name="transpose", perm=[0, 2, 1]),
    helper.make_node("MatMul", ["Q", "KT"], ["score"], name="score"),
    helper.make_node("Mul", ["score", "scale"], ["scaled"], name="scale_mul"),
    helper.make_node("Softmax", ["scaled"], ["attn"], name="softmax", axis=-1),
    helper.make_node("MatMul", ["attn", "V"], ["Out"], name="output"),
]

graph = helper.make_graph(nodes, "fused_attn", [Q, K, V], [Out],
                          initializer=[scale_const])
model = helper.make_model(graph, producer_name="qwythos",
                          opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 13

path = "/root/qwythos_engine/om_models/fused_attn.onnx"
onnx.save(model, path)
print(f"Fused attention model: {path}")

# Verify
m = onnx.load(path)
print(f"  Inputs: {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}")
print(f"  Outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}")
print(f"  Nodes: {[n.op_type for n in m.graph.node]}")
