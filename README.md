# Qwythos-Ascend

**Qwythos-9B 推理引擎 — 在华为 Ascend 310 NPU 上运行 9B 参数大模型**

> 完整 32 层混合注意力模型（24 Gated DeltaNet + 8 Full Attention）在 4 颗 Ascend 310 芯片上运行，支持多模态输入和 OpenAI 兼容 API。

## 🏆 核心特性

| 特性 | 状态 |
|------|:----:|
| **ATC + aclmdlExecute** — 算子级 NPU 执行 | ✅ 14 个算子模型编译运行 |
| **32 层纯 NPU 前向** — 全部计算在 NPU 上 | ✅ 19s/32层 |
| **混合注意力架构** — 24 DeltaNet + 8 Full Attention | ✅ 完全支持 |
| **GQA 注意力** — Grouped Query Attention | ✅ 16 Q-heads, 4 KV-heads |
| **SwiGLU MLP** — 门控激活 | ✅ NPU 算子 |
| **RMSNorm** — 层归一化 | ✅ NPU 算子 |
| **4-chip 权重分布** — 32GB 显存利用 | ✅ 748 权重预加载 |
| **OpenAI 兼容 API** — Chat + Completions | ✅ FastAPI |
| **TBE 自定义算子** — SSM State Update | ✅ tik DSL 编译 |
| **CPU 推理备选** — 180s/token fallback | ✅ NumPy 实现 |

## 🚀 快速开始

### 环境要求

- 华为 Atlas 300I 3010 (Ascend 310) × 4
- CANN 7.0.0 (Toolkit + NNAE + NNRT)
- 驱动 24.1.1.3
- Python 3.10+

### 安装

```bash
# 1. 配置环境变量
export QWYTHOS_HOME=/path/to/qwythos-ascend
export QWYTHOS_WEIGHT_PATH=/path/to/model/weights
export QWYTHOS_MODEL_DIR=/path/to/om_models
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/7.0.0/lib64:/usr/local/Ascend/driver/lib64

# 2. 安装依赖
pip install -r requirements.txt

# 3. 编译 ONNX 算子（可选，已有预编译 .om 模型）
python scripts/create_onnx.py

# 4. 启动服务器
python npu_server.py
```

### 调用 API

```bash
# 健康检查
curl http://your-server-ip:8000/health

# 模型列表
curl http://your-server-ip:8000/v1/models \
  -H "Authorization: Bearer your-api-key"

# 对话
curl -X POST http://your-server-ip:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwythos-9b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWYTHOS_HOME` | `/root/qwythos_engine` | 项目根目录 |
| `QWYTHOS_WEIGHT_PATH` | `/root/models/...` | 模型权重路径 |
| `QWYTHOS_MODEL_DIR` | `$QWYTHOS_HOME/om_models` | 编译后 .om 模型目录 |
| `QWYTHOS_API_KEY` | `your-api-key` | API 认证密钥 |
| `SOC_VERSION` | `Ascend310` | 芯片型号 |

## 🏗️ 项目结构

```
qwythos-ascend/
├── engine/                     ← NPU 推理引擎
│   ├── qwythos_npu_v5.py       ← 32层 NPU 前向（单芯片，已验证）
│   ├── qwythos_npu_v7.py       ← 4-chip 权重分布版
│   ├── weights.py              ← 760 权重加载器 (18.8GB)
│   ├── sampler.py              ← Top-K/Top-P 采样器
│   ├── model.py                ← CPU 推理参考实现
│   ├── attention.py            ← Full Attention + GQA
│   ├── delta_net.py            ← Gated DeltaNet
│   └── vision_encoder.py       ← 27层 ViT
├── om_models/                  ← ATC 编译的算子模型
│   ├── mm_1_4096_4096.om       ← MatMul (隐藏→隐藏)
│   ├── mm_1_4096_12288.om      ← MatMul (隐藏→MLP)
│   ├── mm_1_12288_4096.om      ← MatMul (MLP→隐藏)
│   ├── ops_rmsnorm.om          ← RMSNorm
│   ├── ops_silu.om             ← SiLU 激活
│   ├── ops_softmax.om          ← Softmax
│   ├── ops_add.om              ← 残差连接
│   └── ops_mul.om              ← 逐元素乘法
├── ops/                        ← TBE 自定义算子
│   ├── tik_matmul_fp32.py      ← Cube Unit 矩阵乘
│   ├── ssm_tbe.py              ← DeltaNet SSM 状态更新
│   ├── delta_net_ssm.py        ← SSM NumPy 参考
│   ├── mrope_3d.py             ← 3D mRoPE
│   └── gelu_tanh.py            ← GELU_tanh 激活
├── csrc/                       ← C 运行时
│   └── libqwythos_npu.c        ← NPU 内存管理 (malloc/free/h2d/d2h)
├── npu_server.py               ← OpenAI 兼容 API 服务器
└── tests/                      ← 测试脚本
```

## ⚙️ 技术方案

### 推理管线

```
输入 → Tokenizer → Embed → 32× NPU Layer → LM Head → Sample → 输出
                            ↓
              全部通过 aclmdlExecute 在 NPU 上运行
              14 个 .om 算子链式执行
              数据在 NPU 显存中流转，不经过 CPU
```

### NPU 算子 (ATC 编译)

| 算子 | ONNX | 用途 |
|------|------|------|
| `mm_1_4096_4096` | MatMul [1,4096]×[4096,4096] | Q/O 投影 |
| `mm_1_4096_1024` | MatMul [1,4096]×[4096,1024] | K/V 投影 |
| `mm_1_4096_12288` | MatMul [1,4096]×[4096,12288] | MLP 门控/上投影 |
| `mm_1_4096_8192` | MatMul [1,4096]×[4096,8192] | QKV 组合投影 |
| `ops_rmsnorm` | Pow+ReduceMean+Sqrt+Div | RMS 层归一化 |
| `ops_silu` | Sigmoid+Mul | SiLU 激活 |
| `ops_softmax` | Softmax | 注意力权重 |

### 混合注意力 (Qwen3.5)

```
每 4 层中: 3 层 DeltaNet (线性注意力) + 1 层 Full Attention
Full Attention: Q @ K^T → softmax → @ V  (GQA: 16Q→4KV)
DeltaNet:      QKV → conv1d → gate → SSM state update
```

## 📊 性能

| 配置 | 每 Token | 备注 |
|------|:--------:|------|
| CPU NumPy | 186s | Python 慢速矩阵乘 |
| NPU 单芯片 (v5) | 19s | 每层权重上传 + 计算 |
| NPU 4-chip 预加载 | ~2s (目标) | 权重分布到 4 芯片 |

## 🔧 开发指南

### 编译新算子

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 创建 ONNX 模型
python3 create_onnx_matmul.py

# ATC 编译为 .om
atc --model=model.onnx --framework=5 \
    --output=model --soc_version=Ascend310 \
    --input_shape="A:1,4096;B:4096,4096"
```

### 编写 TBE 自定义算子

参考 `ops/tik_matmul_fp32.py`，使用 `te`/`tik` DSL 编写：
```python
from te import tik
tik_inst = tik.Tik()
# 定义 GM 张量
# 用 Vector Unit / Cube Unit
tik_inst.BuildCCE(kernel_name="my_kernel", inputs=..., outputs=...)
```

### 编译 C 运行时

```bash
gcc -shared -fPIC -o libqwythos_npu.so qwythos_npu_lib.c \
    -I/usr/local/Ascend/ascend-toolkit/latest/include \
    -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
    -lascendcl -Wl,-rpath,.../lib64
```

## 📝 License

Apache 2.0

## 🙏 致谢

- [Qwen3.5](https://github.com/QwenLM/Qwen) — 基础模型架构
- [Empero AI](https://empero.org) — Qwythos-9B 微调
- [Huawei Ascend](https://www.hiascend.com/) — CANN 工具链
