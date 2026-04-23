


https://github.com/mlcommons/inference/tree/master/language/llama2-70b


# 1 代码总体结构（目录与角色）

`mlcommons/inference/language/llama2-70b/` 关键文件与职责：
- `README.md`：基准说明、运行方式（含 Server/Offline/Interactive 指标 TTFT/TPOT、LoadGen 回调要求等）。[GitHub](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md?utm_source=chatgpt.com)
- `main.py`：解析参数、选择场景、驱动 SUT（System Under Test）与 LoadGen。[GitHub](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/main.py?utm_source=chatgpt.com)
- `SUT_API.py`：**核心适配层**。封装 HuggingFace 模型与分词器、数据加载、与 LoadGen 的交互；定义 `FirstTokenStreamer` 并在合适时机调用 `lg.FirstTokenComplete`、`lg.QuerySamplesComplete`（注意需要上报生成 token 数）。[GitHub+1](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/SUT_API.py?utm_source=chatgpt.com)
- `dataset.py`：数据集抽象与样本读取。[GitHub](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/dataset.py?utm_source=chatgpt.com)
- `processorca.py`：评测数据生成/预处理（issues 中多次被提及）。[GitHub+1](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/processorca.py?utm_source=chatgpt.com)
- `evaluate-accuracy.py`、`consolidate_results.py`：准确率计算与汇总（有已知使用注意点/issue）。[GitHub+1](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/evaluate-accuracy.py?utm_source=chatgpt.com)
- `run_server.sh` / `run_offline.sh` / `run_accuracy.sh`：常用场景的一键脚本。[GitHub](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/run_server.sh?utm_source=chatgpt.com)


**LoadGen 交互要点（来自 README）**
- Server 场景：**每个 query 必须调用** `lg.FirstTokenComplete(resp)` 以统计 TTFT。
- 所有场景：`lg.QuerySamplesComplete([lg.QuerySampleResponse(id, addr, size, n_tokens)])` 中的 `n_tokens` **必须**与真实输出 token 数一致，否则 TEST06 会报错


# 2 执行逻辑与流程（从“接到请求”到“上报结果”）

**启动 → 准备阶段（`main.py`）**
- 解析场景（Server/Offline/Interactive）与后端（参考实现默认 PyTorch/HF）。
- 初始化 `SUT_API`（加载 `AutoModelForCausalLM`/`AutoTokenizer`、DataLoader、线程池等），注册 LoadGen 回调。[GitHub+1](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/main.py?utm_source=chatgpt.com)

**推理阶段（`SUT_API.py`）**
- 接收 LoadGen 派发的样本 id → 从 `Dataset` 取 prompt → 调用 `model.generate()`。
- `FirstTokenStreamer` 在**首 token**产生时向一个“holder”发信号，并触发 `lg.FirstTokenComplete`；完成后收集全部 token，计算 `n_tokens`，构造 `QuerySampleResponse` 并调用 `lg.QuerySamplesComplete`。[GitHub](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/SUT_API.py?utm_source=chatgpt.com)

**收尾与准确率**
- 根据 `evaluate-accuracy.py` 对输出进行打分；如需，调用 `consolidate_results.py` 汇总（该步曾被报告可选/需修修补补）。[GitHub+1](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/evaluate-accuracy.py?utm_source=chatgpt.com)

> 官方文档也对 LLAMA2-70B 基准的运行、提交流程与注意事项有概述，可用于背景页引用。[docs.mlcommons.org](https://docs.mlcommons.org/inference/benchmarks/language/llama2-70b/?utm_source=chatgpt.com)

# 3 可运行演示（两种思路）

- **参考脚本直跑**：在能访问权重的前提下，按 `run_server.sh`/`run_offline.sh` 运行；失败多与权重路径、数据预处理或 `n_tokens` 校验有关。[GitHub](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/run_server.sh?utm_source=chatgpt.com)
- **厂商参考实现路线**（如 AMD/NVIDIA 的开源提交/博客）：用其容器或 TensorRT-LLM/ROCm pipeline 快速复现一条跑通的链路（并对比参考实现的吞吐/延迟差异）。[rocm.blogs.amd.com](https://rocm.blogs.amd.com/artificial-intelligence/reproducing-amd-mlperf-inference-submission/README.html?utm_source=chatgpt.com)[GitHub](https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/code/llama2-70b/tensorrt/README.md?utm_source=chatgpt.com)
    

> 注意：仓库里关于 **Interactive** 场景的延迟描述近期有修正（typo fix），演示时建议注明文档日期。


# 4 改进建议（针对参考实现）

1. **降低 Streamer 开销**：当前实现“线程内推流”被多次反馈存在额外开销；可改为 ring-buffer＋无锁队列/原子标志，或使用后端的**连续批处理（continuous batching）**接口减少 Python 层切换。[GitHub](https://github.com/mlcommons/inference/issues/1568?utm_source=chatgpt.com)
2. **KV-Cache 与注意力优化**：接入 Flash-Attn / Paged-Attn、GQA 优化、动态长度裁剪（token budget），在 `generate()` 前设置更细的 `generation_config`。官方/厂商实现（TRT-LLM、ROCm）均证明这类优化有效。[GitHub](https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/code/llama2-70b/tensorrt/README.md?utm_source=chatgpt.com)[rocm.blogs.amd.com](https://rocm.blogs.amd.com/artificial-intelligence/reproducing-amd-mlperf-inference-submission/README.html?utm_source=chatgpt.com)
3. **异步 I/O 与预取**：DataLoader 预取＋多线程解码；`FirstTokenComplete` 与剩余 tokens 写回并行化。
4. **后端解耦**：抽象 `BackendRunner` 接口，支持 PyTorch / TensorRT-LLM / vLLM / MLU / Ascend 等后端热插。
5. **更健壮的准确率管线**：修复/替换 `consolidate_results.py` 的可选依赖，给出失败时的降级路径。[GitHub](https://github.com/mlcommons/inference/issues/1642?utm_source=chatgpt.com)
6. **配置化/可重复**：提供 Docker 与 `mlc-scripts`/CM 的一键命令，同时允许“跳过下载，走本地权重”的分支（已有 issue 诉求）。[GitHub](https://github.com/mlcommons/inference/issues/1747?utm_source=chatgpt.com)

# 5 若测试其他模型或其他硬件芯片，应如何扩展？（代码结构与示例）


```
language/<model-name>/
  ├─ backends/
  │   ├─ pytorch_runner.py      # HF/PyTorch
  │   ├─ tensorrtllm_runner.py  # NVIDIA
  │   ├─ vllm_runner.py         # 通用高速推理
  │   ├─ ascend_runner.py       # 华为 Ascend (CANN/MindSpore-lite)
  │   └─ mlu_runner.py          # 寒武纪 MLU
  ├─ adapters/
  │   ├─ model_adapter.py       # 统一 forward/generate 接口
  │   ├─ tokenizer_adapter.py   # 统一分词接口（HF/TokenizerX）
  │   └─ dataset_provider.py    # 统一数据接口（不同任务/格式）
  ├─ loadgen/
  │   ├─ sut_api.py             # 只保留与 LoadGen 的粘合层
  │   └─ metrics_hooks.py       # TTFT/TPOT/QPS 统一上报
  ├─ configs/
  │   ├─ server.yaml / offline.yaml
  │   └─ accuracy.yaml          # 任务与判分脚本映射
  ├─ tools/processor_xx.py      # 各模型的数据预处理
  └─ main.py

```



### 5.1.1 需要修改/新增的点清单
- **模型适配**：新增 `ModelAdapter`（加载、prefill/decoding、kv-cache 管理、采样策略）。
- **分词器适配**：例如 Qwen/Baichuan/DeepSeek 需定制 BOS/EOS、特别 token 映射。
- **数据/准确率**：不同任务（中文指令、代码生成、推理题）要替换 `processor*.py` 与 `evaluate-accuracy.py` 规则。[GitHub+1](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/processorca.py?utm_source=chatgpt.com)
- **硬件后端**：实现对应 `BackendRunner`（构建 engine、管理张量并行/流水并行、Pinned Memory、异构 offload 等），并在 `sut_api.py` 中注入    
- **LoadGen 参数**：按不同场景的目标 QPS/最大延迟/样本池配置进行 YAML 化管理（避免硬编码）。


