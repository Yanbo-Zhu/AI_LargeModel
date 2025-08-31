

# 1 dockerfile+launch.sh+build.sh

## 1.1 dockerfile 


```
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV TZ=US/Pacific
ENV DEBIAN_FRONTEND=noninteractive

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN rm -rf /var/lib/apt/lists/* && rm /etc/apt/sources.list.d/* \
 && apt update \
 && apt install -y --no-install-recommends build-essential autoconf \
        libtool git ccache curl wget pkg-config sudo ca-certificates \
        automake libssl-dev bc python3-dev python3-pip google-perftools \
        gdb libglib2.0-dev clang sshfs libre2-dev libboost-dev \
        libnuma-dev numactl sysstat sshpass ntpdate less iputils-ping \
 && apt -y autoremove \
 && apt remove -y cmake \
 && apt install -y --no-install-recommends pkg-config zip g++ zlib1g-dev \
        unzip libarchive-dev
RUN apt install -y --no-install-recommends rsync

# Install setuptools
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --upgrade setuptools wheel virtualenv

# Install conda
WORKDIR /tmp
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
    && bash Miniconda3-* -b -p /opt/miniconda3
ENV PATH="$PATH:/opt/miniconda3/bin"
RUN conda create -n llama2-70b python=3.10
RUN chmod -R 777 /opt/miniconda3

```


整个 Dockerfile 就是：

基于 CUDA 11.8 + cuDNN 的 Ubuntu20.04

安装常用开发工具 & Python 工具链

安装 Miniconda，并建一个 llama2-70b 环境

最后用 chmod -R 777 让所有人都能自由使用 /opt/miniconda3


### 1.1.1 🔹 前面几步（环境准备）

`FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`

👉 基础镜像：Ubuntu 20.04 + CUDA 11.8 + cuDNN8（带开发工具）。

`SHELL ["/bin/bash", "-c"]`

👉 以后 RUN 命令都用 bash 来执行。

`ENV LC_ALL=C.UTF-8 ENV LANG=C.UTF-8`

👉 设置容器的语言/编码为 UTF-8，避免乱码。

`ENV TZ=US/Pacific ENV DEBIAN_FRONTEND=noninteractive`

👉 设置时区为美国太平洋时间（Pacific），同时让 apt 在安装时**不用交互式确认**（避免卡住）。

5. **apt 安装依赖**
    
    - `build-essential autoconf libtool git ...` 等开发工具包
        
    - `python3-dev python3-pip` Python 开发环境
        
    - `gdb` 调试器
        
    - `libnuma-dev numactl` NUMA 相关库
        
    - `rsync` 文件同步工具  
        👉 这些都是构建大模型推理/训练环境时常用的基础依赖。
        

---

### 1.1.2 🔹 Python & Conda 部分

`RUN python3 -m pip install --upgrade pip \     && python3 -m pip install --upgrade setuptools wheel virtualenv`

👉 升级 pip、setuptools、wheel，安装 virtualenv，保证 Python 包管理工具是最新的。

2. **安装 Miniconda**
    

`RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \     && bash Miniconda3-* -b -p /opt/miniconda3`

👉 下载并静默安装 Miniconda 到 `/opt/miniconda3`。

`ENV PATH="$PATH:/opt/miniconda3/bin"`

👉 把 conda 命令加到 PATH，之后就能直接用 `conda` 了。

`RUN conda create -n llama2-70b python=3.10`

👉 新建一个 conda 环境 `llama2-70b`，指定 Python 3.10。

---

### 1.1.3 🔹 最后一行

`RUN chmod -R 777 /opt/miniconda3`

- `chmod -R` = 递归修改权限
    
- `777` = 所有用户都可以 **读/写/执行**
    
- `/opt/miniconda3` = Miniconda 的安装目录
    

👉 作用：确保容器里的任何用户（包括非 root 用户）都能读写 Miniconda 目录，不会遇到 “permission denied” 的错误。

---

### 1.1.4 ⚠️ 注意

- `chmod 777` 虽然粗暴有效，但**安全性差**：任何用户都能改动环境，可能导致包被误删/篡改。
    
- 更常见做法是：
    
    - 把 `/opt/miniconda3` 的属主改成某个普通用户
        
    - 或者只给读/执行权限，写操作交给 conda 环境自身




## 1.2 launch.sh 

这个脚本做的事是：
1. 构建一个 GPU 支持的 Docker 镜像 (`llm/gpubringup`)
2. 启动时挂载代码目录 & 传递宿主机用户信息
3. 运行 `with_the_same_user`，让容器里也用和宿主机一致的用户运行（避免文件变成 root 权限）

你在容器里运行训练/测试时，代码目录直接映射到宿主机，生成的文件/日志属主不会是 root，而是你自己。

在 Docker 里构建并运行一个带 GPU 的容器，且容器内用户环境与宿主机一致，方便调试 MLPerf/大模型相关代码。




```
#!/bin/bash

MLCOMMONS_REPO_PATH="$(dirname "$(dirname "$PWD")")"

# Add any volume mounts here with the following syntax
# /path/to/src:/path/to/dir/in/container
MOUNTS=(
    $MLCOMMONS_REPO_PATH:$MLCOMMONS_REPO_PATH
)

# Set up docker environment file for current user
rm -f .docker_env
echo "CI_BUILD_USER=`id -u -n`" >> .docker_env
echo "CI_BUILD_UID=`id -u`" >> .docker_env
echo "CI_BUILD_GROUP=`id -g -n`" >> .docker_env
echo "CI_BUILD_GID=`id -g`" >> .docker_env
cat .docker_env

# Build container
docker build . -t llm/gpubringup

# Build mount flags
declare -a MOUNT_FLAGS
for _mount in ${MOUNTS[@]}; do
    _split=($(echo $_mount | tr ':' '\n'));
    MOUNT_FLAGS+=("--mount type=bind,source=${_split[0]},target=${_split[1]}");
done

set -x
nvidia-docker run -it --rm --net=host --runtime=nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH \
  --security-opt seccomp=unconfined \
  -w $PWD \
  --env-file `pwd`/.docker_env \
  ${MOUNT_FLAGS[*]} \
  llm/gpubringup \
  bash ./with_the_same_user
```


获取项目路径

`MLCOMMONS_REPO_PATH="$(dirname "$(dirname "$PWD")")"`
- `$PWD` = 当前工作目录
- `dirname $PWD` = 上一级目录
- `dirname $(dirname $PWD)` = 上上一级目录  
    👉 这里取 **MLCommons 仓库的根路径**（假设当前脚本在 `.../inference/...` 里）。
    

---

设置要挂载的目录

`MOUNTS=(     $MLCOMMONS_REPO_PATH:$MLCOMMONS_REPO_PATH )`

👉 建立一个数组，定义 **宿主机目录 → 容器目录** 的映射。这里挂载的是整个 MLCommons repo。

例如：

`/home/user/mlcommons:/home/user/mlcommons`

这样容器里就能访问宿主机的代码。

---

设置 `.docker_env`

``
```
rm -f .docker_env 
echo "CI_BUILD_USER=`id -u -n`" >> .docker_env 
echo "CI_BUILD_UID=`id -u`" >> .docker_env 
echo "CI_BUILD_GROUP=`id -g -n`" >> .docker_env 
echo "CI_BUILD_GID=`id -g`" >> .docker_env 
cat .docker_env
```

- 生成一个 `.docker_env` 文件，里面写当前宿主机用户的 **用户名/UID/组名/GID**。
- 后面会传进容器，让容器里也用同一个用户（避免文件属主变成 root）。
    

例如：

`CI_BUILD_USER=user CI_BUILD_UID=1000 CI_BUILD_GROUP=user CI_BUILD_GID=1000`

---


构建 Docker 镜像

`docker build . -t llm/gpubringup`

👉 以当前目录的 Dockerfile 构建一个镜像，名字叫 `llm/gpubringup`。

---

构造挂载参数

```
# Build mount flags
declare -a MOUNT_FLAGS
for _mount in ${MOUNTS[@]}; do
    _split=($(echo $_mount | tr ':' '\n'));
    MOUNT_FLAGS+=("--mount type=bind,source=${_split[0]},target=${_split[1]}");
done
```

- 把 `MOUNTS` 数组里的路径对，转换成 Docker 的 `--mount` 语法。
- 最终得到类似：

`--mount type=bind,source=/home/user/mlcommons,target=/home/user/mlcommons`

---

启动容器（重点）

```
nvidia-docker run -it --rm --net=host --runtime=nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH \
  --security-opt seccomp=unconfined \
  -w $PWD \
  --env-file `pwd`/.docker_env \
  ${MOUNT_FLAGS[*]} \
  llm/gpubringup \
  bash ./with_the_same_user
```

逐项解释：

- `nvidia-docker run` → 用 **NVIDIA Runtime** 启动容器（GPU 支持）
    
- `-it` → 交互模式
    
- `--rm` → 退出时自动删除容器
    
- `--net=host` → 网络与宿主机共享
    
- `--ipc=host` → 共享宿主机的进程通信 (IPC)，利于多进程/分布式训练
    
- `--ulimit memlock=-1` → 允许锁住无限制内存（ML 框架常需要）
    
- `--ulimit stack=67108864` → 设置栈大小
    
- `--cap-add=SYS_PTRACE ...` → 给容器额外的调试/系统权限（调试器 gdb、性能分析等需要）
    
- `--security-opt seccomp=unconfined` → 关闭 seccomp 限制，允许更自由的系统调用
    
- `-w $PWD` → 设置容器启动目录为当前目录
    
- `--env-file .docker_env` → 把之前写的用户 UID/GID 环境变量传进容器
    
- `${MOUNT_FLAGS[*]}` → 把宿主机目录挂载进去
    
- `llm/gpubringup` → 镜像名
    
- `bash ./with_the_same_user` → 容器里运行的第一个命令，脚本作用通常是 **根据传入的 UID/GID 在容器里创建同名用户，并切换到这个用户执行后续命令**




## 1.3 build.sh 


Inside the container, set up the environment with bash build.sh. This will install all the dependencies from the CPU-only setup, as well as any GPU versions for applicable libraries like PyTorch.

```
set -e

conda install pybind11==2.10.4 -c conda-forge -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia
python -m pip install transformers==4.31.0 nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece==0.1.99 accelerate==0.21.0


cd ../../loadgen && python3 -m pip install .
```



下面逐行解释你这段安装脚本在做什么，以及它们各自的作用/注意点：

`set -e`

- 打开“出错即退出”模式。任何一条命令非 0 退出码，脚本立刻停止，避免后续步骤在不一致状态下继续执行。
    

`conda install pybind11==2.10.4 -c conda-forge -y`

- 用 conda 从 **conda-forge** 源安装 `pybind11`（C++/Python 绑定工具，很多高性能包的编译依赖）。`-y` 自动确认。
    

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia`

- 安装 **PyTorch 夜ly版本**、`torchvision`、`torchaudio`，以及与之匹配的 **CUDA 11.8** 运行时（`pytorch-cuda=11.8`）。
    
- 渠道顺序：先 `pytorch-nightly` 再 `nvidia`。这会拉取 nightly 版 PyTorch（功能新但更不稳定）。如果想稳定，用 `-c pytorch -c nvidia`。
    

`python -m pip install \   transformers==4.31.0 nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 \   rouge-score==0.1.2 sentencepiece==0.1.99 accelerate==0.21.0`

- 用 pip 装推理/评测相关包：
    
    - `transformers 4.31.0`：HuggingFace 模型与推理接口（与一些较新模型的 API 可能不完全匹配，固定版本便于复现实验）。
        
    - `nltk`：文本处理工具（如分词、停用词等）。需要的话后续可 `nltk.download('punkt')` 等下载数据。
        
    - `evaluate`：HuggingFace 的评测指标工具集。
        
    - `absl-py`：Abseil 日志/flag 等工具（常见于研究代码）。
        
    - `rouge-score`：ROUGE 指标实现（摘要/生成任务常用）。
        
    - `sentencepiece`：子词分词器（LLaMA/LLM 常用分词）。
        
    - `accelerate`：HuggingFace 设备/多卡启动与分布式便捷封装。
        

`cd ../../loadgen && python3 -m pip install .`

- 切到本地 `loadgen` 目录（MLPerf 的 **LoadGen** 负载生成器），以“本地源码包”方式安装（`pip install .`）。这是跑 MLPerf Inference 的核心流量发生器与日志/准确率接口。



# 2 evaluate_accuracy.py


把 **MLPerf accuracy 日志**（`mlperf_log_accuracy.json`）里模型生成的**token 序列**解码成文本，与数据集里的**参考答案**对齐，计算 **ROUGE** 指标，并输出汇总统计（包括生成长度统计）。为加速评测，ROUGE 计算采用 **多进程并行**。

这份脚本将 MLPerf accuracy 日志里的生成 token 还原并解码成文本，和数据集答案按 qsl_idx 对齐，使用 多进程计算 ROUGE 指标，最后输出平均分与生成长度统计，用于离线质量评测与对比。

输出：打印一个字典（各 ROUGE 的平均分 ×100，保留 4 位小数；以及长度统计）。


## 2.1 参数与输入输出

输入
- `--checkpoint-path`：HF 格式的模型或分词器目录（用于 `AutoTokenizer` 解码生成的 token）。
    
- `--mlperf-accuracy-file`：MLPerf 的 accuracy 日志（JSON），每条记录里包含：
    
    - `qsl_idx`：样本索引
        
    - `data`：把生成 token（或logits）按指定 dtype 编码后再 hex 编码的字符串
        
- `--dataset-file`：预处理后的验证集（pkl，含参考答案列 `output`）。
    
- `--dtype`：accuracy 日志中 `data` 字段的底层 dtype（默认 `int64`，可选 `int32` 或 `float`）。


输出：打印一个字典（各 ROUGE 的平均分 ×100，保留 4 位小数；以及长度统计）。


---

## 2.2 主要流程



1  **解析参数 & 依赖准备**
- 下载 NLTK 句子切分资源（`punkt`），并实例化 tokenizer。
- `use_fast=False`：用 Python 版 tokenizer（非 Rust FastTokenizer）。

```
args = get_args()
nltk.download("punkt")
nltk.download("punkt_tab")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, ...)
```

---

2 加载参考答案
```
targets = get_groundtruth(args.dataset_file)  # 读取 pkl 中的 data["output"]

```

----

3 读取 accuracy 日志并还原预测 token

```
results = json.load(open(mlperf_accuracy_file))
eval_dtype = np.int64 / np.int32 / np.float32  # 取决于 --dtype
for pred in results:
    qsl_idx = pred["qsl_idx"]
    arr = np.frombuffer(bytes.fromhex(pred["data"]), eval_dtype)

```

data 原本是二进制 buffer，经 hex 编码存入 json；这里先 fromhex 再 np.frombuffer 得到 Numpy 数组。
脚本用 seen 去重，避免同一 qsl_idx 多次计入。
将参考答案 targets[qsl_idx] 和预测 arr 追加到列表。
统计总生成 token 数 gen_tok_len。

---

4 批量解码预测 token → 文本
```
preds_decoded_text = tokenizer.batch_decode(preds_token_ids, skip_special_tokens=True)

```


---

5 后处理文本并并行计算 ROUGE
```
preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
targets = ["\n".join(nltk.sent_tokenize(t.strip())) for t in targets]
```

ROUGE-LSum 约定每句换行，便于句级对齐。

切分成 cpu_count() 份 chunk，用 multiprocessing.Pool.map 并行计算：
```
metric = evaluate.load("rouge")
metric.compute(predictions=..., references=..., use_stemmer=True, use_aggregator=False)
```

use_aggregator=False → 返回逐样本得分列表（每个 chunk 一份）。

---

6 汇总各 chunk 的结果并计算平均分
```
aggregated_results[k].extend(v)  # 合并 rouge1/rouge2/rougeL 列表
final_result = {k: round(np.mean(v) * 100, 4) for k, v in aggregated_results.items()}

```

同时补充：

- `gen_len`：按 **字符/词**长度（这里其实是 `len(pred)`，即字符串长度），不是 token 数
    
- `gen_num`：样本数
    
- `gen_tok_len`：预测 token 总数（来自还原的 token 序列）
    
- `tokens_per_sample`：平均每样本的生成 token 数

---

7 打印结果 
print(final_result)


## 2.3 关键实现细节与假设

- **对齐方式**：用 `qsl_idx` 对齐预测与参考答案；假设 `dataset_file` 的 `data["output"]` 与 MLPerf 的 QSL 索引一致（`targets[qsl_idx]` 有效）。
    
- **数据类型**：`--dtype` 必须与记录 `data` 时使用的 dtype 一致（否则解码错乱）。
    
- **token→文本**：`batch_decode(..., skip_special_tokens=True)` 会去掉如 `<eos>`、`<pad>` 等特殊符号。
    
- **NLTK 资源**：首次需要下载 `punkt`；`punkt_tab` 并非常见资源，若报错可移除。
    
- **并行计算**：每个进程内各自 `evaluate.load("rouge")`，避免对象在进程间共享的问题。
    

---

## 2.4 易踩坑与改进建议

1. **`punkt_tab` 资源**
    
    - `nltk.download("punkt_tab")` 大概率没必要，若离线或拉取失败会报错。建议删除或包裹 `try/except`。
        
2. **`dtype=float` 的语义**
    
    - 当选 `float` 时用 `np.float32` 还原数组；如果 accuracy 日志存的是 **token id**，应使用整型；若存的是 **logits** 或 **概率**，还原后不能直接 `batch_decode`（会报类型错误）。当前代码默认 `preds_token_ids` 是整数 ID。
        
3. **参考答案长度 & 数据一致性**
    
    - 若 `results` 里的 `qsl_idx` 超过 `targets` 范围会抛异常。可在读取时做范围校验并报更友好的错误信息。
        
4. **并行分块均衡**
    
    - `chunk_size = ceil(N / num_chunks)`；在 `N` 显著小于 CPU 数时会产生许多空块。可设 `num_chunks = min(cpu_count(), N)`。
        
5. **评测可重复性**
    
    - `evaluate.load("rouge")` 首次需要联网缓存；离线环境建议在镜像里预装或固定版本。
        
6. **统计项命名**
    
    - `gen_len` 当前是 **字符长度和**（对 `pred` 字符串取 `len`）。若想统计 **token 长度和**，应使用 `gen_tok_len`（脚本已提供）。建议在输出中注明单位，避免误解。
        
7. **容错处理**
    
    - 若某些样本解码失败或为空，建议在预处理阶段过滤或记录 `bad cases`，避免整个任务中断。
        

---

# 3 复杂度与性能

- 时间主要花在 **tokenizer 解码** 和 **ROUGE** 计算。
    
- 并行能显著缩短 ROUGE 时间；I/O 与 JSON 解析开销较小。
    
- 内存主要由 `preds_token_ids`、`preds_decoded_text` 和 ROUGE 中间结构占用；在 2~3 万样本规模通常可接受。

## 3.1 源码

```
import argparse
from transformers import AutoTokenizer
import nltk
import evaluate
import numpy as np
import json
from multiprocessing import Pool, cpu_count


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", required=True, help="Path to Llama2-70b-hf-chat checkpoint"
    )
    parser.add_argument(
        "--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json"
    )
    parser.add_argument(
        "--dataset-file",
        required=True,
        help="path to processed openorca validation set",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose messages")
    parser.add_argument(
        "--dtype",
        default="int64",
        help="dtype of the accuracy log",
        choices=["int32", "int64", "float"],
    )
    args = parser.parse_args()
    return args


def get_groundtruth(processed_dataset_file):
    import pandas as pd

    data = pd.read_pickle(processed_dataset_file)
    ground_truths = data["output"]
    return ground_truths


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def compute_rouge_chunk(chunk):
    """Compute ROUGE scores for a chunk of predictions and references."""
    metric = evaluate.load("rouge")
    preds, targets = chunk
    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False
    )
    return result


def main():

    args = get_args()
    dataset_path = args.dataset_file
    checkpoint_path = args.checkpoint_path
    nltk.download("punkt")
    nltk.download("punkt_tab")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,
    )

    targets = get_groundtruth(args.dataset_file)

    target_required = []
    preds_token_ids = []

    eval_dtype = np.int64
    if args.dtype == "int32":
        eval_dtype = np.int32
    elif args.dtype == "float":
        eval_dtype = np.float32

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    gen_tok_len = 0
    for pred in results:
        qsl_idx = pred["qsl_idx"]
        if qsl_idx in seen:
            continue

        seen.add(qsl_idx)
        target = targets[qsl_idx]
        target_required.append(target)
        pred = np.frombuffer(bytes.fromhex(pred["data"]), eval_dtype)

        gen_tok_len += len(pred)
        preds_token_ids.append(pred)

    preds_decoded_text = tokenizer.batch_decode(
        preds_token_ids, skip_special_tokens=True
    )

    preds, targets = postprocess_text(preds_decoded_text, target_required)

    # Split data into chunks for parallel processing
    num_chunks = cpu_count()  # Number of parallel processes
    chunk_size = len(preds) // num_chunks + (len(preds) % num_chunks > 0)

    chunks = [
        (preds[i:i + chunk_size], targets[i:i + chunk_size])
        for i in range(0, len(preds), chunk_size)
    ]

    # Use multiprocessing Pool to compute ROUGE scores in parallel
    with Pool(num_chunks) as pool:
        results_list = pool.map(compute_rouge_chunk, chunks)

    # Aggregate results from all chunks
    aggregated_results = {}

    for result in results_list:
        for k, v in result.items():
            if k not in aggregated_results:
                aggregated_results[k] = []
            aggregated_results[k].extend(v)

    final_result = {k: round(np.mean(v) * 100, 4)
                    for k, v in aggregated_results.items()}

    prediction_lens = [len(pred) for pred in preds]
    gen_num = len(preds)

    final_result.update({
        "gen_len": np.sum(prediction_lens),
        "gen_num": gen_num,
        "gen_tok_len": gen_tok_len,
        "tokens_per_sample": round(gen_tok_len / gen_num, 1),
    })

    print("\nResults\n")
    print(final_result)


if __name__ == "__main__":
    main()

```



# 4 consolidate_results.py 


把大模型推理生成的 token 序列（存放在 run_outputs/ 里）解码成文本，和参考答案比对，计算 ROUGE 分数，并把结果保存到一个新的 `.pkl` 文件。

这段代码实现了一个 **评测管道**：
1. 读入数据集和模型推理结果（token id 序列）
2. 去掉 EOS，解码成文本
3. 和参考答案对比，计算 ROUGE 分数
4. 把结果保存到新的 DataFrame（含生成文本和评测分数）    

用途：**在 MLPerf 或其他 LLM 评测任务中，统一收集并保存模型的输出质量指标**。



## 4.1 导入依赖

- `argparse`：解析命令行参数
    
- `evaluate`：HuggingFace 的评估库（这里用 ROUGE）
    
- `nltk`：自然语言处理工具（用来做句子分割）
    
- `pandas` / `pickle`：读写数据集 `.pkl` 文件
    
- `transformers.LlamaTokenizerFast`：加载 LLaMA 系列的 tokenizer
    
- `tqdm`：进度条
    

---

## 4.2 参数解析 (`get_args()`)

脚本支持几个命令行参数：

- `--dataset-path`：原始数据集（pkl 文件，来自 `processorca.py`）
    
- `--run-outputs`：推理生成的输出目录（里面有很多 `q*.pkl`）
    
- `--model-dir`：HuggingFace 格式的 LLaMA v2 模型目录（用于加载 tokenizer）
    
- `--output-pkl-path`：最终结果保存路径（默认 `full_output.pkl`）
    

---

## 4.3 加载函数

- `load_dataset(p)`：加载原始数据集（pandas DataFrame，里面有 `output` 列作为参考答案）。
    
- `load_run_outputs(p)`：
    
    - 找到 `run_outputs` 目录下的所有 `q*.pkl` 文件。
        
    - 每个文件里包含：
        
        - `query_ids`：样本编号
            
        - `outputs`：对应的 token 序列
            
    - 合并成一个字典 `{qid: outputs}`。
        
    - **断言**：`query_ids` 和 `outputs` 数量相等，且 `qid` 不重复。
        

---

## 4.4 主流程 (`main(args)`)

1. **准备工具**
    
    - 加载 tokenizer：`LlamaTokenizerFast.from_pretrained(args.model_dir)`
        
    - 加载评测指标：`evaluate.load("rouge")`
        
    - 下载 NLTK 的句子切分模型：`nltk.download("punkt")`
        
2. **加载数据**
    
    - 原始数据集 DataFrame `df`
        
    - 运行输出 `run_outputs`（dict）
        
    - 断言总共有 24576 条（写死的数量要求）
        
3. **准备结果容器**
    
    - 三个列表（长度 24576）：
        
        - `output_tok_ids_col`：保存生成的 token id 列表
            
        - `output_text_col`：保存生成的文本（解码后）
            
        - `output_lens`：保存生成序列长度
            
4. **处理每个样本**
    
    - 遍历 `qid, output`：
        
        - 转成 list：`L = list(output)`
            
        - 如果有 **EOS token（id=2）**，截断到第一个 `2` 之前；否则记到 `no_eos_ids`
            
        - 确认最后一个 token 不是 2
            
        - 保存：
            
            - `output_tok_ids_col[qid] = L`
                
            - `output_lens[qid] = len(L)`
                
            - 解码成文本：`tokenizer.decode(L, skip_special_tokens=True)`
                
    - 打印多少条没有 EOS token
        
5. **计算 ROUGE**
    
    - 定义 `_preproc(s)`：先 `strip()`，再用 `nltk.sent_tokenize` 分句，然后用换行符拼接
        
    - `preds = list(map(_preproc, output_text_col))` （模型生成的文本）
        
    - `targets = list(map(_preproc, df["output"]))` （参考答案）
        
    - `rouge_scores = metric.compute(...)`
        
        - 返回每条样本的 `rouge1`、`rouge2`、`rougeL` 分数
            
    - 断言结果长度都是 24576
        
    - 打印平均 ROUGE（乘 100 保留 4 位小数）
        
    - 打印平均输出序列长度
        
6. **写回 DataFrame**
    
    - 新增列：
        
        - `gen_output_tok_id`
            
        - `gen_output_text`
            
        - `gen_output_tok_len`
            
        - `rouge1`、`rouge2`、`rougeL`
            
    - 保存到 `args.output_pkl_path`



## 4.5 源码

```
import argparse
import evaluate
import glob
import nltk
import numpy as np
import os
import pandas as pd
import pickle

from pathlib import Path
from transformers import LlamaTokenizerFast
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to .pkl generated by processorca.py",
    )
    parser.add_argument(
        "--run-outputs",
        type=str,
        default="run_outputs",
        help="Output dir generated by accuracy run.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to Llamav2 HuggingFace repo clone",
    )
    parser.add_argument(
        "--output-pkl-path",
        type=str,
        default="full_output.pkl",
        help="Path to dump output to",
    )
    args = parser.parse_args()
    return args


def load_dataset(p: os.PathLike):
    print(f"Loading from {p}...")
    return pd.read_pickle(p)


def load_run_outputs(p: os.PathLike):
    g = glob.glob(str(Path(p) / "q*.pkl"))

    by_query_idx = dict()
    for pkl_file in g:
        print(f"Loading from {pkl_file}...")
        with open(pkl_file, "rb") as f:
            d = pickle.load(f)
        assert len(d["query_ids"]) == len(d["outputs"])

        for i in range(len(d["query_ids"])):
            qid = d["query_ids"][i]
            assert qid not in by_query_idx
            by_query_idx[qid] = d["outputs"][i]

    return by_query_idx


def main(args):
    # Set up decode and evaluation objects
    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_dir)
    metric = evaluate.load("rouge")
    nltk.download("punkt")

    # Load Data
    df = load_dataset(args.dataset_path)
    run_outputs = load_run_outputs(args.run_outputs)
    assert len(run_outputs) == 24576

    # Set up columns to add
    output_tok_ids_col = [None] * 24576
    output_text_col = [None] * 24576
    output_lens = [None] * 24576

    # Process data
    no_eos_ids = []
    for qid, output in tqdm(run_outputs.items()):
        L = list(output)
        # Prune trailing 2s (EOS token)
        try:
            first2 = L.index(2)
            L = L[:first2]
        except ValueError:
            # Do nothing
            no_eos_ids.append(qid)

        assert L[-1] != 2
        output_tok_ids_col[qid] = L
        output_lens[qid] = len(L)

        # Decode tokens
        output_text_col[qid] = tokenizer.decode(
            output_tok_ids_col[qid], skip_special_tokens=True
        )
    print(f"Found {len(no_eos_ids)} samples with no EOS token")

    print("Calculating rouge scores...")
    def _preproc(s): return "\n".join(nltk.sent_tokenize(s.strip()))
    preds = list(map(_preproc, output_text_col))
    targets = list(map(_preproc, list(df["output"])))
    rouge_scores = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False
    )

    assert len(rouge_scores["rouge1"]) == 24576
    assert len(rouge_scores["rouge2"]) == 24576
    assert len(rouge_scores["rougeL"]) == 24576

    agg = {k: round(np.mean(v) * 100, 4) for k, v in rouge_scores.items()}
    print(agg)
    print("Avg output seqlen:", np.mean(output_lens))

    # Set columns
    df["gen_output_tok_id"] = output_tok_ids_col
    df["gen_output_text"] = output_text_col
    df["gen_output_tok_len"] = output_lens
    df["rouge1"] = rouge_scores["rouge1"]
    df["rouge2"] = rouge_scores["rouge2"]
    df["rougeL"] = rouge_scores["rougeL"]

    p = Path(args.output_pkl_path)
    p.parent.mkdir(exist_ok=True)
    df.to_pickle(p)
    print(f"Dumped to {p}")


if __name__ == "__main__":
    main(get_args())

```
