# gguf 工具安装与使用说明

本目录下的 `gguf/` 是从 llama.cpp 移植的 GGUF 读写库，配合 `convert_hf_to_gguf.py`
可以把 HuggingFace 格式的模型转换为 GGUF 格式。

> 结论先行：本仓库的代码统一以 **`import tools.gguf`** 的方式引用该库
> （见 [convert_hf_to_gguf.py](convert_hf_to_gguf.py) 第 30 行），
> 因此 **不需要、也不建议** 把它单独 `pip install` 成一个顶层 `gguf` 包，
> 只需装好依赖并从仓库根目录运行即可。

---

## 一、目录结构

```
tools/
├── convert_hf_to_gguf.py         # HF -> GGUF 转换主脚本
├── convert_hf_to_gguf_update.py
├── requirements.txt              # Python 依赖清单
└── gguf/                         # GGUF 读写库（以 tools.gguf 方式导入）
    ├── __init__.py
    ├── constants.py
    ├── gguf_reader.py / gguf_writer.py
    ├── metadata.py               # 需要 pyyaml
    ├── vocab.py                  # 需要 sentencepiece，可选 mistral-common
    └── ...
```

注意：`tools/` 下没有 `__init__.py`，`tools` 是一个**命名空间包**，
只有当仓库根目录在 `sys.path` 上时，`import tools.gguf` 才能解析到本地源码。

---

## 二、安装依赖（推荐方式）

在目标 conda 环境（例如 `llm_infer`）中安装依赖，**不安装 gguf 包本身**：

```bash
conda run -n llm_infer pip install -r tools/requirements.txt
```

依赖说明：

| 包 | 用途 | 必需 |
| --- | --- | --- |
| numpy | 张量数据处理 | 是 |
| tqdm | 进度条 | 是 |
| sentencepiece | 分词器（SPM 词表） | 是 |
| pyyaml | `metadata.py` 读取元数据 | 是 |
| transformers | 读取 HF config / 分词器 | 转换时必需 |
| torch | 加载模型权重 | 转换时必需 |
| safetensors | 读取 `.safetensors` 权重 | 转换时必需 |
| protobuf | `sentencepiece_model_pb2` | 转换时必需 |
| mistral-common | 仅 Mistral 官方格式模型 | 可选 |

---

## 三、运行方式

### 方式 A：从仓库根目录直接运行（最简单）

因为当前工作目录会被加入 `sys.path`，`import tools.gguf` 可直接解析：

```bash
cd /home/dear/code/frostfall.llm
conda run -n llm_infer python tools/convert_hf_to_gguf.py --help
```

### 方式 B：设置 PYTHONPATH（可在任意目录运行）

```bash
export PYTHONPATH=/home/dear/code/frostfall.llm
conda run -n llm_infer python tools/convert_hf_to_gguf.py <模型目录> --outfile model.gguf
```

### 方式 C：可编辑安装（可选，不推荐）

若确实想让 `gguf` 在任意环境下作为包被导入，可临时创建 `pyproject.toml`
并执行 `pip install -e tools/`。但要注意：源码内部使用的是绝对导入
`tools.gguf`，与顶层 `gguf` 包名不一致，容易产生「安装了 gguf 却仍依赖
仓库路径」的混淆，因此一般无需这样做。

---

## 四、常见问题

- **`ModuleNotFoundError: No module named 'yaml'`**
  未安装 pyyaml。执行 `pip install -r tools/requirements.txt` 即可。

- **`ModuleNotFoundError: No module named 'transformers' / 'torch'`**
  转换脚本的必需依赖未安装，见上表，按 requirements.txt 安装。

- **`import gguf` 报错 / 指向 site-packages**
  说明环境里残留了一个独立安装的 `gguf` 包。本仓库应使用
  `import tools.gguf`，可先卸载残留包：
  ```bash
  conda run -n llm_infer pip uninstall -y gguf
  ```

- **`import tools.gguf` 找不到 tools**
  未从仓库根目录运行，或未设置 `PYTHONPATH`，参见「方式 A / B」。
