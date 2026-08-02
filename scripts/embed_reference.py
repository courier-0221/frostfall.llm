#!/usr/bin/env python
"""frostfall.embed v0.1 —— 对拍参考实现（transformers 版）。

用法（在 conda 环境 llm 里）:
    conda run -n llm python scripts/embed_reference.py \
        --model /home/lil72/data/models/Qwen3-Embedding-0.6B \
        --text "The capital of China is Beijing."

    # query 侧带 instruction 前缀:
    conda run -n llm python scripts/embed_reference.py \
        --model /home/lil72/data/models/Qwen3-Embedding-0.6B \
        --text "What is the capital of China?" \
        --is-query

    # MRL 截断到 256 维:
    conda run -n llm python scripts/embed_reference.py \
        --model /home/lil72/data/models/Qwen3-Embedding-0.6B \
        --text "..." --target-dim 256

输出 stdout: 与 C++ 侧 v0_1_smoke 一致的一行 JSON
    {"dim": 1024, "n_tokens": 12, "norm": 1.0, "v8": [f0, ..., f7]}
外加 v_all 字段，包含完整向量:
    {..., "v_all": [f0, f1, ..., f_{dim-1}]}
供 diff / cosine 计算使用。
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def build_query_text(text: str, task: str) -> str:
    # 官方模板: "Instruct: {task}\nQuery:{text}"（Query: 冒号后无空格）
    return f"Instruct: {task}\nQuery:{text}"


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # 官方 README 里的 last_token_pool 实现：单条无 padding 时等价 hidden[:, -1]。
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    bs = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(bs, device=last_hidden_states.device), seq_lens]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Qwen3-Embedding-0.6B HF model dir")
    ap.add_argument("--text", default=None, help="input text (else read stdin)")
    ap.add_argument("--is-query", action="store_true")
    ap.add_argument("--task", default="", help="task description (default = official)")
    ap.add_argument("--target-dim", type=int, default=0, help="MRL truncation; 0 = full")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="model dtype (default float32 for stable reference)")
    args = ap.parse_args()

    text = args.text if args.text is not None else sys.stdin.read().rstrip("\r\n")
    if args.is_query:
        task = args.task or DEFAULT_TASK
        text = build_query_text(text, task)

    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    mdl = AutoModel.from_pretrained(args.model, torch_dtype=torch_dtype)
    mdl.eval()

    # 单条推理，不做 padding；tokenizer 会根据 add_eos_token=True 自动追加 <|endoftext|>。
    batch = tok([text], padding=False, truncation=True, max_length=32768, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**batch)
        h = out.last_hidden_state          # [1, n, n_embd]
    v = last_token_pool(h, batch["attention_mask"])  # [1, n_embd]

    n_embd = v.shape[-1]
    if 0 < args.target_dim < n_embd:
        v = v[:, : args.target_dim]

    v = F.normalize(v, p=2, dim=1)
    vec = v[0].to(torch.float32).cpu().numpy()

    result = {
        "dim": int(vec.shape[0]),
        "n_tokens": int(batch["input_ids"].shape[1]),
        "norm": float(math.sqrt(float((vec * vec).sum()))),
        "v8": [float(x) for x in vec[:8]],
        "v_all": [float(x) for x in vec],
    }
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
