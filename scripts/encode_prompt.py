#!/usr/bin/env python3
"""v0.1 tokenizer 占位方案。

frostfall v0.1 还没有自研分词器（推迟到 v0.2，见 doc/design.md），
这里直接用 transformers 把 prompt 编码成 token id，写到一个空白分隔的文本文件，
供 C++ 端（frostfall -i tokens_in.txt）直接读取。

用法:
    python scripts/encode_prompt.py \
        --model /home/lil72/data/model/Qwen3-0___6B \
        --prompt "你好，介绍一下你自己" \
        --out tokens_in.txt
"""
import argparse

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型目录（含 tokenizer 文件）")
    ap.add_argument("--prompt", required=True, help="用户输入的文本")
    ap.add_argument("--out", required=True, help="输出的 token id 文件（空白分隔）")
    ap.add_argument("--no-chat-template", action="store_true",
                     help="不套用 chat 模板，直接对纯文本编码（对拍脚本用这个，方便和 llama-cli 默认行为对齐）")
    ap.add_argument("--enable-thinking", action="store_true",
                     help="Qwen3 chat 模板启用思考模式（默认关闭）")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.no_chat_template:
        ids = tok(args.prompt).input_ids
    else:
        messages = [{"role": "user", "content": args.prompt}]
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

    with open(args.out, "w") as f:
        f.write(" ".join(map(str, ids)))

    print(f"encoded {len(ids)} tokens -> {args.out}")
    print("ids:", ids)


if __name__ == "__main__":
    main()
