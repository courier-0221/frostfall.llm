#!/usr/bin/env python3
"""把 frostfall 输出的完整 token id 序列（空白分隔）反解码成文本，用于人工检查生成结果是否通顺。

用法:
    python scripts/decode_tokens.py \
        --model /home/lil72/data/model/Qwen3-0___6B \
        --ids-file tokens_out.txt
"""
import argparse

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型目录（含 tokenizer 文件）")
    ap.add_argument("--ids-file", required=True, help="frostfall 输出的完整 token id 文件")
    ap.add_argument("--prompt-len", type=int, default=0,
                     help="prompt 占用的 token 数，若提供则额外单独打印“新生成部分”的解码结果")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    with open(args.ids_file) as f:
        ids = [int(x) for x in f.read().split()]

    print(f"总 token 数: {len(ids)}")
    print("--- 完整解码文本（含 prompt）---")
    print(tok.decode(ids, skip_special_tokens=False))

    if args.prompt_len > 0 and args.prompt_len < len(ids):
        print("--- 新生成部分解码文本 ---")
        print(tok.decode(ids[args.prompt_len:], skip_special_tokens=False))


if __name__ == "__main__":
    main()
