#!/usr/bin/env python3
"""v0.1 对拍脚本 —— 分步执行版（见 doc/design.md §8.4）。

四个子命令独立运行，结果持久化到 --workdir，每步执行完即可退出释放内存。
transformers 只在 encode / compare 两步加载；run-ff 和 run-llama 不依赖任何大 Python 包。

典型用法：

  WORKDIR=work/cmp
  HF=/home/lil72/data/model/Qwen3-0___6B
  MODEL=models/qwen3-0.6b-f16.gguf

  # 步骤 1：编码 prompt（加载 transformers，完成后进程退出，内存释放）
  conda run -n llm python scripts/compare_llamacpp.py encode \\
      --hf-model $HF --prompt "The capital of France is" --workdir $WORKDIR

  # 步骤 2：运行 frostfall（无需 transformers）
  python scripts/compare_llamacpp.py run-ff \\
      --frostfall-bin build/frostfall --model $MODEL --workdir $WORKDIR -n 16

  # 步骤 3：运行 llama-cli（无需 transformers）
  python scripts/compare_llamacpp.py run-llama \\
      --llama-cli /home/lil72/data/llama.cpp/build/bin/llama-cli \\
      --model $MODEL --workdir $WORKDIR -n 16

  # 步骤 4：解码并对比（加载 transformers，完成后进程退出）
  conda run -n llm python scripts/compare_llamacpp.py compare \\
      --hf-model $HF --workdir $WORKDIR

workdir 中的中间文件：
  tokens_in.txt  —— prompt token ids（空白分隔整数）
  prompt_n.txt   —— prompt token 数（一个整数）
  prompt.txt     —— prompt 原文
  tokens_ff.txt  —— frostfall 完整输出 token ids（prompt + 新生成）
  llama_out.txt  —— llama-cli stdout 原文
"""
import argparse
import os
import subprocess
import sys

_TOKENS_IN  = "tokens_in.txt"
_PROMPT_N   = "prompt_n.txt"
_PROMPT_TXT = "prompt.txt"
_TOKENS_FF  = "tokens_ff.txt"
_LLAMA_OUT  = "llama_out.txt"


def _wf(workdir, name):
    return os.path.join(workdir, name)


def _require(path):
    if not os.path.exists(path):
        print(f"error: {path} not found — run the preceding step first", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 步骤 1：encode —— 依赖 transformers，完成后进程退出
# ---------------------------------------------------------------------------
def cmd_encode(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    ids = tok(args.prompt).input_ids
    os.makedirs(args.workdir, exist_ok=True)
    with open(_wf(args.workdir, _TOKENS_IN),  "w") as f: f.write(" ".join(map(str, ids)))
    with open(_wf(args.workdir, _PROMPT_N),   "w") as f: f.write(str(len(ids)))
    with open(_wf(args.workdir, _PROMPT_TXT), "w") as f: f.write(args.prompt)
    print(f"encode: {len(ids)} tokens -> {_wf(args.workdir, _TOKENS_IN)}")


# ---------------------------------------------------------------------------
# 步骤 2：run-ff —— 不依赖 transformers，frostfall 日志直接输出到终端
# ---------------------------------------------------------------------------
def cmd_run_ff(args):
    tokens_in  = _wf(args.workdir, _TOKENS_IN)
    tokens_out = _wf(args.workdir, _TOKENS_FF)
    _require(tokens_in)
    cmd = [args.frostfall_bin, "-m", args.model,
           "-i", tokens_in, "-o", tokens_out,
           "-n", str(args.n_predict), "-t", str(args.threads)]
    print("$", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd)   # 不捕获，让进度日志直接打到终端
    if r.returncode != 0:
        print("frostfall 运行失败", file=sys.stderr)
        sys.exit(r.returncode)
    print(f"run-ff: 完成 -> {tokens_out}")


# ---------------------------------------------------------------------------
# 步骤 3：run-llama —— 不依赖 transformers，llama-cli 日志直接输出到终端
# ---------------------------------------------------------------------------
def cmd_run_llama(args):
    prompt_file = _wf(args.workdir, _PROMPT_TXT)
    llama_out   = _wf(args.workdir, _LLAMA_OUT)
    _require(prompt_file)
    with open(prompt_file) as f:
        prompt = f.read()
    cmd = [args.llama_cli, "-m", args.model, "-p", prompt,
           "-n", str(args.n_predict), "--temp", "0",
           "-no-cnv", "--simple-io", "--no-warmup", "-t", str(args.threads)]
    print("$", " ".join(cmd), file=sys.stderr)
    # stdout 捕获到文件（对比用），stderr 透传（进度可见）
    r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("llama-cli 运行失败", file=sys.stderr)
        sys.exit(r.returncode)
    with open(llama_out, "w") as f:
        f.write(r.stdout)
    print(f"run-llama: 完成 -> {llama_out}")


# ---------------------------------------------------------------------------
# 步骤 4：compare —— 依赖 transformers，完成后进程退出
# ---------------------------------------------------------------------------
def cmd_compare(args):
    from transformers import AutoTokenizer

    for name in [_TOKENS_FF, _LLAMA_OUT, _PROMPT_N, _PROMPT_TXT]:
        _require(_wf(args.workdir, name))

    with open(_wf(args.workdir, _PROMPT_N))   as f: n_prompt   = int(f.read().strip())
    with open(_wf(args.workdir, _PROMPT_TXT)) as f: prompt     = f.read()
    with open(_wf(args.workdir, _TOKENS_FF))  as f: all_ids    = [int(x) for x in f.read().split()]
    with open(_wf(args.workdir, _LLAMA_OUT))  as f: llama_full = f.read()

    tok = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)

    ff_new_ids = all_ids[n_prompt:]
    ff_text    = tok.decode(ff_new_ids, skip_special_tokens=False)
    llama_cont = llama_full[len(prompt):] if llama_full.startswith(prompt) else llama_full

    print("=" * 60)
    print("prompt:", prompt)
    print("prompt token 数:", n_prompt)
    print("-" * 60)
    print("[frostfall] 新生成 token id:", ff_new_ids)
    print("[frostfall] 生成的文本:")
    print(ff_text)
    print("-" * 60)
    print("[llama.cpp] 续写部分:")
    print(llama_cont)
    print("=" * 60)

    if ff_text.strip() == llama_cont.strip():
        print("结果: 完全一致 ✓")
    else:
        print("结果: 存在差异 — 请参考 doc/design.md §8.5 用逐层 dump 定位")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="v0.1 对拍脚本（分步执行版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("encode", help="步骤1: 编码 prompt 为 token ids（需要 transformers）")
    p.add_argument("--hf-model", required=True, help="HF 模型目录")
    p.add_argument("--prompt",   required=True, help="输入文本")
    p.add_argument("--workdir",  required=True, help="中间文件目录（自动创建）")

    p = sub.add_parser("run-ff", help="步骤2: 运行 frostfall 生成 token ids（无需 transformers）")
    p.add_argument("--frostfall-bin", required=True)
    p.add_argument("--model",   required=True, help="GGUF 模型路径")
    p.add_argument("--workdir", required=True)
    p.add_argument("-n", "--n-predict", type=int, default=16)
    p.add_argument("-t", "--threads",   type=int, default=4)

    p = sub.add_parser("run-llama", help="步骤3: 运行 llama-cli 生成文本（无需 transformers）")
    p.add_argument("--llama-cli", required=True)
    p.add_argument("--model",   required=True, help="GGUF 模型路径")
    p.add_argument("--workdir", required=True)
    p.add_argument("-n", "--n-predict", type=int, default=16)
    p.add_argument("-t", "--threads",   type=int, default=4)

    p = sub.add_parser("compare", help="步骤4: 解码并对比结果（需要 transformers）")
    p.add_argument("--hf-model", required=True)
    p.add_argument("--workdir",  required=True)

    args = ap.parse_args()
    {"encode": cmd_encode, "run-ff": cmd_run_ff,
     "run-llama": cmd_run_llama, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
