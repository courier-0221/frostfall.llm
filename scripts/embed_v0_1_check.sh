#!/bin/bash
# frostfall.embed v0.1 —— 端到端对拍：跑官方 README 4 条示例
# (2 query with instruction + 2 doc), 生成 2x2 cosine 相似度矩阵与官方对比。
# 官方期望: [[0.7646, 0.1414], [0.1355, 0.6000]]
# 用法: MODEL_HF=/path/to/Qwen3-Embedding-0.6B bash scripts/embed_v0_1_check.sh

set -e
cd "$(dirname "$0")/.."

MODEL_GGUF=${MODEL_GGUF:-models/qwen3-embedding-0.6b-f16.gguf}
MODEL_HF=${MODEL_HF:-/home/lil72/data/models/Qwen3-Embedding-0.6B}

TEXTS=(
    "What is the capital of China?"
    "Explain gravity"
    "The capital of China is Beijing."
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
)
IS_QUERY=(1 1 0 0)  # 前两条是 query, 后两条是 doc

TMPDIR=$(mktemp -d)
echo "tmp: $TMPDIR" >&2

for i in "${!TEXTS[@]}"; do
    t="${TEXTS[$i]}"
    iq="${IS_QUERY[$i]}"
    flag=""
    [[ "$iq" == "1" ]] && flag="--is-query"

    ./build/examples/embed/v0_1_smoke/v0_1_smoke "$MODEL_GGUF" $flag \
        --text "$t" --out "$TMPDIR/cpp_$i.json" 2>/dev/null >/dev/null

    conda run -n llm python scripts/embed_reference.py --model "$MODEL_HF" $flag \
        --text "$t" --dtype float32 >"$TMPDIR/ref_$i.json" 2>/dev/null
done

python - "$TMPDIR" <<'PY'
import json, sys, numpy as np
d = sys.argv[1]
cpp = [np.array(json.load(open(f"{d}/cpp_{i}.json"))["v_all"]) for i in range(4)]
ref = [np.array(json.load(open(f"{d}/ref_{i}.json"))["v_all"]) for i in range(4)]

def norm(x): return x / np.linalg.norm(x)
cpp = [norm(x) for x in cpp]
ref = [norm(x) for x in ref]

# 2x2 相似度矩阵: query_i · doc_j
cpp_mat = np.array([[float(cpp[i] @ cpp[j+2]) for j in range(2)] for i in range(2)])
ref_mat = np.array([[float(ref[i] @ ref[j+2]) for j in range(2)] for i in range(2)])
official = np.array([[0.7646, 0.1414], [0.1355, 0.6000]])

print("=== v0.1 端到端对拍结果 ===")
print("frostfall(C++, F16):")
print(cpp_mat)
print("transformers ref (fp32):")
print(ref_mat)
print("official (README):")
print(official)
print()
print("frostfall vs official abs diff max:", float(np.abs(cpp_mat - official).max()))
print("frostfall vs ref     abs diff max:", float(np.abs(cpp_mat - ref_mat).max()))

# 逐条 cosine (C++ vs ref)
print("\n逐条 cosine (C++ vs ref):")
for i in range(4):
    cos = float((cpp[i] * ref[i]).sum())
    print(f"  text[{i}] cosine = {cos:.6f}")

ok_cos = all(float((cpp[i] * ref[i]).sum()) >= 0.999 for i in range(4))
ok_mat = float(np.abs(cpp_mat - official).max()) <= 0.01
print()
print("PASS" if (ok_cos and ok_mat) else "FAIL",
      "cosine>=0.999:", ok_cos, " matrix-diff<=0.01:", ok_mat)
PY
