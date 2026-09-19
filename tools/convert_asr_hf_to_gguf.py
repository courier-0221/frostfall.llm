#!/usr/bin/env python3
# ============================================================================
# convert_asr_hf_to_gguf.py —— asr/v0.1 转换入口
#
# 把 Qwen3-ASR-0.6B 原始权重包（ModelScope/HF 目录，架构
# Qwen3ASRForConditionalGeneration）转换为 frostfall 可加载的单文件 GGUF：
#   - 文本 Decoder：thinker.model.* -> 现有 qwen3 命名（token_embd / blk.i.* /
#     output_norm / output），与 convert_hf_to_gguf.py 的 Qwen3ForCausalLM 映射一致；
#   - 音频塔：thinker.audio_tower.* -> asr.audio.* 新命名空间（v0.1 只保存与
#     加载校验，不做计算；布局按 ggml im2col 兼容存放，见各张量注释）；
#   - Tokenizer：vocab.json + merges.txt + added_tokens_decoder；
#   - 契约：general.architecture = "qwen3-asr"；文本超参沿用 {arch}.* 键，
#     ASR 专属配置写入 asr.* / asr.audio.*。
#
# 设计约束（doc/asr/design_asr_v0.x.md §2.2）：
#   - 不改动现有 LLM 转换脚本；LLM GGUF 转换行为完全不变；
#   - 矩阵优先 F16；RMSNorm / LayerNorm / bias 保留 F32；暂不量化；
#   - 不按 tie_word_embeddings 去重，embed_tokens 与 lm_head 两份矩阵都保留；
#   - 输出张量映射清单（全部 612 个存储张量的去向），不静默丢弃任何权重。
#
# 用法：
#   python tools/convert_asr_hf_to_gguf.py <模型目录> \
#       --outfile models/qwen3-asr-0.6b-f16.gguf \
#       [--mapping-out doc/asr/asr_v01_tensor_mapping.json]
#
# 依赖：conda llm 环境（torch、numpy）；gguf 使用本工程 vendored tools/gguf-py。
# ============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR / "gguf-py"))

import gguf  # noqa: E402  (vendored tools/gguf-py)

ARCH = "qwen3-asr"


# ----------------------------------------------------------------------------
# 张量映射表
# ----------------------------------------------------------------------------
# 文本侧：与 convert_hf_to_gguf.py Qwen3Model 的映射一致。
# F32 保留项：RMSNorm / LayerNorm 权重；其余矩阵 F16。

TEXT_MAP = {
    "thinker.model.embed_tokens.weight": "token_embd.weight",
    "thinker.model.norm.weight":         "output_norm.weight",
    "thinker.lm_head.weight":            "output.weight",
}

AUDIO_TOWER_MAP = {
    # CNN：PyTorch [out,in,KH,KW] contiguous -> ggml ne={KW,KH,in,out}，
    # 数据顺序不变，直接满足 ggml_conv_2d/im2col 的 filter 布局（v0.2 使用）。
    "thinker.audio_tower.conv2d1.weight": "asr.audio.conv1.weight",
    "thinker.audio_tower.conv2d1.bias":   "asr.audio.conv1.bias",
    "thinker.audio_tower.conv2d2.weight": "asr.audio.conv2.weight",
    "thinker.audio_tower.conv2d2.bias":   "asr.audio.conv2.bias",
    "thinker.audio_tower.conv2d3.weight": "asr.audio.conv3.weight",
    "thinker.audio_tower.conv2d3.bias":   "asr.audio.conv3.bias",
    "thinker.audio_tower.conv_out.weight": "asr.audio.conv_out.weight",
    "thinker.audio_tower.ln_post.weight": "asr.audio.ln_post.weight",
    "thinker.audio_tower.ln_post.bias":   "asr.audio.ln_post.bias",
    "thinker.audio_tower.proj1.weight":   "asr.audio.proj1.weight",
    "thinker.audio_tower.proj1.bias":     "asr.audio.proj1.bias",
    "thinker.audio_tower.proj2.weight":   "asr.audio.proj2.weight",
    "thinker.audio_tower.proj2.bias":     "asr.audio.proj2.bias",
}

# 每个 ASR 音频层内的子模块映射（前缀 thinker.audio_tower.layers.{i}.）
AUDIO_LAYER_SUB = {
    "self_attn.q_proj.weight":       "attn_q.weight",
    "self_attn.q_proj.bias":         "attn_q.bias",
    "self_attn.k_proj.weight":       "attn_k.weight",
    "self_attn.k_proj.bias":         "attn_k.bias",
    "self_attn.v_proj.weight":       "attn_v.weight",
    "self_attn.v_proj.bias":         "attn_v.bias",
    "self_attn.out_proj.weight":     "attn_out.weight",
    "self_attn.out_proj.bias":       "attn_out.bias",
    "self_attn_layer_norm.weight":   "attn_norm.weight",
    "self_attn_layer_norm.bias":     "attn_norm.bias",
    "fc1.weight":                    "fc1.weight",
    "fc1.bias":                      "fc1.bias",
    "fc2.weight":                    "fc2.weight",
    "fc2.bias":                      "fc2.bias",
    "final_layer_norm.weight":       "final_norm.weight",
    "final_layer_norm.bias":         "final_norm.bias",
}

# 文本 Decoder 每层的子模块映射（前缀 thinker.model.layers.{i}.）
TEXT_LAYER_SUB = {
    "input_layernorm.weight":         "attn_norm.weight",
    "self_attn.q_proj.weight":        "attn_q.weight",
    "self_attn.k_proj.weight":        "attn_k.weight",
    "self_attn.v_proj.weight":        "attn_v.weight",
    "self_attn.o_proj.weight":        "attn_output.weight",
    "self_attn.q_norm.weight":        "attn_q_norm.weight",
    "self_attn.k_norm.weight":        "attn_k_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight":           "ffn_gate.weight",
    "mlp.up_proj.weight":             "ffn_up.weight",
    "mlp.down_proj.weight":           "ffn_down.weight",
}


# 规则无法覆盖的 F32 保留项（不以 norm.weight 结尾的 LayerNorm 权重）
F32_KEEP_EXTRA = {"asr.audio.ln_post.weight"}


def is_f32_keep(gguf_name: str) -> bool:
    """保留 F32 的张量：全部 bias、RMSNorm/LayerNorm 权重（文档 §2.2 第 6 条）。

    按后缀规则匹配，对层内（blk.i.* / asr.audio.blk.i.*）与顶层张量统一适用。
    """
    return (gguf_name.endswith(".bias") or gguf_name.endswith("norm.weight")
            or gguf_name in F32_KEEP_EXTRA)


# ----------------------------------------------------------------------------
# 映射与转换
# ----------------------------------------------------------------------------

def map_tensor_name(hf_name: str):
    """HF 张量名 -> (gguf 名, 层号或 None)。返回 None 表示无法映射。"""
    if hf_name in TEXT_MAP:
        return TEXT_MAP[hf_name], None
    if hf_name in AUDIO_TOWER_MAP:
        target = AUDIO_TOWER_MAP[hf_name]
        return (target, None) if target else (None, None)

    parts = hf_name.split(".")
    # thinker.model.layers.{i}.<sub>
    if hf_name.startswith("thinker.model.layers."):
        i = int(parts[3])
        sub = ".".join(parts[4:])
        if sub in TEXT_LAYER_SUB:
            return f"blk.{i}.{TEXT_LAYER_SUB[sub]}", i
        return None, i
    # thinker.audio_tower.layers.{i}.<sub>
    if hf_name.startswith("thinker.audio_tower.layers."):
        i = int(parts[3])
        sub = ".".join(parts[4:])
        if sub in AUDIO_LAYER_SUB:
            return f"asr.audio.blk.{i}.{AUDIO_LAYER_SUB[sub]}", i
        return None, i
    return None, None


def convert_data(t: torch.Tensor, gguf_name: str):
    """BF16/其他 dtype -> F16（矩阵）或 F32（norm/bias），返回 numpy 数组。"""
    keep_f32 = is_f32_keep(gguf_name)
    src = t.detach().float()
    if not keep_f32:
        amax = src.abs().max().item() if src.numel() else 0.0
        if amax > 65504.0:
            raise ValueError(
                f"{gguf_name}: |max|={amax} exceeds F16 range, refuse lossy conversion")
        src = src.half()
    return src.contiguous().numpy()


def build_tokenizer(model_dir: Path, vocab_size: int):
    """从 vocab.json / merges.txt / tokenizer_config.json 构建 GGUF tokenizer 字段。"""
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    tc = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    added = tc.get("added_tokens_decoder", {})  # id -> {content, special}

    id_to_token = [""] * vocab_size
    token_type = [1] * vocab_size  # 1=NORMAL
    claimed = [False] * vocab_size

    for tok, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        if tid < 0 or tid >= vocab_size:
            raise ValueError(f"token {tok!r} id {tid} out of range")
        id_to_token[tid] = tok
        claimed[tid] = True

    for sid, info in added.items():
        tid = int(sid)
        tok = info["content"]
        if tid < 0 or tid >= vocab_size:
            raise ValueError(f"added token {tok!r} id {tid} out of range")
        id_to_token[tid] = tok
        # special -> CONTROL(3)；非 special added token（如 <asr_text>）-> USER_DEFINED(4)
        token_type[tid] = 3 if info.get("special") else 4
        claimed[tid] = True

    # 补齐空位（151705 -> 151936），与 embed_tokens 行数保持一致
    n_dummy = 0
    for i in range(vocab_size):
        if not claimed[i]:
            id_to_token[i] = ""
            token_type[i] = 1
            n_dummy += 1

    merges = []
    for line in (model_dir / "merges.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#version"):
            continue
        merges.append(line)

    def find_added_id(content: str) -> int:
        for sid, info in added.items():
            if info["content"] == content:
                return int(sid)
        return -1

    def vocab_id(content: str) -> int:
        return vocab.get(content, -1)

    eos_id = find_added_id(tc["eos_token"])
    bos_id = find_added_id(tc.get("bos_token")) if tc.get("bos_token") else -1
    pad_id = find_added_id(tc["padding_token"]) if tc.get("padding_token") else -1
    pad_id = pad_id if pad_id >= 0 else vocab_id("<|endoftext|>")
    if pad_id < 0:
        pad_id = eos_id

    meta = {
        "tokens": id_to_token,
        "types": token_type,
        "merges": merges,
        "eos_id": eos_id,
        "bos_id": bos_id,
        "pad_id": pad_id,
        "add_bos": bool(tc.get("add_bos_token", False)),
        "n_dummy": n_dummy,
    }
    return meta


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Qwen3-ASR-0.6B to frostfall GGUF")
    ap.add_argument("model_dir", type=Path, help="模型目录（ModelScope snapshot 或 HF repo）")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--mapping-out", type=Path, default=None,
                    help="输出张量映射清单 JSON（默认 <outfile>.mapping.json）")
    args = ap.parse_args()

    model_dir = args.model_dir
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("architectures") != ["Qwen3ASRForConditionalGeneration"]:
        raise SystemExit(f"unexpected architectures: {cfg.get('architectures')}")

    thinker = cfg["thinker_config"]
    acfg, tcfg = thinker["audio_config"], thinker["text_config"]

    print(f"loading {model_dir / 'model.safetensors'} ...")
    tensors = load_file(str(model_dir / "model.safetensors"))
    print(f"  {len(tensors)} tensors")

    # ---- 建立映射清单（先全部映射，未映射的会报错而不是静默丢弃）----
    mapping = {}      # hf_name -> dict(gguf, dtype, shape, layer)
    converted = {}    # gguf_name -> numpy
    for hf_name, t in tensors.items():
        gguf_name, layer = map_tensor_name(hf_name)
        if gguf_name is None:
            raise SystemExit(f"unmapped tensor: {hf_name} (shape={tuple(t.shape)})")
        arr = convert_data(t, gguf_name)
        if gguf_name in converted:
            raise SystemExit(f"duplicate gguf name: {gguf_name}")
        converted[gguf_name] = arr
        mapping[hf_name] = {
            "gguf": gguf_name,
            "dtype": "F32" if arr.dtype == np.float32 else "F16",
            "shape": list(arr.shape),
            "layer": layer,
        }

    # ---- 契约校验：必需张量必须齐全，数量必须对上文档基线 612 ----
    expect_text = set(TEXT_MAP.values()) | {f"blk.{i}.{s}" for i in range(tcfg["num_hidden_layers"])
                                            for s in TEXT_LAYER_SUB.values()}
    expect_audio = set(v for v in AUDIO_TOWER_MAP.values() if v) | \
        {f"asr.audio.blk.{i}.{s}" for i in range(acfg["encoder_layers"])
         for s in AUDIO_LAYER_SUB.values()}
    missing = (expect_text | expect_audio) - set(converted)
    if missing:
        raise SystemExit(f"missing expected tensors: {sorted(missing)[:8]} ...")
    if len(tensors) != 612:
        raise SystemExit(f"tensor count {len(tensors)} != documented baseline 612; "
                         "verify model revision first")
    print(f"mapping OK: {len(converted)} tensors "
          f"(text={len(expect_text)}, audio={len(expect_audio)})")

    # ---- tokenizer ----
    tmeta = build_tokenizer(model_dir, tcfg["vocab_size"])
    print(f"tokenizer: {len(tmeta['tokens'])} tokens, {len(tmeta['merges'])} merges, "
          f"{tmeta['n_dummy']} dummy, eos={tmeta['eos_id']}, pad={tmeta['pad_id']}")

    # ---- 写 GGUF ----
    writer = gguf.GGUFWriter(args.outfile, arch=ARCH)
    writer.add_architecture()
    writer.add_name("Qwen3-ASR-0.6B")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    # 文本超参（键名 = {arch}.xxx）
    writer.add_vocab_size(tcfg["vocab_size"])
    writer.add_embedding_length(tcfg["hidden_size"])
    writer.add_block_count(tcfg["num_hidden_layers"])
    writer.add_feed_forward_length(tcfg["intermediate_size"])
    writer.add_head_count(tcfg["num_attention_heads"])
    writer.add_head_count_kv(tcfg["num_key_value_heads"])
    writer.add_key_length(tcfg["head_dim"])
    writer.add_context_length(tcfg["max_position_embeddings"])
    writer.add_layer_norm_rms_eps(tcfg["rms_norm_eps"])
    writer.add_rope_freq_base(tcfg["rope_theta"])
    writer.add_rope_dimension_count(tcfg["head_dim"])

    # ASR 专属配置
    writer.add_key_value("asr.audio_start_token_id", thinker["audio_start_token_id"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio_end_token_id", thinker["audio_end_token_id"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio_token_id", thinker["audio_token_id"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.pad_token_id", tmeta["pad_id"], gguf.GGUFValueType.UINT32)
    # 生成 EOS 集合（generation_config.json: eos_token_id = [151645, 151643]）
    gen_cfg = {}
    gc_path = model_dir / "generation_config.json"
    if gc_path.exists():
        gen_cfg = json.loads(gc_path.read_text(encoding="utf-8"))
    eos_set = gen_cfg.get("eos_token_id", [thinker["audio_end_token_id"]])
    if isinstance(eos_set, int):
        eos_set = [eos_set]
    writer.add_key_value("asr.eos_token_ids", [int(v) for v in eos_set], gguf.GGUFValueType.ARRAY)
    writer.add_key_value("asr.chat_template",
                         (model_dir / "chat_template.json").read_text(encoding="utf-8")
                         if (model_dir / "chat_template.json").exists() else "",
                         gguf.GGUFValueType.STRING)
    writer.add_key_value("asr.support_languages", json.dumps(cfg.get("support_languages", [])),
                         gguf.GGUFValueType.STRING)

    # 音频塔超参（asr.audio.*）
    writer.add_key_value("asr.audio.d_model", acfg["d_model"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.encoder_layers", acfg["encoder_layers"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.attention.head_count", acfg["encoder_attention_heads"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.encoder_ffn_dim", acfg["encoder_ffn_dim"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.downsample_hidden_size", acfg["downsample_hidden_size"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.output_dim", acfg["output_dim"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.num_mel_bins", acfg["num_mel_bins"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.n_window", acfg["n_window"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.n_window_infer", acfg["n_window_infer"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.conv_chunksize", acfg["conv_chunksize"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.max_source_positions", acfg["max_source_positions"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.layer_norm_eps", 1e-5, gguf.GGUFValueType.FLOAT32)
    writer.add_key_value("asr.audio.conv.kernel", 3, gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.conv.stride", 2, gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.conv.padding", 1, gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.audio.activation", acfg["activation_function"], gguf.GGUFValueType.STRING)
    writer.add_key_value("asr.audio.scale_embedding", bool(acfg.get("scale_embedding", False)), gguf.GGUFValueType.BOOL)

    # 音频前处理契约（v0.2 C++ Log-Mel 必须使用同一参数）
    pp = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    writer.add_key_value("asr.mel.sampling_rate", pp.get("sampling_rate", 16000), gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.mel.n_fft", pp["n_fft"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.mel.hop_length", pp["hop_length"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.mel.feature_size", pp["feature_size"], gguf.GGUFValueType.UINT32)
    writer.add_key_value("asr.mel.padding_side", pp.get("padding_side", "right"), gguf.GGUFValueType.STRING)

    # tokenizer
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tmeta["tokens"])
    writer.add_token_types(tmeta["types"])
    writer.add_token_merges(tmeta["merges"])
    if tmeta["bos_id"] >= 0:
        writer.add_bos_token_id(tmeta["bos_id"])
    if tmeta["eos_id"] >= 0:
        writer.add_eos_token_id(tmeta["eos_id"])
    # tokenizer_config.json 未定义 padding_token；对齐 generation_config.json 的 pad_token_id
    pad_final = int(gen_cfg.get("pad_token_id", tmeta["pad_id"]))
    if pad_final >= 0:
        writer.add_pad_token_id(pad_final)
    writer.add_add_bos_token(tmeta["add_bos"])

    # 张量（按名字写入，GGUF 不要求顺序）
    for gguf_name, arr in converted.items():
        writer.add_tensor(gguf_name, arr)

    print(f"writing {args.outfile} ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_mb = args.outfile.stat().st_size / 1024 / 1024
    print(f"done: {args.outfile} ({size_mb:.1f} MB)")

    # ---- 张量映射清单 ----
    mapping_path = args.mapping_out or args.outfile.with_suffix(".mapping.json")
    report = {
        "source": {
            "dir": str(model_dir),
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "n_stored_tensors": len(tensors),
        },
        "contract": {
            "general.architecture": ARCH,
            "text_tensor_names": "qwen3 命名（与 convert_hf_to_gguf.py Qwen3Model 一致）",
            "audio_tensor_names": "asr.audio.* 新命名空间",
            "f16": "矩阵权重",
            "f32": "RMSNorm/LayerNorm 权重与全部 bias",
        },
        "summary": {
            "text": len(expect_text), "audio": len(expect_audio),
            "total": len(converted),
            "f16": sum(1 for m in mapping.values() if m["dtype"] == "F16"),
            "f32": sum(1 for m in mapping.values() if m["dtype"] == "F32"),
        },
        "mapping": mapping,
    }
    mapping_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tensor mapping report -> {mapping_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
