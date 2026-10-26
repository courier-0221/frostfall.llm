#!/usr/bin/env python3
# ============================================================================
# export_asr_window_reference.py —— asr/v0.4 跨窗口参考基线导出
#
# 目的（doc/asr/design_asr.md §5.2 第 5 条）：
#   固定官方跨窗口参考语义并导出 golden，供 C++ 块对角双向注意力对齐。
#
# 参考后端选择（架构文档 §5.5 第 2/3 点）：
#   官方 FlashAttention varlen 路径按 cu_seqlens 分块（窗口内双向、窗口间
#   隔离）；其自定义 eager 路径不消费 cu_seqlens（退化为全局双向）。本脚本
#   按 get_audio_cu_seqlens 的语义在 eager SDPA 上显式构造块对角 attn_mask
#   复现 varlen 语义 —— 窗口语义以 cu_seqlens 定义为准，不以 eager 缺陷为准。
#
#   get_audio_cu_seqlens（transformers models/qwen3_asr/modeling_qwen3_asr.py）：
#     aftercnn_lens = A 公式逐块输出；window_aftercnn = max(aftercnn_lens) *
#     (n_window_infer/(n_window*2))；逐块累加，凑满 window_aftercnn 关窗，
#     尾块收尾。本模型：chunk=100，window_aftercnn = 13*8 = 104 位置。
#
# 导出内容（work/asr_ref_win/cross/）：
#   pcm.npy / mel.npy / audio_conv_out.npy / audio_features.npy /
#   audio_hidden.npy（逐层）/ window_bounds.npy / meta.json
#
# 用法（cv 环境）：
#   python scripts/asr/export_asr_window_reference.py <模型目录> \
#       --out-dir work/asr_ref_win --seconds 20
# ============================================================================

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

SR = 16000


# ---- 合成音频（与 v0.1 基线同公式，仅时长不同；20s -> T≈2000，A=260，3 窗口） ----

def synth_audio(n_samples: int) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / SR
    f0 = 165.0 + 55.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.7 * t)
    rng = np.random.default_rng(20260919)
    sig = sig * env + 0.01 * rng.standard_normal(n_samples)
    peak = np.abs(sig).max()
    return (sig / max(peak, 1e-9) * 0.95).astype(np.float32)


# ---- 官方整数公式与窗口边界（逐行对齐 get_audio_cu_seqlens） ----

def feat_extract_output_lengths(length: int) -> int:
    # 官方 _get_feat_extract_output_lengths(n_window=50)：完整块(len=100) -> 13
    leave = length % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (length // 100) * 13


def audio_positions(T: int) -> int:
    # A(T) = 13*(T//100) + ceil((T%100)/8)
    return 13 * (T // 100) + ((T % 100) + 7) // 8


def get_audio_cu_seqlens(chunk_lengths, n_window_infer=800, n_window=50):
    """official semantics: window_aftercnn = max(aftercnn(chunk_lens)) * ratio."""
    aftercnn_per_chunk = [feat_extract_output_lengths(c) for c in chunk_lengths]
    max_len_after_cnn = max(aftercnn_per_chunk)
    ratio = n_window_infer // (n_window * 2)
    window_aftercnn = max_len_after_cnn * ratio
    # 官方：先产窗口长度列表，再 cumsum 成边界
    win_lens = []
    cur = 0
    for alen in aftercnn_per_chunk:
        cur += alen
        if cur >= window_aftercnn:
            win_lens.append(cur)
            cur = 0
    if cur > 0:
        win_lens.append(cur)
    cu = list(np.cumsum([0] + win_lens))
    return cu, window_aftercnn


# ---- 音频塔 forward（v0.1 脚本逐行复现 + 窗口化注意力） ----

def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> torch.Tensor:
    inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-inc * torch.arange(channels // 2).float())
    scaled = torch.arange(length)[:, None].float() * inv[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


def gelu(x):
    return torch.nn.functional.gelu(x)


def audio_tower_forward_windowed(mel: torch.Tensor, T: int, w: dict, acfg: dict,
                                 collect: dict):
    """mel: [128, T] fp32。跨窗口：块对角双向 attn_mask（SDPA 显式复现 varlen 语义）。"""
    d_model = acfg["d_model"]
    n_window = acfg["n_window"]                     # 50
    n_window_infer = acfg["n_window_infer"]         # 800
    n_head = acfg["encoder_attention_heads"]
    head_dim = d_model // n_head

    input_features = mel.T  # [T,128]（scale_embedding=false）
    feature_lens = torch.tensor([T], dtype=torch.long)

    # ---- 分块（官方 forward 前 12 行） ----
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.tensor([n_window * 2] * int(chunk_num.sum()), dtype=torch.long)
    tail_idx = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_idx] = feature_lens % (n_window * 2)
    chunk_lengths[chunk_lengths == 0] = n_window * 2

    chunk_list = input_features.split(chunk_lengths.tolist(), dim=0)
    padded = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    # 官方 _get_feat_extract_output_lengths 作用在单块 chunk_len 上（完整块=100 -> 13）
    def per_chunk_len(c):
        x = int(c)
        leave = x % 100
        feat = (leave - 1) // 2 + 1
        return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (x // 100) * 13
    aftercnn_lens = [per_chunk_len(c) for c in chunk_lengths.tolist()]
    padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(l, dtype=torch.bool) for l in aftercnn_lens], batch_first=True)

    # ---- 3 层 Conv2d + GELU ----
    x = padded.unsqueeze(1)
    x = gelu(torch.nn.functional.conv2d(x, w["conv2d1.weight"], w["conv2d1.bias"], stride=2, padding=1))
    x = gelu(torch.nn.functional.conv2d(x, w["conv2d2.weight"], w["conv2d2.bias"], stride=2, padding=1))
    x = gelu(torch.nn.functional.conv2d(x, w["conv2d3.weight"], w["conv2d3.bias"], stride=2, padding=1))
    b, c, f, t = x.size()
    x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    x = torch.nn.functional.linear(x, w["conv_out.weight"])
    pos = sinusoids(acfg["max_source_positions"], d_model)[: x.shape[1], :].unsqueeze(0)
    x = x + pos
    hidden = x[padded_mask_after_cnn]  # [A,896]
    n_states = hidden.shape[0]

    # ---- 窗口边界 + 块对角双向 attn_mask ----
    cu, window_aftercnn = get_audio_cu_seqlens(chunk_lengths.tolist(), n_window_infer, n_window)
    A = n_states
    assert cu[-1] == A, f"cu_seqlens 终点 {cu[-1]} != A {A}"
    win_id = torch.zeros(A, dtype=torch.long)
    for i, (s0, s1) in enumerate(zip(cu[:-1], cu[1:])):
        win_id[s0:s1] = i
    attn_mask = (win_id[None, :] != win_id[:, None])  # True = 屏蔽
    mask_sdpa = torch.zeros(A, A)
    mask_sdpa[attn_mask] = float("-inf")
    mask_4d = mask_sdpa[None, None, :, :]  # [1,1,A_q,A_k]

    collect["window_bounds"] = cu
    collect["window_aftercnn"] = window_aftercnn
    collect["conv_out"] = hidden.clone()

    # ---- 18 层 Transformer（SDPA + 显式块对角 mask） ----
    eps = 1e-5
    layers_out = [hidden.clone()]
    for il in range(acfg["encoder_layers"]):
        p = f"layers.{il}."
        residual = hidden
        h = torch.nn.functional.layer_norm(hidden, (d_model,), w[p + "self_attn_layer_norm.weight"],
                                           w[p + "self_attn_layer_norm.bias"], eps)
        q = torch.nn.functional.linear(h, w[p + "self_attn.q_proj.weight"], w[p + "self_attn.q_proj.bias"])
        k = torch.nn.functional.linear(h, w[p + "self_attn.k_proj.weight"], w[p + "self_attn.k_proj.bias"])
        v = torch.nn.functional.linear(h, w[p + "self_attn.v_proj.weight"], w[p + "self_attn.v_proj.bias"])
        q = q.view(n_states, n_head, head_dim).transpose(0, 1).unsqueeze(0)
        k = k.view(n_states, n_head, head_dim).transpose(0, 1).unsqueeze(0)
        v = v.view(n_states, n_head, head_dim).transpose(0, 1).unsqueeze(0)
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask_4d, dropout_p=0.0, is_causal=False,
            scale=head_dim ** -0.5)
        o = o.squeeze(0).transpose(0, 1).reshape(n_states, -1).contiguous()
        o = torch.nn.functional.linear(o, w[p + "self_attn.out_proj.weight"], w[p + "self_attn.out_proj.bias"])
        hidden = residual + o
        residual = hidden
        h = torch.nn.functional.layer_norm(hidden, (d_model,), w[p + "final_layer_norm.weight"],
                                           w[p + "final_layer_norm.bias"], eps)
        h = torch.nn.functional.linear(h, w[p + "fc1.weight"], w[p + "fc1.bias"])
        h = gelu(h)
        h = torch.nn.functional.linear(h, w[p + "fc2.weight"], w[p + "fc2.bias"])
        hidden = residual + h
        layers_out.append(hidden.clone())

    hidden = torch.nn.functional.layer_norm(hidden, (d_model,), w["ln_post.weight"], w["ln_post.bias"], eps)
    hidden = torch.nn.functional.linear(hidden, w["proj1.weight"], w["proj1.bias"])
    hidden = gelu(hidden)
    hidden = torch.nn.functional.linear(hidden, w["proj2.weight"], w["proj2.bias"])  # [A,1024]

    collect["hidden"] = torch.stack(layers_out)  # [n_layer+1, A, 896]
    return hidden


def main() -> int:
    ap = argparse.ArgumentParser(description="Export asr/v0.4 cross-window audio reference")
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("work/asr_ref_win"))
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="合成音频时长；20s -> T≈2000，A=260 -> 3 个注意力窗口")
    args = ap.parse_args()

    cfg = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    thinker = cfg["thinker_config"]
    audio_cfg = thinker["audio_config"]

    from transformers import WhisperFeatureExtractor
    mel_extractor = WhisperFeatureExtractor.from_pretrained(args.model_dir)

    weights = load_file(str(args.model_dir / "model.safetensors"))
    audio_w = {k[len("thinker.audio_tower."):]: v.float()
               for k, v in weights.items() if k.startswith("thinker.audio_tower.")}

    pcm = synth_audio(int(SR * args.seconds))
    feat = mel_extractor(pcm, sampling_rate=SR, return_tensors="pt", padding=True,
                         return_attention_mask=True)
    T = int(feat.attention_mask[0].sum())
    mel_valid = feat.input_features[0][:, :T].float()

    collect: dict = {}
    feats = audio_tower_forward_windowed(mel_valid, T, audio_w, audio_cfg, collect)
    A = feats.shape[0]
    A_expect = audio_positions(T)
    assert A == A_expect, f"A mismatch: {A} != {A_expect} (T={T})"

    out_dir = args.out_dir / "cross"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pcm.npy", pcm)
    np.save(out_dir / "mel.npy", mel_valid.numpy().astype(np.float32))
    np.save(out_dir / "audio_conv_out.npy", collect["conv_out"].numpy().astype(np.float32))
    np.save(out_dir / "audio_hidden.npy", collect["hidden"].numpy().astype(np.float32))
    np.save(out_dir / "audio_features.npy", feats.numpy().astype(np.float32))
    np.save(out_dir / "window_bounds.npy", np.array(collect["window_bounds"], dtype=np.int32))

    meta = {
        "case": "cross",
        "sample_rate": SR,
        "n_samples": len(pcm),
        "seconds": args.seconds,
        "T_mel_frames": int(T),
        "A_audio_positions": int(A),
        "window_aftercnn": int(collect["window_aftercnn"]),
        "window_bounds": [int(b) for b in collect["window_bounds"]],
        "n_windows": len(collect["window_bounds"]) - 1,
        "dtype": "float32 reference",
        "notes": "block-diagonal bidirectional attention per official get_audio_cu_seqlens "
                 "(window_aftercnn = max(aftercnn_len)*(n_window_infer/(n_window*2)) = 104); "
                 "explicit SDPA attn_mask reproduces the varlen cu_seqlens semantics",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"[cross] T={T} A={A} windows={meta['n_windows']} bounds={collect['window_bounds']}")
    print(f"reference exported to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
