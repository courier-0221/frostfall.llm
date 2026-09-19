#!/usr/bin/env python3
# ============================================================================
# export_asr_reference.py —— asr/v0.1 Python 参考基线导出
#
# 生成 C++ 数值对齐工具（examples/asr/api_test/asr_align_v01）所需的参考数据：
#   PCM / Mel / Prompt / IDs / 融合 Embedding / 逐层 hidden / logits / 生成 token。
# 自动语言与指定语言（Chinese）分别导出到 work/asr_ref/auto 与 work/asr_ref/lang。
#
# 实现对齐策略：
#   - Mel：官方 WhisperFeatureExtractor（transformers 4.57.6，与本模型 preprocessor 配置
#     同源），C++ 端 Log-Mel 属于 asr/v0.3，本版不涉及；
#   - 音频塔：按官方 modeling_qwen3_asr.py（commit 7c6daf77）逐行用 torch fp32 复现
#     forward（分块卷积 / 正弦位置编码 / 18 层编码器 / ln_post / proj1 / proj2）；
#   - Decoder：transformers Qwen3ForCausalLM（ASR 文本子模型与 Qwen3 同构；
#     MRoPE 三轴相同退化为普通 1D RoPE，见架构文档 §6.3），fp32 eager attention；
#   - 融合：inputs_embeds = embed(input_ids)，audio_pad 位置替换为音频特征（非相加）；
#   - 生成：贪心 argmax，EOS 集合 [151645, 151643]（generation_config.json）。
#
# 参考精度为 fp32，高于官方推理用的 bf16；容差在 C++ 对齐报告里按实测建立。
#
# 用法：
#   python tools/export_asr_reference.py <模型目录> [--out-dir work/asr_ref]
#       [--wav path] [--max-new-tokens N]
#   --wav：用真人语音 wav 替代合成音频（mono/stereo，8/16/32-bit PCM；
#          非 16k 采样率线性插值重采样到 16k）；不传则用确定性合成音频。
# ============================================================================

import argparse
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

SR = 16000
AUDIO_SECONDS = 5.0          # 5 秒 -> 约 500 个有效 Mel 帧 = 5 个完整 CNN 块，A=65，单注意力窗口内
MAX_NEW_TOKENS = 32
TOPK = 10


# ----------------------------------------------------------------------------
# 1. 测试音频（合成语音样信号；v0.1 验收目标是数值对齐，不是识别质量）
# ----------------------------------------------------------------------------

def synth_audio(n_samples: int) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / SR
    # 基频 110~220Hz 慢扫描 + 2/3 次谐波，模拟有律动的浊音段
    f0 = 165.0 + 55.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    # 音节化幅度包络（4 Hz）+ 轻噪声，避免长静音触发空转写
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.7 * t)
    rng = np.random.default_rng(20260919)
    sig = sig * env + 0.01 * rng.standard_normal(n_samples)
    peak = np.abs(sig).max()
    return (sig / max(peak, 1e-9) * 0.95).astype(np.float32)


def save_wav(path: Path, pcm: np.ndarray):
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.tobytes())


def load_wav(path: Path) -> np.ndarray:
    """读 WAV（PCM 8/16/32-bit），单声道化并重采样到 SR=16k，返回 float32 [-1,1]。"""
    with wave.open(str(path), "rb") as w:
        nch, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise SystemExit(f"{path}: {width*8}-bit PCM not supported (need 8/16/32-bit; "
                         "24-bit / mp3 请先转 16-bit PCM wav)")
    if nch > 1:  # 立体声取平均
        data = data.reshape(-1, nch).mean(axis=1)
    if sr != SR:  # 线性插值重采样（兑底路径；导出端直接选 16k 质量最佳）
        n_out = int(round(len(data) * SR / sr))
        idx = np.linspace(0.0, len(data) - 1.0, num=n_out)
        data = np.interp(idx, np.arange(len(data), dtype=np.float64), data)
    return data.astype(np.float32)


# ----------------------------------------------------------------------------
# 2. 音频塔 forward（逐行对齐官方 Qwen3ASRAudioEncoder.forward，fp32）
# ----------------------------------------------------------------------------

def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    # 官方 _get_feat_extract_output_lengths：A(T) = 13*(T//100) + ceil((T%100)/8)
    leave = input_lengths % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> torch.Tensor:
    # 官方 SinusoidsPositionEmbedding（非持久 buffer，公式生成）
    inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-inc * torch.arange(channels // 2).float())
    scaled = torch.arange(length)[:, None].float() * inv[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


def gelu(x):
    return torch.nn.functional.gelu(x)


def audio_tower_forward(mel: torch.Tensor, feature_len: int, w: dict, acfg: dict):
    """mel: [128, T] 有效帧（fp32）。返回 [A, 1024] 音频特征。"""
    d_model = acfg["d_model"]            # 896
    n_window = acfg["n_window"]          # 50 -> CNN 分块 100 帧
    n_mels = acfg["num_mel_bins"]        # 128
    n_head = acfg["encoder_attention_heads"]
    scale_embedding = acfg.get("scale_embedding", False)
    embed_scale = math.sqrt(d_model) if scale_embedding else 1.0

    input_features = mel.T * embed_scale  # [T,128]（scale_embedding=false 时不缩放）
    feature_lens = torch.tensor([feature_len], dtype=torch.long)

    # ---- 按 100 帧分块 + 尾块处理（官方 forward 前 12 行）----
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.tensor([n_window * 2] * int(chunk_num.sum()), dtype=torch.long)
    tail_idx = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_idx] = feature_lens % (n_window * 2)
    chunk_lengths[chunk_lengths == 0] = n_window * 2

    chunk_list = input_features.split(chunk_lengths.tolist(), dim=0)
    padded = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    aftercnn_lens = _get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(int(l), dtype=torch.bool) for l in aftercnn_lens], batch_first=True)

    # ---- 3 层 Conv2d + GELU ----
    x = padded.unsqueeze(1)  # [N,1,128,T_pad]
    x = gelu(torch.nn.functional.conv2d(x, w["conv2d1.weight"], w["conv2d1.bias"],
                                        stride=2, padding=1))
    x = gelu(torch.nn.functional.conv2d(x, w["conv2d2.weight"], w["conv2d2.bias"],
                                        stride=2, padding=1))
    x = gelu(torch.nn.functional.conv2d(x, w["conv2d3.weight"], w["conv2d3.bias"],
                                        stride=2, padding=1))
    b, c, f, t = x.size()
    # 展平顺序：channel 再 frequency（官方 permute(0,3,1,2).view(b,t,c*f)）
    x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    x = torch.nn.functional.linear(x, w["conv_out.weight"])  # conv_out 无 bias

    # ---- 局部正弦位置编码（每个块从 0 重新开始；加在 padded 长度上）----
    pos = sinusoids(acfg["max_source_positions"], d_model)[: x.shape[1], :].unsqueeze(0)
    x = x + pos

    # ---- 去掉 padding 位置 -> [A,896] ----
    hidden = x[padded_mask_after_cnn]  # [A,896]
    n_aftercnn = int(padded_mask_after_cnn.shape[-1])
    # v0.1 约束：单注意力窗口内的短音频（A <= 13*(800/100)=104），此时块内全局双向注意力
    assert n_aftercnn <= (acfg["n_window_infer"] // (n_window * 2)) * 13, \
        f"audio too long for v0.1 single-window baseline: A={n_aftercnn}"

    # ---- 18 层音频 Transformer（LayerNorm -> MHA -> 残差 -> LN -> fc1 -> GELU -> fc2 -> 残差）----
    eps = 1e-5
    n_states = hidden.shape[0]
    for il in range(acfg["encoder_layers"]):
        p = f"layers.{il}."
        residual = hidden
        h = torch.nn.functional.layer_norm(hidden, (d_model,), w[p + "self_attn_layer_norm.weight"],
                                           w[p + "self_attn_layer_norm.bias"], eps)
        q = torch.nn.functional.linear(h, w[p + "self_attn.q_proj.weight"], w[p + "self_attn.q_proj.bias"])
        k = torch.nn.functional.linear(h, w[p + "self_attn.k_proj.weight"], w[p + "self_attn.k_proj.bias"])
        v = torch.nn.functional.linear(h, w[p + "self_attn.v_proj.weight"], w[p + "self_attn.v_proj.bias"])
        # [A,896] -> [1, n_head, A, 64]；非因果、无 mask（块内双向），scale=1/sqrt(64)
        q = q.view(n_states, n_head, -1).transpose(0, 1).unsqueeze(0)
        k = k.view(n_states, n_head, -1).transpose(0, 1).unsqueeze(0)
        v = v.view(n_states, n_head, -1).transpose(0, 1).unsqueeze(0)
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=(d_model // n_head) ** -0.5)
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

    # ---- 输出投影 ----
    hidden = torch.nn.functional.layer_norm(hidden, (d_model,), w["ln_post.weight"], w["ln_post.bias"], eps)
    hidden = torch.nn.functional.linear(hidden, w["proj1.weight"], w["proj1.bias"])
    hidden = gelu(hidden)
    hidden = torch.nn.functional.linear(hidden, w["proj2.weight"], w["proj2.bias"])  # [A,1024]
    return hidden


# ----------------------------------------------------------------------------
# 3. ASR prompt 构造（官方 chat_template，单音频）
# ----------------------------------------------------------------------------

def build_prompt(mode: str, audio_pad_count: int, context: str = "") -> str:
    pad = "<|audio_pad|>" * audio_pad_count
    prompt = (f"<|im_start|>system\n{context}<|im_end|>\n"
              f"<|im_start|>user\n<|audio_start|>{pad}<|audio_end|><|im_end|>\n"
              f"<|im_start|>assistant\n")
    if mode == "lang":
        prompt += "language Chinese<asr_text>"
    return prompt


# ----------------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------------

def export_case(case: str, pcm: np.ndarray, mel_extractor, tok, weights, cfg,
                audio_cfg, text_cfg, out_dir: Path, audio_token_id: int,
                max_new_tokens: int = MAX_NEW_TOKENS):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    out_dir.mkdir(parents=True, exist_ok=True)
    # ---- Mel ----
    feat = mel_extractor(pcm, sampling_rate=SR, return_tensors="pt", padding=True,
                         return_attention_mask=True)
    input_features = feat.input_features[0]          # [128, T_pad]
    # transformers 4.57.6 的 WhisperFeatureExtractor 返回键名为 attention_mask（即官方
# Processor 里的 feature_attention_mask）
    feat_mask = feat.attention_mask[0]               # [T_pad]
    T = int(feat_mask.sum())
    mel_valid = input_features[:, :T].float()        # [128,T]
    assert mel_valid.shape[0] == audio_cfg["num_mel_bins"]

    # ---- 音频塔 ----
    # 剕离 thinker.audio_tower. 前缀，并转 fp32（源权重 BF16）
    audio_w = {k[len("thinker.audio_tower."):]: v.float()
               for k, v in weights.items() if k.startswith("thinker.audio_tower.")}
    audio_features = audio_tower_forward(mel_valid, T, audio_w, audio_cfg)  # [A,1024]
    A = audio_features.shape[0]
    # A(T) = 13*floor(T/100) + ceil((T%100)/8)，即官方整数公式
    A_expect = 13 * (T // 100) + ((T % 100) + 7) // 8
    assert A == A_expect, f"A mismatch: got {A}, expect {A_expect} (T={T})"

    # ---- prompt / tokenize ----
    prompt = build_prompt(case, A)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    S = len(ids)
    audio_positions = [i for i, t in enumerate(ids) if t == audio_token_id]
    assert len(audio_positions) == A, \
        f"audio_pad count {len(audio_positions)} != A {A}"

    # ---- 融合（替换，非相加）----
    tok_embd = weights["thinker.model.embed_tokens.weight"]
    inputs_embeds = tok_embd[torch.tensor(ids, dtype=torch.long)].float()  # [S,1024] fp32
    inputs_embeds[audio_positions] = audio_features

    # ---- Decoder（Qwen3ForCausalLM，重映射权重，fp32 eager）----
    qcfg = Qwen3Config(
        vocab_size=text_cfg["vocab_size"], hidden_size=text_cfg["hidden_size"],
        intermediate_size=text_cfg["intermediate_size"],
        num_hidden_layers=text_cfg["num_hidden_layers"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        head_dim=text_cfg["head_dim"], rms_norm_eps=text_cfg["rms_norm_eps"],
        rope_theta=text_cfg["rope_theta"], tie_word_embeddings=False,
        max_position_embeddings=4096, use_cache=True,
    )
    qcfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(qcfg)

    sd = {}
    for k in weights:
        if k.startswith("thinker.model."):
            sd["model." + k[len("thinker.model."):]] = weights[k].float()
        elif k == "thinker.lm_head.weight":
            sd["lm_head.weight"] = weights[k].float()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # tie_word_embeddings=False：embed_tokens 与 lm_head 作为两份独立矩阵加载，
    # 与文件内实际存储的两份数据一致（官方配置 tie=true 但两份数据区间不同）。
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not missing, f"missing keys: {missing[:5]}"
    model.eval()

    with torch.no_grad():
        out = model(inputs_embeds=inputs_embeds.unsqueeze(0), use_cache=True,
                    output_hidden_states=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()  # [vocab]
        # hidden_states: [0]=embed 输出,[i]=第 i 层输出（norm 前）,[28]=final norm 后
        layer_hidden = torch.stack([h[0].float() for h in out.hidden_states])  # [29,S,1024]

        # ---- 贪心 decode（teacher forcing + 自由生成同源）----
        eos_set = [151645, 151643]
        gen_ids, gen_logits = [], []
        cur = logits.argmax().item()
        pos = S
        while cur not in eos_set and len(gen_ids) < max_new_tokens:
            gen_ids.append(cur)
            gen_logits.append(logits.clone())
            o = model(input_ids=torch.tensor([[cur]]), past_key_values=past,
                      use_cache=True, position_ids=torch.tensor([[pos]]))
            past = o.past_key_values
            logits = o.logits[0, -1].float()
            cur = logits.argmax().item()
            pos += 1
        # 最后一步（产出 EOS 或到达上限的那一步）也保存，用于 teacher forcing 末步对齐
        gen_ids.append(cur)
        gen_logits.append(logits.clone())

    decode_ids = np.array(gen_ids, dtype=np.int32)
    decode_logits = torch.stack(gen_logits).numpy().astype(np.float32)  # [n_gen, vocab]
    n_gen = len(gen_ids)

    # ---- 保存 ----
    np.save(out_dir / "pcm.npy", pcm.astype(np.float32))
    np.save(out_dir / "mel.npy", mel_valid.numpy().astype(np.float32))
    np.save(out_dir / "ids.npy", np.array(ids, dtype=np.int32))
    np.save(out_dir / "audio_positions.npy", np.array(audio_positions, dtype=np.int32))
    np.save(out_dir / "embd.npy", inputs_embeds.numpy().astype(np.float32))  # [S,1024] 行=token
    np.save(out_dir / "layer_hidden.npy", layer_hidden.numpy().astype(np.float32))
    np.save(out_dir / "prefill_logits.npy", logits.numpy().astype(np.float32))
    np.save(out_dir / "prefill_topk_ids.npy",
            torch.topk(out.logits[0, -1].float(), TOPK).indices.numpy().astype(np.int32))
    np.save(out_dir / "prefill_topk_vals.npy",
            torch.topk(out.logits[0, -1].float(), TOPK).values.numpy().astype(np.float32))
    np.save(out_dir / "decode_ids.npy", decode_ids)
    np.save(out_dir / "decode_logits.npy", decode_logits)
    save_wav(out_dir / "test.wav", pcm)

    meta = {
        "case": case,
        "prompt": prompt,
        "sample_rate": SR,
        "n_samples": len(pcm),
        "T_mel_frames": T,
        "A_audio_positions": A,
        "S_prefill_tokens": S,
        "n_gen": n_gen,
        "max_new_tokens": max_new_tokens,
        "audio_token_id": audio_token_id,
        "eos_token_ids": [151645, 151643],
        "topk": TOPK,
        "mel": {"n_fft": 400, "hop_length": 160, "n_mels": 128, "fmin": 0.0, "fmax": 8000.0},
        "dtype": "float32 reference (official inference uses bfloat16)",
        "notes": "audio tower forward is a line-aligned fp32 reimplementation of "
                 "modeling_qwen3_asr.py@7c6daf77; decoder is transformers Qwen3ForCausalLM",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{case}] T={T} A={A} S={S} n_gen={n_gen} "
          f"gen_preview={tok.decode(gen_ids[:8], skip_special_tokens=False)!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export ASR v0.1 numerical reference data")
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("work/asr_ref"))
    ap.add_argument("--wav", type=Path, default=None,
                    help="real speech wav (mono/stereo, 8/16/32-bit PCM); "
                         "auto-resampled to 16k; default: synthetic audio")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                    help="greedy decode cap (real speech may need >32)")
    args = ap.parse_args()
    if args.wav and args.out_dir == Path("work/asr_ref"):
        # 防呆：真人 wav 的参考数据默认另存，避免覆盖合成音频基线
        args.out_dir = Path("work/asr_ref_wav")
        print("note: --wav given; default out-dir switched to work/asr_ref_wav "
              "(synthetic baseline in work/asr_ref is kept)")

    cfg = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    thinker = cfg["thinker_config"]
    audio_cfg, text_cfg = thinker["audio_config"], thinker["text_config"]
    audio_token_id = thinker["audio_token_id"]

    print("loading safetensors ...")
    weights = load_file(str(args.model_dir / "model.safetensors"))

    from transformers import WhisperFeatureExtractor, AutoTokenizer
    mel_extractor = WhisperFeatureExtractor.from_pretrained(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    if args.wav:
        pcm = load_wav(args.wav)
        print(f"loaded real wav: {args.wav} -> {len(pcm)} samples "
              f"({len(pcm) / SR:.2f}s @ {SR} Hz)")
    else:
        pcm = synth_audio(int(SR * AUDIO_SECONDS))

    export_case("auto", pcm, mel_extractor, tok, weights, cfg,
                audio_cfg, text_cfg, args.out_dir / "auto", audio_token_id,
                max_new_tokens=args.max_new_tokens)
    export_case("lang", pcm, mel_extractor, tok, weights, cfg,
                audio_cfg, text_cfg, args.out_dir / "lang", audio_token_id,
                max_new_tokens=args.max_new_tokens)
    print(f"reference data exported to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
