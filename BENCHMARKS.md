# Benchmark journal

Date: 2026-11-02. Everything was measured on this machine with
`llama.cpp b12034 (0.8.0-dev, CUDA 12.4)`.

## Hardware

| | |
|---|---|
| CPU | Intel i7-11800H, 8 cores / 16 threads (llama.cpp uses 8) |
| RAM | 32 GB DDR4-3200, dual channel |
| GPU | RTX 3060 Laptop, 6 GB (6143 MiB), ~5.1 GB free with a browser open |
| OS | Windows, WDDM driver |

## Model

`Qwen3.6-35B-A3B-Q4_K_M.gguf`, 19.01 GiB (34.66B parameters):

* `qwen35moe`: 40 layers, **256 experts, 8 active**, shared expert;
* full attention every 4th layer (10 layers), the rest are gated delta-net (SSM);
* 16 heads / 2 KV heads, head dim 256, trained context 262 144;
* KV cache ~10.9 KiB/token (q8_0) → **32k ≈ 0.36 GB, 128k ≈ 1.43 GB**;
* experts: ~432 MB per layer (Q4_K_M), ~560 MB read from RAM per generated token.

This hybrid architecture is what makes a 128k context possible on 6 GB VRAM.

---

## 1. Baseline from `commands.txt`

`--n-cpu-moe 35`, `q8_0/q8_0`, 32k, `--load-mode none`.

| test | prompt | pp t/s | gen t/s |
|---|---|---|---|
| short code (458t) | 1.8 s | 255.9 | **34.0** |
| short chat (439t) | 1.5 s | 290.4 | 33.7 |
| 8k code | 27.1 s | 304.4 | 31.4 |
| 8k + cached follow-up | 14 t | 37.3 | 28.2 |
| 30k code | 95.0 s | 316.8 | 26.9 |

The user's reported 27–31 t/s reproduces. VRAM 5619 MiB — almost at the ceiling.

## 2. `n-cpu-moe`: each MoE layer on GPU ≈ +2.5%

`llama-bench`, Qwen3.6 Q4_K_M, q8_0/q8_0, 3 repetitions:

| n-cpu-moe | GPU MoE layers | pp512 | tg128 |
|---|---|---|---|
| 40 | 0 | 527.6 | 31.5 |
| 36 | 4 | 540.3 | 35.7 |
| 35 | 5 | 586.6 | 35.1 |
| **34** | **6** | **590.5** | **36.8** |
| 33 | 7 | 497.0 | 35.4 |
| 32 | 8 | 51.6 | **11.2 (spill)** |
| 31 | 9 | 340.0 | 5.1 (spill) |

**Rule:** exceeding VRAM by even ~100 MB makes WDDM move buffers to shared memory
and speed drops 3–6×. Keep headroom instead of fitting exactly.

## 3. Context depth (llama-bench, no MTP, ncmoe 35)

| depth | pp512 | tg128 |
|---|---|---|
| 0 | 429 | 39.7 |
| 16k | 554 | 33.5 |
| 49k | 468 | 26.9 |
| 98k | 407 | **21.1** |

So a 128k context is usable even without MTP.

## 4. MTP (multi-token prediction)

Official model plus the separate head `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` (1 GB),
`--spec-type draft-mtp --spec-draft-n-max 2`.

| config | VRAM | 32k short | 32k 8k | note |
|---|---|---|---|---|
| ncmoe 36 + MTP nmax2 | 5813 | 41.8 | 31.3 | tight VRAM, prefill drops |
| ncmoe 36 + MTP nmax3 | 5876 | 39.2 | 32.5 | worse than nmax2 |
| **ncmoe 38 + MTP nmax2** | **5191** | **43.3** | **38.6** | best 32k balance |
| ncmoe 40 + MTP nmax2 | 4489 | 40.1 | 34.5 | extra VRAM headroom |
| **ncmoe 40 + MTP, 128k** | **5738** | **37.9** | — | best 128k |
| ncmoe 40 + MTP + moe-cache 1G, 128k | 5844 | 14.9 | — | spill, do not use |

MTP acceptance: 0.73–0.85, mean accepted length 2.2–2.7 tokens. Gain +9–15%.

Integrated MTP model `unsloth/...-UD-Q3_K_XL` was **slower** (24.2 t/s):
its experts are IQ3_XXS, and i-quants are slower than K-quants on CPU.
Official Q4_K_M plus the MTP head is both faster and higher quality.

### 128k end-to-end validation

`bench/configs/50-qwen36-mtp-128k-ncmoe40.json`, a real 119,980-token prompt:

| metric | value |
|---|---|
| prompt tokens | 119 980 |
| prefill | 309.5 t/s (387.7 s) |
| generation | **25.0 t/s** (32 tokens) |
| peak VRAM | 5872 MiB |
| min free RAM | 4827 MiB |
| max GPU temp | 89 °C |
| max power | 102.5 W |

## 5. `--load-mode`: `none` vs `mmap`

Same configs, only the load mode changed.

| config | load | 32k short | 32k 8k | 128k short | VRAM |
|---|---|---|---|---|---|
| `none` ncmoe38 | 23.8 s | 43.3 | 38.6 | — | 5191 |
| `mmap` ncmoe38 | 12.9 s | 35.1 | 36.2 | — | 5257 |
| `none` ncmoe40 | 26.3 s | 37.9 | — | — | 5738 |
| `mmap` ncmoe40 | 4.7 s | 35.1 | 33.0 | 35.1 | 5816 |

`mmap` loads faster and is safer when free RAM is low (file pages can be evicted),
but generation is ~10% slower and the first prefill is much slower (96 t/s vs 294 t/s
while the page cache is cold). **Use `--load-mode none`** when at least ~22 GB of RAM
is available; use `mmap` as the fallback.

## 6. n-gram speculative decoding (no draft model)

| spec-type | 32k short | 8k |
|---|---|---|
| none (ncmoe34) | 36.8 | 31.4 |
| `ngram-simple` | 34.1 | 29.4 |
| `ngram-mod` | 32.2 | 19.8 |

No gain: generation is memory-bound, and the verify batch reads the same expert bytes.
Not enabled.

## 7. Threads and batches

`llama-bench`, Q4_K_M, ncmoe 34:

| threads | pp512 | tg128 |
|---|---|---|
| 6 | 656.5 | 36.3 |
| **8** | 646.8 | **36.3** |
| 10 | 649.7 | 35.4 |
| 12 | 635.4 | 34.8 |
| 16 | 637.6 | 28.3 |

| ubatch | pp512 | tg128 |
|---|---|---|
| 256 | 412.6 | 36.0 |
| **512** | **643.8** | 33.6 |
| 1024 | 631.8 | 33.0 |
| 2048 | 635.3 | 33.5 |

Keep `-t 8 -ub 512`. Six threads is nearly identical and is a good choice when the IDE
or opencode runs on the same machine.

## 8. Reasoning flags

Tested through `/v1/chat/completions` (not `/completion`, which bypasses the chat template):

| flag | content | reasoning_content | note |
|---|---|---|---|
| default (`auto`) | empty in 16 tokens | 381 chars | model spends tokens thinking first |
| `-rea off` | `OK` | none | thinking fully disabled |

Use `-rea off` for fast agent loops and tool calls. `--no-reasoning-preserve` keeps only
the last reasoning trace in history and saves context in long sessions. The Qwen3.6
template does not expose `reasoning_effort` levels, so `--reasoning-effort` has no effect
on this model.

## 9. Other models

### Qwen3.8-35B-A3B-Distill (unofficial distill, IQ4_XS 18.27 GiB)

32k:

| ncmoe | VRAM | tg128 (bench) | server short | server 8k |
|---|---|---|---|---|
| 40 | — | 34.4 | — | — |
| 35 | 4662 | — | 38.0 | 35.4 |
| 34 | 5388 | — | 35.0 | 26.6 (throttled run) |
| 31 | — | 42.7 | — | — |
| 30 | — | 10.4 (spill) | — | — |

128k:

| ncmoe | VRAM | short | chat |
|---|---|---|---|
| 36 | 5551 | 35.4 | 35.7 |
| 38 | 4955 | 34.1 | 35.0 |
| 40 | 4120 | 34.9 | 34.9 |

At 128k, too many GPU layers (ncmoe 33 and below) caused a spill and 8–9 t/s.
Slightly faster than 3.6 because it is smaller, but it is a third-party distill,
so the official 3.6 remains the primary model.

### Qwen3.6 UD-IQ4_XS (16.5 GiB) — do not use

Experts are mixed IQ3_S/IQ4_XS → 21–25 t/s, slower than Q4_K_M despite pp ~700 t/s.

### Qwen3.6 UD-Q3_K_XL with built-in MTP (16.04 GiB) — do not use

24.2 short / 21.6 @8k. Same IQ3 expert problem. The separate MTP head on Q4_K_M is better.

### Qwen3.8-27B (dense, UD-IQ3_XXS 10.9 GiB)

| -ngl | pp512 | tg128 |
|---|---|---|
| 0 | 308 | 1.76 |
| 8 | 261 | 1.94 |
| 12 | 382 | 2.09 |
| 16 | 393 | 2.18 |
| 20 | 403 | 2.34 |
| 24 | 414 | 2.42 |

**1.8–2.4 t/s.** A dense 27B cannot be used for agents on this hardware: every token
reads ~10 GB of weights from CPU. Even with a more aggressive quant and 28–30 GPU
layers the realistic ceiling is 3–4 t/s.

## 10. Final configs

| # | context | MTP | n-cpu-moe | load mode | VRAM | gen short | gen 8k | gen 120k |
|---|---|---|---|---|---|---|---|---|
| 1 | 32 768 | yes | 38 | none | 5.2 GB | **43.3** | 38.6 | — |
| 2 | 131 072 | yes | 40 | none | 5.9 GB | 37.9 | 35.0 | 25.0 |
| 3 | 32 768 | no | 35 | none | 5.6 GB | 34.0 | 31.4 | — |

These are the best runs, measured on a cool machine. A later re-check of config 1 in a hot
session gave 37.4 short / 34.5 chat / 33.9 @8k. Laptop thermals move the result by 10–15%;
compare configs inside one session.

## 11. NInfer-3090 mechanics: what transfers

Source: `ninfer-3090-master` (RTX 3090/5090, groupwise `.ninfer` artifacts, no CPU offload).
Its headline DFlash2/MTP3 numbers need 24 GB VRAM and cannot run on 6 GB. On a 5090 the same
35B-A3B decodes at 600–900 t/s because all experts stay in VRAM; here experts stream from
32 GB RAM at ~43 t/s. The portable parts were tested here.

### DFlash2 draft (`--spec-type draft-dflash`)

A compatible GGUF draft exists for exactly this target:
`aminya/Qwen3.6-35B-A3B-DFlash2-GGUF`, Q4_K_M 315 MB. llama.cpp b12034 loads it and runs
without errors. It is still slower than MTP:

| config | nmax | 32k short | 32k chat | 8k | VRAM |
|---|---|---|---|---|---|
| MTP (same session) | 2 | 37.4 | 34.5 | 33.9 | 5409 |
| DFlash2 nmax3 | 3 | 37.6 | 34.9 | — | 5435 |
| DFlash2 Q4_K_M nmax4 | 4 | 36.4 | 37.4 | 28.5 | 5530 |
| DFlash2 Q8_0 nmax4 | 4 | 36.0 | 36.4 | 28.7 | 5610 |
| DFlash2 nmax5 | 5 | 33.0 | 34.5 | — | 5590 |
| DFlash2 nmax7 | 7 | 28.1 | 28.3 | — | 5850 |

Draft acceptance was 0.46–0.56 with mean accepted length 2.8–4.9. The draft is a 6-layer
block model that reads target hidden states; on this 6 GB laptop its overhead and the larger
verify batches cost more than the accepted tokens save. The recommended Q8_0 file is no
better (28.7 @8k). At 8k DFlash2 is 16% slower than MTP. **Not used.**

### n-gram copy drafting (NInfer `--ngram-draft-tokens 15`)

llama.cpp `ngram-simple`, optionally mixed with MTP:
`--spec-ngram-simple-size-n 12 --spec-ngram-simple-size-m 16 --spec-ngram-simple-min-hits 2`.
Tested on a 6k-token prompt that asks the model to reproduce the text (a file-rewrite turn):

| spec | copy 512 tokens | normal 32k short / chat / 8k | 30k |
|---|---|---|---|
| none | 27.4 | — | — |
| MTP nmax2 | 40.9 | 37.4 / 34.5 / 33.9 | 27.7 |
| ngram-simple m16 | 42.6 | — | — |
| MTP + ngram m16, hits 1 | 42.9 | 37.5 / 35.4 / 33.4 | — |
| **MTP + ngram m16, hits 2** | **43.0** | 37.9 / 35.9 / 33.0 | 28.0 |

On copy turns ngram drafts 13–14 token spans at ~80% acceptance and gains ~5% over MTP.
On normal code it is neutral within run-to-run noise. This is the one NInfer idea that
transfers, but the win is workload-specific. It is not enabled by default; config
`82-mtp2-ngram16-hits2` is the optional copy-heavy profile.

### `--cache-reuse`

The server refuses it for this model:
`cache_reuse is not supported by this context, it will be disabled`.
The hybrid gated-delta-net/SSM state cannot be shifted like a pure attention KV cache.
An edited 22.7k prompt still re-prefilled 22,731 tokens (70 s) with and without
`--cache-reuse 256`. Normal append-only prefix reuse still works (4 tokens, 0.1 s).

### q4_0 V cache and one more GPU layer

`-ctk q8_0 -ctv q4_0` saves ~0.36 GB at 128k, enough for `--n-cpu-moe 39`. At 32k it gave
39.0 short / 34.7 @8k (vs 37.4 / 33.9), but at 120k it fell to 19.35 t/s. The reason is in
the log:

```
no FlashAttention vector kernel compiled for K/V types q8_0-q4_0,
converting K and V to f16 instead (slow)
```

So the stock binary cannot use q4_0 V efficiently at long context. q8_0/q8_0 cfg 2 stays
at 25.0 t/s @120k. A custom build with `GGML_CUDA_FA_QUANTS=q8_0-q4_0` might change the
trade-off, but the quality and dequant costs remain; not worth it here.

### Other NInfer ideas

* **MTP3**: already tested — 39.2/32.5 vs 41.8/31.3 for nmax2 at ncmoe36; at 120k it gave
  21.1 t/s vs 25.0 for nmax2 (5933 MiB). nmax2 wins.
* **Lossy long prefill**: dense-27B-only in NInfer and explicitly refused for the 35B-A3B;
  it also risks breaking exact file edits by coding agents.
* **Decode during prefill / layer slices**: needs multiple concurrent lanes; this setup runs
  one slot, and llama.cpp has no equivalent scheduler.
* **rk4v4 / paged KV / ReplaySSM / cuBLAS prefill**: engine-specific and tied to the 24 GB
  `.ninfer` artifact; not portable to llama.cpp on 6 GB.
* **DSpark / EAGLE-3 drafts**: both exist for this target, but DFlash2 already loses 16% at
  8k. Another draft changes acceptance, not the fact that every verified token still reads
  experts from RAM. Not pursued.

Conclusion: the three configs remain the best. DFlash2 does not transfer; ngram copy is a
small optional win for file-rewrite turns; q4_0 V cache is a trap with the stock binary.

## Reproducing

```bat
cd D:\AI
python bench\suite.py --configs "bench/configs/4[34]*.json" --tests short,mid
python bench\run.py --config bench\configs\50-qwen36-mtp-128k-ncmoe40.json --tests ctx
python bench\suite.py --configs "bench/configs/6[0-2]*.json" --tests short,mid
python bench\reasoning_probe.py

:: NInfer comparison
python bench\run.py --config bench\configs\73-dflash2-32k-nmax4.json --tests short,mid
python bench\copy_probe.py --config bench\configs\82-mtp2-ngram16-hits2.json
python bench\cache_reuse_probe.py
```

Results: `bench/results/`, logs: `bench/logs/`.
