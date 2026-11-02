# Qwen3.6-35B-A3B on RTX 3060 Laptop (6 GB VRAM, 32 GB RAM)

Local llama.cpp setup for coding agents (opencode or any OpenAI-compatible client).
The official **Qwen3.6-35B-A3B Q4_K_M** runs at **~43 tok/s** on a 32k context and
**~25 tok/s** at 120k tokens, with a full 128k context inside the 6 GB VRAM budget
(experts streamed from RAM).

## Results

All numbers are measured on this machine with llama.cpp b12034 (CUDA 12.4).
Generation is greedy (`temperature 0`) with `ignore_eos`; "short" is a ~450-token prompt,
"8k" and "30k"/"120k" are synthetic code contexts. VRAM includes the desktop.

| Model | Quant | Mode | Context | Prefill | Generation | VRAM | Verdict |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | Q4_K_M | MTP, n-cpu-moe 38 | 32k | 294 t/s | **43.3** short, 38.6 @8k | 5.2 GB | config 1 |
| Qwen3.6-35B-A3B | Q4_K_M | MTP, n-cpu-moe 40 | 128k | 309 t/s @120k | **25.0** @120k, 37.9 short | 5.9 GB | config 2 |
| Qwen3.6-35B-A3B | Q4_K_M | no MTP, n-cpu-moe 35 | 32k | 256 t/s | 34.0 short, 31.4 @8k, 26.9 @30k | 5.6 GB | config 3 |
| Qwen3.6-35B-A3B | Q4_K_M | no MTP, n-cpu-moe 34 | — | 590 t/s (pp512) | 36.8 (tg128) | — | best no-MTP |
| Qwen3.6-35B-A3B | Q4_K_M | MTP, n-cpu-moe 38, mmap | 32k | 267 t/s | 38.1 short, 36.2 @8k | 5.3 GB | ~10% slower |
| Qwen3.6-35B-A3B | Q4_K_M | MTP, n-cpu-moe 40, mmap | 128k | 239 t/s | 35.1 short, 33.0 @8k | 5.8 GB | ~10% slower |
| Qwen3.6-35B-A3B | UD-Q3_K_XL | built-in MTP, n-cpu-moe 35 | 32k | 249 t/s | 24.2 short, 21.6 @8k | 5.9 GB | slow IQ3 experts |
| Qwen3.6-35B-A3B | UD-IQ4_XS | no MTP, n-cpu-moe 35 | — | 741 t/s (pp512) | 24.7 (tg128) | — | slow IQ3 experts |
| Qwen3.8-35B-A3B-Distill | IQ4_XS | no MTP, n-cpu-moe 35 | 32k | 287 t/s | 38.0 short, 35.4 @8k | 4.7 GB | unofficial, faster |
| Qwen3.8-35B-A3B-Distill | IQ4_XS | no MTP, n-cpu-moe 36 | 128k | 313 t/s | 35.4 short | 5.6 GB | unofficial |
| Qwen3.8-27B (dense) | UD-IQ3_XXS | 24 layers on GPU | — | 414 t/s (pp512) | **2.4** (tg128) | — | unusable for agents |

The official Q4_K_M + the separate **MTP head** (`mtp-Qwen3.6-35B-A3B-Q4_0.gguf`, 1 GB)
is the fastest correct option: **+9–15%** generation for ~0.8 GB VRAM.
UD-IQ4_XS / UD-Q3_K_XL are faster in prefill but only ~24 tok/s in generation because
their IQ3 experts are slower on CPU.

## Quick start

```bat
:: 1. download the model and the MTP head (~20 GB)
download-models.bat

:: 2. pick a config (1-3)
start.bat

:: 3. stop the server
stop.bat
```

`llama-cuda\` is not committed. Place a llama.cpp Windows CUDA build (b12034 or newer)
there so that `llama-cuda\llama-server.exe` exists.

Server: `http://127.0.0.1:8080/v1`, model alias `qwen3.6-35b-a3b`.

## Configs (`start.bat`)

| # | Purpose | Context | MTP | n-cpu-moe | Generation | VRAM |
|---|---|---|---|---|---|---|
| 1 | Fast agent loop | 32 768 | yes | 38 | ~43 tok/s | 5.2 GB |
| 2 | Long context | 131 072 | yes | 40 | ~38 tok/s short, ~25 tok/s @120k | 5.9 GB |
| 3 | No MTP / compatibility | 32 768 | no | 35 | ~34 tok/s | 5.6 GB |

Common flags: `-fa on`, KV `q8_0/q8_0`, `-np 1`, `--jinja`, `--load-mode none`,
`-t 8`, `-ub 512`, `-fit off`.

Run only one `llama-server` at a time. The scripts kill the previous process before start.
A second server on a 6 GB GPU forces shared memory and cuts speed by 4×.

Extra llama-server flags can be appended:

```bat
:: disable thinking (much lower latency in agent loops)
start.bat 1 -rea off

:: reduce history tokens in long sessions
start.bat 1 --no-reasoning-preserve

:: safer but ~10% slower if RAM is tight
start.bat 1 --load-mode mmap
```

`-rea off` produces clean `content` with no `reasoning_content`. The default is `auto`
(thinking on for this template).

## opencode (V2)

Copy `opencode.example.jsonc` into your project as `opencode.jsonc`, or merge this provider:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "llama-local/qwen3.6-35b-a3b",
  "providers": {
    "llama-local": {
      "name": "llama.cpp local",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen3.6-35B-A3B (local)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 131072, "output": 8192 }
        }
      }
    }
  }
}
```

The model supports the OpenAI tool-calling format through llama.cpp's `--jinja` template.
If the client shows no reasoning, add
`"compatibility": { "reasoningField": "reasoning_content" }` to the model.

## Notes

* **Memory bandwidth is the limit.** Experts stay on CPU; every token reads ~560 MB
  of expert weights. Moving a MoE layer to the GPU adds ~2.5% speed.
* **VRAM headroom matters.** Exceeding VRAM by ~100 MB on WDDM moves buffers to shared
  memory and drops speed 3–6×; the configs keep 0.3–1 GB of headroom.
* 128k KV is only ~1.4 GB because full attention runs every 4th layer.
* A 120k-token prefill takes ~6.5 minutes (309 t/s); cached-prefix follow-ups are instant.
  The GPU reaches ~89 °C and clocks down during long prefills, so compare configs
  within one session.

Full journal: [BENCHMARKS.md](BENCHMARKS.md). Harness: [bench/README.md](bench/README.md).

## Layout

```
start.bat              config menu (1-3)
stop.bat               kill llama-server
download-models.bat    fetch Q4_K_M + MTP head with hf
opencode.example.jsonc ready-to-use opencode V2 provider
bench/                 Python benchmark harness (stdlib only)
bench/configs/         reproducible server configs
BENCHMARKS.md          full experiment log
```
