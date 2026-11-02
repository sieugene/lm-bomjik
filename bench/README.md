# bench — llama-server measurement harness

Python 3 standard library only, no dependencies. It always runs **one** server and
kills the previous one (`taskkill /F /IM llama-server.exe`).

## Quick start

```bat
cd D:\AI

:: one config, full test set (short, 8k, cached, 30k)
python bench\run.py --config bench\configs\43-qwen36-mtp-ncmoe38.json --tests full

:: a series of configs, sequential, with a summary table
python bench\suite.py --configs "bench/configs/4*.json" --tests short,mid

:: 128k validation on a 120k-token prompt (slow: 6-15 minutes)
python bench\run.py --config bench\configs\50-qwen36-mtp-128k-ncmoe40.json --tests ctx

:: reasoning-flag behavior through /v1/chat/completions
python bench\reasoning_probe.py

:: speculative-mode comparisons (DFlash2, ngram copy, cache reuse)
python bench\run.py --config bench\configs\73-dflash2-32k-nmax4.json --tests short,mid
python bench\copy_probe.py --config bench\configs\82-mtp2-ngram16-hits2.json
python bench\cache_reuse_probe.py
```

Output:

* `bench/results/<name>.json` — timings, VRAM, temperature, startup log lines;
* `bench/results/summary.md` — table for the latest series;
* `bench/results/all.jsonl` — append-only history;
* `bench/logs/<name>.log` — full llama-server log.

## Test presets (`run.py --tests`)

| preset | what it does |
|---|---|
| `short` | 400-token code and chat prompts, 192 generated tokens (`ignore_eos`) |
| `mid` | 8k prompt, 128 generated tokens |
| `long` | 30k prompt, 64 generated tokens |
| `full` | short + mid + cached + long |
| `ctx` | 120k prompt, 32 generated tokens |

`cached` sends two requests: the second adds ~15 tokens to the first and should reuse
the prefix from the prompt cache (this is how opencode works). It shows the real cost
of a new message in a long session.

## Config format

```json
{
  "name": "43-qwen36-mtp-ncmoe38",
  "notes": "what and why",
  "model": "D:\\AI\\Models\\other.gguf",
  "load_timeout": 900,
  "args": ["-m", "{model}", "--jinja", "..."]
}
```

`{model}` is replaced from the `model` field (or the default Q4_K_M model).

## Metrics

Taken from the `/completion` response `timings`:

* `prompt_per_second` — prefill speed;
* `predicted_per_second` — generation speed.

A background thread polls `nvidia-smi` (VRAM, temperature, SM clock, power) and free RAM;
peaks are stored in the result.

Laptop temperature and throttling affect the numbers noticeably: compare configs within
one series, never draw conclusions from a single run.
