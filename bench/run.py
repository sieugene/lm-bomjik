#!/usr/bin/env python3
"""Run one llama-server config, execute benchmark prompts, save JSON result."""
import argparse, json, sys, threading, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama
import prompts

ROOT = llama.ROOT
RESULTS = ROOT / "bench" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


class Monitor(threading.Thread):
    def __init__(self, server):
        super().__init__(daemon=True)
        self.s = server
        self.stop_ev = threading.Event()
        self.peak_vram = 0
        self.min_free_ram = 10**9
        self.peak_rss = 0
        self.samples = 0
        self.max_temp = 0
        self.min_sm_clock = 10**9
        self.max_power = 0

    def run(self):
        while not self.stop_ev.is_set():
            g = llama.gpu_mem_used_mib(full=True)
            free, _ = llama.free_ram_mib()
            if g.get("vram_used"):
                self.peak_vram = max(self.peak_vram, g["vram_used"])
                self.max_temp = max(self.max_temp, g.get("gpu_temp", 0))
                self.max_power = max(self.max_power, g.get("power_w", 0))
                if g.get("sm_clock"):
                    self.min_sm_clock = min(self.min_sm_clock, g["sm_clock"])
            if free:
                self.min_free_ram = min(self.min_free_ram, free)
            self.s.peak_vram = self.peak_vram
            self.s.peak_rss = self.peak_rss
            self.samples += 1
            self.stop_ev.wait(1.5)

    def stop(self):
        self.stop_ev.set()
        self.join(timeout=10)


def get_prompt(server, target_tokens, kind="code", seed=1234):
    chars = int(target_tokens * 3.3)
    text = prompts.make_text(chars, seed=seed, kind=kind)
    for _ in range(5):
        n = len(server.tokenize(text))
        if abs(n - target_tokens) <= max(64, target_tokens * 0.04):
            break
        chars = max(200, int(chars * target_tokens / max(1, n)))
        text = prompts.make_text(chars, seed=seed, kind=kind)
    n = len(server.tokenize(text))
    return text, n


def run_test(server, name, target_tokens, n_predict, kind="code", seed=1234,
             extra=None, warm_cache=False):
    text, ntok = get_prompt(server, target_tokens, kind=kind, seed=seed)
    prompt = text + prompts.INSTRUCTIONS[kind]
    body_extra = {"ignore_eos": True}
    if extra:
        body_extra.update(extra)
    t0 = time.time()
    r = server.complete(prompt, n_predict=n_predict, cache_prompt=warm_cache, extra=body_extra)
    wall = time.time() - t0
    tm = r.get("timings", {})
    out = {
        "test": name,
        "target_tokens": target_tokens,
        "prompt_tokens": tm.get("prompt_n", ntok),
        "predicted_n": tm.get("predicted_n"),
        "prompt_ms": round(tm.get("prompt_ms", 0), 1),
        "prompt_tps": round(tm.get("prompt_per_second", 0), 2),
        "pred_ms": round(tm.get("predicted_ms", 0), 1),
        "pred_tps": round(tm.get("predicted_per_second", 0), 2),
        "wall_s": round(wall, 1),
        "peak_vram_mib": server.peak_vram,
        "min_free_ram_mib": MON.min_free_ram if MON else None,
        "gpu_temp": MON.max_temp if MON else None,
        "reply_head": (r.get("content") or "")[:120],
    }
    print(f"  {name:10s} prompt={out['prompt_tokens']:7d}t pp={out['prompt_tps']:8.2f} t/s "
          f"({out['prompt_ms']/1000:7.1f}s)  gen={out['pred_tps']:6.2f} t/s "
          f"({out['predicted_n']}t)  vram={out['peak_vram_mib']}MiB", flush=True)
    return out


def warmup(server):
    try:
        server.complete("Hello", n_predict=4, extra={"ignore_eos": True})
    except Exception as e:
        print(f"  warmup failed: {e}")


MON = None


def run_config(cfg, tests, out_path=None):
    global MON
    name = cfg["name"]
    args = [a.replace("{model}", str(cfg.get("model") or llama.MODEL)) for a in cfg["args"]]
    print(f"\n=== {name} ===\n  args: {' '.join(args)}", flush=True)
    server = llama.Server(args, tag=name, port=cfg.get("port", 8080),
                          env=cfg.get("env"))
    result = {"config": name, "args": args, "notes": cfg.get("notes", ""),
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": []}
    try:
        t0 = time.time()
        server.start(timeout=cfg.get("load_timeout", 900))
        result["load_s"] = round(server.load_s, 1)
        result["startup"] = llama.parse_startup(server.logpath)
        result["startup_log"] = llama.startup_mem_lines(server.logpath)
        print(f"  loaded in {server.load_s:.1f}s  {result['startup']}", flush=True)
        MON = Monitor(server)
        MON.start()
        warmup(server)
        for t in tests:
            spec = t
            if spec["type"] == "cached_second":
                continue
            if spec["type"] == "cached":
                # two requests: warm the prefix, then ask an incremental question
                text, ntok = get_prompt(server, spec["tokens"], seed=spec.get("seed", 7))
                instr = prompts.INSTRUCTIONS["code"]
                first = text + instr + "\n\nList only the function names."
                r1 = server.complete(first, n_predict=8, cache_prompt=True,
                                     extra={"ignore_eos": True})
                t1 = r1.get("timings", {})
                second = first + "\n\nNow describe the second function in one paragraph."
                r2 = server.complete(second, n_predict=64, cache_prompt=True,
                                     extra={"ignore_eos": True})
                t2 = r2.get("timings", {})
                out = {
                    "test": "cached", "target_tokens": spec["tokens"],
                    "first_prompt_tokens": t1.get("prompt_n"),
                    "first_prompt_tps": round(t1.get("prompt_per_second", 0), 2),
                    "second_prompt_tokens": t2.get("prompt_n"),
                    "second_prompt_tps": round(t2.get("prompt_per_second", 0), 2),
                    "second_pred_tps": round(t2.get("predicted_per_second", 0), 2),
                    "second_predicted_n": t2.get("predicted_n"),
                    "peak_vram_mib": server.peak_vram,
                    "min_free_ram_mib": MON.min_free_ram,
                }
                print(f"  cached      first pp={out['first_prompt_tokens']}t "
                      f"{out['first_prompt_tps']} t/s | second pp={out['second_prompt_tokens']}t "
                      f"{out['second_prompt_tps']} t/s | gen={out['second_pred_tps']} t/s", flush=True)
            else:
                out = run_test(server, spec["name"], spec["tokens"], spec["predict"],
                               kind=spec.get("kind", "code"), seed=spec.get("seed", 1234),
                               extra=spec.get("extra"))
            result["tests"].append(out)
        result["ok"] = True
    except Exception as e:
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(f"  FAILED: {result['error']}", flush=True)
        try:
            result["startup_log"] = llama.startup_mem_lines(server.logpath)
        except Exception:
            pass
    finally:
        if MON:
            MON.stop()
            result["peak_vram_mib"] = MON.peak_vram
            result["min_free_ram_mib"] = MON.min_free_ram
            result["gpu_max_temp"] = MON.max_temp
            result["gpu_min_sm_clock"] = MON.min_sm_clock if MON.min_sm_clock < 10**9 else None
            result["gpu_max_power"] = round(MON.max_power, 1)
            MON = None
        server.stop()
    result["total_s"] = round(time.time() - t0, 1)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(RESULTS / "all.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


TEST_PRESETS = {
    "short": [{"type": "gen", "name": "short", "tokens": 400, "predict": 192},
              {"type": "gen", "name": "chat", "tokens": 400, "predict": 192, "kind": "chat", "seed": 55}],
    "mid": [{"type": "gen", "name": "mid8k", "tokens": 8192, "predict": 128}],
    "long": [{"type": "gen", "name": "long30k", "tokens": 30000, "predict": 64}],
    "full": [{"type": "gen", "name": "short", "tokens": 400, "predict": 192},
             {"type": "gen", "name": "mid8k", "tokens": 8192, "predict": 128},
             {"type": "cached", "tokens": 8192},
             {"type": "gen", "name": "long30k", "tokens": 30000, "predict": 64}],
    "ctx": [{"type": "gen", "name": "ctx120k", "tokens": 120000, "predict": 32}],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config json")
    ap.add_argument("--tests", default="full", help="preset name or comma list of presets")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    if a.tests in TEST_PRESETS:
        tests = TEST_PRESETS[a.tests]
    else:
        tests = []
        for k in a.tests.split(","):
            tests += TEST_PRESETS[k.strip()]
    r = run_config(cfg, tests, a.out)
    print(json.dumps({k: v for k, v in r.items() if k not in ("startup", "startup_log")},
                     indent=1, ensure_ascii=False))
    sys.exit(0 if r.get("ok") else 1)


if __name__ == "__main__":
    main()
