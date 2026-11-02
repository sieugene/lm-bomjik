#!/usr/bin/env python3
"""Measure whether --cache-reuse avoids re-prefilling after a mid-prompt edit."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama
import prompts

CFG = Path(__file__).resolve().parent / "configs" / "43-qwen36-mtp-ncmoe38.json"
MARK = "\n// INSERTED BY A LATER TOOL CALL: this line changes the middle of the file.\n"


def run(tag, extra):
    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    args = [a.replace("{model}", str(llama.MODEL)) for a in cfg["args"]]
    i = args.index("--host")
    args = args[:i] + extra + args[i:]
    s = llama.Server(args, tag=tag, port=8080)
    text = prompts.make_text(60_000, seed=77, kind="code")
    at = int(len(text) * 0.30)
    edited = text[:at] + MARK + text[at:]
    rows = []
    try:
        s.start(timeout=900)
        for name, prompt in (("cold", text + prompts.INSTRUCTIONS["code"]),
                             ("same", text + prompts.INSTRUCTIONS["code"]),
                             ("edited", edited + prompts.INSTRUCTIONS["code"])):
            t0 = time.time()
            r = s.complete(prompt, n_predict=4, cache_prompt=True)
            t = r["timings"]
            rows.append((name, t["prompt_n"], t["prompt_ms"], t["prompt_per_second"],
                         t["predicted_per_second"], time.time() - t0))
            print(f"[{tag}] {name:7s} prompt_n={t['prompt_n']:6d} "
                  f"pp={t['prompt_per_second']:7.1f} t/s ({t['prompt_ms']/1000:6.1f}s) "
                  f"gen={t['predicted_per_second']:5.1f} t/s", flush=True)
    finally:
        s.stop()
    return rows


if __name__ == "__main__":
    run("no-reuse", [])
    run("reuse256", ["--cache-reuse", "256"])
