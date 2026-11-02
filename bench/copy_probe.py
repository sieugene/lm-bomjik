#!/usr/bin/env python3
"""Run one file-rewrite (copy-from-context) request against a saved config."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama
import prompts

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--tag", default=None)
p.add_argument("--chars", type=int, default=16_000)
p.add_argument("--n-predict", type=int, default=512)
a = p.parse_args()

cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
args = [x.replace("{model}", str(llama.MODEL)) for x in cfg["args"]]
tag = a.tag or cfg["name"]
s = llama.Server(args, tag=tag, port=8080)
text = prompts.make_text(a.chars, seed=99, kind="code")
prompt = ("Repeat the following text exactly, character for character. "
          "Do not add commentary, do not summarize, output only the text.\n\n"
          + text + "\n\nOUTPUT:\n")
try:
    s.start(timeout=900)
    r = s.complete(prompt, n_predict=a.n_predict, cache_prompt=False)
    t = r["timings"]
    print(f"[{tag}] prompt={t['prompt_n']}t gen={t['predicted_n']}t "
          f"gen={t['predicted_per_second']:.1f} t/s pp={t['prompt_per_second']:.1f} t/s")
finally:
    s.stop()
