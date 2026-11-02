#!/usr/bin/env python3
"""Compare speculative modes on a file-rewrite (copy-from-context) task."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama
import prompts

CFG = Path(__file__).resolve().parent / "configs" / "43-qwen36-mtp-ncmoe38.json"
MTP = r"D:\AI\Models\mtp-Qwen3.6-35B-A3B-Q4_0.gguf"


def base_args():
    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    args = [a.replace("{model}", str(llama.MODEL)) for a in cfg["args"]]
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("-md",):
            skip = True
            continue
        if a in ("--spec-type", "--spec-draft-n-max"):
            skip = True
            continue
        out.append(a)
    return out


def run(tag, extra):
    args = base_args()
    i = args.index("--host")
    args = args[:i] + extra + args[i:]
    s = llama.Server(args, tag=tag, port=8080)
    text = prompts.make_text(16_000, seed=99, kind="code")
    prompt = ("Repeat the following text exactly, character for character. "
              "Do not add commentary, do not summarize, output only the text.\n\n"
              + text + "\n\nOUTPUT:\n")
    try:
        s.start(timeout=900)
        r = s.complete(prompt, n_predict=512, cache_prompt=False)
        t = r["timings"]
        print(f"[{tag}] prompt={t['prompt_n']}t gen={t['predicted_n']}t "
              f"gen={t['predicted_per_second']:.1f} t/s "
              f"pp={t['prompt_per_second']:.1f} t/s", flush=True)
    finally:
        s.stop()


if __name__ == "__main__":
    run("no-spec", ["--spec-type", "none"])
    run("mtp2", ["-md", MTP, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])
    run("ngram16", ["--spec-type", "ngram-simple",
                    "--spec-ngram-simple-size-n", "12",
                    "--spec-ngram-simple-size-m", "16"])
    run("mtp+ngram", ["-md", MTP, "--spec-type", "draft-mtp,ngram-simple",
                      "--spec-draft-n-max", "2",
                      "--spec-ngram-simple-size-n", "12",
                      "--spec-ngram-simple-size-m", "16"])
