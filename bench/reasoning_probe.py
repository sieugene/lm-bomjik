#!/usr/bin/env python3
"""Check how llama-server reasoning flags affect /v1/chat/completions."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama

CFG = Path(__file__).resolve().parent / "configs" / "43-qwen36-mtp-ncmoe38.json"


def probe(tag, extra):
    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    args = [a.replace("{model}", str(llama.MODEL)) for a in cfg["args"]]
    args = [a for a in args if a not in ("-fit", "off")]
    # config ends with: --host ... --port ... -fit off; append variant flags before host
    i = args.index("--host")
    args = args[:i] + extra + args[i:]
    s = llama.Server(args, tag=tag, port=8080)
    try:
        s.start(timeout=900)
        r = s.complete_chat([{"role": "user",
                              "content": "Reply with the single word OK and nothing else."}],
                            n_predict=96)
        msg = r["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        print(f"[{tag}] content={content.strip()[:70]!r} "
              f"reasoning_len={len(reasoning)} reasoning={reasoning.strip()[:70]!r}")
    finally:
        s.stop()


if __name__ == "__main__":
    probe("rea-auto", [])
    probe("rea-off", ["-rea", "off"])
