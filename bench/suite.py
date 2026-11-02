#!/usr/bin/env python3
"""Run a list of configs sequentially (never parallel) and print a comparison table."""
import argparse, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as runner

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="glob, e.g. bench/configs/0*.json")
    ap.add_argument("--tests", default="full")
    ap.add_argument("--outdir", default="bench/results")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(a.configs))
    if not files:
        raise SystemExit(f"no configs match {a.configs}")
    preset = runner.TEST_PRESETS.get(a.tests)
    tests = preset if preset else [t for k in a.tests.split(",") for t in runner.TEST_PRESETS[k]]
    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        cfg = json.loads(Path(f).read_text(encoding="utf-8-sig"))
        out = outdir / f"{cfg['name']}.json"
        if a.skip_done and out.exists():
            print(f"skip {cfg['name']} (done)")
            r = json.loads(out.read_text(encoding="utf-8"))
        else:
            r = runner.run_config(cfg, tests, out)
        results.append(r)
        # incremental summary
        write_summary(results, outdir / "summary.md")
        with open(outdir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, ensure_ascii=False)


def write_summary(results, path):
    lines = ["# Benchmark summary", "",
             "| config | load s | test | prompt t | pp t/s | pp s | gen t/s | VRAM MiB | free RAM MiB | ok |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if not r.get("tests"):
            lines.append(f"| {r['config']} | {r.get('load_s','-')} | FAILED | - | - | - | - | "
                         f"{r.get('peak_vram_mib','-')} | {r.get('min_free_ram_mib','-')} | {r.get('ok')} |")
            continue
        for t in r["tests"]:
            if "prompt_tokens" in t:
                lines.append(f"| {r['config']} | {r.get('load_s','-')} | {t['test']} | {t['prompt_tokens']} | "
                             f"{t['prompt_tps']} | {t['prompt_ms']/1000:.1f} | {t['pred_tps']} | "
                             f"{t.get('peak_vram_mib','-')} | {t.get('min_free_ram_mib','-')} | {r.get('ok')} |")
            else:
                lines.append(f"| {r['config']} | {r.get('load_s','-')} | {t['test']} | "
                             f"{t.get('second_prompt_tokens')} | {t.get('second_prompt_tps')} | - | "
                             f"{t.get('second_pred_tps')} | {t.get('peak_vram_mib','-')} | "
                             f"{t.get('min_free_ram_mib','-')} | {r.get('ok')} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
