#!/usr/bin/env python3
"""llama-server process control + measurement helpers (stdlib only)."""
import json, os, re, signal, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = ROOT / "llama-cuda"
SERVER = LLAMA_DIR / "llama-server.exe"
MODEL = ROOT / "Models" / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
LOGDIR = ROOT / "bench" / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)


def kill_all_servers():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # also any leftover from this harness
    time.sleep(0.5)


def gpu_mem_used_mib(full=False):
    try:
        q = "memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        vals = [x.strip() for x in out.split(",")]
        used, total, util = int(float(vals[0])), int(float(vals[1])), int(float(vals[2]))
        if full:
            return {"vram_used": used, "vram_total": total, "gpu_util": util,
                    "gpu_temp": int(float(vals[3])), "sm_clock": int(float(vals[4])),
                    "power_w": float(vals[5])}
        return used, total, util
    except Exception:
        return (None, None, None) if not full else {}


def free_ram_mib():
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys // 2**20, m.ullAvailPageFile // 2**20
    except Exception:
        return None, None


def http_json(url, payload=None, timeout=3600, method=None):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Server:
    def __init__(self, args, tag, port=8080, env=None):
        self.args = [str(a) for a in args]
        self.tag = tag
        self.port = port
        self.proc = None
        self.logpath = LOGDIR / f"{tag}.log"
        self.env = env
        self.load_s = None
        self.peak_vram = 0
        self.peak_rss = 0

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout=600):
        kill_all_servers()
        e = os.environ.copy()
        e.update({"LLAMA_ARG_PORT": str(self.port)})
        if self.env:
            e.update(self.env)
        log = open(self.logpath, "w", encoding="utf-8", errors="replace")
        t0 = time.time()
        self.proc = subprocess.Popen(
            [str(SERVER)] + self.args,
            cwd=str(LLAMA_DIR), stdout=log, stderr=subprocess.STDOUT,
            env=e, creationflags=subprocess.CREATE_NO_WINDOW)
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}, see {self.logpath}")
            try:
                st = http_json(self.base + "/health", timeout=3)
                if st.get("status") in ("ok", "no slot available"):
                    self.load_s = time.time() - t0
                    return self
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"server did not become ready in {timeout}s, see {self.logpath}")

    def sample(self):
        used, _, _ = gpu_mem_used_mib()
        if used:
            self.peak_vram = max(self.peak_vram, used)
        try:
            import subprocess as sp
            out = sp.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-Process -Id {self.proc.pid}).WorkingSet64"],
                         capture_output=True, text=True, timeout=15).stdout.strip()
            if out.isdigit():
                self.peak_rss = max(self.peak_rss, int(out) // 2**20)
        except Exception:
            pass
        return used

    def stop(self):
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        kill_all_servers()

    def props(self):
        return http_json(self.base + "/props", timeout=30)

    def tokenize(self, content):
        r = http_json(self.base + "/tokenize", {"content": content}, timeout=300)
        return r.get("tokens", [])

    def complete(self, prompt, n_predict=128, cache_prompt=False, temperature=0.0,
                 top_k=1, extra=None, timeout=7200):
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "cache_prompt": cache_prompt,
            "stream": False,
        }
        if extra:
            body.update(extra)
        return http_json(self.base + "/completion", body, timeout=timeout)

    def complete_chat(self, messages, n_predict=128, timeout=7200):
        body = {"messages": messages, "n_predict": n_predict, "temperature": 0.0,
                "top_k": 1, "stream": False, "cache_prompt": False}
        return http_json(self.base + "/v1/chat/completions", body, timeout=timeout)


def parse_startup(logpath):
    """Extract useful numbers printed by llama.cpp at load time."""
    txt = Path(logpath).read_text(encoding="utf-8", errors="replace")
    info = {}
    for pat, key in [
        (r"n_ctx\s*=\s*(\d+)", "n_ctx"),
        (r"n_batch\s*=\s*(\d+)", "n_batch"),
        (r"n_ubatch\s*=\s*(\d+)", "n_ubatch"),
        (r"n_threads\s*=\s*(\d+)", "n_threads"),
        (r"n_threads_batch\s*=\s*(\d+)", "n_threads_batch"),
        (r"n_parallel\s*=\s*(\d+)", "n_parallel"),
        (r"n_layer\s*=\s*(\d+)", "n_layer"),
    ]:
        m = re.search(pat, txt)
        if m:
            info[key] = int(m.group(1))
    for pat, key in [
        (r"(?:KV self|KV) size\s*=\s*([\d.]+)\s*MiB", "kv_mib"),
        (r"CPU_Mapped model buffer size\s*=\s*([\d.]+)\s*MiB", "cpu_mapped_mib"),
        (r"CPU_Mapped model buffer size\s*=\s*([\d.]+)\s*MiB", "cpu_mapped_mib2"),
        (r"CUDA0 model buffer size\s*=\s*([\d.]+)\s*MiB", "cuda_model_mib"),
        (r"CUDA0 compute buffer size\s*=\s*([\d.]+)\s*MiB", "cuda_compute_mib"),
        (r"CUDA0 KV buffer size\s*=\s*([\d.]+)\s*MiB", "cuda_kv_mib"),
        (r"CUDA0 (?:RS|host) buffer size\s*=\s*([\d.]+)\s*MiB", "cuda_rshost_mib"),
        (r"graph splits\s*=\s*(\d+)\s*/\s*(\d+)", "graph_splits"),
        (r"load time\s*=\s*([\d.]+)\s*ms", "load_ms"),
        (r"model size\s*=\s*([\d.]+)\s*MiB", "model_mib"),
    ]:
        m = re.search(pat, txt)
        if m:
            try:
                info[key] = float(m.group(1))
            except Exception:
                pass
    return info


def startup_mem_lines(logpath):
    txt = Path(logpath).read_text(encoding="utf-8", errors="replace")
    lines = [l for l in txt.splitlines() if re.search(
        r"(buffer size|KV self|KV buffer|model size|n_ctx|n_ubatch|n_batch|n_threads|offloaded|"
        r"graph splits|load time|warming up|n_cpu_moe|moe cache|type_k|type_v)", l, re.I)]
    return lines[-80:]
