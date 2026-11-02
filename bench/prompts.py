#!/usr/bin/env python3
"""Deterministic synthetic prompts for benchmarking (agent/code oriented)."""
import random

KEYWORDS = ["ingest", "resolver", "scheduler", "transport", "codec", "ledger", "router",
            "cache", "journal", "planner", "worker", "shard", "filter", "encoder",
            "registry", "session", "buffer", "matcher", "queue", "snapshot"]


def _ident(rng):
    return f"{rng.choice(KEYWORDS)}_{rng.choice(KEYWORDS)}_{rng.randrange(100000):05d}"


def code_block(rng, lines=40):
    out = []
    names = [_ident(rng) for _ in range(6)]
    out.append(f"// module: {names[0]}")
    out.append("#include <cstdint>")
    out.append("#include <vector>")
    out.append("#include <string>")
    out.append("")
    out.append(f"namespace {names[1]} {{")
    out.append(f"struct {names[2].title().replace('_','')} {{")
    out.append("    std::uint64_t id;")
    out.append("    std::vector<std::uint8_t> payload;")
    out.append("    std::string label;")
    out.append("};")
    out.append("")
    for i in range(lines):
        n = _ident(rng)
        k = _ident(rng)
        out.append(f"static inline std::uint64_t {n}(const {names[2].title().replace('_','')}& rec, std::uint64_t seed) {{")
        out.append(f"    std::uint64_t acc = seed ^ rec.id ^ 0x{rng.randrange(1<<32):08x}ULL;")
        out.append(f"    for (std::size_t i = 0; i < rec.payload.size(); ++i) {{")
        out.append(f"        acc = (acc * 0x{rng.randrange(1<<20, 1<<31):08x}ULL) + rec.payload[i];")
        out.append(f"        acc ^= acc >> {rng.choice([7, 11, 13, 17, 23])};")
        out.append("    }")
        out.append(f"    return acc + {rng.randrange(1000)};")
        out.append("}")
        out.append("")
        out.append(f"// note: {k} depends on {names[3]} and must stay deterministic")
        out.append(f"static const char* {k}_name = \"{k}\";")
        out.append("")
        if i % 3 == 2:
            out.append("/*")
            out.append(f" * {names[4]} contract: callers must never mutate rec.payload")
            out.append(f" * while {names[5]} holds a reference. See docs/{names[0]}.md.")
            out.append(" */")
            out.append("")
    out.append(f"}} // namespace {names[1]}")
    return "\n".join(out)


def chat_block(rng):
    topics = ["memory bandwidth", "KV cache", "expert routing", "quantization", "batch size",
              "flash attention", "speculative decoding", "thread affinity", "page cache"]
    t = rng.choice(topics)
    return (f"Note {rng.randrange(10000)}: when discussing {t}, remember that "
            f"memory bandwidth limits token generation speed. "
            f"We measured {rng.randrange(10, 90)} tokens per second with "
            f"{rng.randrange(2, 64)} layers and a {rng.randrange(4, 128)}k token context. "
            f"The result depends on quantization and the number of active experts.")


def make_text(target_chars, seed=1234, kind="code", code_ratio=0.85):
    rng = random.Random(seed)
    parts = []
    n = 0
    while n < target_chars:
        if kind == "code" or rng.random() < code_ratio:
            b = code_block(rng, lines=rng.choice([12, 24, 40]))
        else:
            b = chat_block(rng)
        parts.append(b)
        n += len(b) + 1
    return "\n\n".join(parts)[:target_chars]


INSTRUCTIONS = {
    "code": ("\n\nYou are given a large C++ codebase above. "
             "Task: add a function that validates every record, then list the three most "
             "likely bugs in the code above. Answer concisely, with code."),
    "chat": ("\n\nSummarize the notes above and answer: what limits token generation speed "
             "on a laptop with 6 GB VRAM and 32 GB RAM?"),
}


def build_prompt(kind="code", chars=12000, seed=1234):
    return make_text(chars, seed=seed, kind=kind) + INSTRUCTIONS[kind]
