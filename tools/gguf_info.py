#!/usr/bin/env python3
"""Minimal GGUF metadata reader (no deps)."""
import struct, sys, json

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

class R:
    def __init__(self, f): self.f = f
    def read(self, n):
        b = self.f.read(n)
        if len(b) != n: raise EOFError
        return b
    def u32(self): return struct.unpack('<I', self.read(4))[0]
    def u64(self): return struct.unpack('<Q', self.read(8))[0]
    def i32(self): return struct.unpack('<i', self.read(4))[0]
    def i64(self): return struct.unpack('<q', self.read(8))[0]
    def f32(self): return struct.unpack('<f', self.read(4))[0]
    def f64(self): return struct.unpack('<d', self.read(8))[0]
    def string(self):
        n = self.u64()
        return self.read(n).decode('utf-8', 'replace')

def val(r, t):
    if t == T_U8: return r.read(1)[0]
    if t == T_I8: return struct.unpack('<b', r.read(1))[0]
    if t == T_U16: return struct.unpack('<H', r.read(2))[0]
    if t == T_I16: return struct.unpack('<h', r.read(2))[0]
    if t == T_U32: return r.u32()
    if t == T_I32: return r.i32()
    if t == T_F32: return r.f32()
    if t == T_BOOL: return bool(r.read(1)[0])
    if t == T_STR: return r.string()
    if t == T_U64: return r.u64()
    if t == T_I64: return r.i64()
    if t == T_F64: return r.f64()
    if t == T_ARR:
        et = r.u32(); n = r.u64()
        return [val(r, et) for _ in range(n)]
    raise ValueError(f'type {t}')

f = open(sys.argv[1], 'rb')
r = R(f)
magic = r.read(4)
if magic != b'GGUF': raise SystemExit('not gguf')
ver = r.u32(); n_tensors = r.u64(); n_kv = r.u64()
print(f'# gguf v{ver} tensors={n_tensors} kv={n_kv}')
meta = {}
for _ in range(n_kv):
    k = r.string(); t = r.u32(); v = val(r, t)
    meta[k] = v
    if isinstance(v, list) and len(v) > 8: v = f'<list len={len(v)}>'
    print(f'{k} = {v}')
print()
print('## tensors')
tinfo = []
for _ in range(n_tensors):
    name = r.string(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]
    tt = r.u32(); off = r.u64()
    tinfo.append((name, dims, tt, off))
    print(name, dims, 'type', tt)
json.dump(meta, open(sys.argv[1] + '.meta.json', 'w'), indent=1, default=str)
