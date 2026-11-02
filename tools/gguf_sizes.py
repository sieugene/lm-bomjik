#!/usr/bin/env python3
"""Summarize GGUF tensor sizes per layer/type."""
import struct, sys, re
from collections import defaultdict

TYPES = {0: ('F32', 4), 1: ('F16', 2), 2: ('Q4_0', 18/32), 3: ('Q4_1', 20/32), 6: ('Q5_0', 22/32),
         7: ('Q5_1', 24/32), 8: ('Q8_0', 34/32), 9: ('Q8_1', 36/32), 10: ('Q2_K', 84/256),
         11: ('Q3_K', 110/256), 12: ('Q4_K', 144/256), 13: ('Q5_K', 176/256), 14: ('Q6_K', 210/256),
         15: ('Q8_K', 292/256), 16: ('IQ2_XXS', 66/256), 17: ('IQ2_XS', 74/256), 18: ('IQ3_XXS', 98/256),
         19: ('IQ1_S', 50/256), 20: ('IQ4_NL', 18/32), 21: ('IQ3_S', 110/256), 22: ('IQ2_S', 82/256),
         23: ('IQ4_XS', 136/256), 24: ('I8', 1), 25: ('I16', 2), 26: ('I32', 4), 27: ('I64', 8),
         28: ('F64', 8), 29: ('IQ1_M', 56/256), 30: ('BF16', 2), 34: ('TQ1_0', 54/256), 35: ('TQ2_0', 66/256)}

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

f = open(sys.argv[1], 'rb')
assert f.read(4) == b'GGUF'
ver = struct.unpack('<I', f.read(4))[0]
nt = struct.unpack('<Q', f.read(8))[0]
nkv = struct.unpack('<Q', f.read(8))[0]

def skip_val(f, t):
    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sizes: f.read(sizes[t])
    elif t == 8: read_str(f)
    elif t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n): skip_val(f, et)
    else: raise ValueError(t)

for _ in range(nkv):
    read_str(f); t = struct.unpack('<I', f.read(4))[0]; skip_val(f, t)

tensors = []
for _ in range(nt):
    name = read_str(f)
    nd = struct.unpack('<I', f.read(4))[0]
    dims = list(struct.unpack('<%dQ' % nd, f.read(8*nd)))
    tt = struct.unpack('<I', f.read(4))[0]
    off = struct.unpack('<Q', f.read(8))[0]
    n = 1
    for d in dims: n *= d
    bpw = TYPES.get(tt, ('?', 1))[1]
    tensors.append((name, dims, TYPES.get(tt, (str(tt),))[0], n * bpw, off))

total = sum(t[3] for t in tensors)
print(f'total tensor bytes: {total/2**30:.2f} GiB, tensors={nt}')
bytype = defaultdict(float)
bylayer = defaultdict(float)
for name, dims, ty, size, off in tensors:
    bytype[ty] += size
    m = re.match(r'blk\.(\d+)\.', name)
    key = f'blk.{m.group(1)}' if m else 'other'
    bylayer[key] += size

print('\nby type (GiB):')
for k, v in sorted(bytype.items(), key=lambda x: -x[1]): print(f'  {k:8s} {v/2**30:7.3f}')
print('\nper layer (MiB):')
for k in sorted(bylayer, key=lambda x: (x != 'other', int(x.split(".")[1]) if x.startswith('blk.') else 0)):
    print(f'  {k:8s} {bylayer[k]/2**20:7.1f}')
print('\nnon-expert tensors per layer (MiB):')
for name, dims, ty, size, off in tensors:
    if 'expert' not in name and name.startswith('blk.0.'):
        print(f'  {name:40s} {ty:6s} {size/2**20:7.2f}')
print('\ntop-level tensors (MiB):')
for name, dims, ty, size, off in tensors:
    if not name.startswith('blk.'):
        print(f'  {name:40s} {ty:6s} {size/2**20:7.2f}')
print('\nexpert tensor total per layer (MiB):')
for l in range(0, 100):
    s = sum(t[3] for t in tensors if t[0].startswith(f'blk.{l}.') and 'expert' in t[0])
    if s: print(f'  blk.{l}: {s/2**20:.1f}')
